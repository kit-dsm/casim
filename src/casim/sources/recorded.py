"""Replay a picking day a WMS recorded: its lines become CASIM's state, plan and events.

The input is one row per recorded line (``RecordedRow``): picker, order,
article, quantity, pick node, start and end in seconds of the day. From it,

- the state: one order per WMS order with one position per line, available
  at the shift start; dedicated storage of each article at its recorded pick
  nodes; one picker per recorded picker, present from the shift start, with
  a cart for one order;
- the recorded tours: one tour per recorded order (the WMS order is the
  picking tour), its lines in recorded order, the tours of a picker in the
  order of their first line. A picker who worked on several orders at once
  (overlapping line intervals) does them one after another; the replay
  invents no batches. The tours go to the planner as
  ``information.recorded_plan``;
- the timing, in one of two modes:
  - chained (``pause_threshold_s`` a number): the first tour of each picker
    has a binding recorded start. A later gap over the threshold requests a
    ``PickerDeparture`` at its recorded start; the full duration begins when
    the picker finishes the running tour. Shorter gaps vanish, so errors
    accumulate over the day;
  - anchored (``pause_threshold_s`` None): every recorded tour carries its
    recorded start in the plan, and starts no earlier; no breaks. Each tour's
    error stays its own unless the previous tour overruns its start.

Presence (``presence``): ``shift`` makes every picker present from the
shift start; ``recorded`` makes a picker arrive at the setup start of its
first recorded tour and leave, without return, at the end of its last
recorded line (after its running tour, as every departure). Work still queued
for a picker that has left stays open and is reported as such. ``arrival``
makes a picker arrive as in ``recorded`` but never leave, so all work is
done and late work shows as a later finish.

Gaps (pauses and stops) count only time in which the picker has no
recorded line open, so time spent on another order is never an absence.

CASIM does not interrupt a tour. With ``stops_in_tours`` (chained runs
only), a stop inside a tour over the same threshold, a gap between two lines
or the excess of one line over the model's pick time, also requests a
``PickerDeparture`` of its recorded duration; it begins when the running
tour ends, so the day keeps its recorded absence and only the stop's place
moves to the tour end. Without it, stops inside a tour are not modelled. The
counts of what the replay cannot represent are in ``ReplaySummary``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ware_ops_algos.domain_models import (
    Article, Articles, ArticleType, DimensionType, InformationType, LayoutData, LinePickTime,
    Location,
    ManualPicker, Order, OrderPosition, OrdersDomain, OrderType, PickCart, PlannerInformation,
    RecordedLine, RecordedPickingPlan, RecordedTour, Resources, ResourceType, StorageLocations,
    StorageType, WarehouseInfo, WarehouseInfoType,
)

from casim.domain_objects.sim_domain import SimWarehouseDomain, initial_dynamic_info
from casim.events.base_events import Event
from casim.events.operational_events import (
    PickerDeparture, OrderArrival, OrderStreamClosed, PickerArrival,
)


@dataclass(frozen=True)
class RecordedRow:
    picker: int
    order: int
    article: int
    quantity: int
    node: tuple[float, float]
    start_s: float
    end_s: float


@dataclass(frozen=True)
class TimeModel:
    """The picker's time parameters: walking speed (length per s), pick and setup (s).

    With ``per_unit_s`` the pick time of a line is ``time_per_pick + per_unit_s
    * quantity`` (``LinePickTime``); without it, ``time_per_pick`` per line.
    """

    speed: float
    time_per_pick: float
    tour_setup_time: float
    per_unit_s: float | None = None


@dataclass
class ReplaySummary:
    tours: int = 0
    pauses: int = 0
    pause_s: float = 0.0
    gaps_inside_tours: int = 0
    gap_s_inside_tours: float = 0.0
    stops_in_tours: int = 0
    stop_s_in_tours: float = 0.0
    largest_tour_orders: int = 0


def recorded_tours(rows: Sequence[RecordedRow]) -> list[tuple[int, list[RecordedRow]]]:
    """One tour per (picker, order), its lines in recorded order; tours by first line."""
    by_order: dict[tuple[int, int], list[RecordedRow]] = {}
    for row in sorted(rows, key=lambda r: (r.picker, r.start_s, r.end_s)):
        by_order.setdefault((row.picker, row.order), []).append(row)
    return sorted(((picker, lines) for (picker, _), lines in by_order.items()),
                  key=lambda tour: (tour[1][0].start_s, tour[0]))


def _uncovered(start: float, end: float, busy: Sequence[tuple[float, float]]
               ) -> list[tuple[float, float]]:
    """The parts of [start, end) that no interval of ``busy`` (sorted by start) covers."""
    free, cursor = [], start
    for low, high in busy:
        if high <= cursor:
            continue
        if low >= end:
            break
        if low > cursor:
            free.append((cursor, low))
        cursor = max(cursor, high)
    if cursor < end:
        free.append((cursor, end))
    return free


class RecordedReplay:
    """State, recorded plan and events of one recorded day."""

    def __init__(self, rows: Sequence[RecordedRow], layout: LayoutData, time_model: TimeModel,
                 shift_start_s: float, pause_threshold_s: float | None,
                 setup_in_gap: bool = False,
                 presence: str = "shift", stops_in_tours: bool = False,
                 absences: Sequence[tuple[int, float, float]] | None = None):
        """Build a replay.

        ``absences`` (picker, request time, duration) replace the recorded
        pauses and stops, for example fixed breaks or absences drawn from a
        fitted model; the run is then chained.
        """
        self.rows = list(rows)
        self.layout = layout
        self.time_model = time_model
        self.shift_start_s = float(shift_start_s)
        self.pause_threshold_s = pause_threshold_s
        self.absences = None if absences is None else [
            (int(p), float(t), float(d)) for p, t, d in absences]
        self.setup_in_gap = bool(setup_in_gap)
        self.stops_in_tours = bool(stops_in_tours)
        if presence not in {"shift", "recorded", "arrival"}:
            raise ValueError(f"unknown presence: {presence}")
        self.presence = presence
        self.tours = recorded_tours(self.rows)
        self.pickers = sorted({row.picker for row in self.rows})
        self.summary = ReplaySummary(tours=len(self.tours))

    def plan(self) -> RecordedPickingPlan:
        anchored = self.pause_threshold_s is None and self.absences is None
        seen_pickers: set[int] = set()
        planned = []
        for index, (picker, lines) in enumerate(self.tours):
            first_tour = picker not in seen_pickers
            seen_pickers.add(picker)
            planned.append(RecordedTour(
                tour_id=index, picker_id=picker,
                lines=tuple(RecordedLine(row.order, row.article, row.node) for row in lines),
                start_s=self._setup_start(lines) if anchored or first_tour else None))
        return RecordedPickingPlan(tours=tuple(planned))

    def _setup_start(self, lines: Sequence[RecordedRow]) -> float:
        """Target setup start, optionally before the WMS first-line interval."""
        first_start = min(row.start_s for row in lines)
        return (first_start - self.time_model.tour_setup_time
                if self.setup_in_gap else first_start)

    def domain(self) -> SimWarehouseDomain:
        orders: dict[int, Order] = {}
        for row in sorted(self.rows, key=lambda r: (r.order, r.start_s)):
            order = orders.setdefault(row.order, Order(order_id=row.order,
                                                       order_date=self.shift_start_s))
            order.order_positions.append(OrderPosition(order_number=row.order,
                                                       article_id=row.article,
                                                       amount=row.quantity))
        stored = sorted({(row.article, row.node) for row in self.rows})
        storage = StorageLocations(tpe=StorageType.DEDICATED, locations=[
            Location(x=node[0], y=node[1], article_id=article, amount=10 ** 9)
            for article, node in stored])
        storage.build_article_location_mapping()
        largest = max(len({row.order for row in lines}) for _, lines in self.tours)
        self.summary.largest_tour_orders = largest
        cart = PickCart(n_dimension=1, capacities=[1], dimensions=[DimensionType.ORDERS],
                        n_boxes=largest, box_can_mix_orders=False)
        start = self.layout.layout_network.start_node
        pickers = [ManualPicker(
            id=picker, tpe=ResourceType.HUMAN, capacity=largest, speed=self.time_model.speed,
            time_per_pick=self.time_model.time_per_pick,
            tour_setup_time=self.time_model.tour_setup_time, pick_cart=cart,
            pick_time_model=(None if self.time_model.per_unit_s is None else
                             LinePickTime(self.time_model.time_per_pick,
                                          self.time_model.per_unit_s)),
            available=True, occupied=False, current_location=start)
            for picker in self.pickers]
        if self.pickers != list(range(len(self.pickers))):
            raise ValueError("pickers must be numbered 0 .. n-1 (resource ids are list indices)")
        return SimWarehouseDomain(
            problem_class="OBRSP", objective="mean_order_completion_time", layout=self.layout,
            articles=Articles(ArticleType.STANDARD, [Article(article_id=article)
                                                     for article in sorted({a for a, _ in stored})]),
            orders=OrdersDomain(OrderType.STANDARD, list(orders.values())),
            resources=Resources(ResourceType.HUMAN, pickers), storage=storage,
            information=PlannerInformation(InformationType.PROCESS_INFORMATION,
                                           {"recorded_plan": self.plan()}),
            dynamic_warehouse_info=initial_dynamic_info(),
            warehouse_info=WarehouseInfo(tpe=WarehouseInfoType.ONLINE),
        )

    def _pauses(self) -> list[PickerDeparture]:
        if self.absences is not None:
            return [PickerDeparture(time=time, picker_id=picker, duration=duration)
                    for picker, time, duration in self.absences]
        if self.pause_threshold_s is None:
            return []  # anchored: the plan holds each tour until its recorded start
        pauses = []
        busy: dict[int, list[tuple[float, float]]] = {}
        for row in sorted(self.rows, key=lambda r: r.start_s):
            busy.setdefault(row.picker, []).append((row.start_s, row.end_s))
        previous_end: dict[int, float] = {}
        for picker, lines in self.tours:
            recorded_start = min(row.start_s for row in lines)
            start = self._setup_start(lines)
            end = max(row.end_s for row in lines)
            ordered = sorted(lines, key=lambda r: r.start_s)
            reach = ordered[0].end_s
            for row in ordered[1:]:
                # Only time without any recorded line of the picker: time on
                # another order of the picker is work, not a gap.
                for low, high in _uncovered(reach, row.start_s, busy[picker]):
                    self.summary.gaps_inside_tours += 1
                    self.summary.gap_s_inside_tours += high - low
                    if self.stops_in_tours and high - low > self.pause_threshold_s:
                        pauses.append(self._stop(picker, low, high - low))
                reach = max(reach, row.end_s)
            if self.stops_in_tours:
                for row in ordered:
                    pick = self._pick_time(row)
                    if row.end_s - row.start_s - pick > self.pause_threshold_s:
                        pauses.append(self._stop(picker, row.start_s + pick,
                                                 row.end_s - row.start_s - pick))
            if picker not in previous_end:
                # The first tour has a binding start in the recorded plan. A
                # departure at shift start would hide this picker from the
                # shift-start decision that must schedule the recorded plan.
                pause_from = self.shift_start_s
                is_pause = False
            else:
                pause_from = previous_end[picker]
                gap = recorded_start - pause_from
                is_pause = gap > self.pause_threshold_s
            if is_pause:
                duration = start - pause_from
                if duration > 0:
                    pauses.append(PickerDeparture(time=pause_from, picker_id=picker,
                                                  duration=duration))
                    self.summary.pauses += 1
                    self.summary.pause_s += duration
            # Tours run in the order of their first line; an order inside an
            # earlier one ends before it, so the picker is free from the later end.
            previous_end[picker] = max(previous_end.get(picker, end), end)
        return pauses

    def _pick_time(self, row: RecordedRow) -> float:
        model = self.time_model
        if model.per_unit_s is None:
            return model.time_per_pick
        return model.time_per_pick + model.per_unit_s * row.quantity

    def _stop(self, picker: int, start: float, duration: float) -> PickerDeparture:
        self.summary.stops_in_tours += 1
        self.summary.stop_s_in_tours += duration
        return PickerDeparture(time=start, picker_id=picker, duration=duration)

    def roster(self) -> dict[int, tuple[float, float]]:
        """Each picker's recorded presence: setup start of its first tour to its last line."""
        first: dict[int, float] = {}
        last: dict[int, float] = {}
        for picker, lines in self.tours:
            first.setdefault(picker, self._setup_start(lines))
            last[picker] = max(last.get(picker, float("-inf")),
                               max(row.end_s for row in lines))
        return {picker: (first[picker], last[picker]) for picker in self.pickers}

    def recorded_absences(self) -> list[tuple[int, float, float]]:
        """(picker, request time, duration) of the absences this replay runs with."""
        return [(d.picker_id, d.time, d.duration) for d in self._pauses()]

    def _presence(self) -> list[Event]:
        if self.presence == "shift":
            return [PickerArrival(time=self.shift_start_s, picker_id=picker)
                    for picker in self.pickers]
        roster = self.roster()
        arrivals = [PickerArrival(time=roster[picker][0], picker_id=picker)
                    for picker in self.pickers]
        if self.presence == "arrival":
            return arrivals
        return [*arrivals, *(PickerDeparture(time=roster[picker][1], picker_id=picker)
                             for picker in self.pickers)]

    def events(self, domain: SimWarehouseDomain) -> Iterator[Event]:
        events: list[Event] = [
            *self._presence(),
            *(OrderArrival(order.order_date, order) for order in domain.orders.orders),
            OrderStreamClosed(self.shift_start_s),
            *self._pauses(),
        ]
        return iter(sorted(events, key=lambda event: event.time))
