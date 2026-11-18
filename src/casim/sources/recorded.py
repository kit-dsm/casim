"""Replay a picking day a WMS recorded: its lines become CASIM's state, plan and events.

The input is one row per recorded line (``RecordedRow``): picker, order,
article, quantity, pick node, start and end in seconds of the day. From it,

- the state: one order per WMS order with one position per line, available
  at the shift start; dedicated storage of each article at its recorded pick
  nodes; one picker per recorded picker, present from the shift start, with
  a cart for as many orders as the largest recorded tour holds;
- the recorded tours: a picker's lines in time order, cut where the picker
  has no order open; orders whose intervals overlap (interleaving) share one
  tour. They go to the planner as ``information.recorded_plan``;
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

A gap inside a tour is not a pause: CASIM does not interrupt a tour. The
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
    largest_tour_orders: int = 0


def recorded_tours(rows: Sequence[RecordedRow], interleaving: bool = True
                   ) -> list[tuple[int, list[RecordedRow]]]:
    """Per picker, lines in time order, cut where no order of the picker is open.

    Without ``interleaving`` every order is a tour of its own, its lines in
    recorded order (the submitted model's one order per tour).
    """
    tours = []
    by_picker: dict[int, list[RecordedRow]] = {}
    for row in sorted(rows, key=lambda r: (r.picker, r.start_s, r.end_s)):
        by_picker.setdefault(row.picker, []).append(row)
    if not interleaving:
        by_order: dict[tuple[int, int], list[RecordedRow]] = {}
        for picker, lines in by_picker.items():
            for line in lines:
                by_order.setdefault((picker, line.order), []).append(line)
        return sorted(((picker, lines) for (picker, _), lines in by_order.items()),
                      key=lambda tour: (tour[1][0].start_s, tour[0]))
    for picker, lines in by_picker.items():
        last_end = {}
        for line in lines:
            last_end[line.order] = max(last_end.get(line.order, line.end_s), line.end_s)
        current: list[RecordedRow] = []
        open_until = float("-inf")
        for line in lines:
            if current and line.start_s >= open_until and line.order not in {r.order for r in current}:
                tours.append((picker, current))
                current, open_until = [], float("-inf")
            current.append(line)
            open_until = max(open_until, last_end[line.order])
        if current:
            tours.append((picker, current))
    return sorted(tours, key=lambda tour: (tour[1][0].start_s, tour[0]))


class RecordedReplay:
    """State, recorded plan and events of one recorded day."""

    def __init__(self, rows: Sequence[RecordedRow], layout: LayoutData, time_model: TimeModel,
                 shift_start_s: float, pause_threshold_s: float | None,
                 interleaving: bool = True,
                 breaks: Sequence[tuple[float, float]] | None = None,
                 origin_mode: str = "depot", setup_in_gap: bool = False,
                 sequence_clock: str = "start", presence: str = "shift"):
        """Build a replay; ``sequence_clock=end`` tests confirmation-time order.

        ``breaks`` replaces recorded pauses by fixed breaks for all pickers.
        Tour membership remains inferred from start-time order in either mode.
        """
        self.rows = list(rows)
        self.layout = layout
        self.time_model = time_model
        self.shift_start_s = float(shift_start_s)
        self.pause_threshold_s = pause_threshold_s
        self.breaks = None if breaks is None else [(float(a), float(b)) for a, b in breaks]
        if origin_mode not in {"depot", "first_pick", "aisle_head"}:
            raise ValueError(f"unknown recorded tour origin mode: {origin_mode}")
        self.origin_mode = origin_mode
        self.setup_in_gap = bool(setup_in_gap)
        if presence not in {"shift", "recorded", "arrival"}:
            raise ValueError(f"unknown presence: {presence}")
        self.presence = presence
        self.tours = recorded_tours(self.rows, interleaving)
        if sequence_clock == "end":
            self.tours = [(picker, sorted(lines, key=lambda row: (row.end_s, row.start_s)))
                          for picker, lines in self.tours]
        elif sequence_clock != "start":
            raise ValueError(f"unknown recorded sequence clock: {sequence_clock}")
        self.pickers = sorted({row.picker for row in self.rows})
        self.summary = ReplaySummary(tours=len(self.tours))

    def plan(self) -> RecordedPickingPlan:
        anchored = self.pause_threshold_s is None and self.breaks is None
        seen_pickers: set[int] = set()
        planned = []
        for index, (picker, lines) in enumerate(self.tours):
            first_tour = picker not in seen_pickers
            seen_pickers.add(picker)
            planned.append(RecordedTour(
                tour_id=index, picker_id=picker,
                lines=tuple(RecordedLine(row.order, row.article, row.node) for row in lines),
                start_s=self._setup_start(lines) if anchored or first_tour else None,
                start_node=self._tour_origin(lines[0])))
        return RecordedPickingPlan(tours=tuple(planned))

    def _tour_origin(self, first: RecordedRow) -> tuple[float, float] | None:
        """Diagnostic origins; neither is claimed as an observed WMS position."""
        if self.origin_mode == "depot":
            return None
        if self.origin_mode == "first_pick":
            return first.node
        graph = self.layout.layout_network.graph
        return min((node for node, data in graph.nodes(data=True)
                    if node[0] == first.node[0] and data.get("node_type") == "intersection"),
                   key=lambda node: node[1])

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
        if self.breaks is not None:
            return [PickerDeparture(time=start, picker_id=picker, duration=end - start)
                    for start, end in self.breaks for picker in self.pickers]
        if self.pause_threshold_s is None:
            return []  # anchored: the plan holds each tour until its recorded start
        pauses = []
        previous_end: dict[int, float] = {}
        for picker, lines in self.tours:
            recorded_start = min(row.start_s for row in lines)
            start = self._setup_start(lines)
            end = max(row.end_s for row in lines)
            ordered = sorted(lines, key=lambda r: r.start_s)
            reach = ordered[0].end_s
            for row in ordered[1:]:
                if row.start_s > reach:
                    self.summary.gaps_inside_tours += 1
                    self.summary.gap_s_inside_tours += row.start_s - reach
                reach = max(reach, row.end_s)
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
            previous_end[picker] = end
        return pauses

    def _presence(self) -> list[Event]:
        if self.presence == "shift":
            return [PickerArrival(time=self.shift_start_s, picker_id=picker)
                    for picker in self.pickers]
        first: dict[int, float] = {}
        last: dict[int, float] = {}
        for picker, lines in self.tours:
            first.setdefault(picker, self._setup_start(lines))
            last[picker] = max(last.get(picker, float("-inf")),
                               max(row.end_s for row in lines))
        arrivals = [PickerArrival(time=first[picker], picker_id=picker) for picker in self.pickers]
        if self.presence == "arrival":
            return arrivals
        return [*arrivals,
                *(PickerDeparture(time=last[picker], picker_id=picker) for picker in self.pickers)]

    def events(self, domain: SimWarehouseDomain) -> Iterator[Event]:
        events: list[Event] = [
            *self._presence(),
            *(OrderArrival(order.order_date, order) for order in domain.orders.orders),
            OrderStreamClosed(self.shift_start_s),
            *self._pauses(),
        ]
        return iter(sorted(events, key=lambda event: event.time))
