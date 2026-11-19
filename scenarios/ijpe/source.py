"""One historic picking day of the WMS export, replayed with its recorded decisions.

Lines are assigned to days by timestamp (the order's pick date, R0.4 of
``docs/ijpe/plan.md``), and located on the warehouse graph of
``data/layout.csv`` by (aisle, house, place). Left out, and counted in
``excluded``: lines without a pick node in the layout (aisle 10 and one
location in aisle 7), and orders picked by more than one picker, which a
recorded tour cannot hold. Pickers are numbered per day in the order of their
first line; ``casim.sources.recorded`` builds the state, the recorded plan
and the pauses.

A login is not always one person (the use case partner: the system allows
one order at a time per login, yet the scan windows of some orders of one
login overlap), and the scans do not tell how many people worked. Staffing is
therefore an assumption: ``extra_pickers`` adds pickers to the logins, for a
sensitivity run. An extra picker arrives at the shift start (``presence:
shift``) or at the median arrival of the logins (``presence: arrival``) and
never leaves; it takes breaks and setups like a login's picker.

Disruptions (``disruption``, ``DISRUPTIONS``) are the paper's scenarios built
from the warehouse's lists (``docs/ijpe/data_audit.md``, plan P2.1). A
disruption moves a truck forward; its new picking deadline is the submitted
scenario's due time, and it needs the fixed dispatch plan for every other
order. Scenarios 1 and 2: at the announcement, a ``TruckDisruption`` sets the
new deadline for the orders of the list (S1, S2) that the day replays and
that are due later; an order due earlier keeps its deadline, since the
synthetic slots do not know the warehouse's trucks. Scenario 3: the template
order of list S3 (one store's 53 lines) arrives copied TU / 2.6 times, as new
orders with the new deadline. Scenario 4: the orders of list S5, recorded on
Thursday, arrive on Wednesday at the announcement with the new deadline; on Thursday (``scenario4_next_day``)
they are gone. Orders of a list that the day already holds (picked a day
earlier) stay as they are. The affected orders are marked
(``affected_order_ids``). When orders arrive during the day, the order stream
closes after the last of them.

With ``non_picking=fitted`` the source builds W_0 from the day's fit
(``scripts/ijpe/fit_non_picking.py``, fitted on the other study days):
breaks are timed departures drawn per picker and window; the realized time of
every order line and of every setup of a picker is fixed in the process
truth (``casim.domain_objects.process_truth``), drawn from the fitted
distributions with ``seed``; the resources plan with the means (setup, and
line time by quantity class, ``ClassPickTime``). A line's time is drawn from
a stream seeded by seed, day and order (one draw per position), a picker's
setups from a stream seeded by seed, day and picker, so every policy of a
replication meets the same times. A picker's first setup of the day is the
fixed ``tour_setup_time`` that defines its arrival, since no gap before its
first tour is observed.
"""

from __future__ import annotations

import copy
import json
from bisect import bisect_left
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from ware_ops_algos.domain_models import (
    Article, ClassPickTime, Location, Order, OrderPosition, RecordedLine, RecordedPickingPlan,
    RecordedTour,
)

from casim.domain_objects.process_truth import ProcessTruth
from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.events.operational_events import (
    OrderArrival, OrderStreamClosed, PickerArrival, TruckDisruption,
)
from casim.sources.grid_layout import GridLayout
from casim.sources.recorded import RecordedReplay, RecordedRow, TimeModel
from casim.sources.wms_export import day_lines, disruption_lines, disruption_orders


STUDY_DAYS = ("2026-04-11", "2026-04-13", "2026-04-14", "2026-04-15",
              "2026-04-16", "2026-04-17")
# The scheduled breaks around 08:00 and 12:00: a picker's longest spell
# without a record of at least 10 minutes starting in each window is its break
# (scripts/ijpe/fit_non_picking.py).
BREAK_WINDOWS = ((7.0 * 3600, 9.5 * 3600), (10.5 * 3600, 13.5 * 3600))
BREAK_MIN_S = 600.0
DEPARTURE_HOURS = (10, 12, 14, 16)
STAGING_MINUTES = 30
NON_PICKING = ("recorded", "fitted", "fixed")
TU_PER_ORDER = 2.6                      # the warehouse's workbook (data_audit.md)
H = 3600.0
# The paper's disruption scenarios on the warehouse's lists (plan P2.1): the
# day, what changes, the list, the announcement and the new picking deadline
# (the submitted scenario's due time).
DISRUPTIONS = {
    # Monday 07:30: the orders of list S1 ("120 TU before 13:00") are due at 10:00.
    "scenario1": dict(day="2026-04-13", kind="due", sheet="S1", time_s=7.5 * H,
                      due_s=10 * H),
    # Thursday 10:30: the orders of list S2 ("100 TU before 13:00") are due at 11:00.
    "scenario2": dict(day="2026-04-16", kind="due", sheet="S2", time_s=10.5 * H,
                      due_s=11 * H),
    # Wednesday 08:30: a store orders 6 TU more (template S3), due at 10:00.
    "scenario3": dict(day="2026-04-15", kind="new", sheet="S3", time_s=8.5 * H,
                      due_s=10 * H, tu=6),
    # Wednesday 06:30: Thursday's orders of list S5 ("200 TU after 10:00") are
    # picked on Wednesday, due at 10:00.
    "scenario4": dict(day="2026-04-15", kind="moved", sheet="S5", from_day="2026-04-16",
                      time_s=6.5 * H, due_s=10 * H),
    # Thursday after Scenario 4: those orders were picked the day before.
    "scenario4_next_day": dict(day="2026-04-16", kind="removed", sheet="S5"),
}


def fixed_departure_hour(customer: str) -> int:
    """Synthetic daily store slot, independent of recorded picking times."""
    digest = sha256(str(customer).encode("utf-8")).digest()
    return DEPARTURE_HOURS[int.from_bytes(digest[:8], "big") % len(DEPARTURE_HOURS)]


def _day_key(day: str) -> int:
    return int(pd.Timestamp(day).strftime("%Y%m%d"))


def draw_breaks(fit: dict, roster: dict[int, tuple[float, float]], seed: int, day: int
                ) -> list[tuple[int, float, float]]:
    """(picker, request time, duration): per picker present in a break window, its
    break with the fitted share, start and duration drawn together."""
    rng = np.random.default_rng([seed, day, 0])
    drawn = []
    for picker, (arrival, departure) in sorted(roster.items()):
        for brk in fit["breaks"]:
            low, high = brk["window"]
            present = min(departure, high) > max(arrival, low)
            if brk["samples"] and present and rng.random() < brk["share"]:
                start, duration = brk["samples"][rng.integers(len(brk["samples"]))]
                if arrival <= start <= departure:
                    drawn.append((picker, float(start), float(duration)))
    return drawn


def process_truth(fit: dict, seed: int, day: int, orders, pickers, first_setup_s: float
                  ) -> ProcessTruth:
    """The realized time of every order line and of every setup of a picker.

    A line's time is drawn from the fitted line times of its quantity class,
    with a stream seeded by (seed, day, order) and one draw per position; a
    picker's setups from the fitted setups with a stream seeded by (seed, day,
    picker), after a first setup of ``first_setup_s``.

    The lines of an order are slow or fast together: every line time of an
    order is multiplied by the order's factor, lognormal with mean 1 and the
    fit's ``order_sd`` (the pools hold the within-order variation), drawn from
    a stream seeded by (seed, day, order) of its own, so the line draws do not
    change with it.
    """
    bounds, pools = fit["quantity_bounds"], fit["line_s"]
    sigma = float(np.sqrt(np.log1p(float(fit.get("order_sd", 0.0)) ** 2)))
    line_s: dict[tuple[int, int, int], tuple[float, ...]] = {}
    for order in orders:
        uniforms = np.random.default_rng([seed, day, 1, int(order.order_id)]).random(
            len(order.order_positions))
        factor = 1.0 if sigma == 0.0 else float(np.exp(
            np.random.default_rng([seed, day, 3, int(order.order_id)]).normal(
                -sigma ** 2 / 2, sigma)))
        for position, u in zip(order.order_positions, uniforms):
            pool = pools[bisect_left(bounds, position.amount)]
            key = (int(order.order_id), int(position.article_id), int(position.amount))
            line_s[key] = (*line_s.get(key, ()), factor * float(pool[int(u * len(pool))]))
    setups = fit["setup_s"]
    setup_s = {}
    for picker in pickers:
        uniforms = np.random.default_rng([seed, day, 2, int(picker)]).random(len(orders))
        setup_s[picker] = (float(first_setup_s),
                           *(float(setups[int(u * len(setups))]) for u in uniforms))
    return ProcessTruth(line_s=line_s, setup_s=setup_s)


class ScenarioSource:
    def __init__(self, data_dir: Path, *, day: str, shift_start_s: float,
                 pause_threshold_s: float | None, speed: float, time_per_pick: float,
                 tour_setup_time: float, per_unit_s: float | None = None,
                 setup_in_gap: bool = False,
                 presence: str = "shift", due_plan: str | None = None,
                 stops_in_tours: bool = False, non_picking: str = "recorded",
                 breaks=None, seed: int = 0, extra_pickers: int = 0,
                 disruption: str | None = None):
        """``non_picking`` (time a picker is present and not picking):

        - ``recorded``: the recorded pauses (and with ``stops_in_tours`` the
          stops) over the threshold, as timed departures (the trace-driven
          diagnostic);
        - ``fitted``: breaks, tour setup and line time from the day's fit
          (``data/non_picking/<day>.json``), drawn with ``seed``;
          ``breaks`` as a list of ``[start_s, end_s]`` replaces the fitted
          breaks by the same breaks for every picker (a shared roster);
        - ``fixed``: the ``breaks`` list for every picker, with the
          deterministic setup and pick times.

        ``extra_pickers``: pickers added to the logins (present from the
        shift start or the median arrival, never leaving); needs presence
        ``shift`` or ``arrival``.

        ``disruption``: a key of ``DISRUPTIONS``, on its day; it needs the
        fixed dispatch plan (``due_plan: fixed_dispatch``).
        """
        if non_picking not in NON_PICKING:
            raise ValueError(f"unknown IJPE non-picking time: {non_picking}")
        if non_picking == "fixed" and breaks is None:
            raise ValueError("non_picking fixed needs a break list")
        if non_picking == "recorded" and breaks is not None:
            raise ValueError("a break list goes with non_picking fixed or fitted")
        if extra_pickers < 0:
            raise ValueError("extra_pickers must not be negative")
        if extra_pickers and presence not in ("shift", "arrival"):
            raise ValueError("extra pickers need presence shift or arrival (they never leave)")
        if due_plan not in (None, "fixed_dispatch"):
            raise ValueError(f"unknown IJPE due plan: {due_plan}")
        self.due_plan = due_plan
        self.day = str(day)
        data_dir = Path(data_dir)
        layout, locations = GridLayout().build(data_dir / "layout.csv")
        lines, self.excluded = day_lines(self.day, GridLayout.pick_nodes(locations),
                                         data_dir / "cache")
        midnight = pd.Timestamp(day)
        lines = lines.sort_values(["start", "row"])
        lines, self._arriving_lines = self._disrupt(disruption, lines, due_plan, locations,
                                                    data_dir / "cache")
        if (lines.groupby("order")["customer"].nunique() != 1).any():
            raise ValueError("an order belongs to several stores")
        self.customer_by_order = lines.groupby("order")["customer"].first().to_dict()
        picker_ids = {picker: index for index, picker in enumerate(lines["picker"].unique())}
        rows = [RecordedRow(picker=picker_ids[line.picker], order=int(line.order),
                            article=int(line.article), quantity=int(line.quantity),
                            node=line.node, start_s=(line.start - midnight).total_seconds(),
                            end_s=(line.end - midnight).total_seconds())
                for line in lines.itertuples()]
        # For the fit (scripts/ijpe/fit_non_picking.py): the export rows replayed
        # and the picker numbers.
        self.picker_ids, self.rows_kept = picker_ids, tuple(lines["row"])
        self.replay = RecordedReplay(
            rows, layout, TimeModel(float(speed), float(time_per_pick), float(tour_setup_time),
                                    None if per_unit_s is None else float(per_unit_s)),
            shift_start_s=float(shift_start_s),
            pause_threshold_s=None if pause_threshold_s is None else float(pause_threshold_s),
            setup_in_gap=setup_in_gap, presence=presence,
            stops_in_tours=stops_in_tours)
        self._domain: SimWarehouseDomain | None = None
        self._fit: dict | None = None
        self.seed = int(seed)
        roster = self.replay.roster()
        arrival = (float(shift_start_s) if presence == "shift"
                   else float(np.median([first for first, _ in roster.values()])))
        last = max(end for _, end in roster.values())
        n = len(self.replay.pickers)
        self.extra_pickers = {n + k: (arrival, last) for k in range(int(extra_pickers))}
        self.pickers = [*self.replay.pickers, *self.extra_pickers]
        fixed_breaks = None if breaks is None else [
            (picker, float(start), float(end) - float(start))
            for start, end in breaks for picker in self.pickers]
        if non_picking == "fixed":
            self.replay.absences = fixed_breaks
        elif non_picking == "fitted":
            if pause_threshold_s is None:
                raise ValueError("fitted non-picking time needs a chained run (a pause threshold)")
            path = data_dir / "non_picking" / f"{self.day}.json"
            if not path.exists():
                raise ValueError(f"{path} is missing: python -m scripts.ijpe.fit_non_picking")
            self._fit = json.loads(path.read_text())
            self.replay.absences = (fixed_breaks if fixed_breaks is not None else draw_breaks(
                self._fit, {**roster, **self.extra_pickers}, self.seed, _day_key(self.day)))

    def _disrupt(self, name: str | None, lines: pd.DataFrame, due_plan: str | None,
                 locations, cache: Path) -> tuple[pd.DataFrame, list]:
        """The day's lines as the disruption leaves them, and the orders it adds:
        (order id, [(article, quantity, node), ...])."""
        self.disruption = None
        self.affected_order_ids: frozenset[int] = frozenset()
        self._due_change: tuple[float, list[int], float] | None = None
        if name is None:
            return lines, []
        if name not in DISRUPTIONS:
            raise ValueError(f"unknown IJPE disruption: {name}")
        spec = self.disruption = DISRUPTIONS[name]
        if spec["day"] != self.day:
            raise ValueError(f"{name} runs on {spec['day']}, not {self.day}")
        if spec["kind"] != "removed" and due_plan != "fixed_dispatch":
            raise ValueError("a disruption moves a departure: it needs due_plan fixed_dispatch")
        held = set(lines["order"].astype(str))
        listed = set()
        if spec["kind"] != "new":
            frame = disruption_orders(cache)
            listed = set(frame.loc[frame["sheet"] == spec["sheet"], "order"].astype(str))
        if spec["kind"] == "removed":
            return lines[~lines["order"].astype(str).isin(listed)], []
        deadline = spec["due_s"]
        if spec["kind"] == "due":
            customer = lines.groupby(lines["order"].astype(str))["customer"].first()
            affected = sorted(int(order) for order in listed & held
                              if fixed_departure_hour(customer[order]) * 3600
                              - STAGING_MINUTES * 60 > deadline)
            self.affected_order_ids = frozenset(affected)
            self._due_change = (spec["time_s"], affected, deadline)
            return lines, []
        pick_nodes = GridLayout.pick_nodes(locations)
        if spec["kind"] == "moved":
            other, _ = day_lines(spec["from_day"], pick_nodes, cache)
            other = other[other["order"].astype(str).isin(listed - held)]
            other = other.sort_values(["start", "row"])
            arriving = [(int(order), [(int(r.article), int(r.quantity), r.node)
                                      for r in group.itertuples()])
                        for order, group in other.groupby("order", sort=True)]
        else:
            template = disruption_lines(cache)
            nodes = [pick_nodes.get((int(a), int(h), int(pl)))
                     for a, h, pl in zip(template["aisle"], template["house"], template["place"])]
            positions = [(int(article), int(quantity), node) for article, quantity, node
                         in zip(template["article"], template["quantity"], nodes) if node is not None]
            copies = int(np.ceil(spec["tu"] / TU_PER_ORDER))
            prefix = int(pd.Timestamp(self.day).strftime("%y%m%d"))
            arriving = [(prefix * 10_000 + 9_000 + k, positions) for k in range(1, copies + 1)]
            if {str(order) for order, _ in arriving} & held:
                raise ValueError("a new order's number is taken")
        self.affected_order_ids = frozenset(order for order, _ in arriving)
        self._arrival_s, self._arrival_due = spec["time_s"], deadline
        return lines, arriving

    def initial_state(self) -> SimWarehouseDomain:
        self._domain = self.replay.domain()
        self._arriving: list[Order] = []
        if self._arriving_lines:
            storage, articles = self._domain.storage, self._domain.articles
            stored = {(loc.article_id, (loc.x, loc.y)) for loc in storage.locations}
            known = {article.article_id for article in articles.articles}
            for order_id, positions in self._arriving_lines:
                order = Order(order_id=order_id, order_date=self._arrival_s,
                              due_date=self._arrival_due)
                for article, quantity, node in positions:
                    order.order_positions.append(OrderPosition(
                        order_number=order_id, article_id=article, amount=quantity))
                    if (article, node) not in stored:
                        stored.add((article, node))
                        storage.locations.append(Location(x=node[0], y=node[1], article_id=article,
                                                          amount=10 ** 9))
                    if article not in known:
                        known.add(article)
                        articles.articles.append(Article(article_id=article))
                self._arriving.append(order)
            storage.build_article_location_mapping()
            # The recorded plan says where each order line is picked; an added
            # order gets one tour of its lines at their recorded (S5) or
            # template (S3) places, with no recorded picker on this day (-1).
            processes = self._domain.information.processes
            plan = processes["recorded_plan"]
            first = max(tour.tour_id for tour in plan.tours) + 1
            processes["recorded_plan"] = RecordedPickingPlan(tours=(*plan.tours, *(
                RecordedTour(tour_id=first + k, picker_id=-1, lines=tuple(
                    RecordedLine(order_id, article, node) for article, _, node in positions))
                for k, (order_id, positions) in enumerate(self._arriving_lines))))
        template = self._domain.resources.resources[0]
        for picker in self.extra_pickers:
            extra = copy.deepcopy(template)
            extra.id = picker
            self._domain.resources.resources.append(extra)
        if self._fit is not None:
            # Truth: the realized times. Belief: the resources plan with the means.
            self._domain.truth = process_truth(
                self._fit, self.seed, _day_key(self.day),
                [*self._domain.orders.orders, *self._arriving],
                self.pickers, self.replay.time_model.tour_setup_time)
            planning = ClassPickTime(tuple(self._fit["quantity_bounds"]),
                                     tuple(float(np.mean(pool)) for pool in self._fit["line_s"]))
            for picker in self._domain.resources.resources:
                picker.tour_setup_time = float(np.mean(self._fit["setup_s"]))
                picker.pick_time_model = planning
        if self.due_plan == "fixed_dispatch":
            for order in self._domain.orders.orders:
                hour = fixed_departure_hour(self.customer_by_order[str(order.order_id)])
                order.due_date = hour * 3600 - STAGING_MINUTES * 60
        return self._domain

    def events(self) -> Iterator[Event]:
        events = [*self.replay.events(self._domain),
                  *(PickerArrival(time=arrival, picker_id=picker)
                    for picker, (arrival, _) in self.extra_pickers.items())]
        if self._due_change is not None:
            time, orders, due = self._due_change
            events.append(TruckDisruption(time, orders, due))
        if self._arriving:
            # The stream is complete only after the last order of the day arrived.
            events = [e for e in events if not isinstance(e, OrderStreamClosed)]
            events += [OrderArrival(order.order_date, order) for order in self._arriving]
            events.append(OrderStreamClosed(max(order.order_date for order in self._arriving)))
        return iter(sorted(events, key=lambda event: event.time))
