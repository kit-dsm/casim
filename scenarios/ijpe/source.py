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
from ware_ops_algos.domain_models import ClassPickTime

from casim.domain_objects.process_truth import ProcessTruth
from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.events.operational_events import PickerArrival
from casim.sources.grid_layout import GridLayout
from casim.sources.recorded import RecordedReplay, RecordedRow, TimeModel
from casim.sources.wms_export import day_lines


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
                 breaks=None, seed: int = 0, extra_pickers: int = 0):
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

    def initial_state(self) -> SimWarehouseDomain:
        self._domain = self.replay.domain()
        template = self._domain.resources.resources[0]
        for picker in self.extra_pickers:
            extra = copy.deepcopy(template)
            extra.id = picker
            self._domain.resources.resources.append(extra)
        if self._fit is not None:
            # Truth: the realized times. Belief: the resources plan with the means.
            self._domain.truth = process_truth(
                self._fit, self.seed, _day_key(self.day), self._domain.orders.orders,
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
        arrivals = [PickerArrival(time=arrival, picker_id=picker)
                    for picker, (arrival, _) in self.extra_pickers.items()]
        return iter(sorted([*self.replay.events(self._domain), *arrivals],
                           key=lambda event: event.time))
