"""One historic picking day of the WMS export, replayed with its recorded decisions.

Lines are assigned to days by timestamp (the order's pick date, R0.4 of
``docs/ijpe/plan.md``), and located on the warehouse graph of
``data/layout.csv`` by (aisle, house, place). Left out, and counted in
``excluded``: lines without a pick node in the layout (aisle 10 and one
location in aisle 7), and orders picked by more than one picker, which a
recorded tour cannot hold. Pickers are numbered per day in the order of their
first line; ``casim.sources.recorded`` builds the state, the recorded plan
and the pauses.
"""

from __future__ import annotations

from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path

import pandas as pd

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.sources.grid_layout import GridLayout
from casim.sources.recorded import RecordedReplay, RecordedRow, TimeModel
from casim.sources.wms_export import day_lines


DEPARTURE_HOURS = (10, 12, 14, 16)
STAGING_MINUTES = 30


def fixed_departure_hour(customer: str) -> int:
    """Synthetic daily store slot, independent of recorded picking times."""
    digest = sha256(str(customer).encode("utf-8")).digest()
    return DEPARTURE_HOURS[int.from_bytes(digest[:8], "big") % len(DEPARTURE_HOURS)]


class ScenarioSource:
    def __init__(self, data_dir: Path, *, day: str, shift_start_s: float,
                 pause_threshold_s: float | None, speed: float, time_per_pick: float,
                 tour_setup_time: float, per_unit_s: float | None = None,
                 interleaving: bool = True, breaks: list | None = None,
                 origin_mode: str = "depot", setup_in_gap: bool = False,
                 sequence_clock: str = "start", presence: str = "shift",
                 due_plan: str | None = None, stops_in_tours: bool = False):
        if due_plan not in (None, "fixed_dispatch"):
            raise ValueError(f"unknown IJPE due plan: {due_plan}")
        self.due_plan = due_plan
        layout, locations = GridLayout().build(Path(data_dir) / "layout.csv")
        lines, self.excluded = day_lines(str(day), GridLayout.pick_nodes(locations),
                                         Path(data_dir) / "cache")
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
        self.replay = RecordedReplay(
            rows, layout, TimeModel(float(speed), float(time_per_pick), float(tour_setup_time),
                                    None if per_unit_s is None else float(per_unit_s)),
            shift_start_s=float(shift_start_s),
            pause_threshold_s=None if pause_threshold_s is None else float(pause_threshold_s),
            interleaving=bool(interleaving), breaks=breaks, origin_mode=origin_mode,
            setup_in_gap=setup_in_gap, sequence_clock=sequence_clock, presence=presence,
            stops_in_tours=stops_in_tours)
        self._domain: SimWarehouseDomain | None = None

    def initial_state(self) -> SimWarehouseDomain:
        self._domain = self.replay.domain()
        if self.due_plan == "fixed_dispatch":
            for order in self._domain.orders.orders:
                hour = fixed_departure_hour(self.customer_by_order[str(order.order_id)])
                order.due_date = hour * 3600 - STAGING_MINUTES * 60
        return self._domain

    def events(self) -> Iterator[Event]:
        return self.replay.events(self._domain)
