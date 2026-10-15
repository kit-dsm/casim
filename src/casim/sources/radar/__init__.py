"""Walking vs. Waiting: RADAR benchmark instances, or explicit hand-sized examples."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.sources.events import shift_start_release
from casim.sources.radar.loader import WalkOrWaitDataLoader


class RadarSource:
    """One picker arrives at ``shift_start_s``; the stream closes at the last arrival."""

    def __init__(self, data_dir: Path, *, shift_start_s: float, capacity_orders: int,
                 walking_speed: float, pick_time_s: float,
                 instance: str | None = None, orders: list[dict] | None = None,
                 n_aisles: int | None = None, n_pick_locations: int | None = None,
                 depot_aisle: int | None = None):
        self.shift_start_s = float(shift_start_s)
        self.loader = WalkOrWaitDataLoader(
            data_dir, capacity_orders=capacity_orders, walking_speed=walking_speed,
            pick_time_s=pick_time_s, orders=orders, instance=instance,
            n_aisles=n_aisles, n_pick_locations=n_pick_locations, depot_aisle=depot_aisle)
        self._domain: SimWarehouseDomain | None = None

    def initial_state(self) -> SimWarehouseDomain:
        self._domain = self.loader.load()
        return self._domain

    def events(self) -> Iterator[Event]:
        return shift_start_release(self._domain, self.shift_start_s)
