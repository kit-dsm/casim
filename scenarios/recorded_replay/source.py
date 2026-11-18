"""The recorded-replay verification scenario: recorded lines written out in ``scenario.yaml``.

Each line is (picker, order, article, quantity, [aisle, location], start, end)
on a conventional layout with unit distances; ``casim.sources.recorded``
turns them into the state, the recorded plan and the pauses.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.sources.layout import conventional_layout
from casim.sources.recorded import RecordedReplay, RecordedRow, TimeModel


class ScenarioSource:
    def __init__(self, data_dir: Path, *, lines: list, shift_start_s: float, n_aisles: int,
                 n_pick_locations: int, speed: float, time_per_pick: float,
                 tour_setup_time: float, pause_threshold_s: float | None,
                 per_unit_s: float | None = None, setup_in_gap: bool = False, presence: str = "shift"):
        rows = [RecordedRow(picker=int(picker), order=int(order), article=int(article),
                            quantity=int(quantity), node=(int(node[0]), int(node[1])),
                            start_s=float(start), end_s=float(end))
                for picker, order, article, quantity, node, start, end in lines]
        self.replay = RecordedReplay(
            rows, conventional_layout(n_aisles=int(n_aisles), n_pick_locations=int(n_pick_locations)),
            TimeModel(float(speed), float(time_per_pick), float(tour_setup_time),
                      None if per_unit_s is None else float(per_unit_s)),
            shift_start_s=float(shift_start_s),
            pause_threshold_s=None if pause_threshold_s is None else float(pause_threshold_s),
            setup_in_gap=setup_in_gap, presence=presence)
        self._domain: SimWarehouseDomain | None = None

    def initial_state(self) -> SimWarehouseDomain:
        self._domain = self.replay.domain()
        return self._domain

    def events(self) -> Iterator[Event]:
        return self.replay.events(self._domain)
