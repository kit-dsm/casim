"""Heßler-Irnich instance files with Henn's arrival times (the converted Henn set, F60)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.sources.events import first_arrival_release
from casim.sources.hessler_irnich.henn_online_loader import HennOnlineLoader


class HesslerIrnichSource:
    """``filepath`` relative to the data directory; arrival files in ``arrival_dir``."""

    def __init__(self, data_dir: Path, filepath: str, arrival_dir: str = ".",
                 due_offset_s: float | None = None):
        self.loader = HennOnlineLoader(Path(data_dir) / arrival_dir, due_offset_s=due_offset_s)
        self.filepath = Path(data_dir) / filepath
        self._domain: SimWarehouseDomain | None = None

    def initial_state(self) -> SimWarehouseDomain:
        self._domain = self.loader.load(str(self.filepath))
        return self._domain

    def events(self) -> Iterator[Event]:
        return first_arrival_release(self._domain)
