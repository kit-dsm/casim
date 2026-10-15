"""The OOBP archive of Gil-Borrás et al. (2020): Henn (W5) and Albareda-Sambola (W1 to W4)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.sources.events import first_arrival_release
from casim.sources.oobp.loader import AlbaredaLoader, OobpHennLoader

FORMATS = ("henn", "albareda")


class OobpSource:
    """``filepath`` relative to the data directory; the stream closes at the last
    arrival, or at ``closure_s`` when given (an announced closure)."""

    def __init__(self, data_dir: Path, filepath: str, format: str = "henn",
                 depot: str = "corner", closure_s: float | None = None):
        if format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {format!r}")
        self.loader = (OobpHennLoader(data_dir, depot=depot) if format == "henn"
                       else AlbaredaLoader(data_dir))
        self.filepath = filepath
        self.closure_s = closure_s
        self._domain: SimWarehouseDomain | None = None

    def initial_state(self) -> SimWarehouseDomain:
        self._domain = self.loader.load(self.filepath)
        return self._domain

    def events(self) -> Iterator[Event]:
        return first_arrival_release(self._domain, self.closure_s)
