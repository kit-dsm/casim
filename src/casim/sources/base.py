"""The source of a scenario: its initial state and its exogenous events."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Protocol, runtime_checkable

from casim.data_loader import DataLoader
from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event


@runtime_checkable
class Source(Protocol):
    """Gives W_0 and B_0, then the exogenous events in time order."""

    def initial_state(self) -> SimWarehouseDomain: ...

    def events(self) -> Iterator[Event]: ...


LIBRARY: dict[str, str] = {
    "hessler_irnich": "casim.sources.hessler_irnich.HesslerIrnichSource",
    "oobp": "casim.sources.oobp.OobpSource",
    "casop": "casim.sources.casop.CasopSource",
    "foodmart": "casim.sources.casop.FoodmartSource",
    "kris": "casim.sources.casop.KrisSource",
    "radar": "casim.sources.radar.RadarSource",
}
LOCAL = "local"
LOCAL_CLASS = "ScenarioSource"


def _local_class(scenario_dir: Path) -> type:
    path = scenario_dir / "source.py"
    if not path.exists():
        raise ValueError(f"source 'local' needs {path}")
    spec = spec_from_file_location(f"casim_scenario_source_{scenario_dir.name}", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, LOCAL_CLASS)


def source_class(name: str, scenario_dir: Path) -> type:
    """The library source of that name, or the scenario's own ``source.py``."""
    if name == LOCAL:
        return _local_class(scenario_dir)
    if name not in LIBRARY:
        raise ValueError(f"unknown source {name!r}; allowed: {sorted([*LIBRARY, LOCAL])}")
    module, _, cls = LIBRARY[name].rpartition(".")
    return getattr(import_module(module), cls)


def build_source(spec: Mapping, scenario_dir: Path) -> Source:
    """Instantiate ``source: {name, ...parameters}`` with the scenario's data directory."""
    parameters = {key: value for key, value in spec.items() if key != "name"}
    return source_class(spec["name"], scenario_dir)(
        data_dir=scenario_dir / "data", **parameters)


class SourceLoader(DataLoader):
    """Hands a source's initial state to the engine, which loads it at reset."""

    def __init__(self, source: Source):
        super().__init__(".")
        self.source = source

    def load(self, **_kwargs) -> SimWarehouseDomain:
        return self.source.initial_state()


def release(source: Source):
    """A reset hook that queues the source's events, checking their time order."""

    def hook(sim, _domain) -> None:
        last = float("-inf")
        for event in source.events():
            if event.time < last:
                raise ValueError(f"source events out of time order: {type(event).__name__} "
                                 f"at {event.time} after {last}")
            last = event.time
            sim.add_event(event)
    return hook
