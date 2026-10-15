"""The CASOP benchmark sets as static instances, read with ware_ops_pipes' loaders.

A static instance releases every order at t = 0; every picker arrives at 0 and
the stream closes at 0, so one decision solves the whole instance.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.sources.casop.loader import LOADERS, load_casop_domain, to_static_sim_domain
from casim.sources.events import shift_start_release


class CasopSource:
    """``instance`` of ``instance_set`` below ``instances`` in the data directory."""

    def __init__(self, data_dir: Path, instance_set: str, instance: str,
                 instances: str = "instances"):
        if instance_set not in LOADERS:
            raise ValueError(f"instance_set must be one of {sorted(LOADERS)}, got {instance_set!r}")
        self.root = Path(data_dir) / instances
        self.instance_set = instance_set
        self.instance = instance
        self._domain: SimWarehouseDomain | None = None

    def initial_state(self) -> SimWarehouseDomain:
        self._domain = to_static_sim_domain(
            load_casop_domain(self.root, self.instance_set, self.instance), self.instance_set)
        return self._domain

    def events(self) -> Iterator[Event]:
        return shift_start_release(self._domain, 0.0, closure_s=0.0)


class FoodmartSource(CasopSource):
    """The Foodmart set (Wahlen 2026)."""

    def __init__(self, data_dir: Path, instance: str, instances: str = "instances"):
        super().__init__(data_dir, "FoodmartData", instance, instances)


class KrisSource(CasopSource):
    """The Kris sets (Briant et al.), with their published speed, pick time and setup."""

    def __init__(self, data_dir: Path, instance: str, size: str = "small",
                 instances: str = "instances"):
        sets = {"small": "KrisSmallDataCorrected", "large": "KrisLargeData"}
        if size not in sets:
            raise ValueError(f"size must be small or large, got {size!r}")
        super().__init__(data_dir, sets[size], instance, instances)
