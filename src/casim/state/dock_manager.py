"""Mutable dock fill derived from the warehouse dock."""

from ware_ops_algos.domain_models import Dock


class DockManager:
    """Track staged pallets while retaining the warehouse's dock facts."""

    def __init__(self, dock: Dock | None):
        self.dock = dock
        self.capacity = dock.capacity if dock is not None else None
        self.pallets_per_tour = dock.pallets_per_tour if dock is not None else 1
        self.n_staged_pallets = 0

    def stage_pallets(self, n_pallets: int = 1) -> None:
        self.n_staged_pallets += n_pallets

    def release_pallets(self, n_pallets: int = 1) -> None:
        self.n_staged_pallets = max(0, self.n_staged_pallets - n_pallets)
