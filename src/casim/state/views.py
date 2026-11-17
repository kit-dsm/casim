"""Decision memory and operational facts outside the scoped plan."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationsView:
    n_staged_pallets: int = 0
    dock_capacity: int | None = None
