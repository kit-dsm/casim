"""Decision memory and operational facts outside the scoped plan."""

from dataclasses import dataclass

from casim.domain_objects.sim_domain import DecisionMemory


@dataclass(frozen=True)
class OperationsView:
    is_break: bool = False
    n_staged_pallets: int = 0
