"""Decision memory and operational facts outside the scoped plan."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecisionMemory:
    considered_active_orders: frozenset[tuple[int, int]] = frozenset()
    last_information_version: int = 0
    current_information_version: int = 0


@dataclass(frozen=True)
class OperationsView:
    is_break: bool = False
    n_staged_pallets: int = 0
    extra: dict[str, object] = field(default_factory=dict)
