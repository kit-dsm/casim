"""Simulator-owned decision bookkeeping and domain snapshot types."""

from dataclasses import dataclass
from enum import Enum


class ReconsiderReason(str, Enum):
    WAIT = "wait"
    COMMITMENT_WINDOW = "commitment_window"
    PERIODIC = "periodic"


@dataclass(frozen=True)
class DecisionContext:
    decision: str
    cause: tuple[str, ...]
    reconsider: tuple[ReconsiderReason, ...]
    version: int
    input_closed: bool
    solve: bool


@dataclass(frozen=True)
class DecisionMemory:
    considered_active_orders: frozenset[tuple[int, int]] = frozenset()
    last_information_version: int = 0
    current_information_version: int = 0
    waiting: bool = False
    reconsider_at: float | None = None
