"""Simulator-owned decision bookkeeping and domain snapshot types."""

from dataclasses import dataclass
from enum import Enum

from ware_ops_algos.domain_models.base_domain import BaseWarehouseDomain
from ware_ops_algos.planning.models import PlanView


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


@dataclass(frozen=True)
class DynamicInfo:
    context: DecisionContext
    plan: PlanView
    memory: DecisionMemory


class SimWarehouseDomain(BaseWarehouseDomain):
    def __init__(self, *args, dynamic_warehouse_info: DynamicInfo, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_warehouse_info = dynamic_warehouse_info


def initial_dynamic_info(time: float = 0.0) -> DynamicInfo:
    """Initial empty decision data for a scenario loader."""
    return DynamicInfo(
        context=DecisionContext(
            decision="", cause=(), reconsider=(), version=0,
            input_closed=False, solve=True,
        ),
        plan=PlanView(time=time),
        memory=DecisionMemory(),
    )
