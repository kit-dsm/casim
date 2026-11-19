"""Simulator-owned decision bookkeeping and domain snapshot types."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

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


class ExecutionTimes(Protocol):
    """Realized execution times of a run, from the source (C8 of docs/ijpe/plan.md).

    Without it, a tour's setup takes the resource's ``tour_setup_time`` and a
    line its ``pick_time``. With it, execution asks it for each setup and each
    line; planning still uses the resource's deterministic times, so the
    planners see expected times and the run sees realized ones. The decisions
    never see this object.
    """

    def setup_s(self, picker_id: int, order_numbers: Sequence[int]) -> float: ...

    def pick_s(self, picker_id: int, pick) -> float: ...


class SimWarehouseDomain(BaseWarehouseDomain):
    def __init__(self, *args, dynamic_warehouse_info: DynamicInfo,
                 execution_times: ExecutionTimes | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_warehouse_info = dynamic_warehouse_info
        self.execution_times = execution_times


def initial_dynamic_info(time: float = 0.0) -> DynamicInfo:
    """Initial empty decision data for a scenario loader."""
    return DynamicInfo(
        context=DecisionContext(
            decision="", cause=(), reconsider=(),
            input_closed=False, solve=True,
        ),
        plan=PlanView(time=time),
        memory=DecisionMemory(),
    )
