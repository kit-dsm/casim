from __future__ import annotations

from pathlib import Path
from typing import Any

from ware_ops_algos.algorithms import (
    PickPosition,
    SchedulingSolution,
    WarehouseOrder,
    exact_lorenz_makespan,
)
from ware_ops_algos.domain_models import DataCard, DimensionType


def _resolved_orders(domain, release_time: float | None) -> list[WarehouseOrder]:
    mapping = domain.storage.article_location_mapping
    return [
        WarehouseOrder(
            order_id=order.order_id,
            parent_order_id=order.parent_order_id,
            due_date=order.due_date,
            order_date=(
                order.order_date if release_time is None else release_time
            ),
            pick_positions=tuple(
                PickPosition(
                    order_number=order.order_id,
                    article_id=position.article_id,
                    amount=position.amount,
                    pick_node=(
                        mapping[position.article_id][0].x,
                        mapping[position.article_id][0].y,
                    ),
                    in_store=position.amount,
                    article_name=position.article_name,
                )
                for position in order.order_positions
            ),
        )
        for order in domain.orders.orders
    ]


class LorenzDPSolver:
    """Expose the integrated Lorenz DP through DecisionEngine's solver contract."""

    def __init__(
        self,
        *,
        batch_capacity_orders: int,
        max_states: int,
        max_runtime_s: float,
        release_mode: str,
        problem_class: str,
        instances_dir: Path,
        cache_dir: Path,
        output_dir: Path,
        instance_name: str,
        verbose: bool = False,
        luigi_cfg: Any = None,
    ):
        if release_mode not in {"actual", "current"}:
            raise ValueError(
                "release_mode must be either 'actual' or 'current'"
            )
        self.batch_capacity_orders = int(batch_capacity_orders)
        self.max_states = int(max_states)
        self.max_runtime_s = float(max_runtime_s)
        self.release_mode = release_mode
        self.problem_class = problem_class
        self.instance_name = instance_name

    def build_pipelines(self, data_card: DataCard) -> None:
        if data_card.problem_class != self.problem_class:
            raise ValueError(
                f"Data card problem {data_card.problem_class} does not match "
                f"configured problem {self.problem_class}"
            )
        if self.problem_class != "OBRSP":
            raise ValueError("LorenzDPSolver supports only OBRSP")
        if data_card.objective != "makespan":
            raise ValueError("LorenzDPSolver optimizes makespan only")

    def solve(
        self,
        dynamic_domain,
        action=None,
    ) -> tuple[SchedulingSolution, str, float]:
        if action not in (None, 0):
            raise ValueError("LorenzDPSolver exposes one deterministic strategy")
        picker = dynamic_domain.resources.resources[0]
        cart = picker.pick_cart
        order_dimensions = [
            index
            for index, dimension in enumerate(cart.dimensions)
            if dimension == DimensionType.ORDERS
        ]
        if len(order_dimensions) != 1:
            raise ValueError(
                "LorenzDPSolver requires exactly one ORDERS cart dimension"
            )
        configured_capacity = int(
            cart.capacities[order_dimensions[0]]
        )
        if configured_capacity != self.batch_capacity_orders:
            raise ValueError(
                "Solver batch_capacity_orders does not match the loaded cart"
            )

        current_time = float(
            dynamic_domain.dynamic_warehouse_info.time or 0.0
        )
        immediate_release = (
            current_time if self.release_mode == "current" else None
        )
        solution = exact_lorenz_makespan(
            _resolved_orders(dynamic_domain, immediate_release),
            dynamic_domain.layout,
            picker,
            self.batch_capacity_orders,
            start_time=current_time,
            max_states=self.max_states,
            max_runtime_s=self.max_runtime_s,
        )
        objective = (
            max(job.end_time for job in solution.jobs)
            if solution.jobs
            else current_time
        )
        return solution, "LorenzMakespanDP", float(objective)
