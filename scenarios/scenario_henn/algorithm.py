from __future__ import annotations

from ware_ops_algos.algorithms import CombinedRoutingSolution
from ware_ops_algos.algorithms.waiting import (
    HennDecision,
    decide_henn as _decide_henn,
    route_service_time,
)
from ware_ops_algos.domain_models import OrdersDomain

from casim.domain_objects import SimWarehouseDomain
from casim.events.operational_events import Event


class HennWakeUp(Event):
    """Scenario-local event for reconsidering a timed Henn wait."""

    priority_score = 2


def single_order_service_times(
    snapshot: SimWarehouseDomain,
    solver,
    cache: dict[int, float],
) -> dict[int, float]:
    """Route each newly observed order alone with the configured CoSy solver."""
    picker = snapshot.resources.resources[0]
    for order in snapshot.orders.orders or []:
        if order.order_id in cache:
            continue
        single_snapshot = SimWarehouseDomain(
            problem_class=snapshot.problem_class,
            objective=snapshot.objective,
            layout=snapshot.layout,
            articles=snapshot.articles,
            orders=OrdersDomain(
                tpe=snapshot.orders.tpe,
                orders=[order],
            ),
            resources=snapshot.resources,
            storage=snapshot.storage,
            dynamic_warehouse_info=snapshot.dynamic_warehouse_info,
        )
        result = solver.solve(single_snapshot, action=None)
        if result is None:
            raise RuntimeError(
                f"CoSy generated no single-order route for {order.order_id}"
            )
        solution, _, _ = result
        if not isinstance(solution, CombinedRoutingSolution):
            raise TypeError(
                "The Henn CoSy pipeline must return CombinedRoutingSolution"
            )
        if len(solution.routes) != 1:
            raise ValueError(
                f"Expected one route for order {order.order_id}, "
                f"received {len(solution.routes)}"
            )
        cache[order.order_id] = route_service_time(
            solution.routes[0],
            picker,
        )
    return cache


def decide_henn(
    candidate: CombinedRoutingSolution,
    snapshot: SimWarehouseDomain,
    current_time: float,
    next_arrival: float | None,
    stream_exhausted: bool,
    selector: str,
    single_services: dict[int, float],
    waiting_policy: str = "henn_4_1",
    fill_threshold: float = 0.75,
    max_age_s: float = 300.0,
) -> HennDecision:
    """Thin wrapper: extract picker/resources/orders from snapshot and delegate
    to ware_ops_algos.algorithms.waiting.decide_henn."""
    if not snapshot.resources.resources:
        raise ValueError("Henn decision requires an idle picker")
    picker = snapshot.resources.resources[0]
    return _decide_henn(
        candidate=candidate,
        picker=picker,
        resources=snapshot.resources,
        orders=list(snapshot.orders.orders or []),
        current_time=current_time,
        stream_exhausted=stream_exhausted,
        selector=selector,
        single_services=single_services,
        next_arrival=next_arrival,
        waiting_policy=waiting_policy,
        fill_threshold=fill_threshold,
        max_age_s=max_age_s,
    )
