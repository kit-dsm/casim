from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from ware_ops_algos.algorithms import (
    CombinedRoutingSolution,
    BatchObject,
    NodeType,
    Route,
    RouteNode,
    SchedulingSolution,
    WarehouseOrder,
)
from ware_ops_algos.algorithms.scheduling.scheduling import (
    FIFOScheduling,
    build_jobs,
)
from ware_ops_algos.domain_models import OrdersDomain

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.domain_objects.tour_model import TourPlanningState
from casim.events.base_events import Event


class HennWakeUp(Event):
    """Reconsider a scenario-local waiting decision."""

    priority_score = 2


@dataclass(frozen=True)
class HennDecision:
    action: Literal["dispatch", "wait"]
    solution: SchedulingSolution | None
    wait_until: float | None
    reason: str
    details: dict[str, object]


def route_service_time(route: Route, picker) -> float:
    """Return setup, travel, and item-picking time in seconds."""
    if picker.speed is None or picker.speed <= 0:
        raise ValueError("Henn service time requires a positive picker speed")
    if picker.time_per_pick is None:
        raise ValueError("Henn service time requires picker time_per_pick")
    if picker.tour_setup_time is None:
        raise ValueError("Henn service time requires picker tour_setup_time")

    item_count = sum(
        int(position.in_store)
        for position in route.batch.pick_positions
    )
    if route.item_sequence is not None and len(route.item_sequence) != item_count:
        raise ValueError(
            "The routed item sequence does not match the batch item count: "
            f"{len(route.item_sequence)} != {item_count}"
        )
    return (
        float(picker.tour_setup_time)
        + float(route.distance) / float(picker.speed)
        + item_count * float(picker.time_per_pick)
    )


def single_order_service_times(
    snapshot: SimWarehouseDomain,
    solver,
    cache: dict[int, float],
) -> dict[int, float]:
    """Route each newly observed order alone through the stateless solver."""
    picker = snapshot.resources.resources[0]
    for order in snapshot.orders.orders or []:
        if order.order_id in cache:
            continue
        single_snapshot = SimWarehouseDomain(
            problem_class=snapshot.problem_class,
            objective=snapshot.objective,
            layout=snapshot.layout,
            articles=snapshot.articles,
            orders=OrdersDomain(tpe=snapshot.orders.tpe, orders=[order]),
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
        cache[order.order_id] = route_service_time(solution.routes[0], picker)
    return cache


def _route_key(route: Route) -> tuple[int, tuple[int, ...]]:
    return (
        int(route.batch.batch_id),
        tuple(sorted(int(value) for value in route.batch.order_numbers)),
    )


def _selection_value(
    route: Route,
    selector: str,
    picker,
    single_services: dict[int, float],
) -> float:
    service = route_service_time(route, picker)
    if selector == "first":
        return float(route.batch.batch_id)
    if selector == "short":
        return service
    if selector == "long":
        return -service
    if selector == "sav":
        saving = (
            sum(single_services[order_id] for order_id in route.batch.order_numbers)
            - service
        )
        return -saving
    raise ValueError(f"Unknown Henn selection rule: {selector!r}")


def _select_route(
    routes: list[Route],
    selector: str,
    picker,
    single_services: dict[int, float],
) -> Route:
    return min(
        routes,
        key=lambda route: (
            _selection_value(route, selector, picker, single_services),
            _route_key(route),
        ),
    )


def _order_routes(
    routes: list[Route],
    selector: str,
    picker,
    single_services: dict[int, float],
) -> list[Route]:
    remaining = list(routes)
    ordered: list[Route] = []
    while remaining:
        selected = _select_route(remaining, selector, picker, single_services)
        ordered.append(selected)
        remaining.remove(selected)
    return ordered


def _schedule(
    routes: list[Route],
    snapshot: SimWarehouseDomain,
    current_time: float,
    selector: str,
) -> SchedulingSolution:
    jobs = build_jobs(routes, snapshot.resources, release_time=current_time)
    solution = FIFOScheduling(snapshot.resources).solve(jobs)
    solution.algo_name = f"Henn_{selector.upper()}"
    return solution


def decide_henn(
    candidate: CombinedRoutingSolution,
    snapshot: SimWarehouseDomain,
    current_time: float,
    input_closed: bool,
    selector: str,
    single_services: dict[int, float],
    waiting_policy: str = "henn_4_1",
    fill_threshold: float = 0.75,
    max_age_s: float = 300.0,
) -> HennDecision:
    """Select a batch and decide whether to dispatch it, without mutation."""
    if not isinstance(candidate, CombinedRoutingSolution):
        raise TypeError("Henn candidates must be CombinedRoutingSolution")
    if not candidate.routes:
        raise ValueError("Open orders produced no candidate route")
    if not snapshot.resources.resources:
        raise ValueError("Henn decision requires an idle picker")
    if waiting_policy not in {"henn_4_1", "no_wait", "fill_or_age"}:
        raise ValueError(f"Unknown Henn waiting policy: {waiting_policy!r}")

    picker = snapshot.resources.resources[0]
    routes = list(candidate.routes)
    details: dict[str, object] = {
        "candidate_batches": [
            {
                "batch_id": route.batch.batch_id,
                "order_ids": sorted(route.batch.order_numbers),
                "distance": float(route.distance),
                "service_time_s": route_service_time(route, picker),
            }
            for route in sorted(routes, key=_route_key)
        ],
        "selector": selector,
        "input_closed": input_closed,
        "waiting_policy": waiting_policy,
    }

    if input_closed:
        selected = _order_routes(routes, selector, picker, single_services)
        details["selected_order_ids"] = [
            sorted(route.batch.order_numbers) for route in selected
        ]
        return HennDecision(
            "dispatch",
            _schedule(selected, snapshot, current_time, selector),
            None,
            "input_closed_dispatch_all",
            details,
        )

    if waiting_policy == "fill_or_age":
        orders = list(snapshot.orders.orders or [])
        visible_items = sum(
            int(getattr(position, "in_store", getattr(position, "amount", 0)))
            for order in orders
            for position in (
                getattr(order, "pick_positions", None)
                or getattr(order, "order_positions", ())
            )
        )
        capacity = max(1.0, float(picker.capacity or 1.0))
        fill = min(1.0, visible_items / capacity)
        oldest_release = min(
            (float(order.order_date or 0.0) for order in orders),
            default=current_time,
        )
        oldest_age = current_time - oldest_release
        details.update({
            "fill": fill,
            "fill_threshold": fill_threshold,
            "oldest_age_s": oldest_age,
            "max_age_s": max_age_s,
        })
        if fill < fill_threshold and oldest_age < max_age_s:
            return HennDecision(
                "wait",
                None,
                oldest_release + max_age_s,
                "fill_or_age_below_threshold",
                details,
            )

    if len(routes) > 1:
        selected = _select_route(routes, selector, picker, single_services)
        details["selected_order_ids"] = sorted(selected.batch.order_numbers)
        return HennDecision(
            "dispatch",
            _schedule([selected], snapshot, current_time, selector),
            None,
            "multiple_batches_select_one",
            details,
        )

    route = routes[0]
    if waiting_policy in {"no_wait", "fill_or_age"}:
        details["selected_order_ids"] = sorted(route.batch.order_numbers)
        return HennDecision(
            "dispatch",
            _schedule([route], snapshot, current_time, selector),
            None,
            f"{waiting_policy}_release",
            details,
        )

    critical_order = min(
        route.batch.orders,
        key=lambda order: (
            -single_services[order.order_id],
            int(order.order_id),
        ),
    )
    critical_service = single_services[critical_order.order_id]
    batch_service = route_service_time(route, picker)
    threshold = (
        2 * float(critical_order.order_date)
        + critical_service
        - batch_service
    )
    details.update({
        "critical_order_id": int(critical_order.order_id),
        "critical_order_service_time_s": critical_service,
        "batch_service_time_s": batch_service,
        "threshold_s": threshold,
    })

    if current_time < threshold:
        return HennDecision(
            "wait",
            None,
            threshold,
            "single_batch_threshold",
            details,
        )

    details["selected_order_ids"] = sorted(route.batch.order_numbers)
    return HennDecision(
        "dispatch",
        _schedule([route], snapshot, current_time, selector),
        None,
        "single_batch_release",
        details,
    )


def insert_into_active_tour(
    candidate: CombinedRoutingSolution,
    tour: TourPlanningState,
    current_position: RouteNode,
    layout,
) -> Route | None:
    """Insert feasible newly routed picks into the active pick sequence.

    The OBRP candidate establishes batching feasibility and resolves storage.
    This function then preserves the active tour's remaining pick order and
    inserts only new picks at their cheapest positions.
    """
    old_ids = set(tour.order_numbers)
    eligible = [
        route
        for route in candidate.routes
        if old_ids.issubset(route.batch.order_numbers)
        and set(route.batch.order_numbers) - old_ids
    ]
    if not eligible:
        return None
    selected = min(
        eligible,
        key=lambda route: (
            -len(set(route.batch.order_numbers) - old_ids),
            float(route.distance),
            _route_key(route),
        ),
    )
    inserted_orders = [
        deepcopy(order)
        for order in selected.batch.orders
        if order.order_id not in old_ids
    ]

    residual_by_order = {
        order_id: tuple(
            deepcopy(pick)
            for pick in tour.remaining_picks
            if pick.order_number == order_id
        )
        for order_id in tour.order_numbers
    }
    residual_orders = [
        WarehouseOrder(
            order_id=order.order_id,
            parent_order_id=order.parent_order_id,
            due_date=order.due_date,
            order_date=order.order_date,
            pick_positions=residual_by_order[order.order_id],
        )
        for order in tour.batch.orders
    ]
    replacement_batch = BatchObject(
        batch_id=tour.batch.batch_id,
        orders=[*residual_orders, *inserted_orders],
    )

    remaining_nodes = [
        deepcopy(node)
        for node in tour.annotated_route[tour.cursor + 1:]
        if node.node_type == NodeType.PICK
    ]
    end = deepcopy(tour.annotated_route[-1])
    current_is_unpicked = bool(
        tour.remaining_picks
        and tour.remaining_picks[0].pick_node == current_position.position
    )
    start = RouteNode(
        current_position.position,
        NodeType.PICK if current_is_unpicked else NodeType.ROUTE,
    )
    stops = [start, *remaining_nodes, end]
    distance_matrix = layout.layout_network.distance_matrix

    def distance(left: RouteNode, right: RouteNode) -> float:
        return float(distance_matrix.at[left.position, right.position])

    new_picks = sorted(
        [pick for order in inserted_orders for pick in order.pick_positions],
        key=lambda pick: (
            pick.order_number,
            pick.article_id,
            pick.pick_node,
            pick.amount,
        ),
    )
    for pick in new_picks:
        node = RouteNode(pick.pick_node, NodeType.PICK)
        insert_at = min(
            range(1, len(stops)),
            key=lambda index: (
                distance(stops[index - 1], node)
                + distance(node, stops[index])
                - distance(stops[index - 1], stops[index]),
                index,
            ),
        )
        stops.insert(insert_at, node)

    total_distance = sum(
        distance(left, right)
        for left, right in zip(stops, stops[1:])
    )
    return Route(
        distance=total_distance,
        route=[node.position for node in stops],
        item_sequence=[
            node.position for node in stops if node.node_type == NodeType.PICK
        ],
        batch=replacement_batch,
        annotated_route=stops,
    )
