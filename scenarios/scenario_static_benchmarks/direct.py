"""Solve a static instance by calling the pipeline's ware_ops_algos algorithms directly.

The algorithms are built with the arguments casim's CoSy components use
(`casim.pipelines.subproblems`), so that comparison 1 of Task B.4 differs only
where CASIM's decision and simulation path differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from ware_ops_algos.algorithms import (
    ClarkAndWrightBatching, DueDateBatching, EDDScheduling, FIFOScheduling, FifoBatching,
    GreedyItemAssignment, LocalSearchBatching, MinMinItemAssignment, NearestNeighbourhoodRouting,
    OrderNrFifoBatching, RatliffRosenthalRouting, SShapeRouting, ShiftNeighborhood, SwapNeighborhood,
)
from ware_ops_algos.algorithms.scheduling.scheduling import build_jobs
from ware_ops_algos.domain_models.base_domain import BaseWarehouseDomain

from scenarios.scenario_static_benchmarks.pipelines import Pipeline

TIME_LIMIT_S = 300


@dataclass(frozen=True)
class DirectResult:
    """Objective values and what is needed to compare them."""

    distance: float
    walked_distance: float | None
    total_time: float
    max_lateness: float | None
    route_distances: tuple[float, ...]
    batches: tuple[tuple[int, ...], ...]
    batching_seconds: float


def _network_kwargs(domain: BaseWarehouseDomain, gen_tour: bool) -> dict:
    network = domain.layout.layout_network
    nodes = list(network.graph.nodes)
    return {
        "start_node": network.start_node,
        "end_node": network.end_node,
        "closest_node_to_start": network.closest_node_to_start,
        "min_aisle_position": network.min_aisle_position,
        "max_aisle_position": network.max_aisle_position,
        "distance_matrix": network.distance_matrix,
        "predecessor_matrix": network.predecessor_matrix,
        "picker": domain.resources.resources,
        "gen_tour": gen_tour,
        "gen_item_sequence": gen_tour,
        "node_list": network.node_list,
        "node_to_idx": {node: index for index, node in enumerate(nodes)},
        "idx_to_node": dict(enumerate(nodes)),
    }


def _rr_kwargs(domain: BaseWarehouseDomain) -> dict:
    graph = domain.layout.graph_data
    return {
        "n_aisles": graph.n_aisles,
        "n_pick_locations": graph.n_pick_locations,
        "dist_aisle": graph.dist_aisle,
        "dist_pick_locations": graph.dist_pick_locations,
        "dist_aisle_location": graph.dist_bottom_to_pick_location,
        "dist_start": graph.dist_start,
        "dist_end": graph.dist_end,
    }


ROUTERS = {"SShape": SShapeRouting, "NearestNeighbourhood": NearestNeighbourhoodRouting,
           "RatliffRosenthal": RatliffRosenthalRouting}
START_BATCHERS = {"FiFo": FifoBatching, "OrderNrFiFo": OrderNrFifoBatching}


def make_item_assignment(name: str, domain: BaseWarehouseDomain):
    if name == "GreedyIA":
        return GreedyItemAssignment(domain.storage)
    network = domain.layout.layout_network
    return MinMinItemAssignment(storage_locations=domain.storage,
                                distance_matrix=network.distance_matrix,
                                start_node=network.start_node)


def _cost_router_kwargs(router: type, domain: BaseWarehouseDomain) -> dict:
    """Router arguments of casim's ConfiguredBatchingNode."""
    kwargs = _network_kwargs(domain, gen_tour=False)
    if router is RatliffRosenthalRouting:
        kwargs.update(_rr_kwargs(domain))
    return kwargs


def make_batcher(pipeline: Pipeline, domain: BaseWarehouseDomain):
    cart = domain.resources.resources[0].pick_cart
    articles = domain.articles
    if pipeline.batching == "FiFo":
        return FifoBatching(pick_cart=cart, articles=articles)
    if pipeline.batching == "DueDate":
        return DueDateBatching(pick_cart=cart, articles=articles)
    router = ROUTERS[pipeline.batching_router]
    if pipeline.batching_start is None:
        return ClarkAndWrightBatching(pick_cart=cart, articles=articles, routing_class=router,
                                      routing_class_kwargs=_cost_router_kwargs(router, domain),
                                      time_limit=TIME_LIMIT_S)
    return LocalSearchBatching(pick_cart=cart, articles=articles, routing_class=router,
                               routing_class_kwargs=_cost_router_kwargs(router, domain),
                               start_batching_class=START_BATCHERS[pipeline.batching_start],
                               neighborhood_classes=[SwapNeighborhood, ShiftNeighborhood],
                               time_limit=TIME_LIMIT_S)


def make_router(name: str, domain: BaseWarehouseDomain):
    """Routers as casim's routing components build them."""
    if name == "RatliffRosenthal":
        network = domain.layout.layout_network
        return RatliffRosenthalRouting(
            start_node=domain.layout.graph_data.start_connection_point,
            end_node=network.end_node,
            closest_node_to_start=network.closest_node_to_start,
            min_aisle_position=network.min_aisle_position,
            max_aisle_position=network.max_aisle_position,
            distance_matrix=network.distance_matrix,
            predecessor_matrix=network.predecessor_matrix,
            picker=domain.resources.resources,
            **_rr_kwargs(domain))
    return ROUTERS[name](**_network_kwargs(domain, gen_tour=True))


def _walked(routes, domain: BaseWarehouseDomain) -> float | None:
    """Length of the routes' node sequences; None if a router builds none (RatliffRosenthal)."""
    matrix = domain.layout.layout_network.distance_matrix
    total = 0.0
    for route in routes:
        if not route.annotated_route:
            return None
        nodes = [node.position for node in route.annotated_route]
        total += sum(matrix.at[a, b] for a, b in zip(nodes, nodes[1:]))
    return total


def solve(pipeline: Pipeline, domain: BaseWarehouseDomain) -> DirectResult:
    """Item assignment, batching, routing and (Kris) scheduling of one static instance."""
    resolved = make_item_assignment(pipeline.item_assignment, domain).solve(
        domain.orders.orders).resolved_orders
    started = perf_counter()
    batches = make_batcher(pipeline, domain).solve(resolved).batches
    batching_seconds = perf_counter() - started
    router = make_router(pipeline.routing, domain)
    routes = []
    for batch in batches:
        route = router.solve(batch.pick_positions).route
        route.batch = batch
        routes.append(route)
        router.reset_parameters()
    jobs = build_jobs(routes, domain.resources, release_time=0.0)
    scheduler = (EDDScheduling if pipeline.scheduling == "EDDScheduler" else FIFOScheduling)(
        domain.resources, available_from={})
    scheduled = scheduler.solve(jobs).jobs
    lateness = [job.end_time - job.job.due_date for job in scheduled
                if job.job.due_date not in (None, float("inf"))]
    return DirectResult(
        distance=sum(route.distance for route in routes),
        walked_distance=_walked(routes, domain),
        total_time=sum(job.processing_time for job in jobs)
        + (domain.resources.resources[0].tour_setup_time or 0.0) * len(jobs),
        max_lateness=max(lateness) if lateness else None,
        route_distances=tuple(sorted(route.distance for route in routes)),
        batches=tuple(sorted(tuple(sorted(order.order_id for order in batch.orders))
                             for batch in batches)),
        batching_seconds=batching_seconds,
    )
