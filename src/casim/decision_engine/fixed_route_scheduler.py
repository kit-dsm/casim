"""Concrete direct routing/scheduling used after external batching decisions."""

from ware_ops_algos.algorithms import SShapeRouting
from ware_ops_algos.algorithms.scheduling.scheduling import (
    FIFOScheduling,
    build_jobs,
)


class FixedRouteScheduler:
    """Route buffered batches by S-shape and schedule them FIFO.

    This is the direct counterpart of the existing CoSy
    PickListProvider -> SShape -> FIFO pipeline.  It exists so high-frequency
    learning runs can use the same semantic decision boundary without a
    scenario-local routing/scheduling copy.
    """

    def __init__(self, *, problem_class: str):
        self.problem_class = str(problem_class)
        self._router = None
        self._layout = None
        self.solve_calls = 0

    def prepare(self, data_card) -> None:
        if self.problem_class != "ORSP":
            raise ValueError("FixedRouteScheduler supports only ORSP")
        if data_card.problem_class != self.problem_class:
            raise ValueError(
                f"Data card problem {data_card.problem_class} does not match "
                f"configured problem {self.problem_class}"
            )

    def router_for(self, snapshot):
        layout = snapshot.layout
        if self._router is not None and self._layout is layout:
            return self._router
        network = layout.layout_network
        self._router = SShapeRouting(
            start_node=network.start_node,
            end_node=network.end_node,
            closest_node_to_start=network.closest_node_to_start,
            min_aisle_position=network.min_aisle_position,
            max_aisle_position=network.max_aisle_position,
            distance_matrix=network.distance_matrix,
            predecessor_matrix=network.predecessor_matrix,
            picker=snapshot.resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=network.node_list,
            node_to_idx={
                node: index
                for index, node in enumerate(network.graph.nodes)
            },
            idx_to_node={
                index: node
                for index, node in enumerate(network.graph.nodes)
            },
        )
        self._layout = layout
        return self._router

    def solve(self, snapshot, action=None):
        if action not in (None, 0):
            raise ValueError("FixedRouteScheduler has one deterministic strategy")
        router = self.router_for(snapshot)
        routes = []
        routing_time = 0.0
        for batch in snapshot.dynamic_warehouse_info.buffered_batches or []:
            routed = router.solve(batch.pick_positions)
            routed.route.batch = batch
            routes.append(routed.route)
            routing_time += float(routed.execution_time)
            self.solve_calls += 1
        jobs = build_jobs(
            routes,
            snapshot.resources,
            release_time=float(snapshot.dynamic_warehouse_info.time or 0.0),
        )
        solution = FIFOScheduling(snapshot.resources).solve(jobs)
        solution.execution_time += routing_time
        solution.algo_name = "SShapeRouting+FIFOScheduling"
        objective = (
            max((job.end_time for job in solution.jobs), default=0.0)
            if snapshot.objective == "makespan"
            else sum(route.distance for route in routes)
        )
        return solution, solution.algo_name, float(objective)
