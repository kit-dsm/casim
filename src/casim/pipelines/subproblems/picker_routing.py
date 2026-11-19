"""CoSy routing nodes: each names its router class; ``build_router`` builds it."""

from ware_ops_algos.algorithms import SShapeRouting, WalkOrWaitSShapeRouting, ReroutableSShapeRouting, \
    ReroutableWalkOrWaitSShapeRouting, LargestGapRouting, MidpointRouting, ReturnRouting, \
    NearestNeighbourhoodRouting, ExactTSPRoutingDistance, RatliffRosenthalRouting, UShapeRouting

from collections import defaultdict

from ware_ops_algos.algorithms import CombinedRoutingSolution, GivenSequenceRouting
from casim.pipelines.problem_based_template import (
    AbstractActiveTourRouting, AbstractPickerRouting, PickerRouting, dump_pickle, load_pickle,
    recorded_plan,
)
from casim.pipelines.tour_filters import rebuildable_tours


class SShape(PickerRouting):
    router_class = SShapeRouting


class WalkOrWaitSShape(SShape):
    """Select the S-shape traversal used by the original waiting simulator."""

    router_class = WalkOrWaitSShapeRouting


class ReroutableSShape(AbstractActiveTourRouting):
    """Use the S-shape variant that re-solves an active residual tour."""

    router_class = ReroutableSShapeRouting


class ReroutableWalkOrWaitSShape(AbstractActiveTourRouting):
    """Use paper-style S-shape traversal with active rerouting."""

    router_class = ReroutableWalkOrWaitSShapeRouting


class LargestGap(PickerRouting):
    router_class = LargestGapRouting


class Midpoint(PickerRouting):
    router_class = MidpointRouting


class Return(PickerRouting):
    router_class = ReturnRouting


class NearestNeighbourhood(PickerRouting):
    router_class = NearestNeighbourhoodRouting


class TSPRouting(PickerRouting):
    router_class = ExactTSPRoutingDistance

    def _get_inited_router(self, pickers=None, **options):
        return super()._get_inited_router(
            pickers, set_time_limit=self.pipeline_params.runtime, **options)


class RatliffRosenthal(PickerRouting):
    router_class = RatliffRosenthalRouting


class UShape(PickerRouting):
    router_class = UShapeRouting


class PlannedRoutes(AbstractPickerRouting):
    """Keep the route each planned tour already has; route nothing.

    The routing counterpart of ``PickListProvider``: a decision that only
    schedules tours an earlier decision planned and routed takes each tour's
    route as it is, so a route is computed once. Every batch must be the
    batch of a planned tour the decision may revise, with a route from the
    depot; a pick list without a route needs a router instead.
    """

    def run(self):
        batching_sol = load_pickle(self.input()["batching_sol"]["batching_sol"].path)
        planned = {view.batch.order_numbers: view.route
                   for view in rebuildable_tours(self._load_plan()) if view.route is not None}
        routes = []
        for batch in batching_sol.batches:
            route = planned.get(batch.order_numbers)
            if route is None:
                raise ValueError(f"PlannedRoutes: batch {batch.batch_id} (orders "
                                 f"{sorted(batch.order_numbers)}) has no planned route")
            route.batch = batch
            routes.append(route)
        dump_pickle(self.output()["routing_sol"].path, CombinedRoutingSolution(
            algo_name="PlannedRoutes", execution_time=0.0, routes=routes))


class GivenSequence(PickerRouting):
    """Route each batch through its recorded line sequence (GivenSequenceRouting).

    The sequence of a batch is its recorded tour's lines; each line is matched
    to the batch's pick position of the same order and article, so that the
    route uses the item assignment's pick nodes, recorded or not.
    """

    def run(self):
        plan = recorded_plan(self)
        batching_sol = load_pickle(self.input()["batching_sol"]["batching_sol"].path)
        network = self._load_layout().layout_network
        nodes = list(network.graph.nodes)
        routes, execution_time = [], 0.0
        for batch in batching_sol.batches:
            recorded_tour = plan.tour(batch.batch_id)
            pending = defaultdict(list)
            for position in batch.pick_positions:
                pending[(position.order_number, position.article_id)].append(position.pick_node)
            sequence = [(line.order_id, line.article_id,
                         pending[(line.order_id, line.article_id)].pop(0))
                        for line in recorded_tour.lines]
            router = GivenSequenceRouting(
                start_node=recorded_tour.start_node or network.start_node,
                end_node=network.end_node,
                closest_node_to_start=network.closest_node_to_start,
                min_aisle_position=network.min_aisle_position,
                max_aisle_position=network.max_aisle_position,
                distance_matrix=network.distance_matrix,
                predecessor_matrix=network.predecessor_matrix,
                picker=self._load_resources().resources, gen_tour=True, gen_item_sequence=True,
                node_list=network.node_list,
                node_to_idx={node: index for index, node in enumerate(nodes)},
                idx_to_node=dict(enumerate(nodes)), sequence=sequence)
            solution = router.solve(batch.pick_positions)
            solution.route.batch = batch
            routes.append(solution.route)
            execution_time += solution.execution_time
        dump_pickle(self.output()["routing_sol"].path, CombinedRoutingSolution(
            algo_name=GivenSequenceRouting.algo_name, execution_time=execution_time,
            routes=routes))
