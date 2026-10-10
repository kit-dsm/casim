from ware_ops_algos.algorithms import SShapeRouting, WalkOrWaitSShapeRouting, ReroutableSShapeRouting, \
    ReroutableWalkOrWaitSShapeRouting, LargestGapRouting, MidpointRouting, ReturnRouting, \
    NearestNeighbourhoodRouting, ExactTSPRoutingDistance, RatliffRosenthalRouting, UShapeRouting

from casim.domain_objects.sim_domain import ActiveTourRoutingSolution
from casim.pipelines.problem_based_template import PickerRouting, dump_pickle, load_pickle


class SShape(PickerRouting):
    router_class = SShapeRouting
    uses_active_origin = False

    def _get_inited_router(self, *, routing_origin=None, picker=None):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        return self.router_class(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=[picker] if picker is not None else resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
            routing_origin=routing_origin if self.uses_active_origin else None,
        )


class WalkOrWaitSShape(SShape):
    """Select the S-shape traversal used by the original waiting simulator."""

    router_class = WalkOrWaitSShapeRouting


class _ActiveTourRoutingNode:
    """Route each admitted tour from its own projected origin."""

    def run(self):
        batching_sol = load_pickle(self.input()["batching_sol"]["batching_sol"].path)
        dynamic = self._load_dynamic_info()
        tours = {tour.tour_id: tour for tour in dynamic.admission_tours}
        pickers = {picker.id: picker for picker in self._load_resources().resources}
        routes = []
        execution_time = 0.0
        algo_name = ""
        for batch in batching_sol.batches:
            tour = tours[batch.tour_id]
            router = self._get_inited_router(
                routing_origin=tour.routing_origin,
                picker=pickers[tour.picker_id],
            )
            routing_solution = router.solve(batch.pick_positions)
            route = routing_solution.route
            route.batch = batch
            routes.append(route)
            algo_name = routing_solution.algo_name
            execution_time += routing_solution.execution_time
        dump_pickle(self.output()["routing_sol"].path, ActiveTourRoutingSolution(
            algo_name=algo_name,
            execution_time=execution_time,
            routes=routes,
            considered_pairs=batching_sol.considered_pairs,
        ))


class ReroutableSShape(_ActiveTourRoutingNode, SShape):
    """Use the S-shape variant that re-solves an active residual tour."""

    router_class = ReroutableSShapeRouting
    uses_active_origin = True


class ReroutableWalkOrWaitSShape(_ActiveTourRoutingNode, SShape):
    """Use paper-style S-shape traversal with active rerouting."""

    router_class = ReroutableWalkOrWaitSShapeRouting
    uses_active_origin = True


class LargestGap(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        return LargestGapRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
        )


class Midpoint(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        return MidpointRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
        )


class Return(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        return ReturnRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
        )


class NearestNeighbourhood(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        return NearestNeighbourhoodRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
        )


class TSPRouting(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        return ExactTSPRoutingDistance(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
            set_time_limit=self.pipeline_params.runtime,
        )


class RatliffRosenthal(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        graph_params = layout.graph_data
        layout_network = layout.layout_network
        return RatliffRosenthalRouting(
            start_node=layout.graph_data.start_connection_point,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            n_aisles=graph_params.n_aisles,
            n_pick_locations=graph_params.n_pick_locations,
            dist_aisle=graph_params.dist_aisle,
            dist_pick_locations=graph_params.dist_pick_locations,
            dist_aisle_location=graph_params.dist_bottom_to_pick_location,
            dist_start=graph_params.dist_start,
            dist_end=graph_params.dist_end,
        )

class UShape(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        return UShapeRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
        )
