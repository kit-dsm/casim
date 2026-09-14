from ware_ops_algos.algorithms import (
    ClarkAndWrightBatching,
    DueDateBatching,
    FifoBatching,
    ItemAssignmentSolution,
    LocalSearchBatching,
    NearestNeighbourhoodRouting,
    OrderNrFifoBatching,
    ResidualBatchingInput,
    ResidualFifoBatching,
    SShapeRouting,
)

from casim.pipelines.problem_based_template import (
    BatchingNode,
    dump_pickle,
    load_pickle,
)


class OrderNrFiFo(BatchingNode):
    def _get_inited_batcher(self):
        articles = self._get_articles()
        resources = self._get_resources()
        return OrderNrFifoBatching(pick_cart=resources.resources[0].pick_cart, articles=articles)


class FiFo(BatchingNode):
    def _get_inited_batcher(self):
        articles = self._get_articles()
        resources = self._get_resources()
        return FifoBatching(pick_cart=resources.resources[0].pick_cart, articles=articles)


class ResidualFiFo(BatchingNode):
    """CoSy component for deterministic filling of empty active-tour bins."""

    def run(self):
        dynamic_info = load_pickle(
            self.input()["instance"]["dynamic_warehouse_info"].path
        )
        resources = self._get_resources()
        ia_solution: ItemAssignmentSolution = load_pickle(
            self.input()["item_assignment_sol"]["item_assignment_sol"].path
        )
        solution = ResidualFifoBatching().solve(
            ResidualBatchingInput(
                active_batch=dynamic_info.buffered_batches[0],
                candidate_orders=tuple(ia_solution.resolved_orders),
                bin_order_ids=tuple(dynamic_info.cart_bin_order_ids),
                locked_bin_ids=frozenset(dynamic_info.locked_bin_ids),
                pick_cart=resources.resources[0].pick_cart,
            )
        )
        dump_pickle(self.output()["batching_sol"].path, solution)


class DueDate(BatchingNode):
    def _get_inited_batcher(self):
        articles = self._get_articles()
        resources = self._get_resources()
        return DueDateBatching(pick_cart=resources.resources[0].pick_cart, articles=articles)


class LSBatchingNNFiFo(BatchingNode):
    def _get_inited_batcher(self):
        articles = self._get_articles()
        resources = self._get_resources()
        layout = self._get_layout()
        layout_network = layout.layout_network
        routing_kwargs = {
            "start_node": layout_network.start_node,
            "end_node": layout_network.end_node,
            "closest_node_to_start": layout_network.closest_node_to_start,
            "min_aisle_position": layout_network.min_aisle_position,
            "max_aisle_position": layout_network.max_aisle_position,
            "distance_matrix": layout_network.distance_matrix,
            "predecessor_matrix": layout_network.predecessor_matrix,
            "picker": resources.resources,
            "gen_tour": True,
            "gen_item_sequence": True,
            "node_list": layout_network.node_list,
            "node_to_idx": {node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            "idx_to_node": {idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
        }
        return LocalSearchBatching(
            pick_cart=resources.resources[0].pick_cart,
            articles=articles,
            routing_class=NearestNeighbourhoodRouting,
            routing_class_kwargs=routing_kwargs,
            start_batching_class=FifoBatching,
            time_limit=self.pipeline_params.runtime,
        )


class LSBatchingSShapeFiFo(BatchingNode):
    def _get_inited_batcher(self):
        articles = self._get_articles()
        resources = self._get_resources()
        layout = self._get_layout()
        layout_network = layout.layout_network
        routing_kwargs = {
            "start_node": layout_network.start_node,
            "end_node": layout_network.end_node,
            "closest_node_to_start": layout_network.closest_node_to_start,
            "min_aisle_position": layout_network.min_aisle_position,
            "max_aisle_position": layout_network.max_aisle_position,
            "distance_matrix": layout_network.distance_matrix,
            "predecessor_matrix": layout_network.predecessor_matrix,
            "picker": resources.resources,
            "gen_tour": True,
            "gen_item_sequence": True,
            "node_list": layout_network.node_list,
            "node_to_idx": {node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            "idx_to_node": {idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
        }
        return LocalSearchBatching(
            pick_cart=resources.resources[0].pick_cart,
            articles=articles,
            routing_class=SShapeRouting,
            routing_class_kwargs=routing_kwargs,
            start_batching_class=FifoBatching,
            time_limit=self.pipeline_params.runtime,
        )


class ClarkAndWrightNN(BatchingNode):
    def _get_inited_batcher(self):
        articles = self._get_articles()
        resources = self._get_resources()
        layout = self._get_layout()
        layout_network = layout.layout_network
        routing_kwargs = {
            "start_node": layout_network.start_node,
            "end_node": layout_network.end_node,
            "closest_node_to_start": layout_network.closest_node_to_start,
            "min_aisle_position": layout_network.min_aisle_position,
            "max_aisle_position": layout_network.max_aisle_position,
            "distance_matrix": layout_network.distance_matrix,
            "predecessor_matrix": layout_network.predecessor_matrix,
            "picker": resources.resources,
            "gen_tour": False,
            "gen_item_sequence": False,
        }
        return ClarkAndWrightBatching(
            pick_cart=resources.resources[0].pick_cart,
            articles=articles,
            routing_class=NearestNeighbourhoodRouting,
            routing_class_kwargs=routing_kwargs,
            time_limit=self.pipeline_params.runtime,
        )


class ClarkAndWrightSShape(BatchingNode):
    def _get_inited_batcher(self):
        articles = self._get_articles()
        resources = self._get_resources()
        layout = self._get_layout()
        layout_network = layout.layout_network
        routing_kwargs = {
            "start_node": layout_network.start_node,
            "end_node": layout_network.end_node,
            "closest_node_to_start": layout_network.closest_node_to_start,
            "min_aisle_position": layout_network.min_aisle_position,
            "max_aisle_position": layout_network.max_aisle_position,
            "distance_matrix": layout_network.distance_matrix,
            "predecessor_matrix": layout_network.predecessor_matrix,
            "picker": resources.resources,
            "gen_tour": False,
            "gen_item_sequence": False,
        }
        return ClarkAndWrightBatching(
            pick_cart=resources.resources[0].pick_cart,
            articles=articles,
            routing_class=SShapeRouting,
            routing_class_kwargs=routing_kwargs,
            time_limit=self.pipeline_params.runtime,
        )
