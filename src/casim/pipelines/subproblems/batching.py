from ware_ops_algos.algorithms import OrderNrFifoBatching, FifoBatching, RemainingRouteFifoBatching, DueDateBatching, LocalSearchBatching, \
    NearestNeighbourhoodRouting, SShapeRouting, ClarkAndWrightBatching

from casim.pipelines.problem_based_template import BatchingNode, load_pickle


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


class RemainingRouteFiFo(BatchingNode):
    """Configured paper admission rule, using a detached residual-route projection."""

    def _get_inited_batcher(self):
        dynamic = load_pickle(self.input()["instance"]["dynamic_warehouse_info"].path)
        resources = self._get_resources()
        if len(resources.resources) != 1 or dynamic.active_tour_id is None:
            raise ValueError("Remaining-route admission needs one active picker")
        return RemainingRouteFifoBatching(
            pick_cart=resources.resources[0].pick_cart,
            articles=self._get_articles(),
            active_order_ids=dynamic.active_order_ids,
            candidate_order_ids=dynamic.active_candidate_ids,
            remaining_route=dynamic.remaining_route_positions,
            occupied_bins=dynamic.occupied_bins,
        )


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
