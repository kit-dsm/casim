from ware_ops_algos.algorithms import OrderNrFifoBatching, FifoBatching, DueDateBatching, LocalSearchBatching, \
    NearestNeighbourhoodRouting, SShapeRouting, ClarkAndWrightBatching
from ware_ops_algos.algorithms.batching import IncumbentStartBatching

from cosy_luigi import CoSyLuigiTaskParameter

from casim.pipelines.problem_based_template import (
    BatchingNode, IncumbentProvider, load_pickle,
)


class IncumbentBatchingNode(BatchingNode):
    """Repair candidate: keep selected drafts and insert newly buffered orders."""

    incumbent = CoSyLuigiTaskParameter(IncumbentProvider)

    def _get_inited_batcher(self):
        resources = self._get_resources()
        layout = self._get_layout().layout_network
        router = SShapeRouting(
            start_node=layout.start_node, end_node=layout.end_node,
            closest_node_to_start=layout.closest_node_to_start,
            min_aisle_position=layout.min_aisle_position,
            max_aisle_position=layout.max_aisle_position,
            distance_matrix=layout.distance_matrix,
            predecessor_matrix=layout.predecessor_matrix,
            picker=resources.resources, gen_tour=True,
            gen_item_sequence=True, node_list=layout.node_list,
            node_to_idx={node: index for index, node in
                         enumerate(layout.graph.nodes)},
            idx_to_node={index: node for index, node in
                         enumerate(layout.graph.nodes)},
        )

        def route_cost(batch):
            router.reset_parameters()
            return router.solve(batch.pick_positions).route.distance

        views = load_pickle(self.input()["incumbent"]["incumbent_tours"].path)
        drafts = tuple(view.batch for view in views)
        return IncumbentStartBatching(
            pick_cart=resources.resources[0].pick_cart,
            articles=self._get_articles(), incumbent_batches=drafts,
            route_cost=route_cost,
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
