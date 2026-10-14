from ware_ops_algos.algorithms import (
    Batching, ClarkAndWrightBatching, DueDateBatching, FifoBatching, LocalSearchBatching,
    OrderNrFifoBatching, RatliffRosenthalRouting, ShiftNeighborhood, SwapNeighborhood,
)
from ware_ops_algos.algorithms.batching import IncumbentStartBatching
from ware_ops_algos.algorithms.batching.moves import Neighborhood
from ware_ops_algos.algorithms.routing.routing import Routing

from cosy_luigi import CoSyLuigiTaskParameter

from casim.pipelines.problem_based_template import (
    BatchingNode, IncumbentProvider, load_pickle,
)


class IncumbentBatchingNode(BatchingNode):
    """Repair candidate: keep selected drafts and insert newly buffered orders."""

    incumbent = CoSyLuigiTaskParameter(IncumbentProvider)
    routing_class: type[Routing] | None = None

    def _get_inited_batcher(self):
        resources = self._get_resources()
        layout = self._get_layout().layout_network
        router_class = self.routing_class
        if router_class is None:
            raise ValueError("Incumbent batching needs a configured routing class")
        carts = [resource.pick_cart for resource in resources.resources]
        if not carts or carts[0] is None or any(cart != carts[0] for cart in carts):
            raise ValueError("IncumbentStartBatching requires homogeneous pick carts")
        router = router_class(
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

        views = load_pickle(self.input()["incumbent"]["incumbent_tours"].path)
        drafts = tuple(view.batch for view in views)
        return IncumbentStartBatching(
            pick_cart=carts[0],
            articles=self._get_articles(), incumbent_batches=drafts,
            router=router,
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


class ConfiguredBatchingNode(BatchingNode):
    """Batching whose cost is a route length, configured from its card."""

    routing_class: type[Routing] | None = None

    def _routing_kwargs(self) -> dict:
        """Router arguments as ware_ops_pipes' configured batching passes them."""
        resources = self._get_resources()
        layout = self._get_layout()
        network = layout.layout_network
        nodes = list(network.graph.nodes)
        kwargs = {
            "start_node": network.start_node,
            "end_node": network.end_node,
            "closest_node_to_start": network.closest_node_to_start,
            "min_aisle_position": network.min_aisle_position,
            "max_aisle_position": network.max_aisle_position,
            "distance_matrix": network.distance_matrix,
            "predecessor_matrix": network.predecessor_matrix,
            "picker": resources.resources,
            "gen_tour": False,
            "gen_item_sequence": False,
            "node_list": network.node_list,
            "node_to_idx": {node: index for index, node in enumerate(nodes)},
            "idx_to_node": dict(enumerate(nodes)),
        }
        if self.routing_class is RatliffRosenthalRouting:
            graph = layout.graph_data
            kwargs.update({
                "n_aisles": graph.n_aisles,
                "n_pick_locations": graph.n_pick_locations,
                "dist_aisle": graph.dist_aisle,
                "dist_pick_locations": graph.dist_pick_locations,
                "dist_aisle_location": graph.dist_bottom_to_pick_location,
                "dist_start": graph.dist_start,
                "dist_end": graph.dist_end,
            })
        return kwargs


class ConfiguredLocalSearchNode(ConfiguredBatchingNode):
    """LocalSearchBatching with the neighbourhoods of ware_ops_pipes' generator."""

    start_batching_class: type[Batching] | None = None
    neighborhood_classes: tuple[type[Neighborhood], ...] = (SwapNeighborhood, ShiftNeighborhood)

    def _get_inited_batcher(self):
        return LocalSearchBatching(
            pick_cart=self._get_resources().resources[0].pick_cart,
            articles=self._get_articles(),
            routing_class=self.routing_class,
            routing_class_kwargs=self._routing_kwargs(),
            start_batching_class=self.start_batching_class,
            neighborhood_classes=list(self.neighborhood_classes),
            time_limit=self.pipeline_params.runtime,
        )


class ConfiguredClarkAndWrightNode(ConfiguredBatchingNode):
    """ClarkAndWrightBatching configured with its routing class."""

    def _get_inited_batcher(self):
        return ClarkAndWrightBatching(
            pick_cart=self._get_resources().resources[0].pick_cart,
            articles=self._get_articles(),
            routing_class=self.routing_class,
            routing_class_kwargs=self._routing_kwargs(),
            time_limit=self.pipeline_params.runtime,
        )
