from ware_ops_algos.algorithms import (
    Batching, ClarkAndWrightBatching, DueDateBatching, FifoBatching, FirstFitFifoBatching,
    LocalSearchBatching, OrderNrFifoBatching, ShiftNeighborhood, SwapNeighborhood,
)
from ware_ops_algos.algorithms.batching import IncumbentStartBatching
from ware_ops_algos.algorithms.batching.moves import Neighborhood
from ware_ops_algos.algorithms.routing.routing import Routing

from cosy_luigi import CoSyLuigiTaskParameter

from casim.pipelines.problem_based_template import (
    BatchingNode, IncumbentProvider, load_pickle,
)
from casim.pipelines.subproblems.routers import build_router, router_arguments


class IncumbentBatchingNode(BatchingNode):
    """Repair candidate: keep selected drafts and insert newly buffered orders."""

    incumbent = CoSyLuigiTaskParameter(IncumbentProvider)
    routing_class: type[Routing] | None = None

    def _get_inited_batcher(self):
        resources = self._get_resources()
        router_class = self.routing_class
        if router_class is None:
            raise ValueError("Incumbent batching needs a configured routing class")
        carts = [resource.pick_cart for resource in resources.resources]
        if not carts or carts[0] is None or any(cart != carts[0] for cart in carts):
            raise ValueError("IncumbentStartBatching requires homogeneous pick carts")
        router = build_router(router_class, self._get_layout(), resources.resources)

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


class FirstFitFiFo(BatchingNode):
    def _get_inited_batcher(self):
        articles = self._get_articles()
        resources = self._get_resources()
        return FirstFitFifoBatching(pick_cart=resources.resources[0].pick_cart,
                                    articles=articles)


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
        return router_arguments(self.routing_class, self._get_layout(),
                                self._get_resources().resources,
                                gen_tour=False, gen_item_sequence=False)


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
