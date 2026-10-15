from ware_ops_algos.algorithms import (
    EDDScheduling, ERDScheduling, LPTScheduling, SPTScheduling,
    FIFOScheduling, SAVScheduling,
)
from ware_ops_algos.algorithms.routing.routing import Routing

from casim.pipelines.problem_based_template import AbstractScheduling


class EDDScheduler(AbstractScheduling):
    def _get_inited_scheduler(self, available_from: dict[int, float]):
        resources = self._load_resources()
        return EDDScheduling(resources, available_from=available_from)


class ERDScheduler(AbstractScheduling):
    def _get_inited_scheduler(self, available_from: dict[int, float]):
        resources = self._load_resources()
        return ERDScheduling(resources, available_from=available_from)


class LPTScheduler(AbstractScheduling):
    def _get_inited_scheduler(self, available_from: dict[int, float]):
        resources = self._load_resources()
        return LPTScheduling(resources, available_from=available_from)


class HennLongScheduler(LPTScheduler):
    """Henn's LONG rule: rank batches by service time, with no due dates."""


class SPTScheduler(AbstractScheduling):
    def _get_inited_scheduler(self, available_from: dict[int, float]):
        resources = self._load_resources()
        return SPTScheduling(resources, available_from=available_from)


class FIFOScheduler(AbstractScheduling):
    def _get_inited_scheduler(self, available_from: dict[int, float]):
        return FIFOScheduling(self._load_resources(), available_from=available_from)


class SAVScheduler(AbstractScheduling):
    """Henn's SAV selection; builds its own router from the configured class."""

    routing_class: type[Routing] | None = None

    def _load_layout(self):
        from ware_ops_algos.domain_models import LayoutData
        from casim.pipelines.problem_based_template import load_pickle
        return load_pickle(self.input()["instance"]["layout"].path)

    def _get_inited_scheduler(self, available_from: dict[int, float]):
        resources = self._load_resources()
        network = self._load_layout().layout_network
        if self.routing_class is None:
            raise ValueError("SAVScheduler needs a configured routing_class")
        router = self.routing_class(
            start_node=network.start_node, end_node=network.end_node,
            closest_node_to_start=network.closest_node_to_start,
            min_aisle_position=network.min_aisle_position,
            max_aisle_position=network.max_aisle_position,
            distance_matrix=network.distance_matrix,
            predecessor_matrix=network.predecessor_matrix,
            picker=resources.resources, gen_tour=True, gen_item_sequence=True,
            node_list=network.node_list,
            node_to_idx={node: index for index, node in enumerate(network.graph.nodes)},
            idx_to_node=dict(enumerate(network.graph.nodes)),
        )
        return SAVScheduling(resources, router=router,
                             available_from=available_from)
