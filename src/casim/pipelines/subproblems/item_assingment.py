from ware_ops_algos.algorithms import GreedyItemAssignment, MinMinItemAssignment

from casim.pipelines.problem_based_template import AbstractItemAssignment, load_pickle


class GreedyIA(AbstractItemAssignment):

    def _get_inited_ia(self):
        storage = self._get_storage()
        item_assigner = GreedyItemAssignment(storage)
        return item_assigner


class MinMinIA(AbstractItemAssignment):

    def _get_inited_ia(self):
        network = load_pickle(self.input()["instance"]["layout"].path).layout_network
        return MinMinItemAssignment(
            storage_locations=self._get_storage(),
            distance_matrix=network.distance_matrix,
            start_node=network.start_node,
        )
