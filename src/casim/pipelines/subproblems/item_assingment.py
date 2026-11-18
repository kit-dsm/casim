from ware_ops_algos.algorithms import GreedyItemAssignment, MinMinItemAssignment, RecordedItemAssignment

from casim.pipelines.problem_based_template import AbstractItemAssignment, load_pickle, recorded_plan


class GreedyIA(AbstractItemAssignment):

    def _get_inited_ia(self):
        storage = self._get_storage()
        item_assigner = GreedyItemAssignment(storage)
        return item_assigner


class RecordedIA(AbstractItemAssignment):
    """Keep the recorded pick node for every order position."""

    def _get_inited_ia(self):
        return RecordedItemAssignment(self._get_storage(), recorded_plan(self))


class MinMinIA(AbstractItemAssignment):

    def _get_inited_ia(self):
        network = load_pickle(self.input()["instance"]["layout"].path).layout_network
        return MinMinItemAssignment(
            storage_locations=self._get_storage(),
            distance_matrix=network.distance_matrix,
            start_node=network.start_node,
        )
