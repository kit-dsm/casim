from ware_ops_algos.algorithms import GreedyItemAssignment

from casim.pipelines.problem_based_template import AbstractItemAssignment


class GreedyIA(AbstractItemAssignment):

    def _get_inited_ia(self):
        storage = self._get_storage()
        item_assigner = GreedyItemAssignment(storage)
        return item_assigner
