from ware_ops_algos.algorithms import EDDSequencer

from casim.pipelines.problem_based_template import AbstractSequencing


class EDDSequencing(AbstractSequencing):
    def _get_inited_sequencer(self):
        orders = self._load_orders()
        return EDDSequencer(orders=orders)