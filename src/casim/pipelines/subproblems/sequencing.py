from ware_ops_algos.algorithms import SPTSequencer, EDDSequencer

from casim.pipelines.problem_based_template import AbstractSequencing


class EDDSequencing(AbstractSequencing):
    def _get_inited_sequencer(self):
        return EDDSequencer()


class SPTSequencing(AbstractSequencing):
    def _get_inited_sequencer(self):
        return SPTSequencer()