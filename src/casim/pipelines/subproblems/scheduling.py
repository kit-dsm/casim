from ware_ops_algos.algorithms import EDDScheduling, ERDScheduling, LPTScheduling, SPTScheduling

from casim.pipelines.problem_based_template import AbstractScheduling


class EDDScheduler(AbstractScheduling):
    def _get_inited_scheduler(self):
        return EDDScheduling()


class ERDScheduler(AbstractScheduling):
    def _get_inited_scheduler(self):
        return ERDScheduling()


class LPTScheduler(AbstractScheduling):
    def _get_inited_scheduler(self):
        return LPTScheduling()


class SPTScheduler(AbstractScheduling):
    def _get_inited_scheduler(self):
        return SPTScheduling()
