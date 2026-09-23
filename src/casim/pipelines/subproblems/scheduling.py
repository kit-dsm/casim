from ware_ops_algos.algorithms import EDDScheduling, ERDScheduling, LPTScheduling, SPTScheduling, FIFOScheduling

from casim.pipelines.problem_based_template import AbstractScheduling


class EDDScheduler(AbstractScheduling):
    def _get_inited_scheduler(self):
        resources = self._load_resources()
        return EDDScheduling(resources)


class ERDScheduler(AbstractScheduling):
    def _get_inited_scheduler(self):
        resources = self._load_resources()
        return ERDScheduling(resources)


class LPTScheduler(AbstractScheduling):
    def _get_inited_scheduler(self):
        resources = self._load_resources()
        return LPTScheduling(resources)


class SPTScheduler(AbstractScheduling):
    def _get_inited_scheduler(self):
        resources = self._load_resources()
        return SPTScheduling(resources)


class FIFOScheduler(AbstractScheduling):
    def _get_inited_scheduler(self):
        return FIFOScheduling(self._load_resources())
