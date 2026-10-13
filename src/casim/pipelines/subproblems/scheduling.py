from ware_ops_algos.algorithms import EDDScheduling, ERDScheduling, LPTScheduling, SPTScheduling, FIFOScheduling

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


class SPTScheduler(AbstractScheduling):
    def _get_inited_scheduler(self, available_from: dict[int, float]):
        resources = self._load_resources()
        return SPTScheduling(resources, available_from=available_from)


class FIFOScheduler(AbstractScheduling):
    def _get_inited_scheduler(self, available_from: dict[int, float]):
        return FIFOScheduling(self._load_resources(), available_from=available_from)
