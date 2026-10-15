from ware_ops_algos.algorithms import (
    EDDScheduling, ERDScheduling, LPTScheduling, SPTScheduling,
    FIFOScheduling, SAVScheduling,
)

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
    def _get_inited_scheduler(self, available_from: dict[int, float]):
        routing_task = self.requires()["routing_sol"]
        return SAVScheduling(self._load_resources(),
                             router=routing_task._get_inited_router(),
                             available_from=available_from)
