from ware_ops_algos.algorithms import (
    EDDScheduling, ERDScheduling, LPTScheduling, SPTScheduling,
    FIFOScheduling, SAVScheduling,
)
from ware_ops_algos.algorithms.routing.routing import Routing

from casim.pipelines.problem_based_template import AbstractScheduling, load_pickle
from casim.pipelines.subproblems.routers import build_router


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
        return load_pickle(self.input()["instance"]["layout"].path)

    def _get_inited_scheduler(self, available_from: dict[int, float]):
        resources = self._load_resources()
        if self.routing_class is None:
            raise ValueError("SAVScheduler needs a configured routing_class")
        router = build_router(self.routing_class, self._load_layout(), resources.resources)
        return SAVScheduling(resources, router=router,
                             available_from=available_from)
