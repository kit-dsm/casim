from ware_ops_algos.algorithms import (
    AislePatrolPositioning, CenterRelocationPositioning,
)

from casim.pipelines.problem_based_template import AbstractPositioning


class AislePatrolPositioningNode(AbstractPositioning):
    def _get_positioner(self):
        return AislePatrolPositioning()


class CenterRelocationPositioningNode(AbstractPositioning):
    def _get_positioner(self):
        return CenterRelocationPositioning()
