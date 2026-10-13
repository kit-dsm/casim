from ware_ops_algos.algorithms import (
    AislePatrolPositioning, CenterRelocationPositioning, Positioning,
)

from casim.pipelines.problem_based_template import AbstractPositioning


class AislePatrolPositioningNode(AbstractPositioning):
    def _get_positioner(self) -> Positioning:
        return AislePatrolPositioning()


class CenterRelocationPositioningNode(AbstractPositioning):
    def _get_positioner(self) -> Positioning:
        return CenterRelocationPositioning()
