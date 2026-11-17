"""CoSy routing nodes: each names its router class; ``build_router`` builds it."""

from ware_ops_algos.algorithms import SShapeRouting, WalkOrWaitSShapeRouting, ReroutableSShapeRouting, \
    ReroutableWalkOrWaitSShapeRouting, LargestGapRouting, MidpointRouting, ReturnRouting, \
    NearestNeighbourhoodRouting, ExactTSPRoutingDistance, RatliffRosenthalRouting, UShapeRouting

from casim.pipelines.problem_based_template import AbstractActiveTourRouting, PickerRouting


class SShape(PickerRouting):
    router_class = SShapeRouting


class WalkOrWaitSShape(SShape):
    """Select the S-shape traversal used by the original waiting simulator."""

    router_class = WalkOrWaitSShapeRouting


class ReroutableSShape(AbstractActiveTourRouting):
    """Use the S-shape variant that re-solves an active residual tour."""

    router_class = ReroutableSShapeRouting


class ReroutableWalkOrWaitSShape(AbstractActiveTourRouting):
    """Use paper-style S-shape traversal with active rerouting."""

    router_class = ReroutableWalkOrWaitSShapeRouting


class LargestGap(PickerRouting):
    router_class = LargestGapRouting


class Midpoint(PickerRouting):
    router_class = MidpointRouting


class Return(PickerRouting):
    router_class = ReturnRouting


class NearestNeighbourhood(PickerRouting):
    router_class = NearestNeighbourhoodRouting


class TSPRouting(PickerRouting):
    router_class = ExactTSPRoutingDistance

    def _get_inited_router(self, pickers=None, **options):
        return super()._get_inited_router(
            pickers, set_time_limit=self.pipeline_params.runtime, **options)


class RatliffRosenthal(PickerRouting):
    router_class = RatliffRosenthalRouting


class UShape(PickerRouting):
    router_class = UShapeRouting
