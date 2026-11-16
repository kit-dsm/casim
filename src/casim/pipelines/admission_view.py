"""The started tours an admission decision and its routing see."""

from ware_ops_algos.planning import Accepts, PlanView, TourStates, TourView


def admitting_tours(plan: PlanView) -> tuple[TourView, ...]:
    """Started tours whose revision scope accepts inserted orders, in plan order."""
    return tuple(view for view in plan.tours
                 if view.status is TourStates.STARTED and view.accepts is Accepts.INSERT)
