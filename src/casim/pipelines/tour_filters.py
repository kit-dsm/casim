"""The tours of a decision's plan that each kind of CoSy stage may revise."""

from ware_ops_algos.planning import Accepts, PlanView, TourStates, TourView


def admitting_tours(plan: PlanView) -> tuple[TourView, ...]:
    """Started tours whose revision scope accepts inserted orders, in plan order."""
    return tuple(view for view in plan.tours
                 if view.status is TourStates.STARTED and view.accepts is Accepts.INSERT)


def rebuildable_tours(plan: PlanView) -> tuple[TourView, ...]:
    """Tours whose revision scope permits rebuilding them, in plan order."""
    return tuple(view for view in plan.tours if view.accepts is Accepts.REBUILD)
