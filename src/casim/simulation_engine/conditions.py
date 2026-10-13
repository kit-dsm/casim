"""Feasibility gates over the typed decision snapshot."""

from ware_ops_algos.domain_models import SimWarehouseDomain, TourStates
from ware_ops_algos.domain_models.lifecycle import Accepts


class Condition:
    def get_decision(self, state: SimWarehouseDomain) -> bool:
        raise NotImplementedError


class NewInformationCondition(Condition):
    """Skip solving a draft again when the input has not changed."""

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        return (state.memory.current_information_version !=
                state.memory.last_information_version)


class BreakCondition(Condition):
    def get_decision(self, state: SimWarehouseDomain) -> bool:
        return not state.operations.is_break


class DockCapacityCondition(Condition):
    def __init__(self, threshold: int):
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        return (state.operations.n_staged_pallets + len(state.plan.tours)
                <= self.threshold)


class NbrPickersCondition(Condition):
    def __init__(self, threshold: int):
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        return len(state.plan.resources) >= self.threshold


class NbrOrdersCondition(Condition):
    def __init__(self, threshold: int, allow_when_done: bool = False):
        self.threshold = threshold
        self.allow_when_done = allow_when_done

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        count = len(state.plan.orders)
        return (count >= self.threshold or
                self.allow_when_done and state.context.input_closed and count > 0)


class VisiblePlanningWorkCondition(Condition):
    """A horizon contains a buffered order or a revisable scheduled tour."""

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        return bool(state.plan.orders or any(
            tour.status == TourStates.SCHEDULED and tour.accepts == Accepts.REBUILD
            for tour in state.plan.tours))


class ActiveTourReadyCondition(Condition):
    """A visible candidate can be considered once the current pick has ended."""

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        return ((bool(state.plan.active_candidate_ids) or
                 state.context.input_closed and any(
                     tour.positioning_mode == "center_relocation"
                     and tour.status == TourStates.STARTED
                     and not tour.batch.order_numbers for tour in state.plan.tours)) and
                any(tour.status == TourStates.STARTED and
                    tour.accepts in (Accepts.INSERT, Accepts.REVISE) and
                    tour.picking_until is None for tour in state.plan.tours))


class NbrBatchesCondition(Condition):
    def __init__(self, threshold: int):
        self.threshold = threshold

    def get_decision(self, state: SimWarehouseDomain) -> bool:
        count = (len(state.plan.pick_lists) +
                 sum(tour.accepts == Accepts.REBUILD for tour in state.plan.tours))
        return count >= self.threshold
