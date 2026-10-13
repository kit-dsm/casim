"""Conditions for solving a selected decision snapshot."""

from ware_ops_algos.planning.lifecycle import Accepts, TourStates

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.state.views import OperationsView


class Condition:
    def get_decision(self, state_snapshot: SimWarehouseDomain,
                     operations: OperationsView) -> bool:
        raise NotImplementedError


class NewInformationCondition(Condition):
    """Solve only when information has changed since the last solution."""

    def get_decision(self, state_snapshot, operations):
        memory = state_snapshot.dynamic_warehouse_info.memory
        return memory.current_information_version != memory.last_information_version


class BreakCondition(Condition):
    def get_decision(self, state_snapshot, operations):
        return not operations.is_break


class DockCapacityCondition(Condition):
    def __init__(self, threshold: int):
        self.threshold = threshold

    def get_decision(self, state_snapshot, operations):
        plan = state_snapshot.dynamic_warehouse_info.plan
        return operations.n_staged_pallets + len(plan.tours) <= self.threshold


class NbrPickersCondition(Condition):
    def __init__(self, threshold: int):
        self.threshold = threshold

    def get_decision(self, state_snapshot, operations):
        return len(state_snapshot.resources.resources) >= self.threshold


class NbrOrdersCondition(Condition):
    def __init__(self, threshold: int, allow_when_done: bool = False):
        self.threshold = threshold
        self.allow_when_done = allow_when_done

    def get_decision(self, state_snapshot, operations):
        dynamic = state_snapshot.dynamic_warehouse_info
        count = len(dynamic.plan.order_buffer)
        return (count >= self.threshold or
                self.allow_when_done and dynamic.context.input_closed and count > 0)


class VisiblePlanningWorkCondition(Condition):
    """A horizon contains buffered work or a revisable draft."""

    def get_decision(self, state_snapshot, operations):
        plan = state_snapshot.dynamic_warehouse_info.plan
        return bool(plan.order_buffer or any(
            tour.status is TourStates.PLANNED and tour.accepts is Accepts.REBUILD
            for tour in plan.tours))


class ActiveTourReadyCondition(Condition):
    """A visible order or closure return can use a ready active tour."""

    def get_decision(self, state_snapshot, operations):
        dynamic = state_snapshot.dynamic_warehouse_info
        plan = dynamic.plan
        closure_return = (dynamic.context.input_closed and any(
            tour.holding and tour.status is TourStates.STARTED
            and not tour.batch.order_numbers for tour in plan.tours))
        ready = any(tour.status is TourStates.STARTED
                    and tour.accepts is Accepts.INSERT
                    and tour.picking_until is None for tour in plan.tours)
        return bool(plan.order_buffer or closure_return) and ready


class NbrBatchesCondition(Condition):
    def __init__(self, threshold: int):
        self.threshold = threshold

    def get_decision(self, state_snapshot, operations):
        plan = state_snapshot.dynamic_warehouse_info.plan
        count = (len(plan.pick_list_buffer) +
                 sum(tour.accepts is Accepts.REBUILD for tour in plan.tours))
        return count >= self.threshold
