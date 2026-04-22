import logging

from ware_ops_algos.algorithms import PickList, Route, Job

from casim.events.base_events import ProcessEvent, Event
from casim.events.operational_events import PickerArrival, PickerTourQuery
from casim.state import State

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PickListDone(ProcessEvent):
    def __init__(self, time: float, pick_list: PickList):
        super().__init__(time)
        self.pick_list = pick_list

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        events_to_return = []
        return events_to_return


class RoutingDone(ProcessEvent):
    def __init__(self, time: float, route: Route, picker_id: int | None = None):
        super().__init__(time)
        self.route = route
        self.picker_id = picker_id

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        # state.order_manager.clear_order_buffer(self.route.pick_list.orders)
        # state.add_route_to_planning_state(self.route,
        #                                   self.picker_id)
        events_to_return = []
        picker_id = self.picker_id
        if self.picker_id is not None:
            picker = state.resource_manager.get_resource(picker_id)
            if not picker.occupied:
                events_to_return.append(PickerTourQuery(self.time,
                                                        picker_id))
                state.resource_manager.mark_picker_occupied(picker_id)
        return events_to_return


class SequencingDone(ProcessEvent):
    priority_score = 0
    def __init__(self, time: float, sequencing: Job):
        super().__init__(time)
        self.sequencing = sequencing

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        # state.order_manager.clear_order_buffer(self.sequencing.route.pick_list.orders)
        # state.add_sequencing_to_planning_state(self.sequencing)
        events_to_return = []
        picker_id = self.sequencing.picker_id
        picker = state.resource_manager.get_resource(picker_id)
        if not picker.occupied:
            events_to_return.append(PickerTourQuery(self.time,
                                                    picker_id))
            state.resource_manager.mark_picker_occupied(picker_id)
        return events_to_return