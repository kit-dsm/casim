import logging
import sys

from ware_ops_algos.algorithms import Route, Job, PickerAssignment, PickList, NodeType, RouteNode, WarehouseOrder
from ware_ops_algos.domain_models import Order

from ware_ops_sim.sim.state import SimulationState
# from ware_sim.sim.sim_utils import logger
from ware_ops_sim.sim.state.tour_manager import TourPlanningState, TourStates

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Event:
    event_counter = 0
    priority_score = 1  # Default priority for operational events

    def __init__(self, time: float):
        self.time = time
        self.id = Event.event_counter
        Event.event_counter += 1

    def __lt__(self, other: 'Event'):
        """Sorts by time first, then by priority_score (lower is higher priority),
           then by ID."""
        if self.time != other.time:
            return self.time < other.time

        # Tie-breaker logic
        if self.priority_score != other.priority_score:
            return self.priority_score < other.priority_score

        return self.id < other.id

    def __le__(self, other: 'Event'):
        return self.__lt__(other) or self.__eq__(other)

    def __eq__(self, other: 'Event'):
        return (self.time == other.time and
                self.priority_score == other.priority_score and
                self.id == other.id)

    def handle(self, state: 'SimulationState') -> list['Event']:
        # if self.time > state.current_time:
        # assert self.time >= state.current_time, f"{self.time}, {state.current_time}, {self}"
        #     state.current_time = self.time
        return []


class ProcessEvent(Event):
    """
    A ProcessEvent represents events created as a result of a decision.
    It contains the respective solution for the decision problem which
    is added to the state.
    May return status update events to initiate operational EventChain.
    """
    priority_score = 0
    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: SimulationState) -> list[Event]:
        super().handle(state)
        return []


class OrderSelectionDone(ProcessEvent):
    def __init__(self, time: float, selected_order: WarehouseOrder,
                 picker_id: int | None = None):
        super().__init__(time)
        self.selected_order = selected_order
        self.picker_id = picker_id

    def handle(self, state: SimulationState) -> list[Event]:
        super().handle(state)
        # print("Selected Order added")
        state.add_selected_order_to_planning_state(self.selected_order,
                                                   self.picker_id)
        return []


class PickListSelectionDone(ProcessEvent):
    def __init__(self, time: float, selected_order: PickList,
                 picker_id: int | None = None):
        super().__init__(time)
        self.selected_order = selected_order
        self.picker_id = picker_id

    def handle(self, state: SimulationState) -> list[Event]:
        super().handle(state)
        # print("Selected Order added")
        # state.add_selected_order_to_planning_state(self.selected_order,
        #                                            self.picker_id)
        state.add_selected_pick_list_to_planning_state(self.selected_order,
                                                       self.picker_id)
        return []

class ItemAssignmentDone(ProcessEvent):
    def __init__(self, time: float, resolved_orders: list[Order]):
        super().__init__(time)
        self.resolved_orders = resolved_orders

    def handle(self, state: SimulationState) -> list[Event]:
        super().handle(state)
        state.add_ia_to_planning_state(self.resolved_orders)
        return []


class PickListDone(ProcessEvent):
    def __init__(self, time: float, pick_list: PickList):
        super().__init__(time)
        self.pick_list = pick_list

    def handle(self, state: SimulationState) -> list[Event]:
        super().handle(state)
        events_to_return = []
        state.add_pick_list_to_planning_state(self.pick_list)
        # TODO We push pick lists to pickers here. Maybe add PLDonePush / PLDonePull?

        for picker in state.resource_manager.get_resources().resources:
            if not picker.occupied:
                events_to_return.append(PickerArrival(self.time, picker.id))
        return events_to_return


class PickListDonePull(ProcessEvent):
    def __init__(self, time: float, pick_list: PickList):
        super().__init__(time)
        self.pick_list = pick_list

    def handle(self, state: SimulationState) -> list[Event]:
        super().handle(state)
        events_to_return = []
        state.add_pick_list_to_planning_state(self.pick_list)
        return events_to_return


class RoutingDone(ProcessEvent):
    def __init__(self, time: float, route: Route, picker_id: int | None = None):
        super().__init__(time)
        self.route = route
        self.picker_id = picker_id

    def handle(self, state: SimulationState) -> list[Event]:
        super().handle(state)
        state.add_route_to_planning_state(self.route,
                                          self.picker_id)
        events_to_return = []
        picker_id = self.picker_id
        if self.picker_id is not None:
            picker = state.resource_manager.get_resource(picker_id)
            if not picker.occupied:
                events_to_return.append(PickerTourQuery(self.time,
                                                        picker_id))
                # print("Routing Done", picker_id, events_to_return)
                state.resource_manager.mark_picker_occupied(picker_id)
        return events_to_return


# class AssignmentDone(ProcessEvent):
#     def __init__(self, time: float, assignment: PickerAssignment):
#         super().__init__(time)
#         self.assignment = assignment
#
#     def handle(self, state: SimulationState) -> list[Event]:
#         super().handle(state)
#         state.add_assignment_to_planning_state(self.assignment)
#         events_to_return = []
#         picker_id = self.assignment.picker.id
#         picker = state.resource_manager.get_resource(picker_id)
#         # print("Assignment", self.assignment)
#         offset = 0
#         if not picker.occupied:
#             events_to_return.append(PickerTourQuery(self.time + offset,
#                                                     picker_id))
#             state.resource_manager.mark_picker_occupied(picker_id)
#
#         return events_to_return


class SequencingDone(ProcessEvent):
    def __init__(self, time: float, sequencing: Job):
        super().__init__(time)
        self.sequencing = sequencing

    def handle(self, state: SimulationState) -> list[Event]:
        super().handle(state)
        # print("SequencingDone")
        pick_nodes = [n for n in self.sequencing.route.annotated_route if n.node_type == NodeType.PICK]
        logger.info("SequencingDone", len(pick_nodes))
        state.add_sequencing_to_planning_state(self.sequencing)
        events_to_return = []
        picker_id = self.sequencing.picker_id
        picker = state.resource_manager.get_resource(picker_id)
        if not picker.occupied:
            events_to_return.append(PickerTourQuery(self.time,
                                                    picker_id))
            state.resource_manager.mark_picker_occupied(picker_id)
        return events_to_return


class OrderArrival(Event):
    """Order enters the system and is buffered."""
    priority_score = 0
    def __init__(self, time: float, order: Order):
        super().__init__(time)
        self.order = order

    def handle(self, state: SimulationState) -> list[Event]:
        super().handle(state)
        logger.info(f"Order {self.order.order_id} arrived at t={self.time}")
        state.order_manager.add_order_to_buffer(self.order)
        return []


class ShiftStart(Event):
    priority_score = 0
    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: 'SimulationState') -> list['Event']:
        super().handle(state)
        return []


class FlushRemainingOrders(Event):
    priority_score = 0
    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: 'SimulationState') -> list['Event']:
        super().handle(state)
        return []

class PickerArrival(Event):
    priority_score = 1
    def __init__(self, time: float, picker_id: int):
        super().__init__(time)
        self.picker_id = picker_id

    def handle(self, state: 'SimulationState') -> list['Event']:
        super().handle(state)
        # state.current_picker_id = self.picker_id
        # print(f"Picker {self.picker_id} arrived at {self.time}")
        return []


class PickerIdle(Event):
    priority_score = 1
    def __init__(self, time: float, picker_id: int):
        super().__init__(time)
        self.picker_id = picker_id

    def handle(self, state: 'SimulationState') -> list['Event']:
        super().handle(state)
        # print(f"Picker {self.picker_id} arrived at {self.time}")
        return []


class BreakStart(Event):
    def __init__(self, time: float, duration: float):
        super().__init__(time)
        self.duration = duration

    def handle(self, state: 'SimulationState') -> list['Event']:
        super().handle(state)
        state.is_break = True
        state.break_duration = self.duration
        return [BreakEnd(self.time + self.duration)]


class BreakEnd(Event):
    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: 'SimulationState') -> list['Event']:
        super().handle(state)
        state.is_break = False
        return []



class OrderPriorityChange(Event):
    """Handles the change of an orders priority. E.g. changes the due date"""
    def __init__(self, time: float, order: Order):
        super().__init__(time)
        self.order = order

    def handle(self, state: 'SimulationState') -> list[Event]:
        for order in state.order_manager.get_order_buffer():
            if order.order_id == self.order.order_id:
                order.due_date = self.order.due_date
        return []


class PickerTourQuery(Event):
    """
    A picker registers and queries a pick tour.
    There <should> always be a tour available.
    This means he/she has no active tour or just started the shift.
    We look if a tour for the picker is registered in the planning state.
    If not we either take a neutral pick tour or delay the query.
    The start of the tour is scheduled according to the schedule.
    """

    def __init__(self, time: float, picker_id: int):
        super().__init__(time)
        self.picker_id = picker_id

    def handle(self, state: SimulationState) -> list[Event]:
        super().handle(state)
        picker = state.resource_manager.get_resource(
            self.picker_id)
        # state.current_picker_id = self.picker_id
        # picker.notified = False
        next_tour_id = state.tour_manager.get_next_tour_for_picker(
            self.picker_id)
        logger.info(f"Picker {picker.id} has pending tour with id={next_tour_id} at {self.time}")
        # print((f"Picker {picker.id} has pending tour with id={next_tour_id} at {self.time}"))
        if state.is_break:
            logger.info(f"Break scheduled until {self.time + state.break_duration}")
            return [PickerTourQuery(self.time + state.break_duration, self.picker_id)]
        if next_tour_id is not None:
            # There exists an assigned tour for the picker
            next_tour = state.tour_manager.get_tour(next_tour_id)
            start_time = state.current_time
            if next_tour.start_time:
                # The tour is scheduled, we use the start time of the tour
                # Otherwise we greedily start the tour right away
                # start_time = next_tour.start_time
                start_time = self.time
            next_tour.status = TourStates.PENDING
            if picker.tour_setup_time:
                start_time += picker.tour_setup_time # TODO Check if setup times are not added twice
            return [TourStart(start_time, next_tour_id)]
        else:
            # Nothing to do right now, picker is idle
            picker.occupied = False
            return [PickerIdle(state.current_time, picker.id)]


class BaseTourEvent(Event):
    priority_score = 1
    def __init__(self, time: float, tour_id: int):
        super().__init__(time)
        self.tour_id = tour_id

    def get_tour(self, state: SimulationState) -> TourPlanningState:
        return state.tour_manager.get_tour(self.tour_id)


class TourStart(BaseTourEvent):
    def handle(self, state: 'SimulationState') -> list['Event']:
        super().handle(state)
        tour = self.get_tour(state)
        # print(f"Tour {tour.tour_id}, picker {tour.assigned_resource} orders {tour.order_numbers}")
        tour_id = tour.tour_id
        res = state.resource_manager.get_resource(tour.assigned_resource)
        # assert tour.route_nodes and tour.route_nodes[0] == (1, -1), \
        #     f"Tour {tour.tour_id} does not start at depot (1,-1): {tour.route_nodes[0] if tour.route_nodes else None}"

        assert isinstance(tour.annotated_route[0], RouteNode)

        state.resource_manager.update_resource_location(tour.assigned_resource,
                                                        tour.annotated_route[0])
        state.tour_manager.start_tour(tour_id, self.time)

        logger.info(f"resource {tour.assigned_resource} started tour {tour.tour_id} at t={self.time}")
        if tour.at_end() and res.current_location == (0, -1):
            return [TourEnd(self.time, tour.tour_id)]
        if tour.at_end():
            return [NodeArrival(self.time, tour.tour_id)]

        return [TravelEvent(self.time, tour.tour_id)]


class TravelEvent(BaseTourEvent):
    """Advance along the route."""
    def handle(self, state: SimulationState) -> list[Event]:
        from ware_ops_sim.data_loaders import CobotPicker
        super().handle(state)
        tour = self.get_tour(state)
        res = state.resource_manager.get_resource(tour.assigned_resource)

        assert not tour.at_end(), f"Travel at end of route on tour {tour.tour_id}"

        origin = tour.current_node()
        dest = tour.next_node()

        travel_distance = state.layout_manager.get_distance(origin, dest)
        if travel_distance < 6400 and isinstance(res, CobotPicker):
            travel_time = travel_distance / res.speed_follow_mode
        else:
            travel_time = travel_distance / res.speed

        # aisle_congestion_rate = state.resource_manager
        current_aisle = res.current_location.position[0]
        n_cobots_in_aisle = state.resource_manager.get_aisle_count(aisle_id=current_aisle)

        congestion_rate = 0
        if hasattr(res, "aisle_congestion_rate"):
            congestion_rate: float = res.aisle_congestion_rate  # x% slowdown per additional cobot
        congestion_factor = 1
        if n_cobots_in_aisle > 0:
            congestion_factor = 1.0 + congestion_rate * (n_cobots_in_aisle - 1)
        travel_time = travel_time * congestion_factor
        logger.debug(f"N cobots in aisle: {n_cobots_in_aisle - 1}, Congestion {congestion_factor}")
        arrival_time = state.current_time + travel_time

        # state.resource_manager.set_picker_busy_until(tour.assigned_resource, arrival_time)

        logger.debug(f"Travel: Picker {res.id} {origin} -> {dest} in {travel_time} min. "
                     f"Distance: {travel_distance}, "
                     f"congestion factor: {congestion_factor}")
        # mutate execution state
        state.tour_manager.advance_cursor(tour.tour_id)  # move cursor to dest
        assert isinstance(dest, RouteNode)
        state.resource_manager.update_resource_location(tour.assigned_resource, dest)

        return [NodeArrival(arrival_time, tour.tour_id)]


class NodeArrival(BaseTourEvent):
    """Handle arrival at a node: either pick, continue travel, or end at depot."""
    def handle(self, state: SimulationState) -> list[Event]:
        super().handle(state)
        tour = self.get_tour(state)
        res = state.resource_manager.get_resource(tour.assigned_resource)
        here = res.current_location
        logger.info(f"Debug Picker {res.id} arrives at {here}")
        logger.info(f"Debug picks left {tour.picks_left}")
        # Finish only if we are at end AND at depot (0,-1)
        if tour.at_end():
            return [TourEnd(self.time, tour.tour_id)]

        # If next planned pick is exactly here → start pick
        if here.node_type == NodeType.PICK:
            finish_at = self.time + res.time_per_pick
            return [PickComplete(finish_at, tour.tour_id, pick_start=self.time)]

        # Otherwise keep traveling (must not be at end)
        return [TravelEvent(self.time, tour.tour_id)]


# class PickStartEvent(BaseTourEvent):
#     """Start picking at the current node (must match next planned pick)."""
#     def handle(self, state: SimulationState) -> list[Event]:
#         super().handle(state)
#         tour = self.get_tour(state)
#         res = state.resource_manager.get_resource(tour.assigned_resource)
#
#         here = res.current_location
#         assert tour.picks_left, f"No picks left but PickStart issued on tour {tour.tour_id}"
#         assert tour.picks_left[0] == here, f"PickStart at {here} but next pick is {tour.picks_left[0]}"
#
#         # consume this pick from the execution queue
#         popped = state.tour_manager.pop_next_pick_if_here(tour.tour_id, here)
#         assert popped, "Failed to pop next pick at current location"
#
#         time_per_pick = res.time_per_pick
#
#         logger.debug(f"Pick start: Picker {res.id} at {here} t={self.time}")
#         return [PickEndEvent(self.time + time_per_pick, tour.tour_id)]


# class PickEndEvent(BaseTourEvent):
#     """Complete picking at current node; mark pick positions fulfilled."""
#     def handle(self, state: SimulationState) -> list[Event]:
#         super().handle(state)
#         tour = self.get_tour(state)
#         res = state.resource_manager.get_resource(tour.assigned_resource)
#         here = res.current_location
#
#         state.tour_manager.mark_pick_positions_fulfilled_at(tour.tour_id, here)
#
#         logger.debug(f"Pick end: Picker {res.id} at {here} t={self.time}")
#
#         # If we coincidentally are at the end depot after pick (rare), finish; else move on
#         # if tour.at_end() and here == (0, -1):
#         if tour.at_end():
#             return [TourEnd(self.time, tour.tour_id)]
#         return [TravelEvent(self.time, tour.tour_id)]


class PickComplete(BaseTourEvent):
    """Finish the pick at the current node; pop pick and mark positions fulfilled."""
    def __init__(self, time: float, tour_id: int, pick_start: float):
        super().__init__(time, tour_id)
        self.pick_start = pick_start

    def handle(self, state: SimulationState) -> list[Event]:
        super().handle(state)
        tour = self.get_tour(state)
        res = state.resource_manager.get_resource(tour.assigned_resource)
        here = res.current_location

        state.tour_manager.mark_pick_positions_fulfilled_at(tour.tour_id, here)

        logger.debug(f"Pick complete: Picker {res.id} at {here} t={self.time}")
        state.tracker.update_on_pick_end({
            'type': 'pick_end',
            'order_id': tour.order_numbers[0],
            'picker_id': res.id,
            'picker_type': res.__class__.__name__,
            'aisle': here.position[0],
            'node': here.position[1],
            'pick_start': self.pick_start,
            'pick_end': self.time,
            'tour_id': self.tour_id,
            })
        if tour.at_end():
            return [TourEnd(self.time, tour.tour_id)]
        return [TravelEvent(self.time, tour.tour_id)]


class TourEnd(BaseTourEvent):
    def handle(self, state: SimulationState) -> list[Event]:
        super().handle(state)
        tour = self.get_tour(state)
        res = state.resource_manager.get_resource(tour.assigned_resource)
        here = res.current_location

        # assert here == (0, -1), f"TourEnd at non-depot {here} on tour {tour.tour_id}"
        logger.info(f"resource {res.id}: Tour {tour.tour_id} completed at depot. "
              f"Start time: {tour.start_time}"
              f"Planned end: {tour.end_time_planned}, actual end: {self.time}"
              f"Makespan: {self.time - tour.start_time}")
        # finalize tour and free picker
        state.tour_manager.finish_tour(tour.tour_id, self.time)
        om = state.order_manager
        state.tracker.update_on_tour_end(tour_start=tour.start_time, tour_finish=self.time, order_manager=om)
        assert self.tour_id != state.tour_manager.get_next_tour_for_picker(res.id), (f"{self.tour_id}, " 
                                                                                     f"{state.tour_manager._picker_tour_queues}")
        return [PickerTourQuery(self.time, res.id)]

