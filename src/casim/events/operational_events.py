import logging
import sys

from ware_ops_algos.algorithms import Route, Job, PickerAssignment, PickList, NodeType, RouteNode, WarehouseOrder
from ware_ops_algos.domain_models import Order

from casim.domain_objects.tour_model import TourPlanningState, TourStates
from casim.events.base_events import Event
from casim.state import State

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OrderArrival(Event):
    """Order enters the system and is buffered."""
    priority_score = 1

    def __init__(self, time: float, order: Order):
        super().__init__(time)
        self.order = order

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        logger.info(f"Order {self.order.order_id} arrived at t={self.time}")
        state.order_manager.add_order_to_buffer(self.order)
        return []


class ShiftStart(Event):
    priority_score = 0

    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        return []


class FlushRemainingOrders(Event):
    priority_score = 0

    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        return []


class PickerArrival(Event):

    priority_score = 1
    def __init__(self, time: float, picker_id: int):
        super().__init__(time)
        self.picker_id = picker_id

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        return []


class TruckDeparture(Event):

    priority_score = 1

    def __init__(self, time, capacity):
        super().__init__(time)
        self.capacity = capacity

    def handle(self, state: 'State') -> list['Event']:
        if hasattr(state, "dock_manager"):
            state.dock_manager.release_pallets(self.capacity)
            print(f"released {self.capacity} pallets")
            resources = state.resource_manager.get_resources().resources
            idle_pickers = [p for p in resources if not p.occupied]
            return [PickerArrival(self.time, p.id) for p in idle_pickers]
        return []


class WMSRun(Event):
    def __init__(self, time):
        super().__init__(time)

    def handle(self, state: 'State') -> list['Event']:
        return []


class PickerIdle(Event):
    priority_score = 1

    def __init__(self, time: float, picker_id: int):
        super().__init__(time)
        self.picker_id = picker_id

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        # print(f"Picker {self.picker_id} arrived at {self.time}")
        state.tracker.on_idle_start(self.picker_id, self.time)
        return []


class PickerTourQuery(Event):
    def __init__(self, time: float, picker_id: int):
        super().__init__(time)
        self.picker_id = picker_id

    def handle(self, state: State) -> list[Event]:
        # A picker queries a new tour everytime they are forced by e.g. a scheduling result
        # Or after they finished their last tour to query other scheduled tours.
        super().handle(state)
        picker = state.resource_manager.get_resource(
            self.picker_id)
        # state.current_picker_id = self.picker_id
        # picker.notified = False
        next_tour_id = state.tour_manager.get_next_tour_for_picker(
            self.picker_id)
        logger.info(f"Picker {picker.id} has pending tour with id={next_tour_id} at {self.time}")
        # print((f"Picker {picker.id} has pending tour with id={next_tour_id} at {self.time}"))
        if next_tour_id is not None:
            # There exists an assigned tour for the picker
            next_tour = state.tour_manager.get_tour(next_tour_id)
            start_time = state.current_time
            if next_tour.start_time is not None:
                start_time = max(next_tour.start_time, self.time)
                # The tour is scheduled -> we use the start time of the tour
                # Otherwise we greedily start the tour right away
            next_tour.status = TourStates.PENDING
            if picker.tour_setup_time:
                start_time += picker.tour_setup_time  # TODO Check if setup times are not added twice
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

    def get_tour(self, state: State) -> TourPlanningState:
        return state.tour_manager.get_tour(self.tour_id)


class TourStart(BaseTourEvent):
    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        tour = self.get_tour(state)
        state.tracker.on_idle_end(tour.assigned_resource, self.time)
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
    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        tour = self.get_tour(state)
        res = state.resource_manager.get_resource(tour.assigned_resource)

        assert not tour.at_end(), f"Travel at end of route on tour {tour.tour_id}"

        origin = tour.current_node()
        dest = tour.next_node()

        travel_distance = state.layout_manager.get_distance(origin, dest)
        travel_time = travel_distance / res.speed
        arrival_time = state.current_time + travel_time

        logger.debug(f"Travel: Picker {res.id} {origin} -> {dest} in {travel_time} min. "
                     f"Distance: {travel_distance}")
        state.tracker.on_travel(picker_id=res.id, distance=travel_distance)
        # mutate execution state
        state.tour_manager.advance_cursor(tour.tour_id)  # move cursor to dest
        assert isinstance(dest, RouteNode)
        state.resource_manager.update_resource_location(tour.assigned_resource, dest)

        return [NodeArrival(arrival_time, tour.tour_id)]


class NodeArrival(BaseTourEvent):
    """Handle arrival at a node: either pick, continue travel, or end at depot."""
    def handle(self, state: State) -> list[Event]:
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


class PickComplete(BaseTourEvent):
    """Finish the pick at the current node; pop pick and mark positions fulfilled."""
    def __init__(self, time: float, tour_id: int, pick_start: float):
        super().__init__(time, tour_id)
        self.pick_start = pick_start

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        tour = self.get_tour(state)
        res = state.resource_manager.get_resource(tour.assigned_resource)
        here = res.current_location

        state.tour_manager.mark_pick_positions_fulfilled_at(tour.tour_id, here)

        logger.debug(f"Pick complete: Picker {res.id} at {here} t={self.time}")
        state.tracker.on_pick_end(
            tour_id=self.tour_id,
            picker_id=res.id,
            order_id=tour.order_numbers[0],
            item_id=None,
            start_time=self.pick_start,
            end_time=self.time
        )
        if tour.at_end():
            return [TourEnd(self.time, tour.tour_id)]
        return [TravelEvent(self.time, tour.tour_id)]


class TourEnd(BaseTourEvent):
    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        tour = self.get_tour(state)
        res = state.resource_manager.get_resource(tour.assigned_resource)
        here = res.current_location

        assert here.position == state.layout_manager.get_layout().graph_data.end_location, f"TourEnd at non-depot {here} on tour {tour.tour_id}"
        logger.info(f"resource {res.id}: Tour {tour.tour_id} completed at depot. "
              f"Start time: {tour.start_time}"
              f"Planned end: {tour.end_time_planned}, actual end: {self.time}"
              f"Makespan: {self.time - tour.start_time}")
        # finalize tour and free picker
        state.tour_manager.finish_tour(tour.tour_id, self.time)
        om = state.order_manager
        on_time = []
        delayed = []
        for o_id in tour.order_numbers:
            o = om.get_order_from_history(o_id)
            if o.due_date < self.time:
                delayed.append(o_id)
            else:
                on_time.append(o_id)

        # state.tracker.update_on_tour_end(tour_start=tour.start_time, tour_finish=self.time, order_manager=om)
        state.tracker.on_tour_end(tour.tour_id, tour.start_time, self.time, tour.order_numbers, tour.assigned_resource,
                                  on_time, delayed)

        assert self.tour_id != state.tour_manager.get_next_tour_for_picker(res.id), (f"{self.tour_id}, " 
                                                                                     f"{state.tour_manager._picker_tour_queues}")
        if hasattr(state, "dock_manager"):
            state.dock_manager.stage_pallets(1)
        return [PickerTourQuery(self.time, res.id)]

