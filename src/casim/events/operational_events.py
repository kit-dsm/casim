import logging

from ware_ops_algos.algorithms import NodeType, RouteNode
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
        logger.info("Order %s arrived at t=%s", self.order.order_id, self.time)
        state.order_manager.add_order_to_buffer(self.order)
        return []


class ShiftStart(Event):
    priority_score = 0

    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        state.is_break = False
        print(f"{self.time} shift start")
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
    def __init__(self, time: float, picker_id: int, picker_available = False):
        super().__init__(time)
        self.picker_id = picker_id
        self.picker_available = picker_available

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        if self.picker_available:
            state.resource_manager.set_picker_available(self.picker_id)
        elif not self.picker_available:
            state.resource_manager.set_picker_unavailable(self.picker_id)
        return []

class VolumeShiftAcrossDay(Event):
    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: 'State') -> list['Event']:
        return []

class OrderIngestion(Event):
    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: 'State') -> list['Event']:
        return []



class TruckDisruption(Event):
    def __init__(self, time: float, from_time: float, to_time: float, te_volume: int, palett_te_factor: float):
        super().__init__(time)
        self.from_time = from_time
        self.to_time = to_time
        self.te_volume = te_volume
        self.palett_te_factor = palett_te_factor
        assert from_time > time
        assert to_time > time
        assert to_time < from_time

    def handle(self, state: 'State') -> list['Event']:
        # need to remove the disrupted te's from state
        # in the specified due-date corridor collect orders until te is reached
        # -> From historic ? Sort order history by due date
        # find all tours that have these orders and cancel them
        # re-add the orders with new deadlines to the order buffer.
        # orders can be pre-batched or batched

        print(f"Truck disruption at {self.time}: from {self.from_time} to {self.to_time}")
        # first iterate over already scheduled tours
        orders_pulled = []
        batches_pulled = []
        volume_satisfied = False

        all_tours = state.tour_manager.all_tours
        tours_to_cancel = []
        seen_tours = 0
        for tour_id, tour in all_tours.items():
            seen_tours += 1
            if tour.status not in [TourStates.DONE, TourStates.PENDING, TourStates.STARTED]:
                if tour.batch.earliest_due_date >= self.from_time:
                    for o in tour.batch.orders:
                        orders_pulled.append(o)
                    batches_pulled.append(tour.batch)
                    tour.status = TourStates.CANCELLED
                    tours_to_cancel.append(tour_id)
                if len(orders_pulled) * self.palett_te_factor >= self.te_volume:
                    volume_satisfied = True
                    break

        print("cancelled %d tours", len(tours_to_cancel))
        # If volume not already satisfied, e.g. because no tours are scheduled collect from batch buffer.
        if not volume_satisfied and len(tours_to_cancel) == 0:
            for b in state.order_manager._pick_list_buffer:
                if b.earliest_due_date >= self.from_time:
                    for o in b.orders:
                        o.due_date = self.to_time
                        orders_pulled.append(o)
                    # b.earliest_due_date = self.to_time
                    # batches_pulled.append(b)
                if len(orders_pulled) * self.palett_te_factor >= self.te_volume:
                    volume_satisfied = True
                    break


        print("pulled %d batches", len(batches_pulled))
        # if not volume_satisfied:
        #     orders_pulled = []
        #     for o_id, o in state.order_manager._order_history.items():
        #         if o.due_date == self.from_time:
        #             orders_pulled.append(o)
        #             if len(orders_pulled) * self.palett_te_factor >= self.te_volume:
        #                 break
        #
        #     for order in orders_pulled:
        #         reassigned_order = WarehouseOrder(order_id=order.order_id,
        #               due_date=self.to_time,
        #               order_date=order.order_date,
        #               pick_positions=order.pick_positions,
        #               parent_order_id=order.parent_order_id)
        #         state.order_manager.add_order_to_buffer(reassigned_order)

        for b in batches_pulled:
            # b.earliest_due_date = self.to_time
            for o in b.orders:
                logger.debug("%s cancelled", o)
                o.due_date = self.to_time
            state.order_manager.add_pick_list_to_buffer(b)

        for o in orders_pulled:
            state.tracker.on_volume_shift(o, from_time=self.from_time, to_time=self.to_time)
        return []

class TruckArrival(Event):
    def __init__(self, time: float, wait_buffer=0):
        super().__init__(time)
        self.wait_buffer = wait_buffer

    def handle(self, state: 'State') -> list['Event']:
        n_pallets = 0
        if hasattr(state, "dock_manager"):
            n_pallets = state.dock_manager.n_staged_pallets
        # return [TruckDeparture(self.time + 30 + self.wait_buffer, capacity=n_pallets)]
        return []

class TruckDeparture(Event):

    priority_score = 1

    def __init__(self, time, capacity):
        super().__init__(time)
        self.capacity = capacity

    def handle(self, state: 'State') -> list['Event']:
        ts_delayed = []
        ts_in_progress = []
        ts_expected_finish = []
        for t_id, t in state.tour_manager.all_tours.items():
            if t.status == TourStates.STARTED:
                ts_in_progress.append(t_id)
                if t.batch.earliest_due_date == self.time:
                    ts_delayed.append(t_id)
                    ts_expected_finish.append(t.end_time_planned)

        state.tracker.on_truck_departure_delays(self.time, ts_delayed, ts_expected_finish)
        if hasattr(state, "dock_manager"):
            if self.capacity:
                state.dock_manager.release_pallets(self.capacity)
                logger.warning("released %d pallets", self.capacity)
            resources = state.resource_manager.get_resources().resources
            idle_pickers = [p for p in resources if not p.occupied]
            return [PickerArrival(self.time, p.id, p.available) for p in idle_pickers]
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

class BreakStart(Event):
    priority_score = 0
    def __init__(self, time: float, break_duration: float):
        super().__init__(time)
        self.break_duration = break_duration

    def handle(self, state: State) -> list[Event]:
        state.is_break = True
        logger.debug("break start at %s", self.time)
        state.break_duration = self.break_duration
        return []

class BreakEnd(Event):
    priority_score = 0
    def __init__(self, time: float, picker_id: int):
        super().__init__(time)
        self.picker_id = picker_id

    def handle(self, state: State) -> list[Event]:
        state.is_break = False
        logger.debug("Picker %d break end at %s", self.picker_id, self.time)
        return [PickerTourQuery(self.time, self.picker_id)]


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
        next_tour_id = state.tour_manager.get_next_tour_for_picker(
            self.picker_id)
        logger.info("Picker %s has pending tour with id=%s at %s", picker.id, next_tour_id, self.time)
        if state.is_break:
            print(f"break scheduled for {self.time + state.break_duration}")
            return [BreakEnd(self.time + state.break_duration, self.picker_id)]

        if next_tour_id is not None:
            # There exists an assigned tour for the picker
            next_tour = state.tour_manager.get_tour(next_tour_id)
            if next_tour.status == TourStates.CANCELLED:
                state.tour_manager.remove_canceled_tour_for_picker(picker.id, next_tour_id)
                # peek_next_tour_id = state.tour_manager.get_next_tour_for_picker(picker.id)
                logger.debug("Tour %d for picker %d is cancelled at %s.", next_tour_id, picker.id, self.time)
                # if peek_next_tour_id:
                #     peek_next_tour = state.tour_manager.get_tour(peek_next_tour_id)
                #     print(f"Next tour in queue is {peek_next_tour_id} with start time {peek_next_tour.start_time}")
                return [PickerTourQuery(self.time, self.picker_id)]
                # return [PickerIdle(state.current_time, picker.id)]
            start_time = state.current_time
            if next_tour.start_time is not None:
                start_time = max(next_tour.start_time, self.time)
                # The tour is scheduled -> we use the start time of the tour
                # Otherwise we greedily start the tour right away
            next_tour.status = TourStates.PENDING
            if picker.tour_setup_time:
                start_time += picker.tour_setup_time
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
        if tour.status == TourStates.CANCELLED:
            print(f"{tour.tour_id} cancelled at {self.time}")
        state.tour_manager.start_tour(tour_id, self.time)
        state.resource_manager.mark_picker_occupied(tour.assigned_resource)
        picker = state.resource_manager.get_resource(tour.assigned_resource)
        logger.info("resource %d started tour %d at t=%s", tour.assigned_resource, tour.tour_id, self.time)
        state.tracker.on_tour_start(self.time)
        if tour.at_end() and res.current_location == (0, -1):
            return [TourEnd(self.time, tour.tour_id)]
        if tour.at_end():
            return [NodeArrival(self.time, tour.tour_id)]

        start_time = self.time
        return [TravelEvent(start_time , tour.tour_id)]


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

        logger.debug("Travel: Picker %d %s -> %s in %s min. Distance: %s",
                     res.id, origin, dest, travel_time, travel_distance)
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
        logger.info("Debug Picker %s arrives at %s", res.id, here)
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

        # state.tour_manager.mark_pick_positions_fulfilled_at(tour.tour_id, here)

        logger.debug("Pick complete: Picker %d at %s t=%s", res.id, here, self.time)
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
        logger.info("resource %d: Tour %d completed at depot. Start time: %s Planned end: %s, actual end: %s Makespan: %s",
                    res.id, tour.tour_id, tour.start_time, tour.end_time_planned, self.time, self.time - tour.start_time)
        # finalize tour and free picker
        state.tour_manager.finish_tour(tour.tour_id, self.time)
        om = state.order_manager
        on_time = []
        delayed = []
        for o_id in tour.order_numbers:
            o = om.get_order_from_history(o_id)
            if o.due_date:
                if o.due_date < self.time:
                    delayed.append(o_id)
                else:
                    on_time.append(o_id)

        # state.tracker.update_on_tour_end(tour_start=tour.start_time, tour_finish=self.time, order_manager=om)
        if self.tour_id == state.tour_manager.get_next_tour_for_picker(res.id):
            logger.error("Tour ID %d matches next tour for picker (queues: %s)", self.tour_id, state.tour_manager._picker_tour_queues)
            raise AssertionError(f"Tour {self.tour_id} should not be next tour for picker {res.id}")
        n_pallets_dock = 0
        if hasattr(state, "dock_manager"):
            state.dock_manager.stage_pallets(1)
            n_pallets_dock = state.dock_manager.n_staged_pallets
        n_lines = len(tour.original_route.item_sequence)
        state.tracker.on_tour_end(tour.tour_id,
                                  tour.start_time,
                                  self.time,
                                  tour.order_numbers,
                                  tour.assigned_resource,
                                  on_time,
                                  delayed,
                                  n_pallets_dock,
                                  n_lines)
        return [PickerTourQuery(self.time, res.id)]

