import logging

from ware_ops_algos.domain_models import Order

from casim.events.base_events import BaseTourEvent, Event
from casim.state import State

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OrderArrival(Event):
    """Order enters the system and is buffered."""
    priority_score = 0

    def __init__(
        self,
        time: float,
        order: Order | None = None,
        *,
        order_id=None,
        release_version: int | None = None,
    ):
        super().__init__(time)
        self.order = order
        self.order_id = order.order_id if order is not None else order_id
        self.release_version = release_version
        self.cancelled = False

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        if self.order is not None:
            state.receive_order(self.order)
        elif state.release_order(
            self.order_id,
            int(self.release_version),
        ) is None:
            self.cancelled = True
            return []
        logger.info("Order %s arrived at t=%s", self.order_id, self.time)
        target = state.request_arrival_intervention(self.time)
        if target is None:
            return []
        tour_id, picker_id, version = target
        return [
            InterventionRequest(
                self.time,
                tour_id,
                picker_id,
                version,
                resumes_execution=False,
            )
        ]


class ShiftStart(Event):
    priority_score = 0

    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        state.start_shift(self.time)
        return []


class FlushRemainingOrders(Event):
    priority_score = 0

    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        state.close_input()
        return []


class PickerArrival(Event):

    priority_score = 1
    def __init__(self, time: float, picker_id: int, picker_available = False):
        super().__init__(time)
        self.picker_id = picker_id
        self.picker_available = picker_available

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        state.set_picker_availability(
            self.picker_id,
            bool(self.picker_available),
            self.time,
        )
        if (
            self.picker_available
            and state.picker_should_query(self.picker_id)
        ):
            return [PickerTourQuery(self.time, self.picker_id)]
        return []


class PickerDeparture(Event):
    """End a picker's shift without interrupting active work."""

    def __init__(self, time: float, picker_id: int):
        super().__init__(time)
        self.picker_id = picker_id

    def handle(self, state: 'State') -> list['Event']:
        state.set_picker_availability(self.picker_id, False, self.time)
        return []

class VolumeShiftAcrossDay(Event):
    def __init__(
        self,
        time: float,
        from_due: float | None = None,
        new_due: float | None = None,
        max_orders: int | None = None,
    ):
        super().__init__(time)
        self.from_due = from_due
        self.new_due = new_due
        self.max_orders = max_orders

    def handle(self, state: 'State') -> list['Event']:
        if self.from_due is None or self.new_due is None:
            return []
        shifted = state.reschedule_unreleased_orders(
            from_due=self.from_due,
            new_due=self.new_due,
            new_release=self.time,
            max_orders=self.max_orders,
        )
        return [
            OrderArrival(
                release_time,
                order_id=order_id,
                release_version=version,
            )
            for order_id, release_time, version in shifted
        ] + [WMSRun(self.time + 1e-6)]

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
        max_orders = max(
            1,
            int(self.te_volume / self.palett_te_factor),
        )
        state.disrupt_unstarted_work(
            from_due=self.from_time,
            new_due=self.to_time,
            max_orders=max_orders,
        )
        return []

class TruckArrival(Event):
    def __init__(self, time: float, wait_buffer=0):
        super().__init__(time)
        self.wait_buffer = wait_buffer

    def handle(self, state: 'State') -> list['Event']:
        return []

class TruckDeparture(Event):

    priority_score = 1

    def __init__(self, time, capacity):
        super().__init__(time)
        self.capacity = capacity

    def handle(self, state: 'State') -> list['Event']:
        return [
            PickerTourQuery(self.time, picker_id)
            for picker_id in state.depart_truck(self.time, self.capacity)
        ]


class WMSRun(Event):
    def __init__(self, time):
        super().__init__(time)

    def handle(self, state: 'State') -> list['Event']:
        return []


class PlanningRun(Event):
    """Configured scheduling/replanning checkpoint."""

    def handle(self, state: 'State') -> list['Event']:
        return []


class PickerIdle(Event):
    priority_score = 1

    def __init__(self, time: float, picker_id: int):
        super().__init__(time)
        self.picker_id = picker_id

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        return []

class BreakStart(Event):
    priority_score = 0
    def __init__(self, time: float, break_duration: float):
        super().__init__(time)
        self.break_duration = break_duration

    def handle(self, state: State) -> list[Event]:
        logger.debug("break start at %s", self.time)
        return [BreakEnd(state.start_break(self.time, self.break_duration))]

class BreakEnd(Event):
    priority_score = 0
    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: State) -> list[Event]:
        logger.debug("Facility break end at %s", self.time)
        return [
            PickerTourQuery(self.time, picker_id)
            for picker_id in state.finish_break(self.time)
        ]


class PickerTourQuery(Event):
    def __init__(self, time: float, picker_id: int):
        super().__init__(time)
        self.picker_id = picker_id

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        action, tour_id, action_time = state.query_picker_tour(
            self.picker_id,
            self.time,
        )
        if action == "retry":
            return [PickerTourQuery(action_time, self.picker_id)]
        if action == "start":
            _, version = state.tour_target(tour_id)
            return [TourStart(action_time, tour_id, version)]
        if action == "idle":
            return [PickerIdle(action_time, self.picker_id)]
        return []


class InterventionRequest(BaseTourEvent):
    """Pause execution so the active route suffix can be replanned."""

    def __init__(
        self,
        time: float,
        tour_id: int,
        picker_id: int,
        route_version: int,
        resumes_execution: bool = False,
    ):
        super().__init__(time, tour_id, route_version)
        self.picker_id = picker_id
        self.resumes_execution = resumes_execution
        self.cancelled = False

    def handle(self, state: State) -> list[Event]:
        if not state.accept_intervention_request(
            self.tour_id,
            self.route_version,
        ):
            self.cancelled = True
            return []
        return []


class TourStart(BaseTourEvent):
    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        if self.is_stale(state):
            return []
        if not state.start_tour(self.tour_id, self.time):
            return []
        _, version = state.tour_target(self.tour_id)
        return [TravelEvent(self.time, self.tour_id, version)]


class TravelEvent(BaseTourEvent):
    """Request one traversal and schedule its completion."""
    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        if self.is_stale(state):
            return []
        action, action_time = state.request_next_traversal(
            self.tour_id,
            self.time,
        )
        picker_id, version = state.tour_target(self.tour_id)
        if action == "arrival":
            return [NodeArrival(action_time, self.tour_id, version)]
        if action == "end":
            return [TourEnd(self.time, self.tour_id, version)]
        if action == "intervention":
            return [InterventionRequest(
                self.time,
                self.tour_id,
                picker_id,
                version,
                resumes_execution=True,
            )]
        return []


class NodeArrival(BaseTourEvent):
    """Handle arrival at a node: either pick, continue travel, or end at depot."""
    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        if self.is_stale(state):
            return []
        action, action_time, awakened = state.confirm_node_arrival(
            self.tour_id,
            self.time,
        )
        awakened_events = [
            TravelEvent(self.time, waiting_tour_id, waiting_version)
            for _, waiting_tour_id, waiting_version in awakened
        ]
        picker_id, version = state.tour_target(self.tour_id)
        if action == "end":
            return [*awakened_events, TourEnd(self.time, self.tour_id, version)]
        if action == "intervention":
            return [
                *awakened_events,
                InterventionRequest(
                    self.time,
                    self.tour_id,
                    picker_id,
                    version,
                    resumes_execution=True,
                )
            ]
        if action == "pick":
            return [
                *awakened_events,
                PickComplete(
                    action_time,
                    self.tour_id,
                    pick_start=self.time,
                    route_version=version,
                )
            ]
        if action == "travel":
            awakened_events.append(TravelEvent(self.time, self.tour_id, version))
        return awakened_events


class PickComplete(BaseTourEvent):
    """Finish the pick at the current node; pop pick and mark positions fulfilled."""
    def __init__(
        self,
        time: float,
        tour_id: int,
        pick_start: float,
        route_version: int | None = None,
    ):
        super().__init__(time, tour_id, route_version)
        self.pick_start = pick_start

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        if self.is_stale(state):
            return []
        action, awakened = state.confirm_pick_operation(
            self.tour_id,
            self.pick_start,
            self.time,
        )
        awakened_events = [
            NodeArrival(self.time, waiting_tour_id, waiting_version)
            for _, waiting_tour_id, waiting_version in awakened
        ]
        picker_id, version = state.tour_target(self.tour_id)
        if action == "end":
            return [*awakened_events, TourEnd(self.time, self.tour_id, version)]
        if action == "intervention":
            return [
                *awakened_events,
                InterventionRequest(
                    self.time,
                    self.tour_id,
                    picker_id,
                    version,
                    resumes_execution=True,
                )
            ]
        return [*awakened_events, TravelEvent(self.time, self.tour_id, version)]


class TourEnd(BaseTourEvent):
    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        if self.is_stale(state):
            return []
        picker_id, awakened = state.complete_tour(self.tour_id, self.time)
        resumed = [
            TravelEvent(self.time, waiting_tour_id, waiting_version)
            for _, waiting_tour_id, waiting_version in awakened
        ]
        return [*resumed, PickerTourQuery(self.time, picker_id)]

