from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ware_ops_algos.algorithms import NodeType, RouteNode
from ware_ops_algos.domain_models import Order
from ware_ops_algos.planning.lifecycle import TourStates
from casim.domain_objects.tour_model import TourPlanningState
from casim.domain_objects.sim_domain import ReconsiderReason

from casim.events.base_events import Event

if TYPE_CHECKING:
    from casim.state import State

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
        state.information_version += 1
        return [ActiveTourOpportunity(self.time)]


class ActiveTourOpportunity(Event):
    """Consider the current unserved route after operational events at this time."""

    priority_score = 2

    def __init__(self, time: float, tour_id: int | None = None,
                 route_version: int | None = None):
        super().__init__(time)
        self.tour_id = tour_id
        self.route_version = route_version

    def is_stale(self, state: State) -> bool:
        if self.tour_id is None:
            return False
        tour = state.tour_manager.get_tour(self.tour_id)
        return tour.status != TourStates.STARTED or tour.route_version != self.route_version

    def handle(self, state: State) -> list[Event]:
        return []


class OrderStreamClosed(Event):
    """The order source has announced that this finite stream is complete."""

    def handle(self, state: State) -> list[Event]:
        state.done_flag = True
        state.information_version += 1
        return [ActiveTourOpportunity(self.time)]


class Reconsider(Event):
    """Revisit one decision if its plan version is still current."""

    def __init__(self, time: float, decision: str, version: int,
                 reason: ReconsiderReason):
        super().__init__(time)
        self.decision = decision
        self.version = version
        self.reason = ReconsiderReason(reason)

    def is_stale(self, state: State) -> bool:
        return self.version != state.decision_versions.get(self.decision, 0)


class PickerArrival(Event):
    """The resource becomes present (shift start, end of a break)."""

    priority_score = 1

    def __init__(self, time: float, picker_id: int):
        super().__init__(time)
        self.picker_id = picker_id

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        version = state.resource_manager.arrive(self.picker_id)
        state.information_version += 1
        return [Dispatch(self.time, self.picker_id, version)]


class PickerDeparture(Event):
    """The resource leaves at its next free point: now, or after its current tour.

    With a duration it returns by itself after that time (a break); without
    one it stays absent until its next arrival (a shift end). A tour is never
    interrupted.
    """

    priority_score = 1

    def __init__(self, time: float, picker_id: int, duration: float | None = None):
        super().__init__(time)
        self.picker_id = picker_id
        self.duration = duration

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        version = state.resource_manager.request_departure(self.picker_id, self.duration)
        state.information_version += 1
        return [Dispatch(self.time, self.picker_id, version)]


class Dispatch(Event):
    """Hand a resource its next committed tour, if it is present and free.

    Emitted for every change of a resource's queue or presence and at the end
    of its tour, with the resource's queue version then; an older Dispatch is
    stale. It keeps the priority of the event it replaces: 0 from a decision's
    commands, 1 otherwise. It is not a decision trigger.
    """

    def __init__(self, time: float, picker_id: int, version: int, priority: int = 1):
        super().__init__(time)
        self.picker_id = picker_id
        self.version = version
        self.priority_score = priority

    def is_stale(self, state: 'State') -> bool:
        return self.version != state.resource_manager.version(self.picker_id)

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        events = self._dispatch(state)
        state.project_queue(self.picker_id)
        return events

    def _dispatch(self, state: 'State') -> list['Event']:
        resources, tours = state.resource_manager, state.tour_manager
        picker_id = self.picker_id
        if tours.current_tour(picker_id) is not None:
            return []
        if resources.has_pending_departure(picker_id):
            duration = resources.make_departure_effective(picker_id, self.time)
            return ([] if duration is None
                    else [PickerArrival(self.time + duration, picker_id)])
        if not resources.is_present(picker_id):
            return []
        queue = tours.queue(picker_id)
        if not queue:
            if resources.enter_idle(picker_id):
                return [PickerIdle(self.time, picker_id)]
            return []
        tour = tours.get_tour(queue[0])
        if tour.earliest_start is not None and tour.earliest_start > self.time:
            # The binding start is later; another decision may revise it first.
            return [Dispatch(tour.earliest_start, picker_id, self.version)]
        resources.leave_idle(picker_id)
        tours.advance(tour.tour_id, TourStates.PENDING)
        # The setup begins now: the pending tour's planned times move here.
        tour.dispatched_at = self.time
        duration = tour.planned_end - tour.planned_start
        tour.planned_start, tour.planned_end = self.time, self.time + duration
        picker = resources.get_resource(picker_id)
        return [TourStart(self.time + (picker.tour_setup_time or 0), tour.tour_id)]


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
    """A truck's departure moves: the due time of its orders changes.

    The event changes exogenous information only. It sets the due date of
    the given orders wherever they are still open and cancels nothing; a
    replanning decision triggered by this event revises the plan within its
    revision scope. Which orders a truck carries is the source's choice.
    """

    priority_score = 1

    def __init__(self, time: float, order_ids: list[int], due_date: float):
        super().__init__(time)
        self.order_ids = list(order_ids)
        self.due_date = due_date

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        changed = state.order_manager.set_due_dates(
            self.order_ids, self.due_date, state.tour_manager)
        for order_id, old_due in changed.items():
            state.tracker.on_volume_shift(order_id, from_time=old_due,
                                          to_time=self.due_date)
        missing = set(self.order_ids) - set(changed)
        if missing:
            logger.warning("truck disruption at %s: orders %s are not open",
                           self.time, sorted(missing))
        logger.info("truck disruption at %s: %d orders now due at %s",
                    self.time, len(changed), self.due_date)
        state.information_version += 1
        state.order_revision += 1
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
        ts_delayed = []
        ts_in_progress = []
        ts_expected_finish = []
        for t_id, t in state.tour_manager.all_tours.items():
            if t.status == TourStates.STARTED:
                ts_in_progress.append(t_id)
                if t.batch.earliest_due_date == self.time:
                    ts_delayed.append(t_id)
                    ts_expected_finish.append(t.planned_end)

        state.tracker.on_truck_departure_delays(self.time, ts_delayed, ts_expected_finish)
        if self.capacity:
            state.dock_manager.release_pallets(self.capacity)
            logger.warning("released %d pallets", self.capacity)
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
        return []

class BaseTourEvent(Event):
    priority_score = 1
    def __init__(self, time: float, tour_id: int, route_version: int | None = None):
        super().__init__(time)
        self.tour_id = tour_id
        self.route_version = route_version

    def get_tour(self, state: State) -> TourPlanningState:
        return state.tour_manager.get_tour(self.tour_id)

    def is_stale(self, state: State) -> bool:
        tour = self.get_tour(state)
        return (tour.status in (TourStates.CANCELLED, TourStates.DONE) or
                self.route_version is not None and self.route_version != tour.route_version)


class TourStart(BaseTourEvent):
    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        if self.is_stale(state):
            return []
        tour = self.get_tour(state)
        tour_id = tour.tour_id
        res = state.resource_manager.get_resource(tour.assigned_resource)

        assert isinstance(tour.annotated_route[0], RouteNode)

        state.resource_manager.update_resource_location(tour.assigned_resource,
                                                        tour.annotated_route[0])
        if tour.status == TourStates.CANCELLED:
            logger.warning("tour %d starts although cancelled, at %s", tour.tour_id, self.time)
        state.tour_manager.start_tour(tour_id, self.time)
        state.project_queue(tour.assigned_resource)
        logger.info("resource %d started tour %d at t=%s", tour.assigned_resource, tour.tour_id, self.time)
        state.tracker.on_tour_start(self.time)
        if tour.at_end() and res.current_location == (0, -1):
            return [TourEnd(self.time, tour.tour_id, tour.route_version)]
        if tour.at_end():
            return [NodeArrival(self.time, tour.tour_id, tour.route_version)]

        start_time = self.time
        return [TravelEvent(start_time, tour.tour_id, tour.route_version)]


class TravelEvent(BaseTourEvent):
    """Advance along the route."""
    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        if self.is_stale(state):
            return []
        tour = self.get_tour(state)
        res = state.resource_manager.get_resource(tour.assigned_resource)

        assert not tour.at_end(), f"Travel at end of route on tour {tour.tour_id}"

        origin = tour.current_node()
        dest = tour.next_node()

        if tour.cursor == 0 and tour.routing_origin and tour.routing_origin.edge_destination:
            travel_distance = tour.routing_origin.distance_to_destination
        else:
            travel_distance = state.layout_manager.get_distance(origin, dest)
        travel_time = travel_distance / res.speed
        arrival_time = state.current_time + travel_time

        logger.debug("Travel: Picker %d %s -> %s in %s min. Distance: %s",
                     res.id, origin, dest, travel_time, travel_distance)
        tour.edge_origin = origin
        tour.edge_destination = dest
        tour.edge_start_time = self.time
        tour.edge_end_time = arrival_time
        tour.edge_distance = travel_distance
        return [NodeArrival(arrival_time, tour.tour_id, tour.route_version)]


class NodeArrival(BaseTourEvent):
    """Handle arrival at a node: either pick, continue travel, or end at depot."""
    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        if self.is_stale(state):
            return []
        tour = self.get_tour(state)
        res = state.resource_manager.get_resource(tour.assigned_resource)
        if tour.edge_destination is not None:
            state.tracker.on_travel(picker_id=res.id, distance=tour.edge_distance)
            state.tour_manager.advance_cursor(tour.tour_id)
            state.resource_manager.update_resource_location(tour.assigned_resource, tour.edge_destination)
            tour.edge_origin = tour.edge_destination = None
            tour.edge_start_time = tour.edge_end_time = None
            tour.edge_distance = None
        here = res.current_location
        logger.info("Debug Picker %s arrives at %s", res.id, here)
        # A route ending away from the depot holds for a later suffix replacement.
        if tour.at_end():
            depot = state.layout_manager.get_layout().layout_network.end_node
            if here.position != depot:
                return [ActiveTourOpportunity(self.time, tour.tour_id,
                                              tour.route_version)]
            return [TourEnd(self.time, tour.tour_id, tour.route_version)]

        # If next planned pick is exactly here → start pick
        if here.node_type == NodeType.PICK:
            finish_at = self.time + res.time_per_pick
            tour.picking_until = finish_at
            return [PickComplete(finish_at, tour.tour_id, pick_start=self.time,
                                 route_version=tour.route_version)]

        # Otherwise keep traveling (must not be at end)
        return [TravelEvent(self.time, tour.tour_id, tour.route_version)]


class PickComplete(BaseTourEvent):
    """Finish the pick at the current node; pop pick and mark positions fulfilled."""
    def __init__(self, time: float, tour_id: int, pick_start: float,
                 route_version: int | None = None):
        super().__init__(time, tour_id, route_version)
        self.pick_start = pick_start

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        if self.is_stale(state):
            return []
        tour = self.get_tour(state)
        res = state.resource_manager.get_resource(tour.assigned_resource)
        here = res.current_location
        tour.picking_until = None
        match = next((p for p in tour.remaining_picks if p.pick_node == here.position), None)
        if match is None:
            raise ValueError(f"No remaining pick at {here.position} on tour {tour.tour_id}")
        tour.remaining_picks.remove(match)
        tour.completed_picks.append(match)

        # state.tour_manager.mark_pick_positions_fulfilled_at(tour.tour_id, here)

        logger.debug("Pick complete: Picker %d at %s t=%s", res.id, here, self.time)
        state.tracker.on_pick_end(
            tour_id=self.tour_id,
            picker_id=res.id,
            order_id=match.order_number,
            item_id=match.article_id,
            start_time=self.pick_start,
            end_time=self.time
        )
        if tour.at_end():
            return [TourEnd(self.time, tour.tour_id, tour.route_version)]
        return [TravelEvent(self.time, tour.tour_id, tour.route_version),
                ActiveTourOpportunity(self.time, tour.tour_id, tour.route_version)]


class TourEnd(BaseTourEvent):
    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        if self.is_stale(state):
            return []
        tour = self.get_tour(state)
        res = state.resource_manager.get_resource(tour.assigned_resource)
        here = res.current_location

        assert here.position == state.layout_manager.get_layout().graph_data.end_location, f"TourEnd at non-depot {here} on tour {tour.tour_id}"
        logger.info("resource %d: Tour %d completed at depot. Start time: %s Planned end: %s, actual end: %s Makespan: %s",
                    res.id, tour.tour_id, tour.actual_start, tour.planned_end, self.time, self.time - tour.actual_start)
        # finalize tour and free picker
        state.tour_manager.finish_tour(tour.tour_id, self.time)
        state.project_queue(tour.assigned_resource)
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

        state.dock_manager.stage_pallets(state.dock_manager.pallets_per_tour)
        n_pallets_dock = state.dock_manager.n_staged_pallets
        n_lines = len(tour.completed_picks)
        state.tracker.on_tour_end(tour.tour_id,
                                  tour.actual_start,
                                  self.time,
                                  tour.order_numbers,
                                  tour.assigned_resource,
                                  on_time,
                                  delayed,
                                  n_pallets_dock,
                                  n_lines)
        return [Dispatch(self.time, res.id, state.resource_manager.version(res.id))]

