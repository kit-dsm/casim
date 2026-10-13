"""Check live references, then apply a validated plan delta through managers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from ware_ops_algos.algorithms import (
    ActiveTourBatch, NodeType, Route, RouteNode, RoutingOrigin,
)
from ware_ops_algos.planning.commands import (
    CancelTour, Command, CommitTour, CreatePickList, CreateTour,
    ReplaceSuffix, Reschedule, ViolationCode,
)
from ware_ops_algos.planning.lifecycle import TourStates

from casim.events.decision_events import PickListDone, SequencingDone
from casim.events.operational_events import TravelEvent
from casim.decision_engine.positioning import MarkPositioning, RecallPositioning

if TYPE_CHECKING:
    from casim.state.state import State


class CommandViolation(ValueError):
    def __init__(self, code: ViolationCode, message: str):
        super().__init__(message)
        self.code = code


def _fail(message: str, code: ViolationCode = ViolationCode.STATE_INFORMATION):
    raise CommandViolation(code, message)


def check(state: State, commands: tuple[Command, ...]) -> None:
    """Check all live references before making any mutation."""
    buffered = {order.order_id for order in state.order_manager.get_order_buffer()}
    free = set(buffered)
    free.update(order_id for batch in state.order_manager.get_pick_list_buffer()
                for order_id in batch.order_numbers)
    released = set()
    new_count = 0
    for command in commands:
        if isinstance(command, CancelTour):
            if not state.tour_manager.has_tour(command.tour_id):
                _fail(f"Tour {command.tour_id} no longer exists")
            tour = state.tour_manager.get_tour(command.tour_id)
            if tour.status not in (TourStates.PLANNED, TourStates.SCHEDULED):
                _fail(f"Tour {command.tour_id} cannot be cancelled",
                      ViolationCode.PHYSICAL)
            released.update(tour.order_numbers)
            free.update(tour.order_numbers)
        elif isinstance(command, CreatePickList):
            ids = set(command.batch.order_numbers)
            if not ids <= buffered:
                _fail(f"Pick-list orders are not buffered: {sorted(ids - buffered)}")
            buffered -= ids
            free -= ids
        elif isinstance(command, CreateTour):
            ids = set(command.job.order_numbers)
            if not ids <= free:
                _fail(f"Tour orders are not free: {sorted(ids - free)}")
            if not any(resource.id == command.job.picker_id for resource in
                       state.resource_manager.get_resources().resources):
                _fail(f"Picker {command.job.picker_id} no longer exists")
            free -= ids
            buffered -= ids
            new_count += 1
        elif isinstance(command, ReplaceSuffix):
            if not state.tour_manager.has_tour(command.tour_id):
                _fail(f"Tour {command.tour_id} no longer exists")
            ids = set(command.added_order_ids)
            if not ids <= buffered:
                _fail(f"Added orders are not buffered: {sorted(ids - buffered)}")
            tour = state.tour_manager.get_tour(command.tour_id)
            if tour.status is TourStates.STARTED and (
                    command.route.routing_origin.position !=
                    tour.position_at(state.current_time)):
                _fail("Running tour moved after projection", ViolationCode.EXECUTED_PREFIX)
            buffered -= ids
            free -= ids
        elif isinstance(command, Reschedule):
            if not state.tour_manager.has_tour(command.tour_id):
                _fail(f"Tour {command.tour_id} no longer exists")
            if not any(resource.id == command.picker_id for resource in
                       state.resource_manager.get_resources().resources):
                _fail(f"Picker {command.picker_id} no longer exists")
        elif isinstance(command, CommitTour):
            if command.creation_index is not None:
                if command.creation_index >= new_count:
                    _fail("CommitTour references an absent new tour",
                          ViolationCode.MEMBERSHIP)
            elif not state.tour_manager.has_tour(command.tour_id):
                _fail(f"Tour {command.tour_id} no longer exists")
            elif state.tour_manager.get_tour(command.tour_id).status is not TourStates.PLANNED:
                _fail(f"Tour {command.tour_id} is not planned", ViolationCode.PHYSICAL)
        elif isinstance(command, MarkPositioning):
            if not 0 <= command.creation_index < new_count:
                _fail("Positioning mode references an absent new tour",
                      ViolationCode.MEMBERSHIP)
        elif isinstance(command, RecallPositioning):
            if not state.tour_manager.has_tour(command.tour_id):
                _fail(f"Tour {command.tour_id} no longer exists")
            tour = state.tour_manager.get_tour(command.tour_id)
            if (not state.done_flag or tour.status is not TourStates.STARTED
                    or tour.positioning_mode != "center_relocation"
                    or tour.order_numbers):
                _fail("Only an empty center tour may return at closure",
                      ViolationCode.PHYSICAL)
        else:
            _fail(f"Unknown command {type(command).__name__}", ViolationCode.MEMBERSHIP)


def apply(state: State, commands: tuple[Command, ...]):
    """Apply a complete prechecked decision; return process events."""
    check(state, commands)
    events = []
    created_ids = []
    for command in commands:
        if isinstance(command, CreatePickList):
            state.order_manager.clear_order_buffer_by_ids(
                list(command.batch.order_numbers))
            state.add_pick_list_to_planning_state(command.batch)
            events.append(PickListDone(state.current_time, command.batch))
        elif isinstance(command, CreateTour):
            batch = command.job.job.route.batch
            if state.order_manager.has_pick_list(batch):
                state.order_manager.clear_pick_list_buffer([batch])
                source = "pick_list"
            else:
                state.order_manager.clear_order_buffer_by_ids(
                    list(command.job.order_numbers))
                source = "orders"
            tour_id = state.add_sequencing_to_planning_state(command.job)
            state.tour_source[tour_id] = source
            created_ids.append(tour_id)
        elif isinstance(command, CancelTour):
            tour = state.tour_manager.cancel_tour(command.tour_id)
            if state.tour_source.pop(tour.tour_id, "orders") == "pick_list":
                state.order_manager.add_pick_list_to_buffer(tour.batch)
            else:
                for order_id in tour.order_numbers:
                    state.order_manager.add_order_to_buffer(
                        state.order_manager.get_order_from_history(order_id))
        elif isinstance(command, ReplaceSuffix):
            tour = state.tour_manager.get_tour(command.tour_id)
            if tour.status is TourStates.STARTED:
                version = state.commit_active_route(command.route,
                                                    command.added_order_ids)
                events.append(TravelEvent(state.current_time, tour.tour_id, version))
            else:
                state.order_manager.clear_order_buffer_by_ids(
                    list(command.added_order_ids))
                state.tour_manager.replace_planned_route(
                    tour.tour_id, command.route, command.added_order_ids)
        elif isinstance(command, Reschedule):
            state.tour_manager.reschedule(command.tour_id, command.picker_id,
                                          command.start_time, command.end_time)
        elif isinstance(command, CommitTour):
            tour_id = (created_ids[command.creation_index]
                       if command.creation_index is not None else command.tour_id)
            tour = state.tour_manager.get_tour(tour_id)
            state.tour_manager.advance(tour_id, TourStates.SCHEDULED)
            events.append(SequencingDone(state.current_time,
                                         tour.assigned_resource))
        elif isinstance(command, MarkPositioning):
            state.tour_manager.get_tour(
                created_ids[command.creation_index]).positioning_mode = command.mode
        elif isinstance(command, RecallPositioning):
            tour = state.tour_manager.get_tour(command.tour_id)
            network = state.layout_manager.get_layout().layout_network
            origin = tour.position_at(state.current_time)
            if (tour.edge_destination is not None
                    and state.current_time < tour.edge_end_time):
                first = tour.edge_destination.position
                remaining = (tour.edge_distance *
                             (tour.edge_end_time - state.current_time) /
                             (tour.edge_end_time - tour.edge_start_time))
                nodes = [origin, first, *nx.shortest_path(
                    network.graph, first, network.end_node, weight="weight")[1:]]
                distance = remaining + nx.path_weight(
                    network.graph, nodes[1:], weight="weight")
                route_origin = RoutingOrigin(origin, first, remaining)
            else:
                nodes = nx.shortest_path(network.graph, origin,
                                         network.end_node, weight="weight")
                distance = nx.path_weight(network.graph, nodes, weight="weight")
                route_origin = RoutingOrigin(origin)
            route = Route(
                distance=float(distance), routing_origin=route_origin,
                route=nodes, item_sequence=[],
                annotated_route=[RouteNode(node, NodeType.ROUTE) for node in nodes],
                batch=ActiveTourBatch(batch_id=0, orders=[], tour_id=tour.tour_id),
            )
            version = state.commit_active_route(route, ())
            tour.positioning_mode = None
            tour.positioning_hold = False
            events.append(TravelEvent(state.current_time, tour.tour_id, version))
    return events
