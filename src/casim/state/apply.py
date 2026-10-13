"""Validate and apply one decision's commands as a transaction."""

from __future__ import annotations

from copy import deepcopy

from ware_ops_algos.domain_models.lifecycle import Accepts, PHYSICAL, TourStates, accepts_for

from casim.config.schema import RevisionScope, SelectionConfig
from casim.decision_engine.commands import (
    CancelTour, Command, CommandViolation, CreatePickList, CreateTour,
    InsertOrders, Reassign, RecordConsidered, ReleaseTour, ReplaceSuffix,
    Reschedule, ViolationCode, Wait, RecallPositioning,
)


def _violate(code: ViolationCode, message: str) -> None:
    raise CommandViolation(code, message)


def _check_tour(state, command, selection: SelectionConfig, scope: RevisionScope,
                allowed: set[Accepts]):
    tour = state.tour_manager.all_tours.get(command.tour_id)
    if tour is None:
        _violate(ViolationCode.STALE, f"Tour {command.tour_id} no longer exists")
    status = TourStates(tour.status)
    if status not in selection.tours:
        _violate(ViolationCode.SCOPE, f"Tour {tour.tour_id} was not selected")
    accepts = accepts_for(status, scope.freeze_from, scope.frozen_accepts)
    if accepts not in PHYSICAL[status]:
        _violate(ViolationCode.PHYSICS, f"{status.value} tour cannot accept {accepts.value}")
    if accepts not in allowed:
        _violate(ViolationCode.SCOPE, f"Scope does not allow {type(command).__name__} on tour {tour.tour_id}")
    if tour.route_version != command.expected_route_version:
        _violate(ViolationCode.STALE, f"Tour {tour.tour_id} route version changed")
    if hasattr(command, "expected_cursor") and tour.cursor != command.expected_cursor:
        _violate(ViolationCode.STALE, f"Tour {tour.tour_id} cursor changed")
    return tour


def _apply_one(state, command: Command, selection: SelectionConfig,
               scope: RevisionScope, paired_routes: dict[int, ReplaceSuffix],
               admitted: dict[int, tuple[int, ...]], created_ids: list[int]):
    from casim.events.decision_events import PickListDone
    from casim.events.operational_events import PickerTourQuery, TravelEvent

    if isinstance(command, CreatePickList):
        ids = tuple(command.batch.order_numbers)
        if any(order_id not in state.order_manager._order_buffer for order_id in ids):
            _violate(ViolationCode.ORDER_NOT_FREE, f"Pick-list order is not buffered: {ids}")
        state.order_manager.clear_order_buffer_by_ids(ids)
        state.add_pick_list_to_planning_state(command.batch)
        return [PickListDone(state.current_time, command.batch)]

    if isinstance(command, CreateTour):
        batch = command.job.job.route.batch
        ids = tuple(command.job.order_numbers)
        if batch in state.order_manager._pick_list_buffer:
            state.order_manager.clear_pick_list_buffer([batch])
        elif all(order_id in state.order_manager._order_buffer for order_id in ids):
            state.order_manager.clear_order_buffer_by_ids(ids)
        elif ids:
            _violate(ViolationCode.ORDER_NOT_FREE, f"Tour orders are not free: {ids}")
        created_id = state.add_sequencing_to_planning_state(command.job)
        state.tour_manager.get_tour(created_id).positioning_mode = command.positioning_mode
        created_ids.append(created_id)
        return []

    if isinstance(command, CancelTour):
        tour = _check_tour(state, command, selection, scope, {Accepts.REBUILD})
        tour.status = TourStates.CANCELLED
        if tour.assigned_resource is not None:
            state.tour_manager.remove_canceled_tour_for_picker(
                tour.assigned_resource, tour.tour_id)
        if selection.orders:
            for order_id in tour.order_numbers:
                if order_id not in state.order_manager._order_buffer:
                    state.order_manager.add_order_to_buffer(
                        state.order_manager.get_order_from_history(order_id))
        else:
            state.order_manager.add_pick_list_to_buffer(tour.batch)
        return []

    if isinstance(command, InsertOrders):
        tour = _check_tour(state, command, selection, scope,
                           {Accepts.REBUILD, Accepts.INSERT, Accepts.REVISE})
        if command.tour_id not in paired_routes:
            _violate(ViolationCode.PHYSICS, "InsertOrders has no paired ReplaceSuffix")
        if any(order_id not in state.order_manager._order_buffer
               for order_id in command.order_ids):
            _violate(ViolationCode.ORDER_NOT_FREE,
                     f"Admission order is not buffered: {command.order_ids}")
        picker = state.resource_manager.get_resource(tour.assigned_resource)
        if picker.pick_cart is not None and not picker.pick_cart.box_can_mix_orders:
            free = picker.pick_cart.n_boxes - len(set(tour.cart_bins.values()))
            if len(command.order_ids) > free:
                _violate(ViolationCode.PHYSICS, "Not enough free cart bins")
        admitted[command.tour_id] = command.order_ids
        return []

    if isinstance(command, ReplaceSuffix):
        tour = _check_tour(state, command, selection, scope,
                           {Accepts.REBUILD, Accepts.INSERT, Accepts.REVISE})
        if (tour.status == TourStates.STARTED and
                getattr(command.route.batch, "tour_id", None) != tour.tour_id):
            _violate(ViolationCode.PHYSICS, "Replacement route targets another tour")
        if (tour.status == TourStates.STARTED and
                command.route.routing_origin.position != tour.position_at(state.current_time)):
            _violate(ViolationCode.STALE, "Tour origin moved after projection")
        new_ids = set(command.route.batch.order_numbers)
        added_ids = new_ids - set(tour.order_numbers)
        if not set(tour.order_numbers).issubset(new_ids):
            _violate(ViolationCode.PHYSICS, "Replacement drops an order from the tour")
        if added_ids != set(admitted.get(tour.tour_id, ())):
            _violate(ViolationCode.PHYSICS,
                     "Replacement and InsertOrders must name the same new orders")
        if tour.status == TourStates.STARTED:
            version = state.commit_active_route(
                command.route, admitted.get(tour.tour_id, ()))
            return [TravelEvent(state.current_time, tour.tour_id, version)]
        if tour.status not in (TourStates.PLANNED, TourStates.ASSIGNED,
                               TourStates.SCHEDULED, TourStates.RELEASED):
            _violate(ViolationCode.PHYSICS, "Tour is too far along to rebuild")
        state.order_manager.clear_order_buffer_by_ids(admitted.get(tour.tour_id, ()))
        tour.original_route = command.route
        tour.batch = command.route.batch
        tour.annotated_route = list(command.route.annotated_route)
        tour.remaining_picks = list(command.route.batch.pick_positions)
        tour.order_numbers = sorted(new_ids)
        picker = state.resource_manager.get_resource(tour.assigned_resource)
        if picker.pick_cart is not None:
            tour.cart_bins = ({order_id: 0 for order_id in tour.order_numbers}
                              if picker.pick_cart.box_can_mix_orders else
                              {order_id: index for index, order_id in
                               enumerate(tour.order_numbers)})
        tour.route_version += 1
        return []

    if isinstance(command, Reschedule):
        tour = _check_tour(state, command, selection, scope, {Accepts.REBUILD})
        state.tour_manager.schedule_tour(tour.tour_id,
                                         command.start_time, command.end_time)
        return []

    if isinstance(command, Reassign):
        tour = _check_tour(state, command, selection, scope, {Accepts.REBUILD})
        state.resource_manager.get_resource(command.picker_id)
        if tour.assigned_resource is not None:
            state.tour_manager.remove_canceled_tour_for_picker(
                tour.assigned_resource, tour.tour_id)
        tour.assigned_resource = command.picker_id
        state.tour_manager._picker_tour_queues[command.picker_id].append(tour.tour_id)
        return []

    if isinstance(command, ReleaseTour):
        if command.creation_index is not None:
            try:
                tour = state.tour_manager.get_tour(created_ids[command.creation_index])
            except IndexError:
                _violate(ViolationCode.STALE, "Release references a missing new tour")
        else:
            tour = _check_tour(state, command, selection, scope,
                               {Accepts.REBUILD, Accepts.INSERT, Accepts.REVISE, Accepts.NONE})
        if tour.status != TourStates.SCHEDULED:
            _violate(ViolationCode.PHYSICS, "Only a scheduled tour can be released")
        tour.status = TourStates.RELEASED
        picker = state.resource_manager.get_resource(tour.assigned_resource)
        if picker.occupied:
            return []
        state.resource_manager.mark_picker_occupied(tour.assigned_resource)
        return [PickerTourQuery(state.current_time, tour.assigned_resource)]

    if isinstance(command, Wait):
        return []

    if isinstance(command, RecallPositioning):
        import networkx as nx
        from ware_ops_algos.algorithms import ActiveTourBatch, NodeType, Route, RouteNode, RoutingOrigin

        tour = _check_tour(state, command, selection, scope,
                           {Accepts.INSERT, Accepts.REVISE})
        if (not state.done_flag or tour.status != TourStates.STARTED
                or tour.positioning_mode != "center_relocation" or tour.order_numbers):
            _violate(ViolationCode.PHYSICS, "Only an empty center tour can return at closure")
        network = state.layout_manager.get_layout().layout_network
        origin = tour.position_at(state.current_time)
        if tour.edge_destination is not None and state.current_time < tour.edge_end_time:
            first = tour.edge_destination.position
            remaining = tour.edge_distance * (tour.edge_end_time - state.current_time) / (
                tour.edge_end_time - tour.edge_start_time)
            nodes = [origin, first, *nx.shortest_path(
                network.graph, first, network.end_node, weight="weight")[1:]]
            distance = remaining + nx.path_weight(network.graph, nodes[1:], weight="weight")
            routing_origin = RoutingOrigin(origin, first, remaining)
        else:
            nodes = nx.shortest_path(network.graph, origin,
                                     network.end_node, weight="weight")
            distance = nx.path_weight(network.graph, nodes, weight="weight")
            routing_origin = RoutingOrigin(origin)
        route = Route(distance=float(distance), routing_origin=routing_origin,
                      route=nodes, item_sequence=[],
                      annotated_route=[RouteNode(node, NodeType.ROUTE) for node in nodes],
                      batch=ActiveTourBatch(batch_id=0, orders=[], tour_id=tour.tour_id))
        version = state.commit_active_route(route, ())
        tour.positioning_mode = None
        tour.positioning_hold = False
        return [TravelEvent(state.current_time, tour.tour_id, version)]

    if isinstance(command, RecordConsidered):
        state.considered_active_orders.update(command.pairs)
        return []

    raise TypeError(f"No handler for {type(command).__name__}")


def apply_all(state, decision: str, commands: tuple[Command, ...],
              selection: SelectionConfig, scope: RevisionScope):
    """Validate and apply on a detached state, then publish once.

    A failed command leaves the live state, including its tracker, untouched.
    """
    expected = state.decision_versions.get(decision, 0)
    if any(command.expected_decision_version != expected for command in commands):
        _violate(ViolationCode.STALE, f"Decision {decision} version changed")
    paired_routes = {command.tour_id: command for command in commands
                     if isinstance(command, ReplaceSuffix)}
    staged = deepcopy(state)
    admitted: dict[int, tuple[int, ...]] = {}
    created_ids: list[int] = []
    events = []
    for command in commands:
        events.extend(_apply_one(staged, command, selection, scope,
                                 paired_routes, admitted, created_ids))
    staged.decision_versions[decision] = expected + 1
    state.__dict__.update(staged.__dict__)
    return events
