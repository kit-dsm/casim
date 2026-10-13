import copy

from ware_ops_algos.algorithms import AdmissionTour, RoutingOrigin
from ware_ops_algos.domain_models import Resources, ResourceType, OrdersDomain, OrderType, Order, \
    OrderPosition, SimWarehouseDomain, TourStates, DecisionContext, PlanView, TourView
from ware_ops_algos.domain_models.lifecycle import Accepts, ResourceStates, accepts_for

from casim.config.schema import RevisionScope, SelectionConfig
from casim.state import State
from casim.state.views import DecisionMemory, OperationsView


class StateAdapter:
    """Project one decision's selected state without changing live plan data."""

    def __init__(self, selection=None, scope=None, extensions=()):
        self.selection = (SelectionConfig.model_validate(selection)
                          if selection is not None else None)
        self.scope = RevisionScope.model_validate(scope or {})
        self.extensions = tuple(extensions)

    def transform_state(self, state: State, decision: str,
                        problem_class: str | None = None):
        if self.selection is None:
            raise NotImplementedError("StateAdapter needs a lifecycle selection")

        selection = self.selection
        now = state.current_time
        buffered = (state.order_manager.get_order_buffer()
                    if selection.orders else [])
        pick_lists = (state.order_manager.get_pick_list_buffer()
                      if selection.pick_lists else [])
        selected_tours = [
            tour for tour in state.tour_manager.all_tours.values()
            if TourStates(tour.status) in selection.tours
        ]
        if selection.window is not None:
            end = now + selection.window.horizon_s
            if selection.window.by.value == "due_date":
                if not state.done_flag:
                    buffered = [o for o in buffered
                                if o.due_date is not None and o.due_date <= end]
                    pick_lists = [b for b in pick_lists
                                  if b.earliest_due_date <= end]
            else:
                selected_tours = [t for t in selected_tours
                                  if t.start_time is None or now <= t.start_time <= end]

        resources = state.resource_manager.get_resources()
        selected_resources = [
            resource for resource in resources.resources
            if state.resource_manager.resource_status(
                resource.id, state.tour_manager) in selection.resources
        ]
        # Started tours bring their pickers even when the selection omits a
        # resource status, as admission must know the cart and travel speed.
        selected_ids = {resource.id for resource in selected_resources}
        for tour in selected_tours:
            if tour.status == TourStates.STARTED and tour.assigned_resource not in selected_ids:
                selected_resources.append(
                    state.resource_manager.get_resource(tour.assigned_resource))
                selected_ids.add(tour.assigned_resource)

        tour_views = tuple(self._tour_view(state, tour) for tour in selected_tours)
        residual_orders = self._residual_orders(state, selected_tours, tour_views)
        available_from = {
            resource.id: max(
                now,
                state.resource_manager.picker_busy_until(resource.id),
                *(t.end_time_planned or now for t in selected_tours
                  if t.assigned_resource == resource.id
                  and t.status in (TourStates.RELEASED, TourStates.PENDING, TourStates.STARTED)),
            ) for resource in selected_resources
        }
        plan = PlanView(
            orders=tuple(buffered), pick_lists=tuple(pick_lists),
            tours=tour_views, resources=tuple(selected_resources),
            available_from=available_from,
            active_candidate_ids=frozenset(o.order_id for o in buffered),
        )
        extras = {}
        for extension in self.extensions:
            extras.update(extension(state))
        operations = OperationsView(
            is_break=state.is_break,
            n_staged_pallets=(state.dock_manager.n_staged_pallets
                              if hasattr(state, "dock_manager") else 0),
            extra=extras,
        )
        context = DecisionContext(
            time=now, cause=(), decision=decision,
            version=getattr(state, "decision_versions", {}).get(decision, 0),
            input_closed=state.done_flag,
        )
        memory = DecisionMemory(
            considered_active_orders=frozenset(state.considered_active_orders),
            last_information_version=state.last_solved_information_version.get(decision, -1),
            current_information_version=state.information_version)

        projected = SimWarehouseDomain(
            problem_class=problem_class or decision, objective=state.active_objective,
            layout=state.layout_manager.get_layout(),
            orders=OrdersDomain(tpe=OrderType.STANDARD,
                                orders=[*buffered, *residual_orders]),
            resources=Resources(ResourceType.HUMAN, selected_resources),
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(), dynamic_warehouse_info=None,
            warehouse_info=state.warehouse_info, information=state.information,
            context=context, plan=plan, memory=memory, operations=operations,
        )
        return copy.deepcopy(projected)

    def _tour_view(self, state: State, tour) -> TourView:
        status = TourStates(tour.status)
        picker_id = tour.assigned_resource
        picker = (state.resource_manager.get_resource(picker_id)
                  if picker_id is not None else None)
        edge_in_progress = (tour.edge_destination is not None
                            and tour.edge_end_time is not None
                            and state.current_time < tour.edge_end_time)
        if edge_in_progress:
            duration = tour.edge_end_time - tour.edge_start_time
            fraction = (1.0 if duration == 0 else
                        (state.current_time - tour.edge_start_time) / duration)
            origin = RoutingOrigin(
                position=tour.position_at(state.current_time),
                edge_destination=tour.edge_destination.position,
                distance_to_destination=tour.edge_distance * (1.0 - fraction),
            )
        else:
            position = (tour.position_at(state.current_time) if status == TourStates.STARTED
                        else tour.annotated_route[0].position)
            origin = RoutingOrigin(position=position)
        residual_ids = frozenset(
            pick.order_number for pick in tour.remaining_picks)
        pinned_end = tour.cursor + int(edge_in_progress)
        return TourView(
            tour_id=tour.tour_id, picker_id=picker_id, status=status,
            route_version=tour.route_version, cursor=tour.cursor,
            pinned_prefix=tuple(tour.annotated_route[:pinned_end]),
            accepts=accepts_for(status, self.scope.freeze_from,
                                self.scope.frozen_accepts),
            pick_cart=picker.pick_cart if picker else None,
            active_order_ids=residual_ids,
            remaining_route=tuple(node.position for node in
                                  tour.annotated_route[pinned_end:]),
            occupied_bins=len(set(tour.cart_bins.values())),
            routing_origin=origin, picking_until=tour.picking_until,
            route=tour.original_route, batch=tour.batch,
            start_time=tour.start_time, end_time=tour.end_time_planned,
            processing_time=tour.processing_time,
            positioning_mode=tour.positioning_mode,
            positioning_hold=tour.positioning_hold,
        )

    @staticmethod
    def _residual_orders(state: State, tours, views: tuple[TourView, ...]) -> list[Order]:
        orders = []
        for tour, view in zip(tours, views):
            if view.accepts == Accepts.NONE:
                continue
            if tour.status in (TourStates.PLANNED, TourStates.ASSIGNED,
                               TourStates.SCHEDULED):
                orders.extend(state.order_manager.get_order_from_history(order_id)
                              for order_id in tour.order_numbers)
                continue
            if tour.status != TourStates.STARTED:
                continue
            for order_id in tour.order_numbers:
                picks = [pick for pick in tour.remaining_picks
                         if pick.order_number == order_id]
                if not picks:
                    continue
                known = state.order_manager.get_order_from_history(order_id)
                orders.append(Order(
                    order_id=order_id, order_date=known.order_date,
                    due_date=known.due_date,
                    order_positions=[OrderPosition(
                        order_number=order_id, article_id=pick.article_id,
                        amount=pick.amount) for pick in picks],
                ))
        return orders
