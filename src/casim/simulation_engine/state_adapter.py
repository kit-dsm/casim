"""Project the selected operational state into a typed decision snapshot."""

from __future__ import annotations

from copy import deepcopy

from ware_ops_algos.algorithms import RoutingOrigin
from ware_ops_algos.domain_models import (
    Order, OrderPosition, OrdersDomain, OrderType, ResourceType, Resources,
)
from ware_ops_algos.planning import (
    Accepts, PHYSICAL, PlanView, RevisionScope, StateInformation, TourStates, TourView,
    check_scope,
)
from ware_ops_algos.planning.lifecycle import TimeAttribute

from casim.domain_objects.sim_domain import (
    DecisionContext, DecisionMemory, DynamicInfo, SimWarehouseDomain,
)
from casim.state import State


class StateAdapter:
    """Expose only the plan facts configured for one decision."""

    def __init__(self, state_information: StateInformation,
                 scope: RevisionScope):
        self.state_information = state_information
        self.scope = scope
        forbidden = check_scope(state_information, scope)
        if forbidden:
            status = forbidden[0]
            raise ValueError(
                f"state_information.tours includes {status.value}: scope gives "
                f"{scope.accepts(status).value}, physical bound is "
                f"{PHYSICAL[status].value}"
            )

    def transform_state(self, state: State, decision: str,
                        problem_class: str) -> SimWarehouseDomain:
        """Build one detached snapshot from manager accessors."""
        info = self.state_information
        now = state.current_time
        buffered = state.order_manager.get_order_buffer() if info.orders else []
        pick_lists = (state.order_manager.get_pick_list_buffer()
                      if info.pick_lists else [])
        tours = [tour for tour in state.tour_manager.get_all_tours()
                 if tour.status in info.tours]
        if info.window is not None:
            end = now + info.window.horizon_s
            if info.window.by is TimeAttribute.DUE_DATE and not state.done_flag:
                buffered = [order for order in buffered
                            if order.due_date is not None and order.due_date <= end]
                pick_lists = [batch for batch in pick_lists
                              if batch.earliest_due_date <= end]
            if info.window.by is TimeAttribute.START_TIME:
                tours = [tour for tour in tours
                         if tour.start_time is None or now <= tour.start_time <= end]

        resources = state.resource_manager.get_resources().resources
        selected = [resource for resource in resources
                    if state.resource_manager.resource_status(
                        resource.id, state.tour_manager) in info.resources]
        selected_ids = {resource.id for resource in selected}
        for tour in tours:
            if (tour.status is TourStates.STARTED
                    and tour.assigned_resource not in selected_ids):
                selected.append(state.resource_manager.get_resource(
                    tour.assigned_resource))
                selected_ids.add(tour.assigned_resource)

        tour_views = tuple(self._tour_view(state, tour) for tour in tours)
        residual = self._residual_orders(state, tours, tour_views)
        available_from = {
            resource.id: max(
                now,
                state.resource_manager.picker_busy_until(resource.id),
                *(tour.end_time_planned or now for tour in tours
                  if tour.assigned_resource == resource.id
                  and tour.status in (TourStates.SCHEDULED, TourStates.PENDING,
                                      TourStates.STARTED)),
            ) for resource in selected
        }
        plan = PlanView(
            time=now, order_buffer=tuple(buffered),
            pick_list_buffer=tuple(pick_lists), tours=tour_views,
            available_from=available_from,
        )
        dynamic = DynamicInfo(
            context=DecisionContext(
                decision=decision, cause=(), reconsider=(),
                input_closed=state.done_flag, solve=True,
            ),
            plan=plan,
            memory=DecisionMemory(
                considered_active_orders=frozenset(
                    state.decision_considered_active_orders.get(decision, ())),
                last_information_version=state.last_solved_information_version.get(
                    decision, -1),
                current_information_version=state.information_version,
                waiting=state.decision_waiting.get(decision, False),
                reconsider_at=state.decision_reconsider_at.get(decision),
            ),
        )
        snapshot = SimWarehouseDomain(
            problem_class=problem_class, objective=state.active_objective,
            layout=state.layout_manager.get_layout(),
            orders=OrdersDomain(tpe=OrderType.STANDARD,
                                orders=[*buffered, *residual]),
            resources=Resources(ResourceType.HUMAN, selected),
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            warehouse_info=state.warehouse_info,
            information=state.information,
            dock=state.dock_manager.dock,
            dynamic_warehouse_info=dynamic,
        )
        return deepcopy(snapshot)

    def full_plan_view(self, state: State) -> PlanView:
        """Project the complete plan for optional independent auditing."""
        return PlanView(
            time=state.current_time,
            order_buffer=tuple(state.order_manager.get_order_buffer()),
            pick_list_buffer=tuple(state.order_manager.get_pick_list_buffer()),
            tours=tuple(self._tour_view(state, tour)
                        for tour in state.tour_manager.get_all_tours()),
        )

    def _tour_view(self, state: State, tour) -> TourView:
        status = TourStates(tour.status)
        picker_id = tour.assigned_resource
        picker = (state.resource_manager.get_resource(picker_id)
                  if picker_id is not None else None)
        traveling = (tour.edge_destination is not None
                     and tour.edge_end_time is not None
                     and state.current_time < tour.edge_end_time)
        edge_in_progress = tour.edge_destination is not None
        if traveling:
            duration = tour.edge_end_time - tour.edge_start_time
            fraction = (1.0 if duration == 0 else
                        (state.current_time - tour.edge_start_time) / duration)
            origin = RoutingOrigin(
                position=tour.position_at(state.current_time),
                edge_destination=tour.edge_destination.position,
                distance_to_destination=tour.edge_distance * (1.0 - fraction),
            )
        else:
            position = (tour.position_at(state.current_time)
                        if status is TourStates.STARTED
                        else tour.annotated_route[0].position)
            origin = RoutingOrigin(position=position)
        pinned_end = tour.cursor + int(edge_in_progress)
        return TourView(
            tour_id=tour.tour_id, picker_id=picker_id, status=status,
            accepts=self.scope.accepts(status),
            batch=tour.batch, route=tour.route,
            start_time=tour.start_time, end_time=tour.end_time_planned,
            processing_time=tour.processing_time,
            pick_cart=picker.pick_cart if picker else None,
            active_order_ids=frozenset(pick.order_number
                                       for pick in tour.remaining_picks),
            cursor=tour.cursor,
            pinned_prefix=tour.pinned_prefix(edge_in_progress),
            remaining_route=tuple(node.position for node in
                                  tour.annotated_route[pinned_end:]),
            routing_origin=origin,
            occupied_bins=len(set(tour.cart_bins.values())),
            picking_until=tour.picking_until,
            holding=(status is TourStates.STARTED and tour.at_end()
                     and tour.annotated_route[-1].position !=
                     state.layout_manager.get_layout().layout_network.end_node),
        )

    @staticmethod
    def _residual_orders(state: State, tours,
                         views: tuple[TourView, ...]) -> list[Order]:
        orders = []
        for tour, view in zip(tours, views):
            if view.accepts is Accepts.NONE:
                continue
            if tour.status in (TourStates.PLANNED, TourStates.SCHEDULED):
                orders.extend(state.order_manager.get_order_from_history(order_id)
                              for order_id in tour.order_numbers)
                continue
            if tour.status is not TourStates.STARTED:
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
