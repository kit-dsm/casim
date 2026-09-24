import copy

from ware_ops_algos.algorithms import RoutingOrigin
from ware_ops_algos.domain_models import Resources, WarehouseInfoType, ResourceType, OrdersDomain, OrderType, Order, \
    OrderPosition

from casim.domain_objects.sim_domain import SimWarehouseDomain, DynamicInfo
from casim.domain_objects.tour_model import TourStates
from casim.state import State


class StateAdapter:
    def __init__(self):
        pass

    def transform_state(self, state: State, problem: str):
        pass

class HennWaitingAdapter(StateAdapter):
    def transform_state(self, state: State, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)
        for o in buffered_orders:
            assert isinstance(o, Order)

        layout = state.layout_manager.get_layout()

        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_batches=None,
            active_tours=None,
            done=state.done_flag,
            n_staged_pallets=0
        )
        resources = state.resource_manager.get_resources()
        dynamic_resources_list = []
        for r in resources.resources:
            if state.available_for_planning(r.id):
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )
        return copy.deepcopy(dynamic_information)


class ActiveTourAdapter(StateAdapter):
    """Project visible new orders and the unpicked suffix of one active tour."""

    def transform_state(self, state: State, problem: str):
        new_orders = state.order_manager.get_order_buffer()
        if len(state.resource_manager.get_resources().resources) != 1:
            raise ValueError("Active-tour admission currently requires exactly one picker")
        active = [tour for tour in state.tour_manager.all_tours.values()
                  if tour.status == TourStates.STARTED]
        if len(active) != 1:
            raise ValueError("Active-tour opportunity requires exactly one started tour")
        tour = active[0]
        new_orders = [order for order in new_orders
                      if (tour.tour_id, order.order_id) not in state.considered_active_orders]
        picker = state.resource_manager.get_resource(tour.assigned_resource)
        orders = list(new_orders)
        remaining_active_ids = set()
        for order_id in tour.order_numbers:
            picks = [pick for pick in tour.remaining_picks if pick.order_number == order_id]
            if not picks:
                continue
            remaining_active_ids.add(order_id)
            known = state.order_manager.get_order_from_history(order_id)
            orders.append(Order(
                order_id=order_id,
                order_date=known.order_date,
                due_date=known.due_date,
                order_positions=[OrderPosition(
                    order_number=order_id,
                    article_id=pick.article_id,
                    amount=pick.amount,
                ) for pick in picks],
            ))

        position = tour.position_at(state.current_time)
        if tour.edge_destination is not None and state.current_time < tour.edge_end_time:
            duration = tour.edge_end_time - tour.edge_start_time
            fraction = 1.0 if duration == 0 else (state.current_time - tour.edge_start_time) / duration
            origin = RoutingOrigin(
                position=position,
                edge_destination=tour.edge_destination.position,
                distance_to_destination=tour.edge_distance * (1.0 - fraction),
            )
        else:
            origin = RoutingOrigin(position=position)
        dynamic = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            current_picker=picker,
            active_tours=[tour],
            active_tour_id=tour.tour_id,
            active_route_version=tour.route_version,
            routing_origin=origin,
            active_order_ids=frozenset(remaining_active_ids),
            active_candidate_ids=frozenset(order.order_id for order in new_orders),
            remaining_route_positions=tuple(
                node.position for node in tour.annotated_route[
                    tour.cursor + (1 if tour.edge_destination is not None else 0):
                ]
            ),
            occupied_bins=len(set(tour.cart_bins.values())),
            done=state.done_flag,
        )
        return copy.deepcopy(SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=state.layout_manager.get_layout(),
            orders=OrdersDomain(tpe=OrderType.STANDARD, orders=orders),
            resources=Resources(ResourceType.HUMAN, [picker]),
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=dynamic,
        ))


class OrderWindowAdapter(StateAdapter):
    def transform_state(self, state: State, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()

        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)

        layout = state.layout_manager.get_layout()

        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_batches=None,
            active_tours=None,
            done=state.done_flag,
            n_staged_pallets=0
        )
        resources = state.resource_manager.get_resources()
        dynamic_resources_list = []
        for r in resources.resources:
            if not r.occupied:
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )
        return dynamic_information

class ORSPAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: State, problem: str):
        layout = state.layout_manager.get_layout()
        buffered_pls = state.order_manager.get_pick_list_buffer()
        active_or_scheduled_tours = []
        all_tours = state.tour_manager.all_tours
        for tour_id, tour in all_tours.items():
            if tour.status in [TourStates.STARTED,
                               TourStates.SCHEDULED,
                               TourStates.ASSIGNED]:
                active_or_scheduled_tours.append(tour)
        n_staged_pallets = 0
        if hasattr(state, "dock_manager"):
            n_staged_pallets = state.dock_manager.n_staged_pallets
        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_batches=buffered_pls,
            active_tours=active_or_scheduled_tours,
            done=state.done_flag,
            n_staged_pallets=n_staged_pallets,
            is_break=state.is_break
        )
        resources = state.resource_manager.get_resources()
        dynamic_resources_list = []
        for r in resources.resources:
            if not r.occupied and r.available:
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=OrdersDomain(tpe=OrderType.STANDARD, orders=[]),
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information

class ReORSPAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: State, problem: str):
        layout = state.layout_manager.get_layout()
        buffered_pls = state.order_manager.get_pick_list_buffer()
        scheduled_tours = []
        all_tours = state.tour_manager.all_tours
        for tour_id, tour in all_tours.items():  # collect all unstarted unfinished tours, STARTED and PENDING are executed as planned
            if tour.status not in [TourStates.STARTED,
                                   TourStates.DONE,
                                   TourStates.CANCELLED,
                                   TourStates.PENDING]:
                scheduled_tours.append(tour)
                buffered_pls.append(tour.batch)

        n_staged_pallets = 0
        if hasattr(state, "dock_manager"):
            n_staged_pallets = state.dock_manager.n_staged_pallets
        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_batches=buffered_pls,
            active_tours=scheduled_tours,
            done=state.done_flag,
            n_staged_pallets=n_staged_pallets,
            is_break=state.is_break
        )
        resources = state.resource_manager.get_resources()
        dynamic_resources_list = []
        for r in resources.resources:
            if r.available:
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=OrdersDomain(tpe=OrderType.STANDARD, orders=[]),
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information

class OBPAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: State, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)

        layout = state.layout_manager.get_layout()
        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_batches=None,
            active_tours=None,
            done=state.done_flag,
            n_staged_pallets=0
        )
        resources = state.resource_manager.get_resources()
        dynamic_resources_list = []
        for r in resources.resources:
            # if not r.occupied:
            dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information

class OSBPAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: State, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)

        layout = state.layout_manager.get_layout()
        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_batches=None,
            active_tours=None,
            done=state.done_flag,
            n_staged_pallets=0
        )
        resources = state.resource_manager.get_resources()
        dynamic_resources_list = []
        for r in resources.resources:
            # if not r.occupied:
            dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information

class ReOSBPAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.orders = None

    def transform_state(self, state: State, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)

        layout = state.layout_manager.get_layout()
        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_batches=None,
            active_tours=None,
            done=state.done_flag,
            n_staged_pallets=0
        )
        resources = state.resource_manager.get_resources()
        dynamic_resources_list = []
        for r in resources.resources:
            # if not r.occupied:
            if r.available:
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information

class RLORSPAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: State, problem: str):
        layout = state.layout_manager.get_layout()
        buffered_pls = state.order_manager.get_pick_list_buffer()[:1]
        # sorted_pls = sorted(buffered_pls, key=lambda o: o.due_date)

        active_or_scheduled_tours = []
        all_tours = state.tour_manager.all_tours
        for tour_id, tour in all_tours.items():
            if tour.status in [TourStates.STARTED,
                               TourStates.SCHEDULED,
                               TourStates.ASSIGNED]:
                active_or_scheduled_tours.append(tour)
        n_staged_pallets = 0
        if hasattr(state, "dock_manager"):
            n_staged_pallets = state.dock_manager.n_staged_pallets
        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_batches=buffered_pls,
            active_tours=active_or_scheduled_tours,
            done=state.done_flag,
            n_staged_pallets=n_staged_pallets,
            is_break=state.is_break
        )
        resources = state.resource_manager.get_resources()
        dynamic_resources_list = []
        for r in resources.resources:
            if not r.occupied:
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=OrdersDomain(tpe=OrderType.STANDARD, orders=[]),
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information

class RLOSBPAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: State, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()
        sorted_orders = sorted(buffered_orders, key=lambda o: o.due_date)
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=sorted_orders[:1])

        layout = state.layout_manager.get_layout()
        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_batches=None,
            active_tours=None,
            done=state.done_flag,
            n_staged_pallets=0
        )
        resources = state.resource_manager.get_resources()
        dynamic_resources_list = []
        for r in resources.resources:
            if not r.occupied:
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information
