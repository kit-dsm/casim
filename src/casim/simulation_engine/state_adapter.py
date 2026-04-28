from ware_ops_algos.algorithms import AlgorithmSolution, SchedulingSolution, BatchingSolution
from ware_ops_algos.domain_models import Resources, WarehouseInfoType, ResourceType, OrdersDomain, OrderType, Order

from casim.domain_objects.sim_domain import SimWarehouseDomain, DynamicInfo
from casim.domain_objects.tour_model import TourStates
from casim.state import State


class StateAdapter:
    def __init__(self):
        pass

    def transform_state(self, state: State, problem: str):
        pass

    def cleanup_state(self, state: State, solution: AlgorithmSolution):
        pass


class HennWaitingAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.state_snapshot = None

    def transform_state(self, state: State, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()

        active_or_scheduled_tours = []
        all_tours = state.tour_manager.all_tours
        for tour_id, tour in all_tours.items():
            if tour.status in [TourStates.SCHEDULED,
                               TourStates.PENDING,
                               TourStates.ASSIGNED]:
                active_or_scheduled_tours.append(tour)

        # collect raw orders from unstarted tours
        for t in active_or_scheduled_tours:
            for oid in t.order_numbers:
                order = state.order_manager.get_order_from_history(oid)
                buffered_orders.append(order)
        # orders to be considered -> newly arrived + orders from pending tours
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)
        for o in buffered_orders:
            assert isinstance(o, Order)

        layout = state.layout_manager.get_layout()

        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_pick_lists=None,
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
            objective="Distance",
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )
        self.state_snapshot = dynamic_information
        return dynamic_information

    def cleanup_state(self, state: State, solution: SchedulingSolution):
        input_orders = {}
        for o in self.state_snapshot.orders.orders:
            input_orders[o.order_id] = o

        orders = []
        for j in solution.jobs:
            for o_id in j.route.pick_list.order_numbers:
                orders.append(input_orders[o_id])
            state.add_sequencing_to_planning_state(j)
        state.order_manager.clear_order_buffer(orders)
        print(f"Garbage Collected {len(orders)} orders")


class OrderWindowAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.state_snapshot = None

    def transform_state(self, state: State, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()

        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)

        layout = state.layout_manager.get_layout()

        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_pick_lists=None,
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
            objective="Distance",
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )
        self.state_snapshot = dynamic_information
        return dynamic_information

    def cleanup_state(self, state: State, solution: SchedulingSolution):
        input_orders = {}
        for o in self.state_snapshot.orders.orders:
            input_orders[o.order_id] = o

        orders = []
        for j in solution.jobs:
            for o_id in j.route.pick_list.order_numbers:
                orders.append(input_orders[o_id])
            state.add_sequencing_to_planning_state(j)
        state.order_manager.clear_order_buffer(orders)
        print(f"Garbage Collected {len(orders)} orders")



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
            buffered_pick_lists=buffered_pls,
            active_tours=active_or_scheduled_tours,
            done=state.done_flag,
            n_staged_pallets=n_staged_pallets
        )
        resources = state.resource_manager.get_resources()
        dynamic_resources_list = []
        for r in resources.resources:
            if not r.occupied:
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective="distance",
            layout=layout,
            orders=OrdersDomain(tpe=OrderType.STANDARD, orders=[]),
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information

    def cleanup_state(self, state: State, solution: SchedulingSolution):
        pls = []
        for j in solution.jobs:
            pls.append(j.route.pick_list)
            state.add_sequencing_to_planning_state(j)
        state.order_manager.clear_pick_list_buffer(pls)
        # picker_id = solution.jobs[0].picker_id
        # state.resource_manager.mark_picker_occupied(picker_id)
        print(f"Garbage Collected {len(pls)} batches")


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
            buffered_pick_lists=None,
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
            objective="distance",
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information

    def cleanup_state(self, state: State, solution: BatchingSolution):
        orders = [o for pl in solution.pick_lists for o in pl.orders]
        state.order_manager.clear_order_buffer(orders)
        for pl in solution.pick_lists:
            state.add_pick_list_to_planning_state(pl)
        print(f"Garbage Collected {len(orders)} orders")