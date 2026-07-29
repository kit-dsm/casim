from ware_ops_algos.algorithms import AlgorithmSolution, SchedulingSolution, BatchingSolution
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
        self.state_snapshot = dynamic_information
        return dynamic_information

    def cleanup_state(self, state: State, solution: SchedulingSolution):
        input_orders = {}
        for o in self.state_snapshot.orders.orders:
            input_orders[o.order_id] = o

        orders = []
        for j in solution.jobs:
            for o_id in j.route.batch.order_numbers:
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
        self.state_snapshot = dynamic_information
        return dynamic_information

    def cleanup_state(self, state: State, solution: SchedulingSolution):
        input_orders = {}
        for o in self.state_snapshot.orders.orders:
            input_orders[o.order_id] = o

        orders = []
        for j in solution.jobs:
            for o_id in j.route.batch.order_numbers:
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

    def cleanup_state(self, state: State, solution: SchedulingSolution):
        pls = []
        for j in solution.jobs:
            pls.append(j.job.route.batch)
            state.add_sequencing_to_planning_state(j)
        state.order_manager.clear_pick_list_buffer(pls)
        # picker_id = solution.jobs[0].picker_id
        # state.resource_manager.mark_picker_occupied(picker_id)
        print(f"Garbage Collected {len(pls)} batches")

class ReORSPAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: State, problem: str):
        layout = state.layout_manager.get_layout()
        buffered_pls = state.order_manager.get_pick_list_buffer()  ## this should contain stuff
        scheduled_tours = []
        batches = []
        all_tours = state.tour_manager.all_tours
        for tour_id, tour in all_tours.items():  # collect all unstarted unfinished tours, STARTED and PENDING are executed as planned
            if tour.status not in [TourStates.STARTED,
                                   TourStates.DONE,
                                   TourStates.CANCELLED,
                                   TourStates.PENDING]:
                scheduled_tours.append(tour)
                tour.status = TourStates.CANCELLED
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

    def cleanup_state(self, state: State, solution: SchedulingSolution):
        pls = []
        for j in solution.jobs:
            pls.append(j.job.route.batch)
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

    def cleanup_state(self, state: State, solution: BatchingSolution):
        orders = [o for b in solution.batches for o in b.orders]
        state.order_manager.clear_order_buffer(orders)
        for b in solution.batches:
            state.add_pick_list_to_planning_state(b)


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

    def cleanup_state(self, state: State, solution: BatchingSolution):
        parent_ids = {
            o.parent_order_id or o.order_id
            for b in solution.batches
            for o in b.orders
        }
        state.order_manager.clear_order_buffer_by_ids(parent_ids)
        for b in solution.batches:
            state.add_pick_list_to_planning_state(b)
        split_orders = [
            Order(
                order_id=o.order_id,
                parent_order_id=o.parent_order_id,
                order_date=o.order_date,
                due_date=o.due_date,
                order_positions=[
                    OrderPosition(
                        order_number=o.order_id,
                        article_id=pp.article_id,
                        article_name=pp.article_id,
                        amount=pp.amount,
                    )
                ],
            )
            for b in solution.batches
            for o in b.orders
            for pp in o.pick_positions
        ]
        for order in split_orders:
            state.order_manager.add_order_to_buffer(order)
            state.order_manager.clear_order_buffer([order])

        print(f"Garbage Collected {len(parent_ids)} orders")


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

    def cleanup_state(self, state: State, solution: BatchingSolution):
        parent_ids = {
            o.parent_order_id or o.order_id
            for b in solution.batches
            for o in b.orders
        }
        state.order_manager.clear_order_buffer_by_ids(parent_ids)
        for b in solution.batches:
            state.add_pick_list_to_planning_state(b)
        split_orders = [
            Order(
                order_id=o.order_id,
                parent_order_id=o.parent_order_id,
                order_date=o.order_date,
                due_date=o.due_date,
                order_positions=[
                    OrderPosition(
                        order_number=o.order_id,
                        article_id=pp.article_id,
                        article_name=pp.article_id,
                        amount=pp.amount,
                    )
                ],
            )
            for b in solution.batches
            for o in b.orders
            for pp in o.pick_positions
        ]
        for order in split_orders:
            state.order_manager.add_order_to_buffer(order)
            state.order_manager.clear_order_buffer([order])

        print(f"Garbage Collected {len(parent_ids)} orders")


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

    def cleanup_state(self, state: State, solution: SchedulingSolution):
        pls = []
        for j in solution.jobs:
            pls.append(j.job.route.batch)
            state.add_sequencing_to_planning_state(j)
        state.order_manager.clear_pick_list_buffer(pls)
        # picker_id = solution.jobs[0].picker_id
        # state.resource_manager.mark_picker_occupied(picker_id)
        print(f"Garbage Collected {len(pls)} batches")

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

    def cleanup_state(self, state: State, solution: BatchingSolution):
        parent_ids = {
            o.parent_order_id or o.order_id
            for b in solution.batches
            for o in b.orders
        }
        state.order_manager.clear_order_buffer_by_ids(parent_ids)
        for b in solution.batches:
            state.add_pick_list_to_planning_state(b)
        split_orders = [
            Order(
                order_id=o.order_id,
                parent_order_id=o.parent_order_id,
                order_date=o.order_date,
                due_date=o.due_date,
                order_positions=[
                    OrderPosition(
                        order_number=o.order_id,
                        article_id=pp.article_id,
                        article_name=pp.article_id,
                        amount=pp.amount,
                    )
                ],
            )
            for b in solution.batches
            for o in b.orders
            for pp in o.pick_positions
        ]
        for order in split_orders:
            state.order_manager.add_order_to_buffer(order)
            state.order_manager.clear_order_buffer([order])

        print(f"Garbage Collected {len(parent_ids)} orders")