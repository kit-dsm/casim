from copy import deepcopy
from pathlib import Path
from typing import Any, Type

from ware_ops_algos.algorithms import SchedulingSolution, SequencingSolution, CombinedRoutingSolution, WarehouseOrder, \
    AlgorithmSolution, OrderSelectionSolution
from ware_ops_algos.domain_models import OrdersDomain, OrderType, BaseWarehouseDomain, Resources, ResourceType, \
    ResolvedOrderPosition, WarehouseInfoType, WarehouseInfo

from ware_ops_sim.sim.sim_domain import SimWarehouseDomain, DynamicInfo
from ware_ops_sim.sim.state import SimulationState
from ware_ops_sim.sim.state.tour_manager import TourStates
from ware_ops_algos.utils.io_helpers import dump_pickle


class StateTransformer:
    def __init__(self):
        pass

    def transform_state(self, state: SimulationState, problem: str):
        pass

    def on_success(self, state: SimulationState,
                   solution: Type[AlgorithmSolution]):
        pass


class OBSRPStateTransformer(StateTransformer):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.orders = None

    def transform_state(self, state: SimulationState, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()
        print("check this", len(buffered_orders))
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)
        self.orders = buffered_orders
        layout = state.layout_manager.get_layout()

        # tours = state.tour_manager.assignable_tours
        # pick_lists = []
        # for t in tours.keys():
        #     pick_lists.append(tours[t].pick_list)

        warehouse_info = WarehouseInfo(
            tpe=WarehouseInfoType.OFFLINE,
        )
        resources = state.resource_manager.get_resources()
        dynamic_resources_list = []
        for r in resources.resources:
            if not r.occupied:
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = BaseWarehouseDomain(
            problem_class=problem,
            objective="Distance",
            layout=layout,
            orders=orders,
            # resources=state.resource_manager.get_resources(),
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            warehouse_info=warehouse_info
        )

        cache_dir = Path(f"{self.domain_cache_path}/sim")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "dynamic_info.pkl"
        cache_path_tours = cache_dir / "tours.pkl"
        cache_path_pl = cache_dir / "pick_lists.pkl"

        self.cache_path = cache_path
        # dump_pickle(str(cache_path), dynamic_information)
        # dump_pickle(str(cache_path_tours), tours)
        # dump_pickle(str(cache_path_pl), pick_lists)
        return dynamic_information

    def on_success(self, state: SimulationState, solution):
        # if isinstance(solution, SchedulingSolution):
        print("successfull state")
        # orders = solution.jobs[0].route.pick_list.orders
        orders = []
        for j in solution.jobs:
            for o in j.route.pick_list.orders:
                orders.append(o)
        # assert len(orders) == 1
        state.order_manager.clear_order_buffer(orders)

        self.orders = None


class OnlineStateTransformer(StateTransformer):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.orders = None

    def transform_state(self, state: SimulationState, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)
        self.orders = buffered_orders
        # print("Orders dumped", len(buffered_orders))
        layout = state.layout_manager.get_layout()

        resources_list = [state.resource_manager.get_resource(state.current_picker_id)]
        current_picker = state.resource_manager.get_resource(state.current_picker_id)
        # print("resources send", resources_list)
        # state.current_picker_id = None
        tours = state.tour_manager.all_tours
        tours_not_done = []
        for i, t in tours.items():
            if t.status != TourStates.DONE:
                tours_not_done.append(t)

        pick_lists = []
        for t in tours.keys():
            pick_lists.append(tours[t].pick_list)
        # print("Pick Tours state", len(tours_not_done))

        resources_dynamic = Resources(ResourceType.MIXED, resources_list)

        dynamic_info = DynamicInfo(
            WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=state.resource_manager.get_aisle_count(),
            active_tours=tours_not_done,
            current_picker=current_picker
        )
        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective="Distance",
            layout=layout,
            orders=orders,
            resources=state.resource_manager.get_resources(),
            # resources=resources_dynamic,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            warehouse_info=dynamic_info
        )

        cache_dir = Path(f"{self.domain_cache_path}/sim")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "dynamic_info.pkl"
        cache_path_tours = cache_dir / "tours.pkl"
        cache_path_pl = cache_dir / "pick_lists.pkl"

        self.cache_path = cache_path
        # dump_pickle(str(cache_path), dynamic_information)
        # dump_pickle(str(cache_path_tours), tours)
        # dump_pickle(str(cache_path_pl), pick_lists)
        return dynamic_information

    def on_success(self, state: SimulationState, solution: Type[AlgorithmSolution]):
        state.current_picker_id = None
        orders: list[WarehouseOrder] = []
        if isinstance(solution, SchedulingSolution):
            orders = solution.jobs[0].route.pick_list.orders
        elif isinstance(solution, CombinedRoutingSolution):
            for s in solution.routes:
                for o in s.pick_list.orders:
                    orders.append(o)
        print(len(orders))
        # assert len(orders) == 1
        state.order_manager.clear_order_buffer(orders)

class RLStateTransformer(StateTransformer):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.orders = None

    def transform_state(self, state: SimulationState, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)
        self.orders = buffered_orders
        layout = state.layout_manager.get_layout()

        resources_list = [state.resource_manager.get_resource(state.current_picker_id)]
        current_picker = state.resource_manager.get_resource(state.current_picker_id)
        # print("resources send", resources_list)
        # state.current_picker_id = None
        tours = state.tour_manager.all_tours
        tours_not_done = []
        for i, t in tours.items():
            if t.status == TourStates.STARTED:
                tours_not_done.append(t)

        pick_lists = []
        for t in tours.keys():
            pick_lists.append(tours[t].pick_list)
        # print("Pick Tours state", len(tours_not_done))

        resources_dynamic = Resources(ResourceType.MIXED, resources_list)

        dynamic_info = DynamicInfo(
            WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=state.resource_manager.get_aisle_count(),
            active_tours=tours_not_done,
            current_picker=current_picker
        )
        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective="Distance",
            layout=layout,
            orders=orders,
            resources=state.resource_manager.get_resources(),
            # resources=resources_dynamic,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            warehouse_info=dynamic_info
        )
        return dynamic_information

    def on_success(self, state: SimulationState, solution: OrderSelectionSolution):
        state.current_picker_id = None
        orders = solution.selected_orders
        assert len(orders) == 1
        state.order_manager.clear_order_buffer(orders)


class SPRPStateTransformer(StateTransformer):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.orders = None

    def transform_state(self, state: SimulationState, problem: str):

        layout = state.layout_manager.get_layout()

        resources_list = [state.resource_manager.get_resource(state.current_picker_id)]
        current_picker = state.resource_manager.get_resource(state.current_picker_id)
        selected_order = state.tour_manager.get_selected_order(
            picker_id=current_picker.id)
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=[selected_order])
        # print("order send state transfo", selected_order)
        # print("resources send", resources_list)
        tours = state.tour_manager.all_tours
        tours_not_done = []
        for i, t in tours.items():
            if t.status != TourStates.DONE:
                tours_not_done.append(t)

        pick_lists = []
        for t in tours.keys():
            pick_lists.append(tours[t].pick_list)
        # print("Pick Tours state", len(tours_not_done))

        dynamic_info = DynamicInfo(
            WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=state.resource_manager.get_aisle_count(),
            active_tours=[],
            current_picker=current_picker
        )

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective="Distance",
            layout=layout,
            orders=orders,
            resources=state.resource_manager.get_resources(),
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            warehouse_info=dynamic_info
        )

        cache_dir = Path(f"{self.domain_cache_path}/sim")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "dynamic_info.pkl"

        self.cache_path = cache_path
        # dump_pickle(str(cache_path), dynamic_information)
        return dynamic_information

    def on_success(self, state: SimulationState,
                   solution: Type[AlgorithmSolution]):
        pass


class ReoptStateTransformer(StateTransformer):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.tours_to_prune = set()

    def transform_state(self, state: SimulationState, problem: str):
        # In this setting already started tours remain untouched.
        # We collect the planned tours and re-create the orders from them.
        # The scenario is that on order arrival we re-opt all outstanding tours with the new order
        # After solving the scheduling tours are added to the tour planning state/ queues
        # Idle Pickers get notified. If no picker is idle the picking cycle has started already and they will pull the tour when finishing the current tour
        self.tours_to_prune = set()
        all_tours = state.tour_manager.all_tours
        unstarted_tours = []
        for t_id in all_tours.keys():
            t = all_tours[t_id]
            if t.status not in [TourStates.STARTED, TourStates.PENDING]:
                unstarted_tours.append(t)
                self.tours_to_prune.add(t_id)

        buffered_orders = deepcopy(state.order_manager.get_order_buffer())  # To be added to existing tours / batches
        print("buffer", [o.order_id for o in buffered_orders])
        # original_orders = []
        for t in unstarted_tours:
            print("unstarted order", t)
            print(f"contains {t.order_numbers}")

            for o_id in t.order_numbers:
                original_order = state.order_manager.get_order_from_history(o_id)
                buffered_orders.append(original_order)
        for o in buffered_orders:
            for op in o.order_positions:
                if isinstance(op, ResolvedOrderPosition):
                    print("Deebug", o)

        for o in buffered_orders:
            ops = []
            for op in o.order_positions:
                if isinstance(op, ResolvedOrderPosition):
                    ops.append(op.position)
            if len(ops) > 0:
                assert len(ops) == len(o.order_positions)
                o.order_positions = ops

        print("collected orders", buffered_orders)
        o_ids = [o.order_id for o in buffered_orders]
        print(len(o_ids))
        print(len(set(o_ids)))
        # for o in buffered_orders:
        #     original_orders.append(o)
        # # We need to consider the open pick positions per tour
        #
        #
        # order_to_pp = {}
        # open_pick_positions = []
        # for tour in open_tours:
        #     for pp in tour.pick_positions:
        #         if not pp.picked:
        #             open_pick_positions.append(pp)
        #             order_number = pp.position.order_number
        #             if order_number not in order_to_pp.keys():
        #                 order_to_pp[order_number] = []
        #             if pp not in order_to_pp[order_number]:
        #                 order_to_pp[order_number].append(pp)
        #
        # for od in order_to_pp.keys():
        #     # TODO need to track pending orders to rebuild order info (due date and order date)
        #     dynamic_order = Order(order_number=od, order_positions=order_to_pp[od])
        #     buffered_orders.append(dynamic_order)

        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)
        layout = state.layout_manager.get_layout()

        resources_list = []
        # for resource in state.resource_manager.get_resources().resources:
        #     # if not resource.occupied:
        #     used_capa = 0
        #     for tour in unstarted_tours:
        #         # TODO this doesnt work anymore, track capacity in tour object
        #         if resource.id == tour.assigned_resource:  # find used capacity
        #             used_capa = len(tour.pick_positions)
        #
        #     dynamic_resource = Resource(resource.id,
        #                                 resource.capacity - used_capa,  # should approach 0
        #                                 resource.speed,
        #                                 resource.time_per_pick,
        #                                 resource.occupied,
        #                                 resource.current_location)
        #     resources_list.append(dynamic_resource)
        #
        # resources_dynamic = Resources(ResourceType.HUMAN, resources_list)
        resources_domain = state.resource_manager.get_resources()
        # resources_domain.resources = resources_dynamic
        articles = state.storage_manager.get_articles()
        storage = state.storage_manager.get_storage()

        dynamic_information = BaseWarehouseDomain(
            problem_class=problem,
            objective="Distance",
            layout=layout,
            orders=orders,
            resources=resources_domain,
            articles=articles,
            storage=storage,
            # sequencing=open_tours
        )

        # TODO fix path mess
        cache_dir = Path(f"{self.domain_cache_path}/sim")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "dynamic_info.pkl"
        cache_path_tours = cache_dir / "tours.pkl"

        self.cache_path = cache_path
        dump_pickle(str(cache_path), dynamic_information)
        print("cache", self.cache_path)
        return dynamic_information

    def on_success(self, state: SimulationState, problem):
        state.order_manager.clear_order_buffer()
        all_tours = state.tour_manager.all_tours
        for t_id in self.tours_to_prune:
            t = all_tours[t_id]

            # if t.status != TourStates.STARTED:
            assert t.status not in [TourStates.STARTED, TourStates.PENDING]
            del all_tours[t_id]

            for p_id in state.tour_manager._picker_tour_queues.keys():
                picker_queue = state.tour_manager._picker_tour_queues[p_id]
                print(picker_queue)
                print(t_id)
                if t_id in picker_queue:
                    picker_queue.remove(t_id)

