from copy import deepcopy
from pathlib import Path
from typing import Any, Type

from ware_ops_algos.algorithms import SchedulingSolution, SequencingSolution, CombinedRoutingSolution, WarehouseOrder, \
    AlgorithmSolution, OrderSelectionSolution, BatchingSolution, PickListSelectionSolution
from ware_ops_algos.domain_models import OrdersDomain, OrderType, BaseWarehouseDomain, Resources, ResourceType, \
    ResolvedOrderPosition, WarehouseInfoType, WarehouseInfo

from ware_ops_sim.sim.sim_domain import SimWarehouseDomain, DynamicInfo
from ware_ops_sim.sim.state import SimulationState
from ware_ops_sim.sim.state.tour_manager import TourStates
from ware_ops_algos.utils.io_helpers import dump_pickle


class StateSnapshot:
    def __init__(self):
        pass

    def transform_state(self, state: SimulationState, problem: str):
        pass

    def on_success(self, state: SimulationState,
                   solution: Type[AlgorithmSolution]):
        pass


class OBSRPStateView(StateSnapshot):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: SimulationState, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)
        self.orders = buffered_orders
        layout = state.layout_manager.get_layout()

        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_pick_lists=None,
            active_tours=None,
            done=state.done_flag
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
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            warehouse_info=warehouse_info
        )

        return dynamic_information


class OBSRPStateSnapshot(StateSnapshot):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: SimulationState, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()
        # print("check this", len(buffered_orders))
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)
        self.orders = buffered_orders
        layout = state.layout_manager.get_layout()

        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.OFFLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_pick_lists=None,
            active_tours=None,
            done=state.done_flag
        )
        resources = state.resource_manager.get_resources()
        dynamic_resources_list = []
        for r in resources.resources:
            if not r.occupied:
                dynamic_resources_list.append(r)
                break # TODO This is a greedy assignment baked into batching at this point
        # try:
        #     assert len(dynamic_resources_list) == 1
        # except:
        #     pass
        if len(dynamic_resources_list) > 0:
            self.selected_picker = dynamic_resources_list[0].id
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

        return dynamic_information

    # def on_success(self, state: SimulationState, solution: SchedulingSolution):
    #     orders = []
    #     for j in solution.jobs:
    #         for o in j.route.pick_list.orders:
    #             orders.append(o)
    #     state.order_manager.clear_order_buffer(orders)
    #
    #     self.orders = None
    def on_success(self, state: SimulationState,
                   solution: CombinedRoutingSolution):
        # TODO -> this is the step() function? This should contain all domain specific logic on how
        # to handle solutions, including which parts of the solution to commit and what events to return
        # Then we could e.g. only commit the first sequencing solution and return everything else to planning
        # orders remain in order buffer, pickers remain free. -> non-interv. reopt?
        orders = []

        if isinstance(solution, SchedulingSolution):
            # for j in solution.jobs:
            #     for o in j.route.pick_list.orders:
            #         orders.append(o)
            orders = solution.jobs[0].route.pick_list.orders

        elif isinstance(solution, CombinedRoutingSolution):
            for s in solution.routes:
                for o in s.pick_list.orders:
                    orders.append(o)
        state.order_manager.clear_order_buffer(orders)
        # state.current_picker_id = self.selected_picker


class BatchStateSnapshot(StateSnapshot):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.orders = None

    def transform_state(self, state: SimulationState, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)
        self.orders = buffered_orders
        layout = state.layout_manager.get_layout()


        dynamic_info = DynamicInfo(
            WarehouseInfoType.OFFLINE,
            time=state.current_time,
            congestion_rate=state.resource_manager.get_aisle_count(),
            active_tours=None,
            current_picker=None,
            buffered_pick_lists=None,
            done=state.done_flag
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

        return dynamic_information

    def on_success(self, state: SimulationState, solution: Type[AlgorithmSolution]):
        orders: list[WarehouseOrder] = []
        print("Got this solution", type(solution))
        if isinstance(solution, SchedulingSolution):
            orders = solution.jobs[0].route.pick_list.orders
        elif isinstance(solution, CombinedRoutingSolution):
            for s in solution.routes:
                for o in s.pick_list.orders:
                    orders.append(o)
        elif isinstance(solution, BatchingSolution):
            for pl in solution.pick_lists:
                for o in pl.orders:
                    orders.append(o)
        else:
            raise print(type(solution))
        print("Before clearing", len(state.order_manager.get_order_buffer()))
        state.order_manager.clear_order_buffer(orders)
        print("After clearing", len(state.order_manager.get_order_buffer()))


class ReBatchingView(StateSnapshot):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path

    def transform_state(self, state: SimulationState, problem: str):

        buffered_orders = state.order_manager.get_order_buffer()
        buffered_pls = state.order_manager.get_pick_list_buffer()
        for pl in buffered_pls:
            for o_id in pl.order_numbers:
                original_order = state.order_manager.get_order_from_history(o_id)
                buffered_orders.append(original_order)

        all_tours = state.tour_manager.all_tours
        unstarted_tours = []
        for t_id in all_tours.keys():
            t = all_tours[t_id]
            if t.status == TourStates.STARTED:
                unstarted_tours.append(t)

        layout = state.layout_manager.get_layout()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)

        dynamic_info = DynamicInfo(
            WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            active_tours=None,
            current_picker=None,
            buffered_pick_lists=None,
            done=state.done_flag
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

        return dynamic_information

    def on_success(self, state: SimulationState,
                   solution: CombinedRoutingSolution):
        orders = []

        if isinstance(solution, SchedulingSolution):
            # for j in solution.jobs:
            #     for o in j.route.pick_list.orders:
            #         orders.append(o)
            orders = solution.jobs[0].route.pick_list.orders

        elif isinstance(solution, CombinedRoutingSolution):
            for s in solution.routes:
                for o in s.pick_list.orders:
                    orders.append(o)
        state.order_manager.clear_order_buffer(orders)


class OnlineStateSnapshot(StateSnapshot):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.orders = None
        self.pick_lists = None

    def transform_state(self, state: SimulationState, problem: str):
        buffered_pls = state.order_manager.get_pick_list_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=[])
        pick_lists = []
        if len(buffered_pls) > 0:
            pick_lists = [buffered_pls[0]]

        self.pick_lists = pick_lists
        layout = state.layout_manager.get_layout()
        current_picker = state.resource_manager.get_resource(state.current_picker_id)
        tours = state.tour_manager.all_tours
        tours_not_done = []
        for i, t in tours.items():
            if t.status != TourStates.DONE:
                tours_not_done.append(t)

        dynamic_info = DynamicInfo(
            WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=state.resource_manager.get_aisle_count(),
            active_tours=tours_not_done,
            current_picker=current_picker,
            buffered_pick_lists=pick_lists,
            done=state.done_flag
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

        return dynamic_information

    def on_success(self, state: SimulationState, solution: Type[AlgorithmSolution]):
        state.order_manager.clear_pick_list_buffer(self.pick_lists)


class BSRPStateSnapshot(StateSnapshot):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.orders = None
        self.pick_lists = None

    def transform_state(self, state: SimulationState, problem: str):
        buffered_pls = state.order_manager.get_pick_list_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=[])

        layout = state.layout_manager.get_layout()
        current_picker = state.resource_manager.get_resource(state.current_picker_id)
        tours = state.tour_manager.all_tours
        tours_not_done = []
        for i, t in tours.items():
            if t.status != TourStates.DONE:
                tours_not_done.append(t)

        dynamic_info = DynamicInfo(
            WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=state.resource_manager.get_aisle_count(),
            active_tours=tours_not_done,
            current_picker=current_picker,
            buffered_pick_lists=buffered_pls,
            done=state.done_flag
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

        return dynamic_information

    def on_success(self, state: SimulationState, solution: CombinedRoutingSolution):
        pls_to_clear = []
        for r in solution.routes:
            pls_to_clear.append(r.pick_list)
        state.order_manager.clear_pick_list_buffer(pls_to_clear)


class RLStateSnapshot(StateSnapshot):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.orders = None

    def transform_state(self, state: SimulationState, problem: str):
        buffered_orders = state.order_manager.get_order_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)
        self.orders = buffered_orders
        layout = state.layout_manager.get_layout()

        current_picker = state.resource_manager.get_resource(state.current_picker_id)
        tours = state.tour_manager.all_tours
        tours_not_done = []
        for i, t in tours.items():
            if t.status == TourStates.STARTED:
                tours_not_done.append(t)

        pick_lists = []
        for t in tours.keys():
            pick_lists.append(tours[t].pick_list)

        dynamic_info = DynamicInfo(
            WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=state.resource_manager.get_aisle_count(),
            active_tours=tours_not_done,
            current_picker=current_picker,
            buffered_pick_lists=state.order_manager.get_pick_list_buffer(),
            done=state.done_flag
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
        return dynamic_information

    def on_success(self, state: SimulationState, solution: PickListSelectionSolution):
        state.current_picker_id = None
        pls = solution.selected_pick_lists
        assert len(pls) == 1
        state.order_manager.clear_pick_list_buffer(pls)


class SPRPStateSnapshot(StateSnapshot):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.orders = None

    def transform_state(self, state: SimulationState, problem: str):

        layout = state.layout_manager.get_layout()

        resources_list = [state.resource_manager.get_resource(state.current_picker_id)]
        current_picker = state.resource_manager.get_resource(state.current_picker_id)
        # selected_order = state.tour_manager.get_selected_order(
        #     picker_id=current_picker.id)
        selected_pl = state.order_manager.get_selected_pick_list(
            picker_id=current_picker.id)
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=[])
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
            current_picker=current_picker,
            buffered_pick_lists=[selected_pl]
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


class ReoptStateSnapshot(StateSnapshot):
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
        # print("buffer", [o.order_id for o in buffered_orders])
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
            # sequencing=open_tours,
            # warehouse_info=
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

