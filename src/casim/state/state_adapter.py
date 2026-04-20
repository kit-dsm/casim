from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig
from ware_ops_algos.algorithms import AlgorithmSolution, SchedulingSolution
from ware_ops_algos.domain_models import Resources, WarehouseInfoType, ResourceType, OrdersDomain, OrderType

from casim.domain_objects.sim_domain import SimWarehouseDomain, DynamicInfo
from casim.state import State


def build_state_adapters(cfg: DictConfig) -> dict:
    return {
        problem_key: instantiate(st_cfg,
                                 domain_cache_path=Path(cfg.cache_base))
        for problem_key, st_cfg in cfg.decision_engine.state_snapshot.items()
    }


class StateAdapter:
    def __init__(self):
        pass

    def transform_state(self, state: State, problem: str):
        pass

    def cleanup_state(self, state: State, solution: AlgorithmSolution):
        pass


class OBRSPAdapter(StateAdapter):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: State, problem: str):
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

        return dynamic_information

    def cleanup_state(self, state: State, solution: AlgorithmSolution):
        pass


class ORSPAdapter(StateAdapter):
    def __init__(self, domain_cache_path):
        super().__init__()
        self.domain_cache_path = domain_cache_path
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: State, problem: str):
        layout = state.layout_manager.get_layout()
        buffered_pls = state.order_manager.get_pick_list_buffer()

        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_pick_lists=buffered_pls,
            active_tours=None,
            done=state.done_flag
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
        state.order_manager.clear_pick_list_buffer(pls)
        print(f"Garbage Collected {len(pls)} batches")
