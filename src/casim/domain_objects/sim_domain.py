from dataclasses import dataclass

from ware_ops_algos.algorithms import TourPlanningState, PickList
from ware_ops_algos.domain_models import Resources, StorageLocations, LayoutData, Articles, \
    OrdersDomain, WarehouseInfo, Resource, BaseWarehouseDomain


@dataclass
class DynamicInfo(WarehouseInfo):
    time: float | None
    congestion_rate: dict | None
    active_tours: list[TourPlanningState] | None
    current_picker: Resource | None
    buffered_pick_lists: list[PickList] | None
    done: bool


class SimWarehouseDomain(BaseWarehouseDomain):
    def __init__(self,
                 problem_class: str,
                 objective: str,
                 layout: LayoutData,
                 articles: Articles,
                 orders: OrdersDomain,
                 resources: Resources,
                 storage: StorageLocations,
                 dynamic_warehouse_info: DynamicInfo):
        super().__init__(problem_class,
                         objective,
                         layout,
                         articles,
                         orders,
                         resources,
                         storage)
        self.dynamic_warehouse_info = dynamic_warehouse_info

