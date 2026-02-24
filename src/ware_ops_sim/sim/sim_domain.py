from dataclasses import dataclass

from ware_ops_algos.algorithms import TourPlanningState
from ware_ops_algos.domain_models import BaseWarehouseDomain, Resources, StorageLocations, LayoutData, Articles, \
    OrdersDomain, WarehouseInfo, Resource


@dataclass
class DynamicInfo(WarehouseInfo):
    time: float
    congestion_rate: dict
    active_tours: list[TourPlanningState]
    current_picker: Resource


class SimWarehouseDomain(BaseWarehouseDomain):
    def __init__(self,
                 problem_class: str,
                 objective: str,
                 layout: LayoutData,
                 articles: Articles,
                 orders: OrdersDomain,
                 resources: Resources,
                 storage: StorageLocations,
                 warehouse_info: DynamicInfo):
        super().__init__(problem_class,
                         objective,
                         layout,
                         articles,
                         orders,
                         resources,
                         storage,
                         warehouse_info)

