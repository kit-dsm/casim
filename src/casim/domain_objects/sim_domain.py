from dataclasses import dataclass, field

from ware_ops_algos.algorithms import BatchObject
from ware_ops_algos.domain_models import Resources, StorageLocations, LayoutData, Articles, \
    OrdersDomain, WarehouseInfo, Resource, BaseWarehouseDomain

from casim.domain_objects.tour_model import TourPlanningState


@dataclass(kw_only=True)
class DynamicInfo(WarehouseInfo):
    time: float | None = None
    congestion_rate: dict[str, float] = field(default_factory=dict)
    active_tours: list[TourPlanningState] = field(default_factory=list)
    current_picker: Resource | None = None
    buffered_batches: list[BatchObject] = field(default_factory=list)
    done: bool = False
    is_break: bool = False
    n_staged_pallets: int = 0


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

