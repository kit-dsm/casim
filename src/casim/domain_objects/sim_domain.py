from dataclasses import dataclass, field

from ware_ops_algos.algorithms import BatchObject
from ware_ops_algos.domain_models import Resources, StorageLocations, LayoutData, Articles, \
    OrdersDomain, WarehouseInfo, Resource, BaseWarehouseDomain

from casim.domain_objects.tour_model import TourPlanningState


@dataclass(kw_only=True)
class DynamicInfo(WarehouseInfo):
    time: float | None = None
    replannable_tours: list[TourPlanningState] = field(default_factory=list)
    current_picker: Resource | None = None
    buffered_batches: list[BatchObject] = field(default_factory=list)
    done: bool = False
    is_break: bool = False
    n_staged_pallets: int = 0
    active_tour_id: int | None = None
    route_version: int | None = None
    intervention_resumes_execution: bool = False
    origin_type: str | None = None
    edge_origin: tuple[float, float] | None = None
    edge_destination: tuple[float, float] | None = None
    edge_progress: float | None = None
    cart_bin_order_ids: tuple[tuple[int, ...], ...] = ()
    locked_bin_ids: tuple[int, ...] = ()


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

