from dataclasses import dataclass, field

from ware_ops_algos.algorithms import BatchObject, RoutingOrigin
from ware_ops_algos.domain_models import Resources, StorageLocations, LayoutData, Articles, \
    OrdersDomain, WarehouseInfo, Resource, BaseWarehouseDomain, PlannerInformation

from casim.domain_objects.tour_model import TourPlanningState


@dataclass(kw_only=True)
class DynamicInfo(WarehouseInfo):
    time: float | None = None
    congestion_rate: dict[str, float] = field(default_factory=dict)
    active_tours: list[TourPlanningState] = field(default_factory=list)
    current_picker: Resource | None = None
    buffered_batches: list[BatchObject] = field(default_factory=list)
    done: bool = False
    wait_expired: bool = False
    is_break: bool = False
    n_staged_pallets: int = 0
    active_tour_id: int | None = None
    active_route_version: int | None = None
    routing_origin: RoutingOrigin | None = None
    active_order_ids: frozenset[int] = frozenset()
    active_candidate_ids: frozenset[int] = frozenset()
    remaining_route_positions: tuple[tuple[float, float], ...] = ()
    occupied_bins: int = 0


class SimWarehouseDomain(BaseWarehouseDomain):
    def __init__(self,
                 problem_class: str,
                 objective: str,
                 layout: LayoutData,
                 articles: Articles,
                 orders: OrdersDomain,
                 resources: Resources,
                 storage: StorageLocations,
                 dynamic_warehouse_info: DynamicInfo,
                 warehouse_info: WarehouseInfo | None = None,
                 information: PlannerInformation | None = None):
        super().__init__(problem_class,
                         objective,
                         layout,
                         articles,
                         orders,
                         resources,
                         storage,
                         warehouse_info=warehouse_info,
                         information=information)
        self.dynamic_warehouse_info = dynamic_warehouse_info

