"""Load a CASOP benchmark instance with ware_ops_pipes' loader as a static CASIM instance."""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path

from ware_ops_algos.domain_models import (
    DimensionType, OrdersDomain, PickCart, Resources, WarehouseInfo, WarehouseInfoType,
)
from ware_ops_algos.domain_models.base_domain import BaseWarehouseDomain
from ware_ops_pipes.data_loaders import FoodmartLoader, HesslerIrnichLoader, IBRSPLoader

from casim.data_loader import DataLoader
from casim.domain_objects.sim_domain import SimWarehouseDomain, initial_dynamic_info

logger = logging.getLogger(__name__)

# Instance set -> (ware_ops_pipes loader, its keyword arguments), as in CASOP's runners.
LOADERS = {
    "SPRP": (HesslerIrnichLoader, {"mirror_top_depot": True}),
    "SPRP-SS": (HesslerIrnichLoader, {"mirror_top_depot": True}),
    "BahceciOencan": (HesslerIrnichLoader, {"mirror_top_depot": True}),
    "HennWaescherUniform": (HesslerIrnichLoader, {"mirror_top_depot": True}),
    "HennWaescherClassBased": (HesslerIrnichLoader, {"mirror_top_depot": True}),
    "MuterOencanWG": (HesslerIrnichLoader, {"mirror_top_depot": True}),
    "FoodmartData": (FoodmartLoader, {}),
    "KrisSmallDataCorrected": (IBRSPLoader, {}),
    "KrisLargeData": (IBRSPLoader, {}),
}


def load_casop_domain(instances_root: Path, instance_set: str, instance: str) -> BaseWarehouseDomain:
    """The static domain exactly as ware_ops_pipes' loader returns it."""
    loader_class, kwargs = LOADERS[instance_set]
    directory = Path(instances_root) / instance_set
    return loader_class(directory, **kwargs).load(directory / f"{instance}.txt")


def to_static_sim_domain(domain: BaseWarehouseDomain) -> SimWarehouseDomain:
    """Release every order at time 0 and complete the values the simulator needs.

    The routing and batching sets carry no times; walking speed 1, no pick time
    and no setup leave every distance unchanged. A set without a pick cart
    (SPRP, one order per instance) gets one box holding its whole order, so
    batching returns that order as a single batch.
    """
    orders = [replace(order, order_date=0.0) for order in domain.orders.orders]
    total = sum(position.amount for order in orders for position in order.order_positions)
    resources = []
    for resource in domain.resources.resources:
        cart = resource.pick_cart or PickCart(
            n_dimension=1, capacities=[total], dimensions=[DimensionType.ITEMS],
            n_boxes=1, box_can_mix_orders=True)
        resources.append(replace(
            resource,
            speed=1.0 if resource.speed is None else resource.speed,
            time_per_pick=0.0 if resource.time_per_pick is None else resource.time_per_pick,
            tour_setup_time=0.0 if resource.tour_setup_time is None else resource.tour_setup_time,
            capacity=total if resource.capacity is None else resource.capacity,
            pick_cart=cart,
            current_location=resource.current_location or domain.layout.graph_data.start_location,
        ))
    return SimWarehouseDomain(
        problem_class=domain.problem_class,
        objective=domain.objective,
        layout=domain.layout,
        articles=domain.articles,
        orders=OrdersDomain(domain.orders.tpe, orders),
        resources=Resources(domain.resources.tpe, resources),
        storage=domain.storage,
        warehouse_info=WarehouseInfo(tpe=WarehouseInfoType.OFFLINE),
        dynamic_warehouse_info=initial_dynamic_info(),
    )


class StaticBenchmarkLoader(DataLoader):
    """Scenario loader: one CASOP instance, all orders at time 0."""

    def __init__(self, instances_dir: str | Path, cfg):
        super().__init__(instances_dir)
        self.instances_root = Path(cfg.instances_root)
        self.instance_set = cfg.benchmark.instance_set
        self.instance = cfg.benchmark.instance

    def load(self, **kwargs) -> SimWarehouseDomain:
        domain = load_casop_domain(self.instances_root, self.instance_set, self.instance)
        return to_static_sim_domain(domain)
