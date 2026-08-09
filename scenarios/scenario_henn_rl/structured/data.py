from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
from ware_ops_algos.domain_models import (
    Article,
    Articles,
    ArticleType,
    DimensionType,
    Location,
    Order,
    OrderPosition,
    OrdersDomain,
    OrderType,
    PickCart,
    Resource,
    Resources,
    ResourceType,
    StorageLocations,
    StorageType,
    WarehouseInfoType,
)

from casim.domain_objects.sim_domain import DynamicInfo, SimWarehouseDomain
from scenarios.scenario_henn.loader import HennDataLoader
from scenarios.scenario_henn_rl.structured.results import write_json


ROOT = Path(__file__).parents[3].resolve()
HENN_DIR = ROOT / "scenarios" / "scenario_henn"
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}


def generated_instance_splits(spec: dict) -> dict[str, list[str]]:
    """Return deterministic generated IDs balanced over configured sizes."""
    sizes = [int(value) for value in spec["order_counts"]]
    result: dict[str, list[str]] = {}
    for split in SPLIT_CODES:
        count = int(spec[f"{split}_instances"])
        if count % len(sizes):
            raise ValueError(
                f"{split}_instances must be divisible by the number of sizes"
            )
        per_size = count // len(sizes)
        result[split] = [
            f"HL_{split}_{size:03d}_{replica:03d}"
            for size in sizes
            for replica in range(per_size)
        ]
    all_ids = [value for values in result.values() for value in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Generated instance IDs are not disjoint")
    return result


def parse_generated_instance_id(instance_id: str) -> tuple[str, int, int]:
    parts = instance_id.split("_")
    if len(parts) != 4 or parts[0] != "HL" or parts[1] not in SPLIT_CODES:
        raise ValueError(f"Invalid generated Henn instance ID: {instance_id}")
    return parts[1], int(parts[2]), int(parts[3])


def generated_manifest(spec: dict) -> dict[str, object]:
    splits = generated_instance_splits(spec)
    rows = []
    for split, instance_ids in splits.items():
        for instance_id in instance_ids:
            _, order_count, replica = parse_generated_instance_id(instance_id)
            rows.append(
                {
                    "instance_id": instance_id,
                    "split": split,
                    "order_count": order_count,
                    "replica": replica,
                    "seed_components": [
                        int(spec["seed"]),
                        SPLIT_CODES[split],
                        order_count,
                        replica,
                    ],
                }
            )
    return {
        "generator": "lorenz_henn_v1",
        "parameters": dict(spec),
        "splits": splits,
        "instances": rows,
    }


def run_generation(spec: dict, output_dir: Path) -> dict[str, object]:
    manifest = generated_manifest(spec)
    loader = GeneratedHennDataLoader(spec)
    samples = []
    for split, instance_ids in manifest["splits"].items():
        for instance_id in instance_ids:
            domain = loader.load(instance_id)
            orders = domain.orders.orders
            allowances = [
                float(order.due_date) - float(order.order_date)
                for order in orders
            ]
            samples.append(
                {
                    "instance_id": instance_id,
                    "split": split,
                    "orders": len(orders),
                    "mean_items": float(
                        np.mean([len(order.order_positions) for order in orders])
                    ),
                    "last_arrival_s": float(orders[-1].order_date),
                    "minimum_allowance_s": float(min(allowances)),
                    "maximum_allowance_s": float(max(allowances)),
                }
            )
    result = {**manifest, "summary": samples}
    write_json(output_dir / "dataset_manifest.json", result)
    return result


class GeneratedHennDataLoader:
    """Build independent Lorenz-style orders on one cached Henn layout.

    The canonical Henn loader remains unchanged. Only immutable layout data is
    reused; every call creates fresh orders, storage, resources, and articles.
    """

    def __init__(self, spec: dict):
        self.spec = dict(spec)
        base = HennDataLoader(
            instances_dir=HENN_DIR,
            aisle_end_offset=1.5,
            depot_half_span=2.5,
            travel_speed=float(self.spec["travel_speed"]),
            time_per_pick=float(self.spec["time_per_pick"]),
            tour_setup_time=float(self.spec["tour_setup_time"]),
        ).load(
            manifest_path="reproduction/manifest.yaml",
            instance_id=str(self.spec["base_instance_id"]),
        )
        self.layout = base.layout

    def load(self, instance_id: str, **kwargs) -> SimWarehouseDomain:
        del kwargs
        split, order_count, replica = parse_generated_instance_id(instance_id)
        if order_count not in [int(value) for value in self.spec["order_counts"]]:
            raise ValueError(f"Order count is not configured: {order_count}")
        seeds = np.random.SeedSequence(
            [
                int(self.spec["seed"]),
                SPLIT_CODES[split],
                order_count,
                replica,
            ]
        ).spawn(3)
        arrivals_rng = np.random.default_rng(seeds[0])
        sizes_rng = np.random.default_rng(seeds[1])
        locations_rng = np.random.default_rng(seeds[2])

        mean_gap = 4.0 * 3600.0 / float(self.spec["orders_per_four_hours"])
        arrivals = np.cumsum(arrivals_rng.exponential(mean_gap, order_count))
        sizes = sizes_rng.integers(
            int(self.spec["min_order_items"]),
            int(self.spec["max_order_items"]) + 1,
            size=order_count,
        )
        locations = [
            self._locations(locations_rng, int(size)) for size in sizes
        ]
        due_dates = [self._due_date(float(arrival)) for arrival in arrivals]
        demand = Counter(article_id for values in locations for article_id in values)

        articles = Articles(
            ArticleType.STANDARD,
            [Article(article_id=article_id, weight=1.0) for article_id in range(900)],
        )
        storage = StorageLocations(
            tpe=StorageType.DEDICATED,
            locations=[
                Location(
                    x=article_id // 45 // 2 + 1,
                    y=article_id % 45 + 1,
                    article_id=article_id,
                    amount=demand.get(article_id, 0),
                )
                for article_id in range(900)
            ],
        )
        storage.build_article_location_mapping()
        orders = OrdersDomain(
            OrderType.STANDARD,
            [
                Order(
                    order_id=order_id,
                    order_date=float(arrivals[order_id]),
                    due_date=due_dates[order_id],
                    order_positions=[
                        OrderPosition(
                            order_number=order_id,
                            article_id=article_id,
                            amount=1,
                        )
                        for article_id in locations[order_id]
                    ],
                )
                for order_id in range(order_count)
            ],
        )
        capacity = int(self.spec["cart_capacity"])
        cart = PickCart(
            n_dimension=1,
            capacities=[capacity],
            dimensions=[DimensionType.ITEMS],
            n_boxes=1,
            box_can_mix_orders=True,
        )
        resources = Resources(
            ResourceType.HUMAN,
            [
                Resource(
                    id=0,
                    capacity=capacity,
                    speed=float(self.spec["travel_speed"]),
                    time_per_pick=float(self.spec["time_per_pick"]),
                    pick_cart=cart,
                    tour_setup_time=float(self.spec["tour_setup_time"]),
                    available=True,
                    occupied=False,
                    current_location=self.layout.graph_data.start_location,
                )
            ],
        )
        domain = SimWarehouseDomain(
            problem_class="OBRP",
            objective="distance",
            layout=self.layout,
            articles=articles,
            orders=orders,
            resources=resources,
            storage=storage,
            dynamic_warehouse_info=DynamicInfo(
                tpe=WarehouseInfoType.ONLINE,
                time=0.0,
                done=False,
            ),
        )
        domain.henn_metadata = {
            "instance_id": instance_id,
            "generator": "lorenz_henn_v1",
            "capacity": capacity,
            "location_policy": self.spec["location_policy"],
        }
        return domain

    def _locations(self, rng: np.random.Generator, count: int) -> list[int]:
        if count > 900:
            raise ValueError("An order cannot contain more than 900 locations")
        if self.spec["location_policy"] == "uniform":
            return rng.choice(900, size=count, replace=False).astype(int).tolist()
        if self.spec["location_policy"] != "class_based":
            raise ValueError(
                f"Unknown location policy: {self.spec['location_policy']}"
            )
        classes = rng.choice(3, size=count, p=[0.52, 0.36, 0.12])
        ranges = [np.arange(0, 90), np.arange(90, 360), np.arange(360, 900)]
        selected: list[int] = []
        used: set[int] = set()
        for item_class in classes:
            available = np.setdiff1d(ranges[int(item_class)], list(used))
            article_id = int(rng.choice(available))
            selected.append(article_id)
            used.add(article_id)
        return selected

    def _due_date(self, arrival: float) -> float:
        cadence = float(self.spec["cutoff_cadence_hours"]) * 3600.0
        eligible = arrival + float(self.spec["minimum_lead_hours"]) * 3600.0
        return math.ceil(eligible / cadence) * cadence
