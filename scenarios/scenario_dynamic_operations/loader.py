from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall

from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.data_loaders.generators.layout.graph_generator import (
    MultiBlockShelfStorageGraphGenerator,
)
from ware_ops_algos.domain_models import (
    Article,
    Articles,
    ArticleType,
    DimensionType,
    LayoutData,
    LayoutNetwork,
    LayoutParameters,
    LayoutType,
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

DAY = 24 * 60 * 60


def generate_orders(spec: dict, article_ids: list[int]) -> list[Order]:
    """Generate a deterministic backlog-plus-arrivals stream."""
    rng = np.random.default_rng(int(spec["seed"]))
    n_days = int(spec["n_days"])
    n_orders = int(spec["n_orders"])
    base, remainder = divmod(n_orders, n_days)
    day_counts = [base + (day < remainder) for day in range(n_days)]
    shift_start = float(spec.get("shift_start_hour", 6)) * 3600
    shift_end = float(spec.get("shift_end_hour", 18)) * 3600
    backlog_fraction = float(spec.get("backlog_fraction", 0.5))
    due_hours = [float(value) for value in spec["due_hours"]]
    lines_min, lines_max = [int(value) for value in spec["lines_per_order"]]
    urgent = spec.get("urgent_burst") or {}
    urgent_day = int(urgent.get("day", -1))
    urgent_count = int(urgent.get("count", 0))
    urgent_release = float(urgent.get("release_hour", 11)) * 3600
    urgent_due = float(urgent.get("due_hour", 12)) * 3600

    orders = []
    next_id = 1
    for day, count in enumerate(day_counts):
        backlog_count = int(round(count * backlog_fraction))
        burst_count = min(urgent_count if day == urgent_day else 0, count)
        arrival_count = count - backlog_count - burst_count
        if arrival_count > 0:
            span = max(1.0, shift_end - shift_start - 3600)
            gaps = rng.exponential(span / arrival_count, arrival_count)
            arrivals = np.minimum(
                shift_start + np.cumsum(gaps),
                shift_end - 1800,
            )
        else:
            arrivals = np.asarray([], dtype=float)
        releases = [
            (
                day * DAY + (0.0 if day == 0 else shift_start - 3600),
                False,
            )
            for _ in range(backlog_count)
        ]
        releases.extend((day * DAY + value, False) for value in arrivals)
        releases.extend(
            (day * DAY + urgent_release, True)
            for _ in range(burst_count)
        )
        for release, is_urgent in sorted(releases):
            n_lines = int(rng.integers(lines_min, lines_max + 1))
            selected_articles = rng.choice(
                article_ids,
                size=n_lines,
                replace=False,
            )
            if is_urgent:
                due_date = day * DAY + urgent_due
            else:
                eligible_due = [
                    day * DAY + hour * 3600
                    for hour in due_hours
                    if day * DAY + hour * 3600 > release
                ]
                due_date = (
                    eligible_due[int(rng.integers(len(eligible_due)))]
                    if eligible_due
                    else (day + 1) * DAY + due_hours[0] * 3600
                )
            orders.append(
                Order(
                    order_id=next_id,
                    order_date=float(release),
                    due_date=float(due_date),
                    order_positions=[
                        OrderPosition(next_id, int(article_id), 1)
                        for article_id in selected_articles
                    ],
                )
            )
            next_id += 1
    return orders


def build_layout(spec: dict) -> tuple[LayoutData, list[tuple[float, float]]]:
    layout = spec["layout"]
    start = tuple(layout.get("start_location", [0, 0]))
    end = tuple(layout.get("end_location", [-1, 0]))
    start_connection = tuple(layout.get("start_connection_point", [1, 0]))
    end_connection = tuple(layout.get("end_connection_point", [1, 0]))
    generator = MultiBlockShelfStorageGraphGenerator(
        n_aisles=int(layout["n_aisles"]),
        n_pick_locations=int(layout["n_pick_locations"]),
        n_blocks=int(layout["n_blocks"]),
        dist_aisle=float(layout["dist_aisle"]),
        dist_pick_locations=float(layout["dist_pick_locations"]),
        dist_aisle_location=float(layout["dist_aisle_location"]),
        dist_between_blocks=float(layout["dist_between_blocks"]),
        dist_start=float(layout.get("dist_start", 0)),
        dist_end=float(layout.get("dist_end", 0)),
        start_location=start,
        end_location=end,
        start_connection_point=start_connection,
        end_connection_point=end_connection,
    )
    generator.populate_graph()
    graph = generator.G
    pick_nodes = sorted(
        node
        for node, attributes in graph.nodes(data=True)
        if attributes.get("type") == "pick_node"
    )
    for node in pick_nodes:
        graph.nodes[node]["pick_capacity"] = int(
            layout.get("pick_capacity", 1)
        )
    if layout.get("narrow_connector_capacity") is not None:
        capacity = int(layout["narrow_connector_capacity"])
        for origin, destination, attributes in graph.edges(data=True):
            if graph.nodes[origin].get("type") != "pick_node" and (
                graph.nodes[destination].get("type") != "pick_node"
            ):
                attributes["zone_id"] = f"connector:{origin}:{destination}"
                attributes["capacity"] = capacity
    nodes = list(graph.nodes)
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=nodes,
        weight="weight",
        dtype=float,
    )
    distances, predecessors = floyd_warshall(
        adjacency,
        directed=graph.is_directed(),
        return_predecessors=True,
    )
    params = LayoutParameters(
        n_aisles=int(layout["n_aisles"]),
        n_pick_locations=int(layout["n_pick_locations"]),
        n_blocks=int(layout["n_blocks"]),
        dist_top_to_pick_location=float(layout["dist_aisle_location"]),
        dist_bottom_to_pick_location=float(layout["dist_aisle_location"]),
        dist_pick_locations=float(layout["dist_pick_locations"]),
        dist_aisle=float(layout["dist_aisle"]),
        dist_start=float(layout.get("dist_start", 0)),
        dist_end=float(layout.get("dist_end", 0)),
        dist_cross_aisle=float(layout["dist_between_blocks"]),
        start_location=start,
        end_location=end,
        start_connection_point=start_connection,
        end_connection_point=end_connection,
    )
    return LayoutData(
        tpe=LayoutType.CONVENTIONAL,
        graph_data=params,
        layout_network=LayoutNetwork(
            graph=graph,
            distance_matrix=pd.DataFrame(distances, index=nodes, columns=nodes),
            predecessor_matrix=np.asarray(predecessors),
            closest_node_to_start=start_connection,
            start_node=start,
            end_node=end,
            node_list=nodes,
            min_aisle_position=min(node[1] for node in pick_nodes),
            max_aisle_position=max(node[1] for node in pick_nodes),
        ),
    ), pick_nodes


class DynamicOperationsLoader(DataLoader):
    def __init__(self, instances_dir: str | Path):
        super().__init__(Path(instances_dir).resolve())

    def load(self, instance_path: str | Path, **_kwargs) -> SimWarehouseDomain:
        path = Path(instance_path)
        if not path.is_absolute():
            path = self.data_dir / path
        spec = json.loads(path.read_text(encoding="utf-8"))
        layout, pick_nodes = build_layout(spec)
        article_ids = list(range(1, len(pick_nodes) + 1))
        orders = generate_orders(spec, article_ids)
        demand = Counter(
            position.article_id
            for order in orders
            for position in order.order_positions
        )
        articles = Articles(
            ArticleType.STANDARD,
            [Article(article_id=value) for value in article_ids],
        )
        storage = StorageLocations(
            StorageType.DEDICATED,
            [
                Location(
                    x=node[0],
                    y=node[1],
                    article_id=article_id,
                    amount=float(demand[article_id] + 10),
                )
                for article_id, node in zip(article_ids, pick_nodes)
            ],
        )
        storage.build_article_location_mapping()
        n_boxes = int(spec["cart_bins"])
        resources = Resources(
            ResourceType.HUMAN,
            [
                Resource(
                    id=picker_id,
                    capacity=n_boxes,
                    speed=float(spec["picker_speed"]),
                    time_per_pick=float(spec["time_per_pick"]),
                    pick_cart=PickCart(
                        n_dimension=1,
                        capacities=[1.0],
                        dimensions=[DimensionType.ORDERS],
                        n_boxes=n_boxes,
                        box_can_mix_orders=False,
                    ),
                    tour_setup_time=float(spec.get("tour_setup_time", 0)),
                    available=False,
                    occupied=False,
                    current_location=layout.layout_network.start_node,
                )
                for picker_id in range(max(int(v) for v in spec["picker_counts"]))
            ],
        )
        domain = SimWarehouseDomain(
            problem_class="ORSP",
            objective="tardiness",
            layout=layout,
            articles=articles,
            orders=OrdersDomain(OrderType.STANDARD, orders),
            resources=resources,
            storage=storage,
            dynamic_warehouse_info=DynamicInfo(
                tpe=WarehouseInfoType.ONLINE,
                time=0.0,
                done=False,
            ),
        )
        domain.dynamic_operations_spec = spec
        return domain
