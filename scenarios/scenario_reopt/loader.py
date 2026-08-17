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
    distance_matrix_generator,
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

from casim.domain_objects import DynamicInfo, SimWarehouseDomain


def _layout(nodes: list[tuple[int, int]]) -> LayoutData:
    start = (0, 0)
    end = (-1, 0)
    graph = nx.Graph()
    graph.add_nodes_from([start, end, *nodes])
    graph.add_edge(start, end, weight=0.0)
    travel_nodes = [start, *nodes]
    for index, source in enumerate(travel_nodes):
        for target in travel_nodes[index + 1 :]:
            graph.add_edge(
                source,
                target,
                weight=abs(source[0] - target[0])
                + abs(source[1] - target[1]),
            )

    node_list = list(graph.nodes)
    node_to_index = {node: index for index, node in enumerate(node_list)}
    predecessors = np.full((len(node_list), len(node_list)), -9999, dtype=int)
    for source in node_list:
        source_index = node_to_index[source]
        paths = nx.single_source_dijkstra_path(graph, source, weight="weight")
        for target, path in paths.items():
            if target != source:
                predecessors[
                    source_index,
                    node_to_index[target],
                ] = node_to_index[path[-2]]

    return LayoutData(
        tpe=LayoutType.CONVENTIONAL,
        graph_data=LayoutParameters(
            n_aisles=2,
            n_pick_locations=len(nodes),
            n_blocks=2,
            dist_top_to_pick_location=1.0,
            dist_bottom_to_pick_location=1.0,
            dist_pick_locations=1.0,
            dist_aisle=1.0,
            dist_start=0.0,
            dist_end=0.0,
            start_location=start,
            end_location=end,
        ),
        layout_network=LayoutNetwork(
            graph=graph,
            distance_matrix=distance_matrix_generator(graph),
            predecessor_matrix=predecessors,
            closest_node_to_start=start,
            start_node=start,
            end_node=end,
            node_list=node_list,
            min_aisle_position=0,
            max_aisle_position=len(nodes) + 1,
        ),
    )


def _multiblock_layout(raw: dict[str, Any]) -> LayoutData:
    spec = raw["layout"]
    start = tuple(spec.get("start_location", [0, 0]))
    end = tuple(spec.get("end_location", [-1, 0]))
    start_connection = tuple(spec.get("start_connection_point", [1, 0]))
    end_connection = tuple(spec.get("end_connection_point", [1, 0]))
    generator = MultiBlockShelfStorageGraphGenerator(
        n_aisles=int(spec["n_aisles"]),
        n_pick_locations=int(spec["n_pick_locations"]),
        n_blocks=int(spec["n_blocks"]),
        dist_aisle=float(spec["dist_aisle"]),
        dist_pick_locations=float(spec["dist_pick_locations"]),
        dist_aisle_location=float(spec["dist_aisle_location"]),
        dist_between_blocks=float(spec["dist_between_blocks"]),
        dist_start=float(spec.get("dist_start", 0.0)),
        dist_end=float(spec.get("dist_end", 0.0)),
        start_location=start,
        end_location=end,
        start_connection_point=start_connection,
        end_connection_point=end_connection,
    )
    generator.populate_graph()
    graph = generator.G
    nodes = list(graph.nodes)
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=nodes,
        weight="weight",
        dtype=float,
    )
    distances, predecessors = floyd_warshall(
        adjacency,
        directed=False,
        return_predecessors=True,
    )
    params = LayoutParameters(
        n_aisles=int(spec["n_aisles"]),
        n_pick_locations=int(spec["n_pick_locations"]),
        n_blocks=int(spec["n_blocks"]),
        dist_top_to_pick_location=float(spec["dist_aisle_location"]),
        dist_bottom_to_pick_location=float(spec["dist_aisle_location"]),
        dist_pick_locations=float(spec["dist_pick_locations"]),
        dist_aisle=float(spec["dist_aisle"]),
        dist_start=float(spec.get("dist_start", 0.0)),
        dist_end=float(spec.get("dist_end", 0.0)),
        dist_cross_aisle=float(spec["dist_between_blocks"]),
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
            distance_matrix=pd.DataFrame(
                distances,
                index=nodes,
                columns=nodes,
            ),
            predecessor_matrix=np.asarray(predecessors),
            closest_node_to_start=start_connection,
            start_node=start,
            end_node=end,
            node_list=nodes,
            min_aisle_position=0,
            max_aisle_position=int(spec["n_pick_locations"]) + 1,
        ),
    )


class ReoptDataLoader(DataLoader):
    """Load the checked-in three-order Lorenz example into a CASIM domain."""

    def __init__(
        self,
        instances_dir: str | Path,
    ):
        super().__init__(Path(instances_dir).resolve())

    def load(self, instance_path: str | Path, **_kwargs) -> SimWarehouseDomain:
        path = Path(instance_path)
        if not path.is_absolute():
            path = self.data_dir / path
        if not path.is_file():
            raise ValueError(f"Reoptimization instance does not exist: {path}")

        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        self._validate(raw, path)

        releases = {
            int(order_id): float(release)
            for order_id, release in raw["release_times"].items()
        }
        structured = "storage" in raw
        raw_orders = []
        explicit_storage = []
        if structured:
            for entry in raw["storage"]:
                node = tuple(int(value) for value in entry["location"])
                explicit_storage.append(
                    (
                        int(entry["article_id"]),
                        node,
                        float(entry["quantity"]),
                    )
                )
            for order_id_text, positions in sorted(
                raw["orders"].items(),
                key=lambda value: int(value[0]),
            ):
                order_id = int(order_id_text)
                raw_orders.append(
                    (
                        order_id,
                        releases[order_id],
                        [
                            (
                                int(position["article_id"]),
                                int(position["amount"]),
                            )
                            for position in positions
                        ],
                    )
                )
            all_nodes = [node for _, node, _ in explicit_storage]
        else:
            article_id = 1
            for order_id_text, locations in sorted(
                raw["orders"].items(),
                key=lambda value: int(value[0]),
            ):
                order_id = int(order_id_text)
                items = []
                for location in locations:
                    items.append(
                        (
                            article_id,
                            (int(location[0]), int(location[1])),
                        )
                    )
                    article_id += 1
                raw_orders.append((order_id, releases[order_id], items))
            all_nodes = [
                node
                for _, _, items in raw_orders
                for _, node in items
            ]
        layout = (
            _multiblock_layout(raw)
            if "layout" in raw
            else _layout(all_nodes)
        )
        invalid_nodes = sorted(
            set(all_nodes) - set(layout.layout_network.graph.nodes)
        )
        if invalid_nodes:
            raise ValueError(
                f"{path}: pick locations are outside the layout: {invalid_nodes}"
            )
        if structured:
            article_ids = {
                article_id
                for article_id, _, _ in explicit_storage
            }
            demand_ids = {
                article_id
                for _, _, positions in raw_orders
                for article_id, _ in positions
            }
            if not demand_ids.issubset(article_ids):
                raise ValueError(
                    f"{path}: demand contains articles absent from storage: "
                    f"{sorted(demand_ids - article_ids)}"
                )
        else:
            demand = Counter(
                article
                for _, _, items in raw_orders
                for article, _ in items
            )
            locations_by_article = {
                article: node
                for _, _, items in raw_orders
                for article, node in items
            }
            article_ids = set(locations_by_article)
        articles = Articles(
            ArticleType.STANDARD,
            [
                Article(article_id=value)
                for value in sorted(article_ids)
            ],
        )
        storage = StorageLocations(
            StorageType.DEDICATED,
            (
                [
                    Location(
                        x=node[0],
                        y=node[1],
                        article_id=article_id,
                        amount=quantity,
                    )
                    for article_id, node, quantity in explicit_storage
                ]
                if structured
                else [
                    Location(
                        x=node[0],
                        y=node[1],
                        article_id=value,
                        amount=demand[value],
                    )
                    for value, node in sorted(
                        locations_by_article.items()
                    )
                ]
            ),
        )
        storage.build_article_location_mapping()
        orders = OrdersDomain(
            OrderType.STANDARD,
            [
                Order(
                    order_id=order_id,
                    order_date=release,
                    order_positions=[
                        OrderPosition(
                            order_id,
                            article,
                            amount if structured else 1,
                        )
                        for article, amount in items
                    ],
                )
                for order_id, release, items in raw_orders
            ],
        )

        cart_spec = raw.get("pick_cart", {})
        capacity = int(
            cart_spec.get(
                "n_boxes",
                raw["batch_capacity_orders"],
            )
        )
        cart = PickCart(
            n_dimension=1,
            capacities=[
                float(
                    cart_spec.get(
                        "capacity_per_bin",
                        1 if structured else capacity,
                    )
                )
            ],
            dimensions=[DimensionType.ORDERS],
            n_boxes=capacity,
            box_can_mix_orders=bool(
                cart_spec.get("box_can_mix_orders", False)
            ),
        )
        resources = Resources(
            ResourceType.HUMAN,
            [
                Resource(
                    id=picker_id,
                    capacity=capacity,
                    speed=float(raw["picker_speed"]),
                    time_per_pick=float(raw["time_per_pick"]),
                    pick_cart=PickCart(
                        n_dimension=cart.n_dimension,
                        capacities=list(cart.capacities),
                        dimensions=list(cart.dimensions),
                        n_boxes=cart.n_boxes,
                        box_can_mix_orders=cart.box_can_mix_orders,
                    ),
                    tour_setup_time=0.0,
                    available=True,
                    occupied=False,
                    current_location=layout.layout_network.start_node,
                )
                for picker_id in range(int(raw.get("n_pickers", 1)))
            ],
        )
        domain = SimWarehouseDomain(
            problem_class="OBRSP",
            objective="makespan",
            layout=layout,
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
        domain.reopt_metadata = {
            "instance_id": path.stem,
            "source_path": str(path.resolve()),
        }
        return domain

    @staticmethod
    def _validate(raw: dict[str, Any], path: Path) -> None:
        required = {
            "release_times",
            "orders",
            "batch_capacity_orders",
            "picker_speed",
            "time_per_pick",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        if set(raw["orders"]) != set(raw["release_times"]):
            raise ValueError(
                f"{path}: orders and release_times must contain identical ids"
            )
        if int(raw["batch_capacity_orders"]) < 1:
            raise ValueError(f"{path}: batch_capacity_orders must be positive")
        if float(raw["picker_speed"]) <= 0:
            raise ValueError(f"{path}: picker_speed must be positive")
        if float(raw["time_per_pick"]) < 0:
            raise ValueError(f"{path}: time_per_pick must be non-negative")
        if int(raw.get("n_pickers", 1)) < 1:
            raise ValueError(f"{path}: n_pickers must be positive")
        structured = "storage" in raw
        if structured:
            for entry in raw["storage"]:
                if (
                    "article_id" not in entry
                    or "quantity" not in entry
                    or not isinstance(entry.get("location"), list)
                    or len(entry["location"]) != 2
                    or float(entry["quantity"]) < 0
                ):
                    raise ValueError(f"{path}: invalid storage entry {entry}")
        for order_id, positions in raw["orders"].items():
            if not positions:
                raise ValueError(f"{path}: order {order_id} has no items")
            invalid = (
                any(
                    not isinstance(position, dict)
                    or "article_id" not in position
                    or int(position.get("amount", 0)) <= 0
                    for position in positions
                )
                if structured
                else any(
                    not isinstance(location, list) or len(location) != 2
                    for location in positions
                )
            )
            if invalid:
                raise ValueError(
                    f"{path}: order {order_id} has an invalid position"
                )
