"""Configured Walking vs. Waiting orders and benchmark layouts."""

import json
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall
from casim.data_loader import DataLoader
from casim.domain_objects.sim_domain import SimWarehouseDomain, initial_dynamic_info
from ware_ops_algos.domain_models import (
    Article, Articles, ArticleType, DimensionType, Location, Order, OrderPosition,
    OrdersDomain, OrderType, PickCart, Resource, Resources, ResourceType,
    StorageLocations, StorageType, WarehouseInfo, WarehouseInfoType,
    LayoutData, LayoutNetwork, LayoutParameters, LayoutType,
)

from casim.sources.layout import conventional_layout


class WalkOrWaitDataLoader(DataLoader):
    """Load explicit examples or the paper's published instance files.

    With ``orders`` the instance is written out: each order has an arrival, a
    due date and its picks as (aisle, location) on a generated layout of
    ``n_aisles`` aisles with ``n_pick_locations`` locations. With ``instance``
    it is a RADAR instance file, relative to the data directory.
    """

    def __init__(self, instances_dir: str | Path, *, capacity_orders: int,
                 walking_speed: float, pick_time_s: float,
                 orders: list[dict] | None = None, instance: str | None = None,
                 n_aisles: int | None = None, n_pick_locations: int | None = None,
                 depot_aisle: int | None = None):
        super().__init__(Path(instances_dir).resolve())
        if (orders is None) == (instance is None):
            raise ValueError("give either explicit orders or a RADAR instance")
        self.capacity_orders = int(capacity_orders)
        self.walking_speed = float(walking_speed)
        self.pick_time_s = float(pick_time_s)
        self.orders = orders
        self.instance = instance
        self.n_aisles = n_aisles
        self.n_pick_locations = n_pick_locations
        self.depot_aisle = depot_aisle

    def load(self, **kwargs) -> SimWarehouseDomain:
        capacity = self.capacity_orders
        if self.orders is not None:
            n_aisles = int(self.n_aisles)
            n_locations = int(self.n_pick_locations)
            depot_aisle = int(self.depot_aisle)
            layout = conventional_layout(
                n_aisles=n_aisles,
                n_pick_locations=n_locations,
                start_connection_point=(depot_aisle, 0),
                end_connection_point=(depot_aisle, 0),
            )
            specs = list(self.orders)
            article_by_position = {}
            order_rows = []
            for order_id, spec in enumerate(specs):
                items = {}
                for raw_position in spec["picks"]:
                    aisle, location = map(int, raw_position)
                    article_id = article_by_position.setdefault(
                        (aisle, location), len(article_by_position)
                    )
                    items[article_id] = items.get(article_id, 0) + 1
                order_rows.append((order_id, float(spec["arrival_s"]), float(spec["due_s"]), items))
            article_locations = {article_id: point for point, article_id in article_by_position.items()}
        else:
            instance_path = (self.data_dir / self.instance).resolve()
            instance = json.loads(instance_path.read_text(encoding="utf-8"))
            meta = instance["meta"]
            layout_path = (instance_path.parent / meta["layout"]).resolve()
            storage_path = (instance_path.parent / meta["storage_assignment"]).resolve()
            layout = self._benchmark_layout(layout_path)
            article_locations = {
                int(article_id): tuple(point)
                for article_id, point in json.loads(storage_path.read_text(encoding="utf-8")).items()
            }
            order_rows = [
                (int(row["order_id"]), float(row["arrival_time"]),
                 float(row["due_date"]),
                 {int(article_id): int(amount) for article_id, amount in row["items"].items()})
                for row in instance["orders"]
            ]

        demand_by_article = {}
        orders = []
        for order_id, arrival, due, items in order_rows:
            positions = []
            for article_id, amount in items.items():
                demand_by_article[article_id] = demand_by_article.get(article_id, 0) + amount
                positions.append(OrderPosition(
                    order_number=order_id, article_id=article_id, amount=amount,
                ))
            orders.append(Order(
                order_id=order_id,
                order_date=arrival,
                due_date=due,
                order_positions=positions,
            ))

        articles = Articles(
            ArticleType.STANDARD,
            [Article(article_id=article_id, weight=1.0)
             for article_id in demand_by_article],
        )
        storage = StorageLocations(
            tpe=StorageType.DEDICATED,
            locations=[Location(x=article_locations[article_id][0],
                                y=article_locations[article_id][1], article_id=article_id,
                                amount=demand_by_article[article_id])
                       for article_id in demand_by_article],
        )
        storage.build_article_location_mapping()
        cart = PickCart(
            n_dimension=1,
            capacities=[1],
            dimensions=[DimensionType.ORDERS],
            n_boxes=capacity,
            box_can_mix_orders=False,
        )
        picker = Resource(
            id=0, capacity=capacity, speed=self.walking_speed,
            time_per_pick=self.pick_time_s, tour_setup_time=0.0,
            pick_cart=cart, available=True, occupied=False,
            current_location=layout.graph_data.start_location,
        )
        return SimWarehouseDomain(
            problem_class="OBRSP",
            objective="mean_order_completion_time",
            layout=layout,
            articles=articles,
            orders=OrdersDomain(OrderType.STANDARD, orders),
            resources=Resources(ResourceType.HUMAN, [picker]),
            storage=storage,
            dynamic_warehouse_info=initial_dynamic_info(),
            warehouse_info=WarehouseInfo(tpe=WarehouseInfoType.ONLINE),
        )

    @staticmethod
    def _benchmark_layout(path: Path) -> LayoutData:
        with path.open("rb") as stream:
            graph = pickle.load(stream)
        pick_nodes = [node for node, data in graph.nodes(data=True)
                      if data.get("type") == "pick_node"]
        starts = [node for node, data in graph.nodes(data=True)
                  if data.get("type") == "start_node"]
        ends = [node for node, data in graph.nodes(data=True)
                if data.get("type") == "end_node"]
        start, end = starts[0], ends[0]
        start_connection = next(iter(graph.neighbors(start)))
        end_connection = next(iter(graph.neighbors(end)))
        nodes = list(graph.nodes)
        adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", dtype=float)
        distances, predecessors = floyd_warshall(
            adjacency, directed=False, return_predecessors=True
        )
        params = LayoutParameters(
            n_aisles=8, n_pick_locations=16, n_blocks=1,
            dist_top_to_pick_location=graph[(1, 16)][(1, 17)]["weight"],
            dist_bottom_to_pick_location=graph[(1, 0)][(1, 1)]["weight"],
            dist_pick_locations=graph[(1, 1)][(1, 2)]["weight"],
            dist_aisle=graph[(1, 0)][(2, 0)]["weight"],
            dist_start=graph[start][start_connection]["weight"],
            dist_end=graph[end][end_connection]["weight"],
            start_location=start, end_location=end,
            start_connection_point=start_connection,
            end_connection_point=end_connection,
            depot_location="front_left",
        )
        network = LayoutNetwork(
            graph=graph, distance_matrix=pd.DataFrame(distances, index=nodes, columns=nodes),
            predecessor_matrix=predecessors, closest_node_to_start=start_connection,
            min_aisle_position=0, max_aisle_position=17,
            start_node=start, end_node=end, node_list=nodes,
        )
        return LayoutData(tpe=LayoutType.CONVENTIONAL, graph_data=params,
                          layout_network=network)
