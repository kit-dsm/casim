from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall
from ware_ops_algos.algorithms import SShapeRouting, GreedyItemAssignment, OrderNrFifoBatching, ExactTSPRoutingDistance

from ware_ops_algos.data_loaders.base_data_loader import DataLoader
from ware_ops_algos.domain_models import (
    Article, Order, OrderPosition, LayoutData,
    Resource, StorageLocations, Location, ResourceType, LayoutType, LayoutParameters,
    ArticleType, StorageType, OrderType, OrdersDomain, Articles, Resources,
    BaseWarehouseDomain, LayoutNetwork, PickCart, DimensionType, WarehouseInfoType, WarehouseInfo, ManualPicker
)
from ware_ops_algos.generators import MultiBlockShelfStorageGraphGenerator, ShelfStorageGraphGenerator
from ware_ops_algos.utils.visualization import plot_route


def build_oibrsp_graph():
    """
    Single-block, 6-aisle warehouse graph.

    Nodes:
        (0, 0)      — depot
        (a, 0)      — bottom cross-aisle node for aisle a (a=1..6)
        (a, c)      — pick node, aisle a, cell c (a=1..6, c=1..30)
        (a, 31)     — top cross-aisle node for aisle a (a=1..6)

    Edges:
        depot → (1,0): dist_start = 14.4m
        (a,0) ↔ (a+1,0): dist_aisle = 4.8m  (bottom cross-aisle)
        (a,31) ↔ (a+1,31): dist_aisle = 4.8m (top cross-aisle)
        (a,0) ↔ (a,1): dist_aisle_location = 1.3m
        (a,c) ↔ (a,c+1): dist_pick_locations = 1.3m  (c=1..29)
        (a,30) ↔ (a,31): dist_aisle_location = 1.3m
    """
    G = nx.Graph()

    dist_start = 14.4
    dist_aisle = 4.8
    dist_aisle_location = 1.3
    dist_pick = 1.3
    n_aisles = 6
    n_cells = 30

    depot = (0, 0)
    G.add_node(depot, type='depot')

    # bottom and top cross-aisle nodes + pick nodes
    for a in range(1, n_aisles + 1):
        G.add_node((a, 0), type='cross_aisle')
        G.add_node((a, n_cells + 1), type='cross_aisle')
        for c in range(1, n_cells + 1):
            G.add_node((a, c), type='pick_node')

    # depot → aisle 1 bottom
    G.add_edge(depot, (1, 0), weight=dist_start)

    # bottom cross-aisle corridor
    for a in range(1, n_aisles):
        G.add_edge((a, 0), (a + 1, 0), weight=dist_aisle)

    # top cross-aisle corridor
    for a in range(1, n_aisles):
        G.add_edge((a, n_cells + 1), (a + 1, n_cells + 1), weight=dist_aisle)

    # aisle edges
    for a in range(1, n_aisles + 1):
        # bottom entry
        G.add_edge((a, 0), (a, 1), weight=dist_aisle_location)
        # picks
        for c in range(1, n_cells):
            G.add_edge((a, c), (a, c + 1), weight=dist_pick)
        # top exit
        G.add_edge((a, n_cells), (a, n_cells + 1), weight=dist_aisle_location)

    return G


class OIBRSPLoader(DataLoader):
    def __init__(
        self,
        instances_dir: str | Path,
        distance_matrix_path: str | Path,
        cache_dir: str | Path = None,
        # unit conversion
        distance_unit: str = "dm",           # "dm" → convert to m; "m" → keep
        # picker parameters (Table 4)
        picker_travel_velocity_mps: float = 1.0,
        batch_setup_time_s: float = 180.0,
        search_and_pick_time_s: float = 10.0,
        batch_capacity_orders: int = 10,
        pickers_per_n_orders: int = 200,
        # Layout parameters
        dist_bottom_to_cell: float = 0.0,
        dist_top_to_cell: float = 0.0,
        depot_aisle: int = 1,
        # Warehouse configuration
        num_warehouse_blocks: int = 2,
        num_aisles: int = 6,
        num_subaisles: int = 24,
        locations_per_aisle: int = 240,
        storage_policy: str = "Across-aisle",
        # Physical dimensions (meters)
        storage_location_length_m: float = 1.3,
        storage_location_width_m: float = 0.9,
        pick_aisle_width_m: float = 3.0,
        cross_aisle_width_m: float = 6.0,
        cfg=None,
    ):
        super().__init__(instances_dir)
        self.cfg = cfg

        # Distance matrix path (may be relative to instances_dir)
        dm_path = Path(distance_matrix_path)
        if not dm_path.is_absolute():
            dm_path = Path(instances_dir) / dm_path
        self.distance_matrix_path = dm_path

        # Cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.distance_unit = distance_unit
        self.dm_scale = 0.1 if distance_unit == "dm" else 1.0  # dm → m

        # Picker parameters
        self.picker_travel_velocity_mps = picker_travel_velocity_mps
        self.batch_setup_time_s = batch_setup_time_s
        self.search_and_pick_time_s = search_and_pick_time_s
        self.batch_capacity_orders = batch_capacity_orders
        self.pickers_per_n_orders = pickers_per_n_orders

        self.num_warehouse_blocks = num_warehouse_blocks
        self.num_aisles = num_aisles
        self.num_subaisles = num_subaisles
        self.locations_per_aisle = locations_per_aisle
        self.storage_policy = storage_policy
        self.storage_location_length_m = storage_location_length_m
        self.storage_location_width_m = storage_location_width_m
        self.pick_aisle_width_m = pick_aisle_width_m
        self.cross_aisle_width_m = cross_aisle_width_m
        self.dist_bottom_to_cell = dist_bottom_to_cell
        self.dist_top_to_cell = dist_top_to_cell
        self.depot_aisle = depot_aisle

        # Lazy-loaded distance matrix (shared across load() calls)
        self._raw_dist_matrix: Optional[pd.DataFrame] = None

    def load(
        self,
        order_list_path: str | Path,
        order_line_path: str | Path,
        online: bool = False,
        use_cache: bool = True,
        **kwargs,
    ) -> BaseWarehouseDomain:
        """
        Load a single instance.

        Parameters
        ----------
        order_list_path : str | Path
            Path to the OrderList file (absolute or relative to *instances_dir*).
        order_line_path : str | Path
            Path to the OrderLineList file (absolute or relative to *instances_dir*).
        online : bool
            ``True`` for online instances (OrderList has a *ReleaseTime* column).
            ``False`` (default) for static instances (no ReleaseTime column).
        use_cache : bool
            Whether to read/write a pickle cache.

        Returns
        -------
        BaseWarehouseDomain
        """
        order_list_path = self._resolve(order_list_path)
        order_line_path = self._resolve(order_line_path)

        cache_key = f"{order_list_path.stem}_{order_line_path.stem}"

        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}_domain.pkl"
            if cache_path.exists():
                from ware_ops_algos.utils.io_helpers import load_pickle
                return load_pickle(str(cache_path))

        parsed = self._parse(str(order_list_path), str(order_line_path), online=online)
        domain = self._build(parsed)

        if use_cache and self.cache_dir:
            from ware_ops_algos.utils.io_helpers import dump_pickle
            cache_path = self.cache_dir / f"{cache_key}_domain.pkl"
            dump_pickle(str(cache_path), domain)

        return domain

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _resolve(self, p: str | Path) -> Path:
        p = Path(p)
        if not p.is_absolute():
            p = self.data_dir / p
        return p

    # ------------------------------------------------------------------ #
    #  Distance matrix                                                     #
    # ------------------------------------------------------------------ #

    def _load_distance_matrix(self) -> pd.DataFrame:
        """
        Read the tab-separated all-pairs distance matrix.

        Row/column 0 is the depot; rows/columns 1..N are pick locations
        in the same order as LocationID values appear in the instance files.

        Values are converted from decimetres to metres when
        ``self.distance_unit == "dm"``.
        """
        if self._raw_dist_matrix is not None:
            return self._raw_dist_matrix

        raw = pd.read_csv(
            self.distance_matrix_path,
            sep=r"\t",
            header=None,
            engine="python",
            encoding="windows-1252",
        ).astype(float)

        # Scale to metres if necessary
        raw *= self.dm_scale

        self._raw_dist_matrix = raw
        return raw

    # ------------------------------------------------------------------ #
    #  Parsing                                                             #
    # ------------------------------------------------------------------ #

    def _parse(
        self,
        order_list_path: str,
        order_line_path: str,
        online: bool,
    ) -> Dict[str, Any]:
        """
        Parse OrderList and OrderLineList files into DataFrames.

        Static OrderList columns:
            OrderID  NumberOfOrderLines  DueTime  FirstOrderLineID

        Online OrderList columns:
            OrderID  NumberOfOrderLines  ReleaseTime  DueTime  FirstOrderLineID

        OrderLineList columns (both flavours):
            OrderID  OrderLineID  AisleID  CellID  LevelID  LocationID
        """
        if online:
            order_cols = [
                "OrderID", "NumberOfOrderLines", "ReleaseTime", "DueTime",
                "FirstOrderLineID",
            ]
            int_cols  = ["OrderID", "NumberOfOrderLines", "FirstOrderLineID"]
            float_cols = ["ReleaseTime", "DueTime"]
        else:
            order_cols = [
                "OrderID", "NumberOfOrderLines", "DueTime", "FirstOrderLineID",
            ]
            int_cols  = ["OrderID", "NumberOfOrderLines", "FirstOrderLineID"]
            float_cols = ["DueTime"]

        order_df = self._load_csv(
            order_list_path,
            sep=r"\s+",
            header=None,
            names=order_cols,
            engine="python",
            encoding="windows-1252",
        )
        order_df[int_cols]   = order_df[int_cols].astype("int64")
        order_df[float_cols] = order_df[float_cols].astype("float64")
        if not online:
            order_df["ReleaseTime"] = 0.0  # static → all orders available at t=0

        line_cols = ["OrderID", "OrderLineID", "AisleID", "CellID", "LevelID", "LocationID"]
        line_df = self._load_csv(
            order_line_path,
            sep=r"\s+",
            header=None,
            names=line_cols,
            engine="python",
            encoding="windows-1252",
        ).astype({c: "int64" for c in line_cols})

        return {"orders": order_df, "order_lines": line_df, "online": online}

    def _build(self, parsed: Dict[str, Any]) -> BaseWarehouseDomain:
        order_df = parsed["orders"]
        line_df  = parsed["order_lines"]
        online   = parsed["online"]

        n_phys_aisles    = self.num_aisles #// 2        # 12 physical aisles (24 sub-aisles / 2)
        n_pick_per_block = self.locations_per_aisle # // self.num_warehouse_blocks  # 120
        block_span       = n_pick_per_block + 2        # 122 — generator internal row offset

        # dist_between_blocks = 4.7m (NOT cross_aisle_width=6.0m):
        # the generator adds dist_aisle_location on both sides of the inter-block edge,
        # so 1.3 + 4.7 + 1.3 = 7.3m total traversal matches the distance matrix.
        dist_aisle_pitch    = self.pick_aisle_width_m # 4.8m
        dist_between_blocks = self.cross_aisle_width_m    # 4.7m
        dist_start          = self.dist_bottom_to_cell #(n_phys_aisles / 2) * dist_aisle_pitch                       # 14.4m

        depot_node = (0, 0)
        start_conn = (1, 0)

        layout_params = LayoutParameters(
            n_aisles=n_phys_aisles,
            n_pick_locations=self.locations_per_aisle,
            n_blocks=self.num_warehouse_blocks,
            dist_pick_locations=self.storage_location_length_m,
            dist_aisle=dist_aisle_pitch,
            dist_top_to_pick_location=self.storage_location_length_m,
            dist_bottom_to_pick_location=self.storage_location_length_m,
            dist_start=dist_start,
            dist_end=dist_start,
            dist_cross_aisle=self.cross_aisle_width_m,
            start_location=depot_node,
            end_location=depot_node,
            start_connection_point=start_conn,
            end_connection_point=start_conn,
        )

        # graph_generator = MultiBlockShelfStorageGraphGenerator(
        #     n_aisles=n_phys_aisles,
        #     n_pick_locations=int(layout_params.n_pick_locations / 2),
        #     n_blocks=self.num_warehouse_blocks,
        #     dist_aisle=dist_aisle_pitch,
        #     dist_between_blocks=dist_between_blocks,
        #     dist_pick_locations=self.storage_location_width_m,
        #     dist_aisle_location=0,
        #     start_location=depot_node,
        #     end_location=depot_node,
        #     start_connection_point=start_conn,
        #     end_connection_point=start_conn,
        #     dist_start=dist_start,
        #     dist_end=dist_start,
        # )
        # start_location = (self.depot_aisle, -1)
        # end_location = (self.depot_aisle - 1, -1)
        # graph_generator = ShelfStorageGraphGenerator(
        #     # n_blocks=2,
        #     # dist_between_blocks=6,
        #     n_aisles=6,
        #     n_pick_locations=30,
        #     dist_aisle=3,
        #     dist_pick_locations=1.3,
        #     dist_aisle_location=1.3,
        #     start_location=(0, 0),
        #     end_location=(0, 0),
        #     start_connection_point=(1, 0),
        #     end_connection_point=(1, 0),
        #     dist_start=15.7,#14.4,
        #     dist_end=15.7,
        # )
        # graph_generator.populate_graph()
        # graph = graph_generator.G
        #
        # nodes = list(graph.nodes())
        # A = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight='weight', dtype=float)
        # dist_mat, predecessors = floyd_warshall(A, directed=False, return_predecessors=True)
        # dima = pd.DataFrame(dist_mat, index=nodes, columns=nodes)
        G = build_oibrsp_graph()
        nodes = list(G.nodes())
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight='weight', dtype=float)
        dist_mat, predecessors = floyd_warshall(A, directed=False, return_predecessors=True)
        dima = pd.DataFrame(dist_mat, index=nodes, columns=nodes)

        pick_nodes            = [n for n, d in G.nodes(data=True) if d.get("type") == "pick_node"]
        min_aisle_position    = min(n[1] for n in pick_nodes)
        max_aisle_position    = max(n[1] for n in pick_nodes)
        closest_node_to_start = min(pick_nodes, key=lambda n: dima[n][depot_node])

        layout_network = LayoutNetwork(
            graph=G,
            distance_matrix=dima,
            predecessor_matrix=predecessors,
            closest_node_to_start=closest_node_to_start,
            min_aisle_position=min_aisle_position,
            max_aisle_position=max_aisle_position,
            start_node=depot_node,
            end_node=depot_node,
            node_list=nodes,
        )

        layout = LayoutData(
            tpe=LayoutType.CONVENTIONAL,
            graph_data=layout_params,
            layout_network=layout_network,
        )

        unique_locs = (
            line_df[["LocationID", "AisleID", "CellID", "LevelID"]]
            .drop_duplicates()
            .sort_values("LocationID")
            .reset_index(drop=True)
        )

        # def to_graph_node(aisle_id: int, cell_id: int) -> tuple:
        #     phys_aisle    = math.ceil(aisle_id / 2)
        #     block         = (cell_id - 1) // n_pick_per_block
        #     cell_in_block = ((cell_id - 1) % n_pick_per_block) + 1
        #     graph_y       = cell_in_block + block * block_span
        #     return (phys_aisle, graph_y)

        article_list = [
            Article(int(row.LocationID), weight=1)
            for row in unique_locs.itertuples(index=False)
        ]
        articles = Articles(tpe=ArticleType.STANDARD, articles=article_list)

        # def location_id_to_graph_node(loc_id: int, n_cells_per_aisle: int = 120) -> tuple:
        #     pair = (loc_id - 1) // 2
        #     aisle = pair // n_cells_per_aisle + 1
        #     cell = pair % n_cells_per_aisle + 1
        #     return (aisle, cell)
        #
        # # in _build:
        # storage_locations = [
        #     Location(
        #         x=location_id_to_graph_node(int(row.LocationID), n_pick_per_block)[0],
        #         y=location_id_to_graph_node(int(row.LocationID), n_pick_per_block)[1],
        #         article_id=int(row.LocationID),
        #         amount=1,
        #     )
        #     for row in unique_locs.itertuples(index=False)
        # ]
        # storage_locations = []
        # for row in unique_locs.itertuples(index=False):
        #     # Adjust aisle ID for multi-block layout
        #     aisle_id = int(row.AisleID)
        #     x = aisle_id - self.num_aisles if aisle_id > self.num_aisles else aisle_id
        #
        #     storage_locations.append(
        #         Location(
        #             x=x,
        #             y=math.ceil(row.CellID / 2),
        #             article_id=int(row.LocationID),
        #             amount=1,
        #         )
        #     )
        # storage_locations = [
        #     Location(
        #         x=math.ceil(int(row.AisleID) / 2),
        #         y=math.ceil(int(row.CellID) / 2),
        #         article_id=int(row.LocationID),
        #         amount=1,
        #     )
        #     for row in line_df[["LocationID", "AisleID", "CellID"]]
        #     .drop_duplicates(subset="LocationID")
        #     .itertuples(index=False)
        # ]
        storage_locations = [
            Location(
                x=int(row.LocationID),  # pick_node = LocationID
                y=0,  # unused
                article_id=int(row.LocationID),
                amount=1,
            )
            for row in line_df[["LocationID", "AisleID", "CellID"]]
            .drop_duplicates(subset="LocationID")
            .itertuples(index=False)
        ]

        # storage_locations = [
        #     Location(
        #         x=math.ceil(row.AisleID / 2),
        #         y=math.ceil(row.CellID / 2),
        #         article_id=int(row.LocationID),
        #         amount=1,
        #     )
        #     for row in unique_locs.itertuples(index=False)
        # ]

        storage = StorageLocations(tpe=StorageType.DEDICATED, locations=storage_locations)
        storage.build_article_location_mapping()

        grouped = (
            line_df.groupby(["OrderID", "LocationID"], as_index=False)
            .size()
            .rename(columns={"size": "amount"})
        )

        order_attrs = order_df.set_index("OrderID")[
            ["ReleaseTime", "DueTime"]
        ].to_dict(orient="index")

        order_list = []
        for oid in sorted(grouped["OrderID"].unique().tolist()):
            sub   = grouped[grouped["OrderID"] == oid]
            attrs = order_attrs.get(oid, {})

            positions = [
                OrderPosition(
                    order_number=oid,
                    article_id=int(r.LocationID),
                    amount=int(r.amount),
                )
                for r in sub.itertuples(index=False)
            ]

            order_list.append(
                Order(
                    order_id=oid,
                    order_date=attrs.get("ReleaseTime", 0.0),
                    due_date=attrs.get("DueTime"),
                    order_positions=positions,
                )
            )

        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=order_list)

        n_orders    = len(orders.orders)
        n_resources = max(1, math.ceil(n_orders / self.pickers_per_n_orders))

        pick_cart = PickCart(
            n_dimension=1,
            n_boxes=1,
            capacities=[int(self.batch_capacity_orders)],
            dimensions=[DimensionType.ORDERS],
            box_can_mix_orders=True,
        )
        resources_list = [
            ManualPicker(
                id=i,
                capacity=int(self.batch_capacity_orders),
                time_per_pick=self.search_and_pick_time_s,
                speed=self.picker_travel_velocity_mps,
                pick_cart=pick_cart,
                tour_setup_time=self.batch_setup_time_s,
            )
            for i in range(n_resources)
        ]
        resources = Resources(ResourceType.HUMAN, resources_list)

        info_type = WarehouseInfoType.ONLINE if online else WarehouseInfoType.OFFLINE
        warehouse_info = WarehouseInfo(tpe=info_type)

        return BaseWarehouseDomain(
            problem_class="OBRSP",
            objective="Distance",
            layout=layout,
            articles=articles,
            orders=orders,
            resources=resources,
            storage=storage,
            warehouse_info=warehouse_info,
        )


def graph_node_to_matrix_idx(node: tuple, n_pick_per_block: int = 120, block_span: int = 122) -> int:
    if node == (0, 0):
        return 0
    phys_aisle, graph_y = node
    block         = (graph_y - 1) // block_span
    cell_in_block = graph_y - block * block_span
    cell_id       = block * n_pick_per_block + cell_in_block
    pair_idx      = (phys_aisle - 1) * n_pick_per_block + (cell_id - 1)
    return pair_idx * 2 + 1


def main():
    from pathlib import Path
    from typing import Tuple

    def find_project_root() -> Path:
        """Find project root by looking for a marker file."""
        current = Path().resolve()
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():  # or setup.py, .git, etc.
                return parent
        raise FileNotFoundError("Could not find project root")

    PROJECT_ROOT = find_project_root()

    DATA_DIR = PROJECT_ROOT / "data"

    instances_base = DATA_DIR / "instances"
    cache_base = DATA_DIR / "instances" / "caches"
    order_list_online = instances_base / "OIBRSP" / "static" / "OrderList_LargeProblems_1_1.txt"
    order_lines_online = instances_base / "OIBRSP" / "static" / "OrderLineList_LargeProblems_1_1.txt"

    loader_online = OIBRSPLoader(
        instances_dir=DATA_DIR / "OIBRSP",
        distance_matrix_path=instances_base / "OIBRSP" / "distance_matrices" / "DistanceMatrix_24SubAisles.txt",
        cache_dir=DATA_DIR / "caches" / "OIBRSP",
    )

    domain = loader_online.load(
        order_list_path=order_list_online,
        order_line_path=order_lines_online,
        online=False,  # online: has ReleaseTime column
        use_cache=False
    )

    raw = pd.read_csv(
    instances_base / "OIBRSP" / "distance_matrices" / "DistanceMatrix_12SubAisles.txt",
    sep = r"\t", header = None, engine = "python", encoding = "windows-1252",
    ).astype(float)
    raw = raw.dropna(axis=1, how="all")
    dm_ref = raw.values * 0.1
    print()
    orders = domain.orders
    layout = domain.layout
    resources = domain.resources
    articles = domain.articles
    storage_locations = domain.storage

    layout_network = layout.layout_network
    graph_data = layout.graph_data
    graph = layout_network.graph
    graph_params = layout.graph_data
    dima = layout_network.distance_matrix

    selector = GreedyItemAssignment(storage_locations)
    ia_sol = selector.solve(orders.orders)
    orders.orders = ia_sol.resolved_orders

    batcher = OrderNrFifoBatching(
        pick_cart=resources.resources[0].pick_cart,
        articles=articles
    )

    batching_sol = batcher.solve(orders.orders)

    print(batching_sol.execution_time)

    pl = []
    o_ids = [53]
    for order in orders.orders:
        if order.order_id in o_ids:
            print(len(order.pick_positions))
            for pos in order.pick_positions:
                pl.append(pos.pick_node[0])
    # Build pick list from batches
    # batches = batching_sol.batches
    # pick_lists = []
    # for batch in batches:
    #     pick_list = []
    #     print(len(batch.orders))
    #     for order in batch.orders:
    #         for pos in order.pick_positions:
    #             pick_list.append(pos)
    #     pick_lists.append(pick_list)
    def load_reference_dima(matrix_path: str | Path) -> pd.DataFrame:
        raw = pd.read_csv(matrix_path, sep=r"\t", header=None, engine="python", encoding="windows-1252").astype(
            float).dropna(axis=1, how="all")
        dm = raw.values * 0.1
        index = list(range(dm.shape[0]))
        return pd.DataFrame(dm, index=index, columns=index)

    d = load_reference_dima(instances_base / "OIBRSP" / "distance_matrices" / "DistanceMatrix_12SubAisles.txt")


    ss_router = SShapeRouting(
        start_node=0,
        end_node=0,
        closest_node_to_start=1,
        min_aisle_position=layout_network.min_aisle_position,
        max_aisle_position=layout_network.max_aisle_position,
        distance_matrix=layout_network.distance_matrix,
        predecessor_matrix=None,
        picker=resources.resources,
        gen_tour=False, # we dont have predecessor matrix yet
        gen_item_sequence=True,
        node_list=layout_network.node_list,
        node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
        idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))}
    )

    tsp_routing = ExactTSPRoutingDistance(
        distance_matrix=d,
        predecessor_matrix=None,
        big_m=1000,
        start_node=0,
        end_node=0,
        closest_node_to_start=layout_network.closest_node_to_start,
        min_aisle_position=layout_network.min_aisle_position,
        max_aisle_position=layout_network.max_aisle_position,
        picker=resources.resources,
        gen_tour=False,  # we dont have predecessor matrix yet
        gen_item_sequence=True,
        set_time_limit=120,
        node_list=layout_network.node_list,
        node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
        idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
    )
    print(pl)
    sol = tsp_routing.solve(pl)
    sol2 = ss_router.solve(pl)
    print()
    print(sol.route.distance)
    print(sol.route.item_sequence)
    # plot_route(graph, sol.route.route)

    item_seq = sol.route.item_sequence
    loc_map = {(s.x, s.y): s.article_id for s in storage_locations.locations}
    loc_map_2 = {s.article_id: (s.x, s.y) for s in storage_locations.locations}

    # for i in item_seq:
    #     print(f"{i}:{loc_map[i]}")


    from python_tsp.exact import solve_tsp_dynamic_programming
    import numpy as np
    # location_ids = [643, 76, 423, 120, 1, 703, 101, 479, 652, 635, 191, 700, 49, 711, 73]
    location_ids = [613, 111, 314, 398, 52, 198, 418, 566, 489, 1, 295, 544, 41, 403, 579]
    # st = (0, 0)
    # for l in location_ids:
    #     print(dima.at[st, loc_map_2[l]])
    #     st = loc_map_2

    nodes = [0] + location_ids
    dm = d.iloc[nodes, nodes].values
    # Make last column (return to depot) zero to allow open path
    permutation, distance = solve_tsp_dynamic_programming(dm)
    print(f"Open route distance: {distance:.1f}m")
    for p in permutation:
        if p == 0:
            continue
        print(location_ids[p-1])
    # total_dist = 0
    # for pl in pick_lists:
    #     sol = ss_router.solve(pl)
    #     total_dist += sol.route.distance
    #     print(sol.route.distance)
    #     # plot_route(graph, sol.route.route)
    #     ss_router.reset_parameters()
    # print(total_dist)





if __name__ == "__main__":
    main()

