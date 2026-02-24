from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Any

import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall

from ware_ops_algos.data_loaders.base_data_loader import DataLoader
from ware_ops_algos.domain_models import (
    Article, Order, OrderPosition, LayoutData,
    Resource, StorageLocations, Location, ResourceType, LayoutType, LayoutParameters,
    ArticleType, StorageType, OrderType, OrdersDomain, Articles, Resources,
    BaseWarehouseDomain, LayoutNetwork, PickCart, DimensionType, WarehouseInfoType, WarehouseInfo, ManualPicker
)


class IOPVRPLoader(DataLoader):
    """
    Loads warehouse instances from OrderList and OrderLineList files.

    These instances represent multi-block warehouse layouts where:
    - Each unique LocationID is treated as an Article
    - Order lines reference LocationID to create OrderPositions
    - Storage locations are derived from aisle/cell positions
    """

    def __init__(
            self,
            instances_dir: str | Path,
            cache_dir: str | Path = None,
            # Warehouse configuration
            num_warehouse_blocks: int = 2,
            num_aisles: int = 12,
            num_subaisles: int = 24,
            locations_per_aisle: int = 240,
            storage_policy: str = "Across-aisle",
            # Physical dimensions (meters)
            storage_location_length_m: float = 1.3,
            storage_location_width_m: float = 0.9,
            pick_aisle_width_m: float = 3.0,
            cross_aisle_width_m: float = 6.0,
            # Picker parameters
            picker_travel_velocity_mps: float = 1.0,
            batch_setup_time_s: float = 180.0,
            search_and_pick_time_s: float = 10.0,
            batch_capacity_orders: int = 10,
            # Routing parameters
            vehicle_velocity_kmph: float = 50.0,
            vehicle_capacity: float = float("inf"),
            orders_per_vehicle: int = 25,
            # Layout parameters
            dist_bottom_to_cell: float = 1.0,
            dist_top_to_cell: float = 1.0,
            depot_aisle: int = 1,
            cfg=None
    ):
        """
        Args:
            instances_dir: Directory containing instance files
            cache_dir: Optional directory for caching parsed domains
            num_warehouse_blocks: Number of warehouse blocks
            num_aisles: Total number of aisles
            num_subaisles: Number of sub-aisles
            locations_per_aisle: Number of pick locations per aisle
            storage_policy: Storage assignment policy
            storage_location_length_m: Length of storage location in meters
            storage_location_width_m: Width of storage location in meters
            pick_aisle_width_m: Width of picking aisles in meters
            cross_aisle_width_m: Width of cross aisles in meters
            picker_travel_velocity_mps: Picker travel speed in m/s
            batch_setup_time_s: Setup time per batch in seconds
            search_and_pick_time_s: Time to search and pick in seconds
            batch_capacity_orders: Maximum orders per batch
            vehicle_velocity_kmph: Vehicle velocity in km/h
            vehicle_capacity: Vehicle capacity
            orders_per_vehicle: Orders per vehicle
            dist_bottom_to_cell: Distance from bottom cross-aisle to cell
            dist_top_to_cell: Distance from top cross-aisle to cell
            depot_aisle: Depot aisle position (1-indexed)
        """
        super().__init__(instances_dir)
        self.cfg = cfg
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = None

        # Store warehouse configuration
        self.num_warehouse_blocks = num_warehouse_blocks
        self.num_aisles = num_aisles
        self.num_subaisles = num_subaisles
        self.locations_per_aisle = locations_per_aisle
        self.storage_policy = storage_policy
        self.storage_location_length_m = storage_location_length_m
        self.storage_location_width_m = storage_location_width_m
        self.pick_aisle_width_m = pick_aisle_width_m
        self.cross_aisle_width_m = cross_aisle_width_m
        self.picker_travel_velocity_mps = picker_travel_velocity_mps
        self.batch_setup_time_s = batch_setup_time_s
        self.search_and_pick_time_s = search_and_pick_time_s
        self.batch_capacity_orders = batch_capacity_orders
        self.vehicle_velocity_kmph = vehicle_velocity_kmph
        self.vehicle_capacity = vehicle_capacity
        self.orders_per_vehicle = orders_per_vehicle
        self.dist_bottom_to_cell = dist_bottom_to_cell
        self.dist_top_to_cell = dist_top_to_cell
        self.depot_aisle = depot_aisle

    def load(
            self,
            order_list_path: str | Path,
            order_line_path: str | Path,
            use_cache: bool = True,
            **kwargs
    ) -> BaseWarehouseDomain:
        """
        Load warehouse domain from order list and order line files.

        Args:
            order_list_path: Path to order list file (absolute or relative)
            order_line_path: Path to order line file (absolute or relative)
            use_cache: Whether to use/create cached version

        Returns:
            BaseWarehouseDomain object
        """
        order_list_path = Path(order_list_path)
        order_line_path = Path(order_line_path)

        # Handle relative paths
        if not order_list_path.is_absolute():
            order_list_path = self.data_dir / order_list_path
        if not order_line_path.is_absolute():
            order_line_path = self.data_dir / order_line_path

        # Create cache key from both filenames
        cache_key = f"{order_list_path.stem}_{order_line_path.stem}"

        # Check cache
        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}_domain.pkl"
            if cache_path.exists():
                from ware_ops_algos.utils.io_helpers import load_pickle
                self.cache_path = cache_path
                return load_pickle(str(cache_path))

        # Parse and build
        parsed = self._parse(str(order_list_path), str(order_line_path))
        domain = self._build(parsed)

        # Save cache
        if use_cache and self.cache_dir:
            from ware_ops_algos.utils.io_helpers import dump_pickle
            cache_path = self.cache_dir / f"{cache_key}_domain.pkl"
            dump_pickle(str(cache_path), domain)
            self.cache_path = cache_path

        return domain

    def _parse(self, order_list_path: str, order_line_path: str) -> Dict[str, Any]:
        """
        Parse order list and order line files.

        Args:
            order_list_path: Path to order list file
            order_line_path: Path to order line file

        Returns:
            Dictionary with keys: orders, order_lines
        """
        # Parse order list
        order_cols = [
            "OrderID", "NumberOfOrderLines", "ReleaseTimeOPP", "DueTimeOPP",
            "FirstOrderLineID", "X", "Y", "EarliestTime", "LatestTime"
        ]
        order_df = self._load_csv(
            order_list_path,
            sep=r"\s+",
            header=None,
            names=order_cols,
            engine="python",
            encoding="windows-1252"
        )

        # Convert types
        int_cols = ["OrderID", "NumberOfOrderLines", "FirstOrderLineID"]
        order_df[int_cols] = order_df[int_cols].astype("int64")
        float_cols = ["ReleaseTimeOPP", "DueTimeOPP", "X", "Y", "EarliestTime", "LatestTime"]
        order_df[float_cols] = order_df[float_cols].astype("float64")

        # Parse order line list
        line_cols = ["OrderID", "OrderLineID", "AisleID", "CellID", "LevelID", "LocationID"]
        line_df = self._load_csv(
            order_line_path,
            sep=r"\s+",
            header=None,
            names=line_cols,
            engine="python",
            encoding="windows-1252"
        )

        # Convert types
        line_df = line_df.astype({
            "OrderID": "int64",
            "OrderLineID": "int64",
            "AisleID": "int64",
            "CellID": "int64",
            "LevelID": "int64",
            "LocationID": "int64",
        })

        return {"orders": order_df, "order_lines": line_df}

    def _build(self, parsed: Dict[str, Any]) -> BaseWarehouseDomain:
        """
        Build BaseWarehouseDomain from parsed data.

        Args:
            parsed: Dictionary from _parse() containing orders and order_lines DataFrames

        Returns:
            BaseWarehouseDomain instance
        """
        from ware_ops_algos.generators import MultiBlockShelfStorageGraphGenerator

        order_df = parsed["orders"]
        line_df = parsed["order_lines"]

        # Depot configuration
        depot_node = 0
        start_location = (self.depot_aisle, -1)
        end_location = (self.depot_aisle - 1, -1)
        closest_node_to_start = (start_location[0], start_location[1] + 1)
        dist_start = self.dist_bottom_to_cell
        dist_end = self.dist_bottom_to_cell

        # Layout parameters
        layout_params = LayoutParameters(
            n_aisles=self.num_aisles,
            n_pick_locations=self.locations_per_aisle,
            n_blocks=self.num_warehouse_blocks,
            dist_pick_locations=self.storage_location_length_m,
            dist_aisle=self.pick_aisle_width_m,
            dist_top_to_pick_location=self.dist_top_to_cell,
            dist_bottom_to_pick_location=self.dist_bottom_to_cell,
            dist_start=dist_start,
            dist_end=dist_end,
            dist_cross_aisle=self.cross_aisle_width_m,
            start_location=start_location,
            end_location=end_location,
            start_connection_point=(self.depot_aisle, depot_node),
            end_connection_point=(self.depot_aisle, depot_node),
        )

        # Build graph
        min_aisle_position = 0
        max_aisle_position = layout_params.n_pick_locations + 1

        graph_generator = MultiBlockShelfStorageGraphGenerator(
            n_aisles=layout_params.n_aisles,
            n_pick_locations=int(layout_params.n_pick_locations / 2),
            n_blocks=layout_params.n_blocks,
            dist_aisle=layout_params.dist_aisle,
            dist_between_blocks=layout_params.dist_cross_aisle,
            dist_pick_locations=layout_params.dist_pick_locations,
            dist_aisle_location=layout_params.dist_bottom_to_pick_location,
            start_location=layout_params.start_location,
            end_location=layout_params.end_location,
            start_connection_point=layout_params.start_connection_point,
            end_connection_point=layout_params.end_connection_point,
            dist_start=layout_params.dist_start,
            dist_end=layout_params.dist_end
        )
        graph_generator.populate_graph()
        graph = graph_generator.G

        # Compute all-pairs shortest paths
        nodes = list(graph.nodes())
        A = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight='weight', dtype=float)
        dist_mat, predecessors = floyd_warshall(A, directed=False, return_predecessors=True)
        dima = pd.DataFrame(dist_mat, index=nodes, columns=nodes)

        layout_network = LayoutNetwork(
            graph=graph,
            distance_matrix=dima,
            predecessor_matrix=predecessors,
            closest_node_to_start=closest_node_to_start,
            min_aisle_position=min_aisle_position,
            max_aisle_position=max_aisle_position,
            start_node=start_location,
            end_node=end_location,
            node_list=nodes
        )

        layout = LayoutData(
            tpe=LayoutType.CONVENTIONAL,
            graph_data=layout_params,
            layout_network=layout_network,
        )

        # Articles & Storage - each LocationID becomes an Article
        unique_locs = (
            line_df[["LocationID", "AisleID", "CellID", "LevelID"]]
            .drop_duplicates()
            .sort_values("LocationID")
        )

        article_list = [
            Article(int(loc_id), weight=1)
            for loc_id in unique_locs["LocationID"].tolist()
        ]
        articles = Articles(tpe=ArticleType.STANDARD, articles=article_list)

        # Build storage locations
        storage_type = StorageType.DEDICATED
        storage_locations = []
        for row in unique_locs.itertuples(index=False):
            # Adjust aisle ID for multi-block layout
            aisle_id = int(row.AisleID)
            x = aisle_id - self.num_aisles if aisle_id > self.num_aisles else aisle_id

            storage_locations.append(
                Location(
                    x=x,
                    y=int(row.CellID),
                    article_id=int(row.LocationID),
                    amount=1,
                )
            )

        storage = StorageLocations(tpe=storage_type, locations=storage_locations)
        storage.build_article_location_mapping()

        # Build orders
        # Group order lines by OrderID and LocationID to aggregate quantities
        grouped = (
            line_df.groupby(["OrderID", "LocationID"], as_index=False)
            .size()
            .rename(columns={"size": "amount"})
        )

        order_ids_sorted = sorted(grouped["OrderID"].unique().tolist())

        # Index order attributes for quick lookup
        order_attrs = (
            order_df.set_index("OrderID")[
                ["ReleaseTimeOPP", "DueTimeOPP", "X", "Y", "EarliestTime", "LatestTime"]
            ].to_dict(orient="index")
        )

        order_list = []
        for oid in order_ids_sorted:
            sub = grouped[grouped["OrderID"] == oid]

            # Get order attributes
            attrs = order_attrs.get(oid, {})
            order_date = attrs.get("ReleaseTimeOPP")
            # order_date = 0
            due_date = attrs.get("DueTimeOPP")

            # Build order positions
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
                    order_date=order_date,
                    due_date=due_date,
                    order_positions=positions,
                )
            )

        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=order_list)

        # Resources
        n_orders = len(orders.orders)
        # n_resources = math.ceil(n_orders / 200)
        n_resources = 1
        pick_cart = PickCart(n_dimension=1,
                             n_boxes=1,
                             capacities=[int(self.batch_capacity_orders)],
                             dimensions=[DimensionType.ORDERS],
                             box_can_mix_orders=True)
        resources_list = [
            ManualPicker(
                id=i,
                capacity=int(self.batch_capacity_orders),
                time_per_pick=self.search_and_pick_time_s,
                speed=self.picker_travel_velocity_mps,
                pick_cart=pick_cart,
                tour_setup_time=self.batch_setup_time_s
            )
            for i in range(n_resources)
        ]

        resources = Resources(ResourceType.HUMAN, resources_list)

        warehouse_info = WarehouseInfo(tpe=WarehouseInfoType.OFFLINE)

        return BaseWarehouseDomain(
            problem_class="OBSRP",
            objective="Distance",
            layout=layout,
            articles=articles,
            orders=orders,
            resources=resources,
            storage=storage,
            warehouse_info=warehouse_info
        )
