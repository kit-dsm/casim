from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall
from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import BaseWarehouseDomain, Order, Location, Article, PickCart, DimensionType, \
    ResourceType, Resource, Resources, OrdersDomain, OrderType, Articles, ArticleType, StorageLocations, StorageType, \
    LayoutData, LayoutType, LayoutNetwork, LayoutParameters, WarehouseInfoType, WarehouseInfo, OrderPosition


class HennOnlineLoader(DataLoader):
    def __init__(
            self,
            instances_dir: str | Path,
            cfg: dict,
            cache_dir: str | Path = None,
            mirror_top_depot: bool = True
    ):
        """
        Args:
            instances_dir: Directory containing instance files
            cache_dir: Optional directory for caching parsed domains
            mirror_top_depot: If True, convert top-depot instances to bottom-depot via mirroring
        """
        super().__init__(instances_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.mirror_top_depot = mirror_top_depot

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, filepath: str, use_cache: bool = True) -> BaseWarehouseDomain:
        """
        Load a single Heßler-Irnich instance file.

        Args:
            filepath: Path to instance file (absolute or relative to instances_dir)
            use_cache: Whether to use/create cached version

        Returns:
            BaseWarehouseDomain object
        """
        filepath = Path(filepath)
        filepath_arrival_times = f"Arrival_Time_abc1_{filepath.stem}.csv"
        if not filepath.is_absolute():
            filepath = self.data_dir / filepath
        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{filepath.stem}_domain.pkl"
            self.cache_path = cache_path
            if cache_path.exists():
                from ware_ops_algos.utils.io_helpers import load_pickle
                return load_pickle(str(cache_path))

        # Parse and build
        parsed = self._parse(str(filepath), filepath_arrival_times)
        domain = self._build(parsed)

        # Save cache
        if use_cache and self.cache_dir:
            from ware_ops_algos.utils.io_helpers import dump_pickle
            cache_path = self.cache_dir / f"{filepath.stem}_domain.pkl"
            dump_pickle(str(cache_path), domain)

        return domain

    def _parse(self, filepath: str, filepath_arrival_times: str) -> dict[str, Any]:
        """
        Parse a Heßler-Irnich format file into structured data.

        Returns:
            Dictionary with keys: header, articles, skus, orders
        """
        lines = self._load_text(filepath, encoding="windows-1252")
        lines_arrival_time = self._load_text(filepath_arrival_times, encoding="windows-1252")
        order_arrival_times = {}
        for l in lines_arrival_time[3:]:
            o_id, arrival_time = l.split(";")
            order_arrival_times[int(o_id)-1] = int(arrival_time) / 1000  # is given in milliseconds

        header: dict[str, str] = {}
        articles: list[dict[str, int]] = []
        sku_entries: list[dict[str, Any]] = []
        order_entries: list[list[dict[str, int]]] = []

        # ---- Parse header until ARTICLE_SECTION
        idx = 0
        while idx < len(lines) and not lines[idx].startswith("ARTICLE_SECTION"):
            if ":" in lines[idx]:
                k, v = lines[idx].split(":", 1)
                header[k.strip().upper()] = v.strip()
            idx += 1

        # Validate required header fields
        required = [
            "NUM_AISLES", "NUM_CELLS", "DEPOT_AISLE", "DEPOT_LOCATION",
            "DISTANCE_AISLE_TO_AISLE", "DISTANCE_CELL_TO_CELL",
            "DISTANCE_TOP_TO_CELL", "DISTANCE_BOTTOM_TO_CELL",
            "DISTANCE_TOP_OR_BOTTOM_TO_DEPOT"
        ]
        missing = [k for k in required if k not in header]
        if missing:
            raise ValueError(f"Missing required header keys: {missing}")

        idx += 1  # skip ARTICLE_SECTION line

        # ---- Parse ARTICLE_SECTION until SKU_SECTION
        while idx < len(lines) and not lines[idx].startswith("SKU_SECTION"):
            if lines[idx].startswith("ID"):
                parts = lines[idx].split()
                if len(parts) == 2:
                    articles.append({"article_id": int(parts[1])})
                else:
                    articles.append({"article_id": int(parts[1]), "weight": int(parts[3])})
            idx += 1

        idx += 1  # skip SKU_SECTION line

        # ---- Parse SKU_SECTION until ORDER_SECTION
        # NOTE: File uses CELL indexed from TOP=0..C-1; we invert to bottom=1..C
        C = int(header["NUM_CELLS"])
        while idx < len(lines) and not lines[idx].startswith("ORDER_SECTION"):
            if lines[idx].startswith("ID"):
                parts = lines[idx].split()
                sku_entries.append({
                    "article_id": int(parts[1]),
                    "aisle": int(parts[3]),  # 0-based; will convert to 1-based later
                    "cell": C - int(parts[5]),  # invert top->bottom
                    "quantity": int(parts[7]),
                    "side": parts[-1]
                })
            idx += 1

        idx += 1  # skip ORDER_SECTION line

        # ---- Parse ORDER_SECTION until EOF
        current_order: list[dict[str, int]] = []
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("NUM_ARTICLES_IN_ORDER"):
                if current_order:
                    order_entries.append(current_order)
                    current_order = []
            elif line.startswith("ID"):
                parts = line.split()
                current_order.append({
                    "article_id": int(parts[1]),
                    "amount": int(parts[3])
                })
            idx += 1

        if current_order:
            order_entries.append(current_order)

        return {
            "header": header,
            "articles": articles,
            "skus": sku_entries,
            "orders": order_entries,
            "order_arrival_times": order_arrival_times
        }

    def _build(self, parsed: dict[str, Any]) -> BaseWarehouseDomain:
        """
        Build BaseWarehouseDomain from parsed data, with optional mirroring for top depot.

        Args:
            parsed: Dictionary from _parse() containing header, articles, skus, orders

        Returns:
            BaseWarehouseDomain instance
        """
        from ware_ops_algos.generators import ShelfStorageGraphGenerator

        header = parsed["header"]
        order_arrival_times = parsed["order_arrival_times"]

        # Basic parameters
        n_aisles = int(header["NUM_AISLES"])
        n_cells = int(header["NUM_CELLS"])  # == C
        depot_location = header["DEPOT_LOCATION"].lower()  # "top" / "bottom"

        if depot_location not in {"top", "bottom"}:
            raise ValueError(f"DEPOT_LOCATION must be 'top' or 'bottom', got {depot_location!r}")

        # Aisle indexing: file aisles are 0-based; internal aisles are 1-based
        depot_aisle = int(header["DEPOT_AISLE"]) + 1

        # Distances
        dist_aisle = float(header["DISTANCE_AISLE_TO_AISLE"])
        dist_cell = float(header["DISTANCE_CELL_TO_CELL"])
        dist_top_stub = float(header["DISTANCE_TOP_TO_CELL"])
        dist_bottom_stub = float(header["DISTANCE_BOTTOM_TO_CELL"])
        dist_to_depot = float(header["DISTANCE_TOP_OR_BOTTOM_TO_DEPOT"])

        # Start/end + connection points (before optional mirroring)
        if depot_location == "bottom":
            start_location = (depot_aisle, -1)
            end_location = (depot_aisle - 1, -1)
            start_conn = (depot_aisle, 0)
            end_conn = (depot_aisle, 0)
            closest_node_to_start = (depot_aisle, 0)
        else:
            start_location = (depot_aisle, n_cells + 1)
            end_location = (depot_aisle - 1, n_cells + 1)
            start_conn = (depot_aisle, n_cells)
            end_conn = (depot_aisle, n_cells)
            closest_node_to_start = (depot_aisle, n_cells)

        # Optional mirroring step: convert top depot -> bottom depot internally
        mirrored = False
        if depot_location == "top" and self.mirror_top_depot:
            mirrored = True

            # Swap stub distances so costs remain identical under reflection
            dist_top_stub, dist_bottom_stub = dist_bottom_stub, dist_top_stub

            # Treat as bottom internally from here on
            depot_location = "bottom"

            # Force canonical bottom-depot coordinates
            start_location = (depot_aisle, -1)
            end_location = (depot_aisle - 1, -1)
            start_conn = (depot_aisle, 0)
            end_conn = (depot_aisle, 0)
            closest_node_to_start = (depot_aisle, 0)

        # Layout parameters
        layout_params = LayoutParameters(
            n_aisles=n_aisles,
            n_pick_locations=n_cells,
            dist_pick_locations=dist_cell,
            dist_aisle=dist_aisle,
            dist_top_to_pick_location=dist_top_stub,
            dist_bottom_to_pick_location=dist_bottom_stub,
            dist_start=dist_to_depot,
            dist_end=dist_to_depot,
            start_location=start_location,
            end_location=end_location,
            start_connection_point=start_conn,
            end_connection_point=end_conn,
            n_blocks=1,
            depot_location=depot_location,
        )

        # Build graph
        min_aisle_position = 0
        max_aisle_position = layout_params.n_pick_locations + 1

        graph_generator = ShelfStorageGraphGenerator(
            n_aisles=layout_params.n_aisles,
            n_pick_locations=layout_params.n_pick_locations,
            dist_aisle=layout_params.dist_aisle,
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

        # Articles
        if "weight" in parsed["articles"][0].keys():
            article_list = [Article(article_id=a["article_id"], weight=a["weight"]) for a in parsed["articles"]]
        else:
            article_list = [Article(article_id=a["article_id"]) for a in parsed["articles"]]
        articles = Articles(tpe=ArticleType.STANDARD, articles=article_list)

        # Storage (cell inversion already done in parser; apply global mirror if enabled)
        storage_type = (
            StorageType.DEDICATED
            if len(parsed["articles"]) == len(parsed["skus"])
            else StorageType.SCATTERED
        )
        storage_raw = StorageLocations(
            tpe=storage_type,
            locations=[
                Location(
                    x=sku["aisle"] + 1,  # convert to 1-based aisles
                    y=sku["cell"],  # already inverted to bottom=1..C in parser
                    article_id=sku["article_id"],
                    amount=sku["quantity"],
                )
                for sku in parsed["skus"]
            ],
        )
        storage_raw.build_article_location_mapping()

        # Apply mirroring if needed
        storage = (
            self._mirror_storage_locations(storage_raw, n_cells)
            if mirrored
            else storage_raw
        )

        # Orders
        order_list = [
            Order(
                order_id=i,
                order_date=order_arrival_times[i],
                order_positions=[
                    OrderPosition(
                        order_number=i,
                        article_id=pos["article_id"],
                        amount=pos["amount"]
                    )
                    for pos in order_positions
                ],
            )
            for i, order_positions in enumerate(parsed["orders"])
        ]
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=order_list)

        # Resources
        if "PICKER_CAPACITY" in header:
            pick_cart = PickCart(n_dimension=1,
                                 n_boxes=1,
                                 capacities=[int(header["PICKER_CAPACITY"])],
                                 dimensions=[DimensionType.WEIGHT],
                                 box_can_mix_orders=True)
            resources_list = [
                Resource(id=0,
                         capacity=int(header["PICKER_CAPACITY"]),
                         speed=1,
                         time_per_pick=10,
                         pick_cart=pick_cart,
                         tour_setup_time=180,
                         current_location=start_location),
            ]
        else:
            resources_list = [Resource(id=0)]
        resources = Resources(ResourceType.HUMAN, resources_list)

        # Problem class
        problem_class = (
            "SPRP"
            if header.get("TYPE") == "Single_picker_routing"
               or header.get("TYPE") == "Single_picker_routing_with_scattered_storage"
            else "OBRP"
        )

        warehouse_info = WarehouseInfo(tpe=WarehouseInfoType.OFFLINE)

        return BaseWarehouseDomain(
            problem_class=problem_class,
            objective="Distance",
            layout=layout,
            articles=articles,
            orders=orders,
            resources=resources,
            storage=storage,
            warehouse_info=warehouse_info
        )

    @staticmethod
    def _mirror_y(y: int, C: int) -> int:
        """
        Mirror y-coordinate across the horizontal midline.

        - Cross-aisles: 0 <-> C+1
        - Pick cells: 1..C -> (C+1) - y

        Args:
            y: Original y-coordinate
            C: Number of pick cells

        Returns:
            Mirrored y-coordinate
        """
        if y == 0:
            return C + 1
        if y == C + 1:
            return 0
        return (C + 1) - y

    def _mirror_storage_locations(
            self,
            storage: StorageLocations,
            C: int
    ) -> StorageLocations:
        """
        Mirror storage locations vertically.

        Args:
            storage: Original storage locations
            C: Number of pick cells

        Returns:
            New StorageLocations with mirrored y-coordinates
        """
        mirrored = StorageLocations(
            tpe=storage.tpe,
            locations=[
                Location(
                    x=loc.x,
                    y=self._mirror_y(loc.y, C),
                    article_id=loc.article_id,
                    amount=loc.amount
                )
                for loc in storage.locations
            ]
        )
        mirrored.build_article_location_mapping()
        return mirrored
