from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall

from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import (
    Article, ArticleType, Articles,
    BoundingBox, DimensionType,
    LayoutData, LayoutNetwork, LayoutParameters, LayoutType,
    Location, ManualPicker,
    OrdersDomain, OrderType, Order, OrderPosition,
    PickCart, ResourceType, Resources,
    StorageLocation, StorageLocations, StorageSlot, StorageType,
    WarehouseInfoType,
)

from casim.domain_objects.sim_domain import DynamicInfo, SimWarehouseDomain
from casim.io_helpers import dump_pickle, load_pickle
from scenarios.scenario_ijpe.schema import *


class IJPELoader(DataLoader):
    """Load a grocery retailer picking domain from a canonical order stream.

    Layout abstraction:
    - each physical gang has two side rails: left and right,
    - pick nodes lie on the side rail adjacent to the storage face,
    - side changes are only possible at cross-aisle / rung levels,
    - upper/lower storage levels share the same physical pick node.
    """

    # Millimetres
    CCG1_WIDTH: int = 800
    CCG1_LENGTH: int = 1200
    WIDTH_AISLE_SEGMENT: int = 800
    LENGTH_AISLE_SEGMENT: int = 16450
    AISLE_WIDTH: int = 3850

    SIDE_LEFT: str = "L"
    SIDE_RIGHT: str = "R"

    AISLE_KEYS: list[int] = [-1, -2, -3, -4, -5, -6, -7, -8]

    RB_RANGES: list[range] = [
        range(2210, 2284),
        range(2310, 2384),
        range(2010, 2084),
    ]

    # Middle cross aisle that is not represented as a full -90 row in the CSV.
    MANUAL_CROSS_AISLE_Y: int | None = 28400

    def __init__(
        self,
        instances_dir: str | Path,
        cache_dir: str | Path = None,
        problem_class: str = "OBRSP",
        objective: str = "Distance",
        n_pickers: int = 25,
        picker_speed: float = 2306,
        time_per_pick: float = 19,
        tour_setup_time: float = 215,
        cart_bins: int = 1,
        depot_access_cost: float = 0,
        inventory_quantity: int = 9999,
    ):
        super().__init__(instances_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.problem_class = problem_class
        self.objective = objective
        self.n_pickers = n_pickers
        self.picker_speed = picker_speed
        self.time_per_pick = time_per_pick
        self.tour_setup_time = tour_setup_time
        self.cart_bins = cart_bins
        self.depot_access_cost = depot_access_cost
        self.inventory_quantity = inventory_quantity
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def load(
        self,
        orders_path: str,
        layout_path: str,
        use_cache: bool = True,
        **kwargs,
    ) -> SimWarehouseDomain:
        cache_key = (
            f"{Path(orders_path).stem}_{self.n_pickers}_"
            f"{self.problem_class}_{self.objective}"
        )

        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}_domain.pkl"
            if cache_path.exists():
                return load_pickle(str(cache_path))

        orders_df, layout_raw = self._parse(orders_path, layout_path)
        domain = self._build_domain(orders_df, layout_raw)

        if use_cache and self.cache_dir:
            dump_pickle(str(self.cache_dir / f"{cache_key}_domain.pkl"), domain)

        return domain

    def _parse(self, orders_path: str, layout_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        orders_path = Path(orders_path)
        layout_path = Path(layout_path)
        if not orders_path.is_absolute():
            orders_path = self.data_dir / orders_path
        if not layout_path.is_absolute():
            layout_path = self.data_dir / layout_path
        orders = pd.read_csv(
            orders_path,
            dtype={
                COL_ARTICLE_ID: str,
                COL_ORDER_ID: str,
            },
        )

        # Important: the layout CSV has no header.
        # The first row is a real -90 cross-aisle row and must not be consumed as column names.
        layout_raw = pd.read_csv(layout_path, sep=";", header=None)

        layout_raw = (
            layout_raw
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
        )

        return orders, layout_raw

    def _build_domain(self, orders_df: pd.DataFrame, layout_raw: pd.DataFrame) -> SimWarehouseDomain:
        articles = self._build_articles(orders_df)
        layout, storage_locations = self._build_layout(layout_raw)
        storage = self._build_storage(orders_df, storage_locations)
        resources = self._build_resources()
        orders = self._build_orders(orders_df)

        dynamic_info = DynamicInfo(
            tpe=WarehouseInfoType.OFFLINE,
            time=0.0,
            congestion_rate={},
            active_tours=[],
            current_picker=None,
            buffered_batches=None,
            done=False,
            n_staged_pallets=0,
        )

        return SimWarehouseDomain(
            problem_class=self.problem_class,
            objective=self.objective,
            layout=layout,
            articles=articles,
            resources=resources,
            storage=storage,
            orders=orders,
            dynamic_warehouse_info=dynamic_info,
        )

    # ------------------------------------------------------------------
    # Articles
    # ------------------------------------------------------------------

    def _build_articles(self, orders: pd.DataFrame) -> Articles:
        df = orders.copy()

        df["weight_per_unit"] = df[COL_LINE_WEIGHT] / df[COL_QUANTITY]
        df["volume_per_unit"] = df[COL_KOLLI_VOLUME] / df[COL_KOLLI_SIZE]

        stats = (
            df.groupby(COL_ARTICLE_ID)
            .agg(
                weight_per_unit=("weight_per_unit", "median"),
                kolli_volume=(COL_KOLLI_VOLUME, "median"),
            )
            .reset_index()
        )

        have_dims = all(
            c in df.columns
            for c in (COL_LENGTH, COL_WIDTH, COL_HEIGHT, COL_KOLLI_VOLUME)
        )

        if have_dims:
            dims = (
                df.groupby(COL_ARTICLE_ID)[
                    [COL_LENGTH, COL_WIDTH, COL_HEIGHT, "volume_per_unit", COL_KOLLI_SIZE]
                ]
                .first()
                .reset_index()
            )
            stats = stats.merge(dims, on=COL_ARTICLE_ID, how="left")

        article_list = []

        for _, row in stats.iterrows():
            article_list.append(
                Article(
                    article_id=row[COL_ARTICLE_ID],
                    weight=row["weight_per_unit"],
                    volume=row["volume_per_unit"],
                    length=(row[COL_LENGTH] if have_dims and pd.notna(row[COL_LENGTH]) else None),
                    width=(row[COL_WIDTH] if have_dims and pd.notna(row[COL_WIDTH]) else None),
                    height=(row[COL_HEIGHT] if have_dims and pd.notna(row[COL_HEIGHT]) else None),
                    kolli_size=(row[COL_KOLLI_SIZE] if have_dims and pd.notna(row[COL_KOLLI_SIZE]) else None),
                )
            )

        return Articles(tpe=ArticleType.STANDARD, articles=article_list)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self, layout_raw: pd.DataFrame) -> tuple[LayoutData, list[StorageLocation]]:
        layout_np = layout_raw.to_numpy()

        storage_locations = self._build_storage_locations(layout_np)
        G, aisle_side_x, cross_aisle_y = self._build_graph(layout_np, storage_locations)

        if not aisle_side_x:
            raise ValueError("No aisle side rails were generated. Check AISLE_KEYS and layout encoding.")

        if not cross_aisle_y:
            raise ValueError("No cross-aisle / rung y-coordinates were generated.")

        bottom_y = min(cross_aisle_y)
        top_y = max(cross_aisle_y)

        # Gang 1 is encoded as -1. Partner says picking starts on the right side.
        start_aisle_id = -1

        if (start_aisle_id, self.SIDE_RIGHT) not in aisle_side_x:
            raise ValueError("Expected Gang 1 / aisle marker -1 with a right-side rail.")

        closest_node_to_start = (
            aisle_side_x[(start_aisle_id, self.SIDE_RIGHT)],
            bottom_y,
        )

        # Keep original depot abstraction.
        start_location = (0, -10000)
        end_location = (-1, -10000)

        G.add_node(start_location, pos=start_location, node_type="start_node")
        G.add_node(end_location, pos=end_location, node_type="end_node")

        G.add_edge(start_location, closest_node_to_start, weight=self.depot_access_cost, edge_type="depot")
        G.add_edge(end_location, closest_node_to_start, weight=self.depot_access_cost, edge_type="depot")

        nodes = list(G.nodes())

        A = nx.to_scipy_sparse_array(
            G,
            nodelist=nodes,
            weight="weight",
            dtype=float,
        )

        dima_raw, predecessors = floyd_warshall(
            A,
            directed=False,
            return_predecessors=True,
        )

        dima = pd.DataFrame(dima_raw, index=nodes, columns=nodes)

        layout_network = LayoutNetwork(
            graph=G,
            distance_matrix=dima,
            predecessor_matrix=predecessors,
            closest_node_to_start=closest_node_to_start,
            min_aisle_position=bottom_y,
            max_aisle_position=top_y,
            start_node=start_location,
            end_node=end_location,
            node_list=nodes,
        )

        layout_params = LayoutParameters(
            n_aisles=len({aisle_id for aisle_id, _ in aisle_side_x}),
            n_pick_locations=len(storage_locations),
            n_blocks=max(1, len(cross_aisle_y) - 1),
            dist_pick_locations=self.CCG1_WIDTH / 1000,
            dist_aisle=self.AISLE_WIDTH / 1000,
            dist_top_to_pick_location=0,
            dist_bottom_to_pick_location=0,
            dist_start=0,
            dist_end=0,
            dist_cross_aisle=self.WIDTH_AISLE_SEGMENT / 1000,
            start_location=start_location,
            end_location=end_location,
            start_connection_point=closest_node_to_start,
            end_connection_point=closest_node_to_start,
        )

        return (
            LayoutData(
                tpe=LayoutType.CONVENTIONAL,
                graph_data=layout_params,
                layout_network=layout_network,
            ),
            storage_locations,
        )

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _is_aisle_marker(self, val: int) -> bool:
        return val in self.AISLE_KEYS

    def _is_cross_aisle_marker(self, val: int) -> bool:
        return val == -90

    def _is_special_pick_marker(self, val: int) -> bool:
        return val == -99

    def _is_storage_value(self, val: int) -> bool:
        return val > 0

    def _is_storage_or_special(self, val: int) -> bool:
        return self._is_storage_value(val) or self._is_special_pick_marker(val)

    def _column_width(self, col_vals) -> int:
        if any(self._is_aisle_marker(int(v)) for v in col_vals):
            return self.LENGTH_AISLE_SEGMENT

        if any(self._is_storage_or_special(int(v)) for v in col_vals):
            return self.CCG1_LENGTH

        return self.WIDTH_AISLE_SEGMENT

    def _column_offsets(self, layout_np) -> dict[int, int]:
        offsets = {}
        x = 0

        for col in range(layout_np.shape[1]):
            offsets[col] = x
            x += self._column_width(layout_np[:, col])

        return offsets

    def _row_y_centers(self, layout_np) -> dict[int, float]:
        y_centers = {}
        y = 0

        for row in reversed(range(layout_np.shape[0])):
            y_centers[row] = y + self.CCG1_WIDTH / 2
            y += self.CCG1_WIDTH

        return y_centers

    def _total_layout_height(self, layout_np) -> int:
        return layout_np.shape[0] * self.CCG1_WIDTH

    def _compute_aisle_center_x(self, layout_np) -> dict[int, float]:
        col_offsets = self._column_offsets(layout_np)
        aisle_center_x: dict[int, float] = {}

        for col in range(layout_np.shape[1]):
            col_vals = [int(v) for v in layout_np[:, col]]
            aisle_ids = [v for v in col_vals if self._is_aisle_marker(v)]

            if not aisle_ids:
                continue

            aisle_id = max(set(aisle_ids), key=aisle_ids.count)

            x_left = col_offsets[col]
            x_center = x_left + self._column_width(layout_np[:, col]) / 2

            aisle_center_x[aisle_id] = x_center

        return aisle_center_x

    def _compute_cross_aisle_y(self, layout_np) -> list[float]:
        """Return y-levels where side switching is allowed.

        KISS rule:
        - full -90 row = real cross aisle,
        - manual middle level = additional cross aisle if configured.
        """
        y_centers = self._row_y_centers(layout_np)
        cross_aisle_y: list[float] = []

        for row in range(layout_np.shape[0]):
            row_vals = [int(v) for v in layout_np[row, :]]

            if all(self._is_cross_aisle_marker(v) for v in row_vals):
                cross_aisle_y.append(y_centers[row])

        total_height = self._total_layout_height(layout_np)

        if self.MANUAL_CROSS_AISLE_Y is not None:
            if 0 < self.MANUAL_CROSS_AISLE_Y < total_height:
                cross_aisle_y.append(float(self.MANUAL_CROSS_AISLE_Y))

        return sorted(set(cross_aisle_y))

    def _adjacent_aisle_and_side(self, layout_np, row: int, col: int) -> tuple[int | None, str | None]:
        val_left = int(layout_np[row, col - 1]) if col > 0 else None
        val_right = int(layout_np[row, col + 1]) if col < layout_np.shape[1] - 1 else None

        if val_left in self.AISLE_KEYS:
            return val_left, self.SIDE_RIGHT

        if val_right in self.AISLE_KEYS:
            return val_right, self.SIDE_LEFT

        return None, None

    def _base_slot_id(self, place: int) -> int:
        last_digit = place % 10

        if last_digit in (1, 3, 5):
            return place - 1

        return place

    def _slots_for_place(self, place: int) -> list[StorageSlot]:
        base = self._base_slot_id(place)
        last_digit = base % 10

        if last_digit in (0, 2, 4):
            return [
                StorageSlot(id=str(base), level=0),
                StorageSlot(id=str(base + 1), level=1),
            ]

        return [StorageSlot(id=str(place), level=0)]

    def _house_from_place(self, place: int) -> str:
        place_str = str(place)

        if len(place_str) == 4:
            return place_str[:2]

        if len(place_str) == 3:
            return place_str[:1]

        return "0"

    def _rb_range_for_place(self, place: int) -> range | None:
        for rb_range in self.RB_RANGES:
            if place in rb_range:
                return rb_range

        return None

    # ------------------------------------------------------------------
    # Storage-location generation
    # ------------------------------------------------------------------

    def _build_storage_locations(self, layout_np) -> list[StorageLocation]:
        storage_locations: list[StorageLocation] = []

        col_offsets = self._column_offsets(layout_np)
        y_centers = self._row_y_centers(layout_np)

        seen_physical_locations: set[tuple] = set()
        seen_rb_locations: set[tuple] = set()

        for col in range(layout_np.shape[1]):
            for row in reversed(range(layout_np.shape[0])):
                val = int(layout_np[row, col])

                if not self._is_storage_value(val):
                    continue

                adjacent_aisle, side = self._adjacent_aisle_and_side(layout_np, row, col)

                if adjacent_aisle is None or side is None:
                    continue

                base_place = self._base_slot_id(val)
                rb_range = self._rb_range_for_place(base_place)

                if rb_range is not None:
                    rb_key = (adjacent_aisle, side, rb_range.start, rb_range.stop)

                    if rb_key in seen_rb_locations:
                        continue

                    seen_rb_locations.add(rb_key)

                    slots = [StorageSlot(id=str(rb_id), level=0) for rb_id in rb_range]

                    y_center = y_centers[row]
                    rb_width = 3 * self.CCG1_WIDTH

                    bbox = BoundingBox(
                        x_min=col_offsets[col],
                        x_max=col_offsets[col] + self.CCG1_LENGTH,
                        y_min=y_center - rb_width / 2,
                        y_max=y_center + rb_width / 2,
                    )

                    location_id_place = rb_range.start

                else:
                    physical_key = (adjacent_aisle, side, base_place)

                    if physical_key in seen_physical_locations:
                        continue

                    seen_physical_locations.add(physical_key)

                    slots = self._slots_for_place(base_place)

                    y_center = y_centers[row]

                    bbox = BoundingBox(
                        x_min=col_offsets[col],
                        x_max=col_offsets[col] + self.CCG1_LENGTH,
                        y_min=y_center - self.CCG1_WIDTH / 2,
                        y_max=y_center + self.CCG1_WIDTH / 2,
                    )

                    location_id_place = base_place

                aisle = str(adjacent_aisle * -1)
                house = self._house_from_place(location_id_place)

                loc = StorageLocation(
                    id=f"{aisle}_{house}",
                    bbox=bbox,
                    aisle_id=adjacent_aisle,
                    slots=slots,
                )

                loc.side = side
                loc.pick_node = None

                storage_locations.append(loc)

        return storage_locations

    # ------------------------------------------------------------------
    # Graph generation
    # ------------------------------------------------------------------

    def _build_graph(self, layout_np, storage_locations: list[StorageLocation]):
        G = nx.Graph()

        aisle_center_x = self._compute_aisle_center_x(layout_np)
        cross_aisle_y = self._compute_cross_aisle_y(layout_np)

        aisle_side_x: dict[tuple[int, str], float] = {}

        for aisle_id, x_center in aisle_center_x.items():
            aisle_side_x[(aisle_id, self.SIDE_LEFT)] = x_center - self.AISLE_WIDTH / 2
            aisle_side_x[(aisle_id, self.SIDE_RIGHT)] = x_center + self.AISLE_WIDTH / 2

        # 1) Create rail nodes at cross-aisle / rung positions.
        for (aisle_id, side), x in aisle_side_x.items():
            for y in cross_aisle_y:
                G.add_node(
                    (x, y),
                    pos=(x, y),
                    node_type="intersection",
                    aisle_id=aisle_id,
                    side=side,
                )

        # 2) Connect left/right side of the same physical gang at rung levels.
        for aisle_id in aisle_center_x:
            x_left = aisle_side_x[(aisle_id, self.SIDE_LEFT)]
            x_right = aisle_side_x[(aisle_id, self.SIDE_RIGHT)]

            for y in cross_aisle_y:
                G.add_edge(
                    (x_left, y),
                    (x_right, y),
                    weight=self.AISLE_WIDTH,
                    edge_type="rung",
                )

        # 3) Connect neighboring rails horizontally along cross aisles.
        for y in cross_aisle_y:
            row_nodes = sorted(
                [(x, y) for x in aisle_side_x.values()],
                key=lambda n: n[0],
            )

            for n1, n2 in zip(row_nodes, row_nodes[1:]):
                # Do not overwrite rung edges.
                if G.has_edge(n1, n2):
                    continue

                G.add_edge(
                    n1,
                    n2,
                    weight=abs(n2[0] - n1[0]),
                    edge_type="cross_aisle",
                )

        # 4) Add pick nodes on the correct side rail.
        for loc in storage_locations:
            rail_x = aisle_side_x.get((loc.aisle_id, loc.side))

            if rail_x is None:
                continue

            pick_y = (loc.bbox.y_min + loc.bbox.y_max) / 2
            pick_node = (rail_x, pick_y)

            loc.pick_node = pick_node

            G.add_node(
                pick_node,
                pos=pick_node,
                node_type="pick_node",
                slot_id=loc.id,
                aisle_id=loc.aisle_id,
                side=loc.side,
            )

        # 5) Connect each side rail vertically, including pick nodes.
        rail_nodes: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)

        for node, data in G.nodes(data=True):
            aisle_id = data.get("aisle_id")
            side = data.get("side")

            if aisle_id is None or side is None:
                continue

            rail_nodes[(aisle_id, side)].append(node)

        for nodes in rail_nodes.values():
            sorted_nodes = sorted(set(nodes), key=lambda n: n[1])

            for n1, n2 in zip(sorted_nodes, sorted_nodes[1:]):
                if n1 == n2:
                    continue

                G.add_edge(
                    n1,
                    n2,
                    weight=abs(n2[1] - n1[1]),
                    edge_type="vertical_aisle",
                )

        return G, aisle_side_x, cross_aisle_y

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _build_storage(
        self,
        orders: pd.DataFrame,
        storage_locations: list[StorageLocation],
    ) -> StorageLocations:
        locations = []
        seen_articles = set()

        for row in orders.itertuples(index=False):
            article_id = getattr(row, COL_ARTICLE_ID)

            if article_id in seen_articles:
                continue

            aisle = int(getattr(row, COL_AISLE))
            house = int(getattr(row, COL_HOUSE))
            place = str(int(getattr(row, COL_PICKLOCATION)))

            loc_key = f"{aisle}_{house}"

            matched = False

            for loc in storage_locations:
                if loc.id != loc_key:
                    continue

                if loc.pick_node is None:
                    continue

                for slot in loc.slots:
                    if slot.id == place:
                        locations.append(
                            Location(
                                x=loc.pick_node[0],
                                y=loc.pick_node[1],
                                article_id=str(article_id),
                                amount=self.inventory_quantity,
                            )
                        )
                        matched = True
                        break

                if matched:
                    break

            seen_articles.add(article_id)

        storage = StorageLocations(
            tpe=StorageType.DEDICATED,
            locations=locations,
            storage_slots=storage_locations,
        )

        storage.build_article_location_mapping()

        return storage

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    def _build_resources(self) -> Resources:
        start_location = (0, -10000)

        def _cart() -> PickCart:
            return PickCart(
                n_dimension=1,
                n_boxes=self.cart_bins,
                capacities=[1],
                dimensions=[DimensionType.ORDERS],
                box_can_mix_orders=False,
            )

        pickers = [
            ManualPicker(
                tpe=ResourceType.HUMAN,
                id=i,
                speed=self.picker_speed,
                current_location=start_location,
                pick_cart=_cart(),
                time_per_pick=self.time_per_pick,
                tour_setup_time=self.tour_setup_time,
            )
            for i in range(self.n_pickers)
        ]

        return Resources(ResourceType.HUMAN, pickers)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def _build_orders(self, orders_df: pd.DataFrame) -> OrdersDomain:
        order_list = []

        for o_id in orders_df[COL_ORDER_ID].unique():
            subset = orders_df[orders_df[COL_ORDER_ID] == o_id]

            positions = [
                OrderPosition(
                    order_number=str(o_id),
                    article_id=str(getattr(r, COL_ARTICLE_ID)),
                    amount=int(getattr(r, COL_QUANTITY)),
                )
                for r in subset.itertuples(index=False)
            ]

            order_list.append(
                Order(
                    order_id=str(o_id),
                    order_date=float(subset[COL_ORDER_DATE].iloc[0]),
                    due_date=float(subset[COL_DUE_DATE].iloc[0]),
                    order_positions=positions,
                )
            )

        return OrdersDomain(tpe=OrderType.STANDARD, orders=order_list)
