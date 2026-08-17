from ware_ops_algos.algorithms import Job, BatchObject


def parse_sim_time(time_str: str) -> float:
    """Convert HH:MM:SS string to seconds since midnight (simulation float time)."""
    h, m, s = map(int, str(time_str).split(":"))
    return float(h * 3600 + m * 60 + s)


from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall

from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import (
    Article,
    ArticleType,
    Articles,
    BoundingBox,
    CobotPicker,
    DimensionType,
    LayoutData,
    LayoutNetwork,
    LayoutParameters,
    LayoutType,
    Location,
    ManualPicker,
    OrdersDomain,
    OrderType,
    PickCart,
    ResourceType,
    Resources,
    StorageLocation,
    StorageLocations,
    StorageSlot,
    StorageType,
    WarehouseInfoType, OrderPosition, Order,
)

from casim.domain_objects.sim_domain import DynamicInfo, SimWarehouseDomain


class GroceryRetailerLoader(DataLoader):
    """
    Loads the grocery retailer warehouse domain.
    """

    CCG1_WIDTH: int = 800
    CCG1_LENGTH: int = 1200
    WIDTH_AISLE_SEGMENT: int = 800
    LENGTH_AISLE_SEGMENT: int = 12600
    AISLE_KEYS: list[int] = [-1, -2, -3, -4, -5, -6, -7, -8]

    def __init__(
        self,
        instances_dir: str | Path,
        cache_dir: str | Path = None,
        cfg=None,
    ):
        super().__init__(instances_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cfg = cfg
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)


    def load(
        self,
        orders_path: str,
        layout_path: str,
        use_cache: bool = True,
        **kwargs,
    ) -> SimWarehouseDomain:
        cache_key = "grocery_retailer"

        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}_domain.pkl"
            if cache_path.exists():
                from ware_ops_algos.utils.io_helpers import load_pickle
                return load_pickle(str(cache_path))

        orders, layout_raw = self._parse(orders_path, layout_path)
        domain = self._build_domain(orders, layout_raw)

        if use_cache and self.cache_dir:
            from ware_ops_algos.utils.io_helpers import dump_pickle
            dump_pickle(str(self.cache_dir / f"{cache_key}_domain.pkl"), domain)

        return domain


    def _parse(self, orders_path: str, layout_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        orders = pd.read_csv(orders_path, sep=";")
        layout_raw = pd.read_csv(layout_path, sep=";")

        orders["ARTIKELNR"] = orders["ARTIKELNR"].astype(str).str[:-4]
        orders["BEGINN_ZEIT"] = pd.to_datetime(orders["BEGINN_ZEIT"], format="%H:%M:%S").dt.time
        orders["ENDE_ZEIT"] = pd.to_datetime(orders["ENDE_ZEIT"], format="%H:%M:%S").dt.time
        orders["hour"] = orders["BEGINN_ZEIT"].apply(lambda t: t.hour)
        orders["GEWICHT_SOLL"] = orders["GEWICHT_SOLL"].str.replace(",", ".").astype(float)
        orders["MENGE_IST"] = orders["MENGE_IST"].astype(float)
        orders["VOLUMEN_KOLLI"] = orders["VOLUMEN_KOLLI"].str.replace(",", ".").astype(float)
        orders = orders[orders["NEU_DATUM"] == "30.08.2024"]
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
            buffered_pick_lists=None,
            done=False,
            n_staged_pallets=0
        )

        return SimWarehouseDomain(
            problem_class=self.cfg.data_card.problem_type,
            objective="Distance",
            layout=layout,
            articles=articles,
            # orders=OrdersDomain(tpe=OrderType.STANDARD, orders=[]),
            resources=resources,
            storage=storage,
            orders=orders,
            dynamic_warehouse_info=dynamic_info,
        )

    # ------------------------------------------------------------------
    # Articles
    # ------------------------------------------------------------------

    def _build_articles(self, orders: pd.DataFrame) -> Articles:
        article_list = []
        seen: set = set()
        for row in orders[["ARTIKELNR", "MENGE_IST", "GEWICHT_SOLL", "VOLUMEN_KOLLI"]].itertuples(index=False):
            if row.ARTIKELNR not in seen:
                article_list.append(
                    Article(
                        article_id=row.ARTIKELNR,
                        weight=row.GEWICHT_SOLL / row.MENGE_IST if row.MENGE_IST else 0.0,
                        volume=row.VOLUMEN_KOLLI,
                    )
                )
                seen.add(row.ARTIKELNR)
        return Articles(tpe=ArticleType.STANDARD, articles=article_list)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self, layout_raw: pd.DataFrame) -> tuple[LayoutData, list[StorageLocation]]:
        layout_np = layout_raw.to_numpy()
        rbs = [
            list(range(2210, 2284)),
            list(range(2310, 2384)),
            list(range(2010, 2084)),
        ]
        rb_width = 3 * self.CCG1_WIDTH

        storage_locations = self._build_storage_locations(layout_np, rbs, rb_width)
        G, vertical_aisle_x, cross_aisle_y = self._build_graph(layout_np, storage_locations)

        start_location = (0, -10000)
        end_location = (-1, -10000)

        G.add_node(start_location, pos=start_location, type="start_node")
        G.add_node(end_location, pos=end_location, type="end_node")
        G.add_edge(start_location, (list(vertical_aisle_x.values())[0], cross_aisle_y[0]), weight=0)
        G.add_edge(end_location, (list(vertical_aisle_x.values())[0], cross_aisle_y[0]), weight=0)

        nodes = list(G.nodes())
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", dtype=float)
        dima_raw, predecessors = floyd_warshall(A, directed=False, return_predecessors=True)
        dima = pd.DataFrame(dima_raw, index=nodes, columns=nodes)

        layout_network = LayoutNetwork(
            graph=G,
            distance_matrix=dima,
            predecessor_matrix=predecessors,
            closest_node_to_start=(7500, 400),
            min_aisle_position=400,
            max_aisle_position=53600,
            start_node=start_location,
            end_node=end_location,
            node_list=nodes,
        )

        layout_params = LayoutParameters(
            n_aisles=8,
            n_pick_locations=60,
            n_blocks=2,
            dist_pick_locations=0.8,
            dist_aisle=15,
            dist_top_to_pick_location=0,
            dist_bottom_to_pick_location=0,
            dist_start=0,
            dist_end=0,
            dist_cross_aisle=2.4,
            start_location=start_location,
            end_location=end_location,
            start_connection_point=(0, 0),
            end_connection_point=(0, 0),
        )

        return LayoutData(
            tpe=LayoutType.CONVENTIONAL,
            graph_data=layout_params,
            layout_network=layout_network,
        ), storage_locations

    def _build_storage_locations(
        self,
        layout_np,
        rbs: list[list[int]],
        rb_width: int,
    ) -> list[StorageLocation]:
        storage_locations = []
        offset_horizontal = 0
        max_y = 0

        for col in range(layout_np.shape[1]):
            offset_vertical = 0
            offset_horizontal += max_y
            for row in reversed(range(layout_np.shape[0])):
                val = layout_np[row, col]
                if val >= 0 or val == -99:
                    max_y = self.CCG1_LENGTH
                    if val > 0:
                        val_left   = layout_np[row, col - 1] if col > 0 else None
                        val_right  = layout_np[row, col + 1] if col < layout_np.shape[1] - 1 else None
                        val_top    = layout_np[row - 1, col] if row > 0 else None
                        val_bottom = layout_np[row + 1, col] if row < layout_np.shape[0] - 1 else None

                        adjacent_aisle = (
                            val_left   if val_left   in self.AISLE_KEYS else
                            val_right  if val_right  in self.AISLE_KEYS else
                            val_top    if val_top    in self.AISLE_KEYS else
                            val_bottom if val_bottom in self.AISLE_KEYS else None
                        )

                        house = (
                            str(val)[:2] if len(str(val)) == 4 else
                            str(val)[:1] if len(str(val)) == 3 else
                            "0"
                        )

                        slots, bbox = None, None
                        for rb in rbs:
                            if val in rb:
                                slots = [StorageSlot(id=str(rb_id), level=0) for rb_id in rb]
                                bbox = BoundingBox(
                                    x_min=offset_horizontal,
                                    x_max=offset_horizontal + self.CCG1_LENGTH,
                                    y_min=offset_vertical,
                                    y_max=offset_vertical + rb_width,
                                )
                                break

                        if not slots:
                            val_int = int(val)
                            slots = [
                                StorageSlot(id=str(val_int), level=0),
                                StorageSlot(id=str(val_int + 1), level=1),
                            ]
                        if not bbox:
                            bbox = BoundingBox(
                                x_min=offset_horizontal,
                                x_max=offset_horizontal + self.CCG1_LENGTH,
                                y_min=offset_vertical,
                                y_max=offset_vertical + self.CCG1_WIDTH,
                            )

                        aisle = str(adjacent_aisle * -1) if adjacent_aisle else "0"
                        storage_locations.append(
                            StorageLocation(
                                id=f"{aisle}_{house}",
                                bbox=bbox,
                                aisle_id=adjacent_aisle,
                                slots=slots,
                            )
                        )
                    offset_vertical += self.CCG1_WIDTH

                elif val in self.AISLE_KEYS:
                    max_y = self.LENGTH_AISLE_SEGMENT
                    offset_vertical += self.LENGTH_AISLE_SEGMENT
                else:
                    max_y = self.LENGTH_AISLE_SEGMENT
                    offset_vertical += self.WIDTH_AISLE_SEGMENT

        return storage_locations



    def _build_graph(
        self,
        layout_np,
        storage_locations: list[StorageLocation],
    ) -> tuple[nx.Graph, dict, list]:
        G = nx.Graph()
        vertical_aisle_x: dict[int, float] = {}
        cross_aisle_y: list[float] = []

        # vertical aisle x positions
        offset_horizontal = 0
        max_y = 0
        for col in range(layout_np.shape[1]):
            offset_horizontal += max_y
            val = layout_np[layout_np.shape[0] // 2, col]
            if val in self.AISLE_KEYS and val != -90:
                vertical_aisle_x[val] = offset_horizontal + self.LENGTH_AISLE_SEGMENT / 2
                max_y = self.LENGTH_AISLE_SEGMENT
            elif val > 0 or val == -99:
                max_y = self.CCG1_LENGTH
            else:
                max_y = self.LENGTH_AISLE_SEGMENT

        # cross-aisle y positions
        offset_vertical = 0
        in_cross_aisle = False
        cross_aisle_start = 0

        for row in reversed(range(layout_np.shape[0])):
            row_vals = layout_np[row, :]
            is_cross_aisle_row = all(v == -90 for v in row_vals)

            if is_cross_aisle_row:
                if not in_cross_aisle:
                    cross_aisle_start = offset_vertical
                    in_cross_aisle = True
                offset_vertical += self.WIDTH_AISLE_SEGMENT
            else:
                if in_cross_aisle:
                    cross_aisle_y.append((cross_aisle_start + offset_vertical) / 2)
                    in_cross_aisle = False
                first_val = next((v for v in row_vals if v != -90), None)
                if first_val is not None and (first_val > 0 or first_val == -99):
                    offset_vertical += self.CCG1_WIDTH
                else:
                    offset_vertical += self.WIDTH_AISLE_SEGMENT

        if in_cross_aisle:
            cross_aisle_y.append((cross_aisle_start + offset_vertical) / 2)

        cross_aisle_y.append(offset_vertical)
        cross_aisle_y.append(28400)
        cross_aisle_y = sorted(set(cross_aisle_y))

        # intersection nodes
        for aisle_id, x in vertical_aisle_x.items():
            for y in cross_aisle_y:
                G.add_node((x, y), pos=(x, y), node_type="intersection", aisle_id=aisle_id)

        # horizontal cross-aisle edges
        x_sorted = sorted(vertical_aisle_x.values())
        for y in cross_aisle_y:
            for i in range(len(x_sorted) - 1):
                n1, n2 = (x_sorted[i], y), (x_sorted[i + 1], y)
                if G.has_node(n1) and G.has_node(n2):
                    G.add_edge(n1, n2, weight=x_sorted[i + 1] - x_sorted[i], edge_type="cross_aisle")

        # initial vertical aisle edges
        for x in vertical_aisle_x.values():
            for i in range(len(cross_aisle_y) - 1):
                G.add_edge(
                    (x, cross_aisle_y[i]),
                    (x, cross_aisle_y[i + 1]),
                    weight=cross_aisle_y[i + 1] - cross_aisle_y[i],
                    edge_type="vertical_aisle",
                )

        # attach pick nodes to storage locations
        for slot in storage_locations:
            slot_y_center = (slot.bbox.y_min + slot.bbox.y_max) / 2
            aisle_x = vertical_aisle_x.get(slot.aisle_id)
            if aisle_x is not None:
                access_node = (aisle_x, slot_y_center)
                slot.pick_node = access_node
                G.add_node(access_node, pos=access_node, type="pick_node",
                           slot_id=slot.id, aisle_id=slot.aisle_id)

        # rebuild vertical edges including pick nodes
        aisle_all_nodes: dict[float, list] = defaultdict(list)
        for aisle_id, x in vertical_aisle_x.items():
            for y in cross_aisle_y:
                aisle_all_nodes[x].append((x, y))
        for slot in storage_locations:
            if slot.pick_node is not None:
                aisle_all_nodes[slot.pick_node[0]].append(slot.pick_node)

        for aisle_x, nodes in aisle_all_nodes.items():
            for i in range(len(cross_aisle_y) - 1):
                n1 = (aisle_x, cross_aisle_y[i])
                n2 = (aisle_x, cross_aisle_y[i + 1])
                if G.has_edge(n1, n2):
                    G.remove_edge(n1, n2)
            sorted_nodes = sorted(set(nodes), key=lambda n: n[1])
            for n1, n2 in zip(sorted_nodes, sorted_nodes[1:]):
                G.add_edge(n1, n2, weight=n2[1] - n1[1], edge_type="vertical_aisle")

        return G, vertical_aisle_x, cross_aisle_y

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _build_storage(self, orders, storage_locations):
        locations = []
        seen = set()
        for row in orders.itertuples(index=False):
            article_id = row.ARTIKELNR
            if article_id in seen:
                continue
            loc_key = f"{int(row.aisle)}_{row.house}"
            place = str(row.picklocation)
            matched = False
            for loc in storage_locations:
                if loc.id != loc_key:
                    continue
                for slot in loc.slots:
                    if slot.id == place and loc.pick_node is not None:
                        locations.append(Location(
                            x=loc.pick_node[0],
                            y=loc.pick_node[1],
                            article_id=str(article_id),
                            amount=9999,
                        ))
                        matched = True
                        break
                if matched:
                    break
            seen.add(article_id)
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
        n_pickers = 24
        picker_speed = 2306
        speed_follow_mode = 1389
        time_per_pick = 19
        time_per_pick_cobot = 15.77
        tour_setup_time = 215
        aisle_congestion_rate = 0.06
        distribution = 0

        n_cobots = int(n_pickers * distribution)
        n_manual = n_pickers - n_cobots

        start_location = (0, -10000)

        def _cart() -> PickCart:
            return PickCart(n_dimension=1, n_boxes=1, capacities=[1], dimensions=[DimensionType.ORDERS])

        manual = [
            ManualPicker(
                tpe=ResourceType.HUMAN,
                id=i,
                speed=picker_speed,
                current_location=start_location,
                pick_cart=_cart(),
                time_per_pick=time_per_pick,
                tour_setup_time=tour_setup_time,
                aisle_congestion_rate=aisle_congestion_rate,
            )
            for i in range(n_manual)
        ]

        cobots = [
            CobotPicker(
                tpe=ResourceType.COBOT,
                id=i,
                speed=picker_speed,
                current_location=start_location,
                speed_follow_mode=speed_follow_mode,
                pick_cart=_cart(),
                time_per_pick=time_per_pick_cobot,
                tour_setup_time=tour_setup_time,
                aisle_congestion_rate=aisle_congestion_rate,
            )
            for i in range(n_manual, n_manual + n_cobots)
        ]

        return Resources(ResourceType.HUMAN, manual + cobots)

    def _build_orders(
            self,
            orders_df: pd.DataFrame,
    ) -> OrdersDomain:
        orders_df["date"] = pd.to_datetime(orders_df["NEU_DATUM"], format="%d.%m.%Y").dt.date
        orders_df["completion_sec"] = (
            pd.to_timedelta(orders_df["ENDE_ZEIT"].astype(str)).dt.total_seconds()
        )
        buffer_min = 15 * 60
        deadlines = (
            orders_df.groupby(["KUNDENNR", "date"])["completion_sec"].max()
            .groupby("KUNDENNR").max()
            .add(buffer_min)
            .to_dict()
        )
        order_list = []
        unique_order_ids = orders_df["AUFTRAGSNR"].unique()
        for o_id in unique_order_ids:
            subset = orders_df[orders_df["AUFTRAGSNR"] == o_id]
            cust_ids = subset["KUNDENNR"].unique()
            cust_id = cust_ids[0]
            assert len(cust_ids) == 1, f"Order {o_id} has multiple customers: {cust_ids}"
            positions = [
                OrderPosition(
                    order_number=str(o_id),
                    article_id=str(r.ARTIKELNR),
                    amount=int(r.MENGE_IST),
                )
                for r in subset.itertuples(index=False)
            ]
            order_list.append(
                Order(
                    order_id=str(o_id),
                    order_date=0,
                    due_date=deadlines[cust_id],
                    order_positions=positions,
                )
            )
        return OrdersDomain(tpe=OrderType.STANDARD, orders=order_list)


    def build_jobs_replay(self,
                          orders_df: pd.DataFrame):
        orders_df["date"] = pd.to_datetime(orders_df["NEU_DATUM"], format="%d.%m.%Y").dt.date
        orders_df["completion_sec"] = (
            pd.to_timedelta(orders_df["ENDE_ZEIT"].astype(str)).dt.total_seconds()
        )

        orders_df["start_sec"] = (
            pd.to_timedelta(orders_df["BEGINN_ZEIT"].astype(str)).dt.total_seconds()
        )
        releases = (
            orders_df.groupby(["AUFTRAGSNR", "date"])["start_sec"].min()
            .groupby("AUFTRAGSNR").min()
            .to_dict()
        )
        deadlines = (
            orders_df.groupby(["AUFTRAGSNR", "date"])["completion_sec"].max()
            .groupby("AUFTRAGSNR").max()
            .to_dict()
        )
        order_list = []
        jobs = []
        processing_times = []
        unique_order_ids = orders_df["AUFTRAGSNR"].unique()
        for i, o_id in enumerate(unique_order_ids):
            subset = orders_df[orders_df["AUFTRAGSNR"] == o_id]
            picker_id = subset["PERS_NR"].unique()
            cust_ids = subset["KUNDENNR"].unique()
            cust_id = cust_ids[0]
            assert len(cust_ids) == 1, f"Order {o_id} has multiple customers: {cust_ids}"
            positions = [
                OrderPosition(
                    order_number=str(o_id),
                    article_id=str(r.ARTIKELNR),
                    amount=int(r.MENGE_IST),
                )
                for r in subset.itertuples(index=False)
            ]
            order = Order(
                    order_id=str(o_id),
                    order_date=0,
                    due_date=deadlines[o_id],
                    order_positions=positions,
                )
            order_list.append(order)

            processing_time = deadlines[o_id] - releases[o_id]
            processing_times.append(processing_time)
            job = Job(i,
                      processing_time,
                      releases[o_id],
                      deadlines[o_id],
                      len(positions),
                      None,
                      BatchObject(batch_id=i,
                                  orders=[order]))

            jobs.append(ScheduledJob(job, picker_id[0], releases[o_id], deadlines[o_id]))

        return jobs
