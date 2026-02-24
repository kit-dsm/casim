from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall

from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import Articles, ArticleType, Article, OrderType, OrdersDomain, \
    Order, OrderPosition, LayoutParameters, Location, StorageLocations, StorageType, LayoutData, LayoutType, \
    LayoutNetwork, PickCart, DimensionType, Resource, Resources, ResourceType, StorageSlot, BoundingBox, \
    StorageLocation, WarehouseInfoType, CobotPicker, ManualPicker

from ware_ops_sim.sim.sim_domain import SimWarehouseDomain, DynamicInfo

class IWSPELoader(DataLoader):
    def __init__(self,
                 instances_dir: str | Path,
                 cache_dir: str | Path = None,
                 cfg=None):
        super().__init__(instances_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cfg = cfg
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = None

    def _parse(self, orders_path: str, layout_path: str):
        # orders = pd.read_excel(orders_path,
        #                        sheet_name="Clean OP Data"
        #                        )
        orders = pd.read_csv(orders_path, sep=";")

        # layout_raw = pd.read_excel(layout_path)
        layout_raw = pd.read_csv(layout_path, sep=";")

        orders['ARTIKELNR'] = orders['ARTIKELNR'].astype(str).str[:-4]
        orders["BEGINN_ZEIT"] = pd.to_datetime(orders["BEGINN_ZEIT"], format="%H:%M:%S").dt.time
        orders["ENDE_ZEIT"] = pd.to_datetime(orders["ENDE_ZEIT"], format="%H:%M:%S").dt.time
        orders["hour"] = orders["BEGINN_ZEIT"].apply(lambda t: t.hour)
        orders = orders[(orders["NEU_DATUM"] == "30.08.2024")] #  & (orders["hour"] == 7)
        orders["GEWICHT_SOLL"] = orders["GEWICHT_SOLL"].str.replace(",", ".").astype(float)
        orders["MENGE_IST"] = orders["MENGE_IST"].astype(float)
        orders["VOLUMEN_KOLLI"] = orders["VOLUMEN_KOLLI"].str.replace(",", ".").astype(float)
        # orders = orders[:10]
        master_data = orders[["ARTIKELNR", "MENGE_IST", "GEWICHT_SOLL", "VOLUMEN_KOLLI", "LHM_TYP"]]
        return orders, master_data, layout_raw

    def _build(self, orders: pd.DataFrame, master_data: pd.DataFrame, layout_raw: pd.DataFrame):
        article_list = []
        articles_seen = []
        for row in master_data.itertuples(index=False):
            if row.ARTIKELNR not in articles_seen:
                article_list.append(
                    Article(article_id=row.ARTIKELNR,
                            weight=row.GEWICHT_SOLL / row.MENGE_IST,
                            volume=row.VOLUMEN_KOLLI)
                )
                articles_seen.append(row.ARTIKELNR)
        articles = Articles(tpe=ArticleType.STANDARD, articles=article_list)

        order_list = []

        unique_order_ids = orders["AUFTRAGSNR"].unique()

        # dummy_ids = [2408300072, 2408300072]
        for o_id in unique_order_ids:
            subset = orders[orders["AUFTRAGSNR"] == o_id]
            positions = [
                OrderPosition(
                    order_number=int(o_id),
                    article_id=r.ARTIKELNR,
                    amount=int(r.MENGE_IST),
                )
                for r in subset.itertuples(index=False)
            ]
            order_date = subset["NEU_DATUM"].unique()[0]
            start_time = subset["BEGINN_ZEIT"].unique()[0]
            # combined = datetime.combine(order_date.date(), start_time)

            # timestamp_float = combined.timestamp()
            timestamp_float = 0
            end = timestamp_float + (2 * 3600)
            order_list.append(
                Order(
                    order_id=int(o_id),
                    order_date=timestamp_float,
                    due_date=end,
                    order_positions=positions,
                )
            )

        order_domain = OrdersDomain(tpe=OrderType.STANDARD, orders=order_list)

        layout_np = layout_raw.to_numpy()

        rollenbahn_1 = [i for i in range(2210, 2283 + 1)]
        rollenbahn_2 = [i for i in range(2310, 2383 + 1)]
        rollenbahn_3 = [i for i in range(2010, 2083 + 1)]

        ccg1_width = 800
        ccg1_length = 1200

        rb_width = 3 * ccg1_width
        rbs = [rollenbahn_1, rollenbahn_2, rollenbahn_3]
        width_aisle_segment = 800
        length_aisle_segment = 12600

        aisle_keys = [-1, -2, -3, -4, -5, -6, -7, -8]

        storage_locations = []

        offset_horizontal = 0
        max_y = 0

        for col in range(layout_np.shape[1]):
            offset_vertical = 0
            offset_horizontal += max_y
            for row in reversed(range(layout_np.shape[0])):
                val = layout_np[row, col]
                if val >= 0 or val == -99:
                    max_y = ccg1_length
                    if val > 0:
                        val_left = layout_np[row, col - 1] if col > 0 else None
                        val_right = layout_np[row, col + 1] if col < layout_np.shape[1] - 1 else None
                        val_top = layout_np[row - 1, col] if row > 0 else None
                        val_bottom = layout_np[row + 1, col] if row < layout_np.shape[0] - 1 else None

                        # determine which aisle this slot belongs to
                        adjacent_aisle = val_left if val_left in aisle_keys else \
                            val_right if val_right in aisle_keys else \
                                val_top if val_top in aisle_keys else \
                                    val_bottom if val_bottom in aisle_keys else None

                        if len(str(val)) == 4:
                            house = str(val)[:2]
                        elif len(str(val)) == 3:
                            house = str(val)[:1]
                        else:
                            house = str(0)
                        slots = None
                        bbox = None
                        for rb in rbs:
                            if val in rb:
                                slots = [StorageSlot(id=str(rb_id), level=0) for rb_id in rb]
                                bbox = BoundingBox(
                                    x_min=offset_horizontal,
                                    x_max=offset_horizontal + ccg1_length,
                                    y_min=offset_vertical,
                                    y_max=offset_vertical + rb_width,
                                )

                        if not slots:
                            slot_bottom = StorageSlot(id=str(val),
                                                      level=0)
                            val_int = int(val)
                            val_top = val_int + 1

                            slot_top = StorageSlot(id=str(val_top),
                                                   level=1)
                            slots = [slot_top, slot_bottom]

                        if not bbox:
                            bbox = BoundingBox(
                                x_min=offset_horizontal,
                                x_max=offset_horizontal + ccg1_length,
                                y_min=offset_vertical,
                                y_max=offset_vertical + ccg1_width,
                            )

                        aisle = str(adjacent_aisle * -1)
                        storage_loc = StorageLocation(
                            id=aisle + "_" + house,
                            bbox=bbox,
                            aisle_id=adjacent_aisle,
                            slots=slots
                        )

                        storage_locations.append(storage_loc)

                    offset_vertical += ccg1_width

                elif val in aisle_keys:
                    max_y = length_aisle_segment
                    offset_vertical += length_aisle_segment
                else:
                    max_y = length_aisle_segment
                    offset_vertical += width_aisle_segment

        G = nx.Graph()

        vertical_aisle_x = {}
        cross_aisle_y = []

        # Find vertical aisle x positions
        offset_horizontal = 0
        max_y = 0

        for col in range(layout_np.shape[1]):
            offset_horizontal += max_y
            val = layout_np[layout_np.shape[0] // 2, col]

            if val in aisle_keys and val != -90:
                vertical_aisle_x[val] = offset_horizontal + length_aisle_segment / 2
                max_y = length_aisle_segment
            elif val > 0 or val == -99:
                max_y = ccg1_length
            else:
                max_y = length_aisle_segment

        # Find cross-aisle y positions by scanning rows
        offset_vertical = 0
        in_cross_aisle = False
        cross_aisle_start = 0

        for row in reversed(range(layout_np.shape[0])):
            # Check if this entire row is a cross-aisle (-90)
            row_vals = layout_np[row, :]
            is_cross_aisle_row = all(v == -90 for v in row_vals)

            if is_cross_aisle_row:
                if not in_cross_aisle:
                    cross_aisle_start = offset_vertical
                    in_cross_aisle = True
                offset_vertical += width_aisle_segment
            else:
                if in_cross_aisle:
                    cross_aisle_y.append((cross_aisle_start + offset_vertical) / 2)
                    in_cross_aisle = False

                # Determine row height from first non-90 value
                first_val = next((v for v in row_vals if v != -90), None)
                if first_val is not None and (first_val > 0 or first_val == -99):
                    offset_vertical += ccg1_width
                else:
                    offset_vertical += width_aisle_segment

        if in_cross_aisle:
            cross_aisle_y.append((cross_aisle_start + offset_vertical) / 2)

        # Add top entry
        top_y = offset_vertical
        cross_aisle_y.append(top_y)
        cross_aisle_y.append(28400)

        cross_aisle_y = sorted(set(cross_aisle_y))

        # Create intersection nodes
        for aisle_id, x in vertical_aisle_x.items():
            for y in cross_aisle_y:
                G.add_node((x, y), pos=(x, y), node_type='intersection', aisle_id=aisle_id)


        # Connect ALL cross-aisles horizontally
        x_sorted = sorted(vertical_aisle_x.values())
        for y in cross_aisle_y:
            for i in range(len(x_sorted) - 1):
                n1 = (x_sorted[i], y)
                n2 = (x_sorted[i + 1], y)
                if G.has_node(n1) and G.has_node(n2):
                    G.add_edge(n1, n2, weight=x_sorted[i + 1] - x_sorted[i], edge_type='cross_aisle')

        # Connect vertical aisles
        for x in vertical_aisle_x.values():
            for i in range(len(cross_aisle_y) - 1):
                G.add_edge((x, cross_aisle_y[i]), (x, cross_aisle_y[i + 1]),
                           weight=cross_aisle_y[i + 1] - cross_aisle_y[i], edge_type='vertical_aisle')

        for slot in storage_locations:
            slot_y_center = (slot.bbox.y_min + slot.bbox.y_max) / 2
            aisle_x = vertical_aisle_x.get(slot.aisle_id)

            if aisle_x is not None:
                access_node = (aisle_x, slot_y_center)
                slot.pick_node = access_node
                G.add_node(access_node, pos=access_node, type='pick_node', slot_id=slot.id,
                           aisle_id=slot.aisle_id)

        # Second pass: collect all nodes per aisle (cross-aisle nodes + pick nodes)
        from collections import defaultdict

        aisle_all_nodes = defaultdict(list)

        # Add cross-aisle intersection nodes
        for aisle_id, x in vertical_aisle_x.items():
            for y in cross_aisle_y:
                aisle_all_nodes[x].append((x, y))

        # Add pick nodes
        for slot in storage_locations:
            if slot.pick_node is not None:
                aisle_x = slot.pick_node[0]
                aisle_all_nodes[aisle_x].append(slot.pick_node)

        # Remove old vertical aisle edges and rebuild with pick nodes included
        for aisle_x, nodes in aisle_all_nodes.items():
            # Remove existing vertical edges in this aisle
            for i in range(len(cross_aisle_y) - 1):
                n1 = (aisle_x, cross_aisle_y[i])
                n2 = (aisle_x, cross_aisle_y[i + 1])
                if G.has_edge(n1, n2):
                    G.remove_edge(n1, n2)

            # Sort all nodes by y and connect consecutive ones
            sorted_nodes = sorted(set(nodes), key=lambda n: n[1])

            for i in range(len(sorted_nodes) - 1):
                n1 = sorted_nodes[i]
                n2 = sorted_nodes[i + 1]
                weight = n2[1] - n1[1]
                G.add_edge(n1, n2, weight=weight, edge_type='vertical_aisle')

        depot_node = 0
        depot_aisle = 0
        start_location = (depot_aisle, -10000)
        end_location = (depot_aisle - 1, -10000)
        closest_node_to_start = (start_location[0], start_location[1] + 1)

        G.add_node(start_location, pos=start_location, type='start_node')
        G.add_node(end_location, pos=end_location, type='end_node')

        # for x in vertical_aisle_x.values():
        #     G.add_edge(start_location, (x, cross_aisle_y[0]),
        #     weight=0)
        #     G.add_edge(end_location, (x, cross_aisle_y[0]),
        #     weight=0)

        G.add_edge(start_location, (list(vertical_aisle_x.values())[0], cross_aisle_y[0]),
                   weight=0)
        G.add_edge(end_location, (list(vertical_aisle_x.values())[0], cross_aisle_y[0]),
                   weight=0)

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
            # start_location=(15625, 400),
            # end_location=(15625, 400),
            start_connection_point=(depot_aisle, depot_node),
            end_connection_point=(depot_aisle, depot_node),
        )

        nodes = list(G.nodes())
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight='weight', dtype=float)
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
            node_list=nodes)

        layout = LayoutData(
            tpe=LayoutType.CONVENTIONAL,
            graph_data=layout_params,
            layout_network=layout_network,
        )

        storage_type = StorageType.DEDICATED
        locations = []

        for row in orders.itertuples(index=False):
            article_id = row.ARTIKELNR
            aisle = str(int(row.aisle))
            house = str(row.house)
            place = str(row.picklocation)

            loc_key = aisle + "_" + house

            for loc in storage_locations:
                # print(loc.id)
                if loc.id == loc_key:
                    for slot in loc.slots:
                        if slot.id == place:
                            x = loc.pick_node[0]
                            y = loc.pick_node[1]
                            # article_id = row.ARTIKELNR
                            locations.append(
                                Location(
                                    x=x,
                                    y=y,
                                    article_id=article_id,
                                    amount=9999,
                                )
                            )
        storage_domain = StorageLocations(tpe=storage_type, locations=locations, storage_slots=storage_locations)
        storage_domain.build_article_location_mapping()

        PICKER_CAPACITY = 1
        # pick_cart = PickCart(n_dimension=1,
        #                      n_boxes=1,
        #                      capacities=[PICKER_CAPACITY],
        #                      dimensions=[DimensionType.ORDERS])
        resource_cfg = self.cfg.data_cards.features.resources.features
        distribution = resource_cfg.distribution_cobots
        n_cobots = int(resource_cfg.n_pickers * distribution)
        n_manual = int(resource_cfg.n_pickers - n_cobots)

        manual_picker_list = [
                 ManualPicker(
                     tpe=ResourceType.HUMAN,
                     id=i,
                     speed=resource_cfg.picker_speed,
                     current_location=start_location,
                     pick_cart=PickCart(
                         n_dimension=1,
                         n_boxes=1,
                         capacities=[PICKER_CAPACITY],
                         dimensions=[DimensionType.ORDERS]),
                     time_per_pick=resource_cfg.time_per_pick,
                     tour_setup_time=resource_cfg.tour_setup_time,
                     aisle_congestion_rate=resource_cfg.aisle_congestion_rate
                 ) for i in range(n_manual)]

        cobot_list = [
            CobotPicker(
                tpe=ResourceType.COBOT,
                id=i,
                speed=resource_cfg.picker_speed,
                current_location=start_location,
                speed_follow_mode=resource_cfg.speed_follow_mode,
                pick_cart=PickCart(
                     n_dimension=1,
                     n_boxes=1,
                     capacities=[PICKER_CAPACITY],
                     dimensions=[DimensionType.ORDERS]),
                time_per_pick=resource_cfg.time_per_pick_cobot,
                tour_setup_time=resource_cfg.tour_setup_time,
                aisle_congestion_rate=resource_cfg.aisle_congestion_rate)
            for i in range(n_manual, n_manual + n_cobots)]

        resources_list = manual_picker_list + cobot_list
        resources = Resources(ResourceType.HUMAN, resources_list)

        dynamic_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=0.0,
            congestion_rate={},
            active_tours=[],
            current_picker=None
        )
        return SimWarehouseDomain(
            problem_class=self.cfg.data_cards.problem_type,
            objective="Distance",
            layout=layout,
            articles=articles,
            orders=order_domain,
            resources=resources,
            storage=storage_domain,
            warehouse_info=dynamic_info
        )

    def load(self, orders_path: str,
             layout_path: str,
             use_cache: bool = True,
             **kwargs) -> SimWarehouseDomain:
        cache_key = f"test"

        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}_domain.pkl"
            if cache_path.exists():
                from ware_ops_algos.utils.io_helpers import load_pickle
                self.cache_path = cache_path
                return load_pickle(str(cache_path))

        orders, master_data, layout_raw = self._parse(orders_path, layout_path)
        domain = self._build(orders, master_data, layout_raw)

        if use_cache and self.cache_dir:
            from ware_ops_algos.utils.io_helpers import dump_pickle
            cache_path = self.cache_dir / f"{cache_key}_domain.pkl"
            dump_pickle(str(cache_path), domain)
            self.cache_path = cache_path

        return domain

