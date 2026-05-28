# from __future__ import annotations
#
# import json
# import math
# import random
# from collections import defaultdict
# from functools import reduce
# from pathlib import Path
# from typing import Optional
#
# import networkx as nx
# import numpy as np
# import pandas as pd
# from scipy.sparse.csgraph import floyd_warshall
#
# from ware_ops_algos.algorithms import Job, BatchObject, ScheduledJob
# from ware_ops_algos.data_loaders import DataLoader
# from ware_ops_algos.domain_models import (
#     Article, ArticleType, Articles,
#     BoundingBox, CobotPicker, DimensionType,
#     LayoutData, LayoutNetwork, LayoutParameters, LayoutType,
#     Location, ManualPicker,
#     OrdersDomain, OrderType, Order, OrderPosition,
#     PickCart, ResourceType, Resources,
#     StorageLocation, StorageLocations, StorageSlot, StorageType,
#     WarehouseInfoType,
# )
# from ware_ops_algos.utils.io_helpers import dump_pickle, load_pickle
#
# from casim.domain_objects.sim_domain import DynamicInfo, SimWarehouseDomain
#
#
# # ---------------------------------------------------------------------
# # Canonical schema
# # ---------------------------------------------------------------------
# # Input dataframes consumed by this loader are expected to follow this schema.
# # Source-specific adapters are responsible for producing dataframes in this
# # form (see docs for the picks/master canonical contract).
#
# COL_DATE = "date"
# COL_ORDER_ID = "order_id"
# COL_CUSTOMER_ID = "customer_id"
# COL_PICKER_ID = "picker_id"
# COL_ARTICLE_ID = "article_id"
# COL_QUANTITY = "quantity"
# COL_LINE_WEIGHT = "line_weight"
# COL_KOLLI_VOLUME = "kolli_volume"
# COL_AISLE = "aisle"
# COL_HOUSE = "house"
# COL_PICKLOCATION = "picklocation"
# COL_START_SEC = "start_sec"
# COL_END_SEC = "end_sec"
#
# COL_LENGTH = "length"
# COL_WIDTH = "width"
# COL_HEIGHT = "height"
#
#
# # ---------------------------------------------------------------------
# # Anonymization
# # ---------------------------------------------------------------------
#
# def _build_id_map(values, prefix: str, seed: int) -> dict:
#     unique = sorted({str(v) for v in values if v is not None and str(v) != ""})
#     rng = random.Random(seed)
#     rng.shuffle(unique)
#     return {real: f"{prefix}{i:05d}" for i, real in enumerate(unique, start=1)}
#
#
# def _build_or_load_id_maps(
#     picks: pd.DataFrame,
#     master: pd.DataFrame,
#     maps_path: Path,
#     seed: int = 42,
# ) -> dict:
#     if maps_path.exists():
#         with open(maps_path) as f:
#             return json.load(f)
#
#     article_ids = set(picks[COL_ARTICLE_ID].astype(str)) | set(master[COL_ARTICLE_ID].astype(str))
#     maps = {
#         "articles": _build_id_map(article_ids, "A", seed),
#         "orders": _build_id_map(picks[COL_ORDER_ID], "O", seed + 1),
#         "customers": _build_id_map(picks[COL_CUSTOMER_ID], "C", seed + 2),
#         "pickers": _build_id_map(picks[COL_PICKER_ID], "P", seed + 3),
#     }
#     maps_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(maps_path, "w") as f:
#         json.dump(maps, f, indent=2)
#     return maps
#
#
# def _apply_id_maps(picks: pd.DataFrame, master: pd.DataFrame, maps: dict):
#     picks = picks.copy()
#     master = master.copy()
#     picks[COL_ARTICLE_ID] = picks[COL_ARTICLE_ID].astype(str).map(maps["articles"])
#     picks[COL_ORDER_ID] = picks[COL_ORDER_ID].astype(str).map(maps["orders"])
#     picks[COL_CUSTOMER_ID] = picks[COL_CUSTOMER_ID].astype(str).map(maps["customers"])
#     picks[COL_PICKER_ID] = picks[COL_PICKER_ID].astype(str).map(maps["pickers"])
#     master[COL_ARTICLE_ID] = master[COL_ARTICLE_ID].astype(str).map(maps["articles"])
#     master = master.dropna(subset=[COL_ARTICLE_ID])
#     return picks, master
#
#
# # ---------------------------------------------------------------------
# # Calibration
# # ---------------------------------------------------------------------
#
# def _calibrate(picks: pd.DataFrame) -> dict:
#     """Build a market-order generator calibration from canonical picks."""
#     pick_lines = (
#         picks.groupby([COL_DATE, COL_CUSTOMER_ID, COL_ORDER_ID, COL_ARTICLE_ID], as_index=False)
#              .agg(qty=(COL_QUANTITY, "sum"), weight=(COL_LINE_WEIGHT, "sum"))
#     )
#     market_lines = (
#         pick_lines.groupby([COL_DATE, COL_CUSTOMER_ID, COL_ARTICLE_ID], as_index=False)
#                   .agg(qty=("qty", "sum"), weight=("weight", "sum"))
#     )
#
#     article_counts = market_lines[COL_ARTICLE_ID].value_counts()
#     article_pmf = article_counts / article_counts.sum()
#
#     return {
#         "markets_per_shift": market_lines.groupby(COL_DATE)[COL_CUSTOMER_ID].nunique().tolist(),
#         "lines_per_order": market_lines.groupby([COL_DATE, COL_CUSTOMER_ID]).size().tolist(),
#         "weight_per_order": market_lines.groupby([COL_DATE, COL_CUSTOMER_ID])["weight"].sum().tolist(),
#         "articles": article_pmf.index.tolist(),
#         "probs": article_pmf.tolist(),
#         "qty_samples": market_lines.groupby(COL_ARTICLE_ID)["qty"].apply(list).to_dict(),
#         "n_observed_shifts": int(market_lines[COL_DATE].nunique()),
#         "n_observed_market_orders": int(len(market_lines.groupby([COL_DATE, COL_CUSTOMER_ID]))),
#     }
#
#
# def _generate_market_orders(
#     calibration: dict,
#     seed: int,
#     shift_duration_sec: float,
# ) -> list[Order]:
#     """Sample one shift of market orders from a calibration."""
#     rng = np.random.default_rng(seed)
#     articles = np.array(calibration["articles"])
#     probs = np.array(calibration["probs"])
#     probs = probs / probs.sum()
#     qty_samples = {a: np.array(v) for a, v in calibration["qty_samples"].items()}
#
#     n_markets = int(rng.choice(calibration["markets_per_shift"]))
#     orders = []
#     for m in range(1, n_markets + 1):
#         target_lines = int(rng.choice(calibration["lines_per_order"]))
#         positions = []
#         used = set()
#         while len(positions) < target_lines:
#             art = str(rng.choice(articles, p=probs))
#             if art in used:
#                 continue
#             qty = int(rng.choice(qty_samples[art]))
#             positions.append(OrderPosition(
#                 order_number=f"M{m:04d}",
#                 article_id=art,
#                 amount=qty,
#             ))
#             used.add(art)
#         orders.append(Order(
#             order_id=f"M{m:04d}",
#             order_date=0,
#             due_date=shift_duration_sec,
#             order_positions=positions,
#         ))
#     return orders
#
#
# # ---------------------------------------------------------------------
# # Loader
# # ---------------------------------------------------------------------
#
# class WarehousePickingLoader(DataLoader):
#     """Load a warehouse picking domain from canonical input or from a bundle.
#
#     Two data sources are supported:
#       - canonical dataframes (picks, master) produced by a source-specific
#         adapter; optional anonymization is applied at load time
#       - a published artifact bundle containing master data + calibration
#
#     Two order modes via cfg.order_source:
#       - "historic":  replays orders from source picks; requires source data
#       - "synthetic": generates market orders from a calibration
#
#     Anonymization happens once at the parse boundary. The id_maps file is
#     the only artifact that carries the real-to-anonymous mapping; downstream
#     artifacts (bundles, caches) contain anonymous identifiers only.
#
#     Order splitting / palletization is not the loader's responsibility.
#     Synthetic orders are returned as market-level orders; whoever consumes
#     OrdersDomain downstream may split them into transport units.
#     """
#
#     # Layout grid constants (warehouse-generic units in mm)
#     BAY_WIDTH: int = 800
#     BAY_LENGTH: int = 1200
#     AISLE_SEGMENT_WIDTH: int = 800
#     AISLE_SEGMENT_LENGTH: int = 12600
#     AISLE_KEYS: list[int] = [-1, -2, -3, -4, -5, -6, -7, -8]
#
#     def __init__(
#         self,
#         instances_dir: str | Path,
#         cache_dir: str | Path = None,
#         cfg=None,
#     ):
#         super().__init__(instances_dir)
#         self.cache_dir = Path(cache_dir) if cache_dir else None
#         self.cfg = cfg
#         if self.cache_dir:
#             self.cache_dir.mkdir(parents=True, exist_ok=True)
#         self.jobs = None
#
#     # ------------------------------------------------------------------
#     # Entry point
#     # ------------------------------------------------------------------
#
#     def load(
#         self,
#         layout_path: str,
#         picks_df: Optional[pd.DataFrame] = None,
#         master_df: Optional[pd.DataFrame] = None,
#         bundle_dir: Optional[str] = "public_bundle",
#         use_cache: bool = True,
#         **kwargs,
#     ) -> SimWarehouseDomain:
#         cache_key = self._cache_key()
#         if use_cache and self.cache_dir:
#             cache_path = self.cache_dir / f"{cache_key}.pkl"
#             if cache_path.exists():
#                 return load_pickle(str(cache_path))
#
#         # Resolve master data + calibration from source or bundle
#         if picks_df is not None and master_df is not None:
#             domain_base, calibration, picks_for_orders = self._build_from_source(
#                 picks_df, master_df, layout_path,
#             )
#             if bundle_dir is not None:
#                 self._publish_bundle(Path(bundle_dir), domain_base, calibration)
#         elif bundle_dir is not None:
#             domain_base, calibration = self._build_from_bundle(Path(bundle_dir), layout_path)
#             picks_for_orders = None
#         else:
#             raise ValueError(
#                 "Must provide either (picks_df + master_df) or bundle_dir"
#             )
#
#         # Resolve orders
#         if self.cfg.data_card.source.order_source == "historic":
#             if picks_for_orders is None:
#                 raise ValueError("Historic orders require source data, not a bundle")
#             orders = self._build_historic_orders(picks_for_orders)
#             self.jobs = self._build_jobs_replay(picks_for_orders)
#         elif self.cfg.data_card.source.order_source == "synthetic":
#             orders = OrdersDomain(
#                 tpe=OrderType.STANDARD,
#                 orders=_generate_market_orders(
#                     calibration,
#                     seed=getattr(self.cfg, "synthetic_seed", 42),
#                     shift_duration_sec=getattr(self.cfg, "shift_duration_sec", 28800.0),
#                 ),
#             )
#         else:
#             raise ValueError(f"Unknown order_source: {self.cfg.data_card.source.order_source}")
#
#         dynamic_info = DynamicInfo(
#             tpe=WarehouseInfoType.OFFLINE,
#             time=0.0,
#             congestion_rate={},
#             active_tours=[],
#             current_picker=None,
#             buffered_batches=None,
#             done=False,
#             n_staged_pallets=0,
#         )
#
#         domain = SimWarehouseDomain(
#             problem_class=self.cfg.data_card.problem_type,
#             objective="Distance",
#             layout=domain_base["layout"],
#             articles=domain_base["articles"],
#             resources=domain_base["resources"],
#             storage=domain_base["storage"],
#             orders=orders,
#             dynamic_warehouse_info=dynamic_info,
#         )
#
#         if use_cache and self.cache_dir:
#             dump_pickle(str(self.cache_dir / f"{cache_key}.pkl"), domain)
#
#         return domain
#
#     def _cache_key(self) -> str:
#         dataset = getattr(self.cfg, "dataset_name", "warehouse")
#         parts = [dataset, self.cfg.data_card.source.order_source]
#         if self.cfg.data_card.source.order_source == "synthetic":
#             parts.append(f"seed{getattr(self.cfg, 'synthetic_seed', 42)}")
#         return "_".join(parts)
#
#     # ------------------------------------------------------------------
#     # Build paths
#     # ------------------------------------------------------------------
#
#     def _build_from_source(
#         self,
#         picks: pd.DataFrame,
#         master: pd.DataFrame,
#         layout_path: str,
#     ):
#         if getattr(self.cfg, "anonymize", True):
#             maps = _build_or_load_id_maps(
#                 picks, master,
#                 maps_path=Path(getattr(self.cfg, "maps_path", "id_maps.json")),
#             )
#             picks, master = _apply_id_maps(picks, master, maps)
#
#         layout_raw = pd.read_csv(layout_path, sep=";")
#
#         articles = self._build_articles(picks, master)
#         layout, storage_locations = self._build_layout(layout_raw)
#         storage = self._build_storage(picks, storage_locations)
#         resources = self._build_resources()
#         calibration = _calibrate(picks)
#
#         return (
#             {
#                 "articles": articles,
#                 "layout": layout,
#                 "storage": storage,
#                 "resources": resources,
#             },
#             calibration,
#             picks,
#         )
#
#     def _build_from_bundle(self, bundle_dir: Path, layout_path: str):
#         master_pkl = bundle_dir / "master_data.pkl"
#         calibration_json = bundle_dir / "calibration.json"
#         if not master_pkl.exists() or not calibration_json.exists():
#             raise FileNotFoundError(
#                 f"Bundle incomplete: need {master_pkl} and {calibration_json}"
#             )
#
#         master_data = load_pickle(str(master_pkl))
#         with open(calibration_json) as f:
#             calibration = json.load(f)
#
#         layout_raw = pd.read_csv(layout_path, sep=";")
#         layout, _ = self._build_layout(layout_raw)
#         resources = self._build_resources()
#
#         return (
#             {
#                 "articles": master_data["articles"],
#                 "layout": layout,
#                 "storage": master_data["storage"],
#                 "resources": resources,
#             },
#             calibration,
#         )
#
#     def _publish_bundle(self, bundle_dir: Path, domain_base: dict, calibration: dict):
#         bundle_dir.mkdir(parents=True, exist_ok=True)
#         dump_pickle(
#             str(bundle_dir / "master_data.pkl"),
#             {"articles": domain_base["articles"], "storage": domain_base["storage"]},
#         )
#         with open(bundle_dir / "calibration.json", "w") as f:
#             json.dump(calibration, f, indent=2)
#
#     # ------------------------------------------------------------------
#     # Articles
#     # ------------------------------------------------------------------
#
#     def _build_articles(self, picks: pd.DataFrame, master: pd.DataFrame) -> Articles:
#         df = picks.copy()
#         df["weight_per_unit"] = df[COL_LINE_WEIGHT] / df[COL_QUANTITY]
#
#         stats = df.groupby(COL_ARTICLE_ID).agg(
#             weight_per_unit=("weight_per_unit", "median"),
#             kolli_volume=(COL_KOLLI_VOLUME, "median"),
#         ).reset_index()
#
#         # Kolli size inferred via GCD of observed quantities per article
#         kolli_size = (
#             df.groupby(COL_ARTICLE_ID)[COL_QUANTITY]
#               .apply(lambda s: int(reduce(math.gcd, s.astype(int).tolist())))
#               .to_dict()
#         )
#         stats["kolli_size"] = stats[COL_ARTICLE_ID].map(kolli_size)
#         stats["volume_per_unit"] = stats["kolli_volume"] / stats["kolli_size"]
#
#         stats = stats.merge(
#             master[[COL_ARTICLE_ID, COL_LENGTH, COL_HEIGHT, COL_WIDTH]],
#             on=COL_ARTICLE_ID, how="left",
#         )
#
#         article_list = [
#             Article(
#                 article_id=row[COL_ARTICLE_ID],
#                 weight=row["weight_per_unit"],
#                 volume=row["volume_per_unit"],
#                 length=row[COL_LENGTH] if pd.notna(row[COL_LENGTH]) else None,
#                 width=row[COL_WIDTH] if pd.notna(row[COL_WIDTH]) else None,
#                 height=row[COL_HEIGHT] if pd.notna(row[COL_HEIGHT]) else None,
#                 kolli_size=int(row["kolli_size"]),
#             )
#             for _, row in stats.iterrows()
#         ]
#         return Articles(tpe=ArticleType.STANDARD, articles=article_list)
#
#     # ------------------------------------------------------------------
#     # Layout
#     # ------------------------------------------------------------------
#
#     def _build_layout(self, layout_raw: pd.DataFrame) -> tuple[LayoutData, list[StorageLocation]]:
#         layout_np = layout_raw.to_numpy()
#         block_ranges = [
#             list(range(2210, 2284)),
#             list(range(2310, 2384)),
#             list(range(2010, 2084)),
#         ]
#         block_width = 3 * self.BAY_WIDTH
#
#         storage_locations = self._build_storage_locations(layout_np, block_ranges, block_width)
#         G, vertical_aisle_x, cross_aisle_y = self._build_graph(layout_np, storage_locations)
#
#         start_location = (0, -10000)
#         end_location = (-1, -10000)
#         G.add_node(start_location, pos=start_location, type="start_node")
#         G.add_node(end_location, pos=end_location, type="end_node")
#         G.add_edge(start_location, (list(vertical_aisle_x.values())[0], cross_aisle_y[0]), weight=0)
#         G.add_edge(end_location, (list(vertical_aisle_x.values())[0], cross_aisle_y[0]), weight=0)
#
#         nodes = list(G.nodes())
#         A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", dtype=float)
#         dima_raw, predecessors = floyd_warshall(A, directed=False, return_predecessors=True)
#         dima = pd.DataFrame(dima_raw, index=nodes, columns=nodes)
#
#         layout_network = LayoutNetwork(
#             graph=G, distance_matrix=dima, predecessor_matrix=predecessors,
#             closest_node_to_start=(7500, 400),
#             min_aisle_position=400, max_aisle_position=53600,
#             start_node=start_location, end_node=end_location,
#             node_list=nodes,
#         )
#         layout_params = LayoutParameters(
#             n_aisles=8, n_pick_locations=60, n_blocks=2,
#             dist_pick_locations=0.8, dist_aisle=15,
#             dist_top_to_pick_location=0, dist_bottom_to_pick_location=0,
#             dist_start=0, dist_end=0, dist_cross_aisle=2.4,
#             start_location=start_location, end_location=end_location,
#             start_connection_point=(0, 0), end_connection_point=(0, 0),
#         )
#         return LayoutData(
#             tpe=LayoutType.CONVENTIONAL,
#             graph_data=layout_params,
#             layout_network=layout_network,
#         ), storage_locations
#
#     def _build_storage_locations(self, layout_np, block_ranges, block_width):
#         storage_locations = []
#         offset_horizontal = 0
#         max_y = 0
#         for col in range(layout_np.shape[1]):
#             offset_vertical = 0
#             offset_horizontal += max_y
#             for row in reversed(range(layout_np.shape[0])):
#                 val = layout_np[row, col]
#                 if val >= 0 or val == -99:
#                     max_y = self.BAY_LENGTH
#                     if val > 0:
#                         val_left = layout_np[row, col - 1] if col > 0 else None
#                         val_right = layout_np[row, col + 1] if col < layout_np.shape[1] - 1 else None
#                         val_top = layout_np[row - 1, col] if row > 0 else None
#                         val_bottom = layout_np[row + 1, col] if row < layout_np.shape[0] - 1 else None
#                         adjacent_aisle = (
#                             val_left if val_left in self.AISLE_KEYS else
#                             val_right if val_right in self.AISLE_KEYS else
#                             val_top if val_top in self.AISLE_KEYS else
#                             val_bottom if val_bottom in self.AISLE_KEYS else None
#                         )
#                         house = (
#                             str(val)[:2] if len(str(val)) == 4 else
#                             str(val)[:1] if len(str(val)) == 3 else "0"
#                         )
#                         slots, bbox = None, None
#                         for block in block_ranges:
#                             if val in block:
#                                 slots = [StorageSlot(id=str(b_id), level=0) for b_id in block]
#                                 bbox = BoundingBox(
#                                     x_min=offset_horizontal, x_max=offset_horizontal + self.BAY_LENGTH,
#                                     y_min=offset_vertical, y_max=offset_vertical + block_width,
#                                 )
#                                 break
#                         if not slots:
#                             val_int = int(val)
#                             slots = [
#                                 StorageSlot(id=str(val_int), level=0),
#                                 StorageSlot(id=str(val_int + 1), level=1),
#                             ]
#                         if not bbox:
#                             bbox = BoundingBox(
#                                 x_min=offset_horizontal, x_max=offset_horizontal + self.BAY_LENGTH,
#                                 y_min=offset_vertical, y_max=offset_vertical + self.BAY_WIDTH,
#                             )
#                         aisle = str(adjacent_aisle * -1) if adjacent_aisle else "0"
#                         storage_locations.append(StorageLocation(
#                             id=f"{aisle}_{house}",
#                             bbox=bbox,
#                             aisle_id=adjacent_aisle,
#                             slots=slots,
#                         ))
#                     offset_vertical += self.BAY_WIDTH
#                 elif val in self.AISLE_KEYS:
#                     max_y = self.AISLE_SEGMENT_LENGTH
#                     offset_vertical += self.AISLE_SEGMENT_LENGTH
#                 else:
#                     max_y = self.AISLE_SEGMENT_LENGTH
#                     offset_vertical += self.AISLE_SEGMENT_WIDTH
#         return storage_locations
#
#     def _build_graph(self, layout_np, storage_locations):
#         G = nx.Graph()
#         vertical_aisle_x: dict[int, float] = {}
#         cross_aisle_y: list[float] = []
#
#         offset_horizontal = 0
#         max_y = 0
#         for col in range(layout_np.shape[1]):
#             offset_horizontal += max_y
#             val = layout_np[layout_np.shape[0] // 2, col]
#             if val in self.AISLE_KEYS and val != -90:
#                 vertical_aisle_x[val] = offset_horizontal + self.AISLE_SEGMENT_LENGTH / 2
#                 max_y = self.AISLE_SEGMENT_LENGTH
#             elif val > 0 or val == -99:
#                 max_y = self.BAY_LENGTH
#             else:
#                 max_y = self.AISLE_SEGMENT_LENGTH
#
#         offset_vertical = 0
#         in_cross_aisle = False
#         cross_aisle_start = 0
#         for row in reversed(range(layout_np.shape[0])):
#             row_vals = layout_np[row, :]
#             is_cross = all(v == -90 for v in row_vals)
#             if is_cross:
#                 if not in_cross_aisle:
#                     cross_aisle_start = offset_vertical
#                     in_cross_aisle = True
#                 offset_vertical += self.AISLE_SEGMENT_WIDTH
#             else:
#                 if in_cross_aisle:
#                     cross_aisle_y.append((cross_aisle_start + offset_vertical) / 2)
#                     in_cross_aisle = False
#                 first_val = next((v for v in row_vals if v != -90), None)
#                 if first_val is not None and (first_val > 0 or first_val == -99):
#                     offset_vertical += self.BAY_WIDTH
#                 else:
#                     offset_vertical += self.AISLE_SEGMENT_WIDTH
#         if in_cross_aisle:
#             cross_aisle_y.append((cross_aisle_start + offset_vertical) / 2)
#         cross_aisle_y.append(offset_vertical)
#         cross_aisle_y.append(28400)
#         cross_aisle_y = sorted(set(cross_aisle_y))
#
#         for aisle_id, x in vertical_aisle_x.items():
#             for y in cross_aisle_y:
#                 G.add_node((x, y), pos=(x, y), node_type="intersection", aisle_id=aisle_id)
#
#         x_sorted = sorted(vertical_aisle_x.values())
#         for y in cross_aisle_y:
#             for i in range(len(x_sorted) - 1):
#                 n1, n2 = (x_sorted[i], y), (x_sorted[i + 1], y)
#                 if G.has_node(n1) and G.has_node(n2):
#                     G.add_edge(n1, n2, weight=x_sorted[i + 1] - x_sorted[i], edge_type="cross_aisle")
#
#         for x in vertical_aisle_x.values():
#             for i in range(len(cross_aisle_y) - 1):
#                 G.add_edge(
#                     (x, cross_aisle_y[i]), (x, cross_aisle_y[i + 1]),
#                     weight=cross_aisle_y[i + 1] - cross_aisle_y[i],
#                     edge_type="vertical_aisle",
#                 )
#
#         for slot in storage_locations:
#             slot_y_center = (slot.bbox.y_min + slot.bbox.y_max) / 2
#             aisle_x = vertical_aisle_x.get(slot.aisle_id)
#             if aisle_x is not None:
#                 access_node = (aisle_x, slot_y_center)
#                 slot.pick_node = access_node
#                 G.add_node(access_node, pos=access_node, type="pick_node",
#                            slot_id=slot.id, aisle_id=slot.aisle_id)
#
#         aisle_all_nodes: dict[float, list] = defaultdict(list)
#         for aisle_id, x in vertical_aisle_x.items():
#             for y in cross_aisle_y:
#                 aisle_all_nodes[x].append((x, y))
#         for slot in storage_locations:
#             if slot.pick_node is not None:
#                 aisle_all_nodes[slot.pick_node[0]].append(slot.pick_node)
#         for aisle_x, nodes in aisle_all_nodes.items():
#             for i in range(len(cross_aisle_y) - 1):
#                 n1 = (aisle_x, cross_aisle_y[i])
#                 n2 = (aisle_x, cross_aisle_y[i + 1])
#                 if G.has_edge(n1, n2):
#                     G.remove_edge(n1, n2)
#             sorted_nodes = sorted(set(nodes), key=lambda n: n[1])
#             for n1, n2 in zip(sorted_nodes, sorted_nodes[1:]):
#                 G.add_edge(n1, n2, weight=n2[1] - n1[1], edge_type="vertical_aisle")
#
#         return G, vertical_aisle_x, cross_aisle_y
#
#     # ------------------------------------------------------------------
#     # Storage
#     # ------------------------------------------------------------------
#
#     def _build_storage(self, picks: pd.DataFrame, storage_locations: list[StorageLocation]) -> StorageLocations:
#         locations = []
#         seen = set()
#         for row in picks.itertuples(index=False):
#             article_id = getattr(row, COL_ARTICLE_ID)
#             if article_id in seen:
#                 continue
#             loc_key = f"{int(getattr(row, COL_AISLE))}_{getattr(row, COL_HOUSE)}"
#             place = str(getattr(row, COL_PICKLOCATION))
#             matched = False
#             for loc in storage_locations:
#                 if loc.id != loc_key:
#                     continue
#                 for slot in loc.slots:
#                     if slot.id == place and loc.pick_node is not None:
#                         locations.append(Location(
#                             x=loc.pick_node[0], y=loc.pick_node[1],
#                             article_id=str(article_id), amount=9999,
#                         ))
#                         matched = True
#                         break
#                 if matched:
#                     break
#             seen.add(article_id)
#         storage = StorageLocations(
#             tpe=StorageType.DEDICATED,
#             locations=locations,
#             storage_slots=storage_locations,
#         )
#         storage.build_article_location_mapping()
#         return storage
#
#     # ------------------------------------------------------------------
#     # Resources
#     # ------------------------------------------------------------------
#
#     def _build_resources(self) -> Resources:
#         n_pickers = 19
#         picker_speed = 2306
#         speed_follow_mode = 1389
#         time_per_pick = 19
#         time_per_pick_cobot = 15.77
#         tour_setup_time = 215
#         aisle_congestion_rate = 0
#         cobot_share = 0
#         n_cobots = int(n_pickers * cobot_share)
#         n_manual = n_pickers - n_cobots
#         start_location = (0, -10000)
#
#         def _cart() -> PickCart:
#             return PickCart(n_dimension=1, n_boxes=2, capacities=[1], dimensions=[DimensionType.ORDERS])
#
#         manual = [
#             ManualPicker(
#                 tpe=ResourceType.HUMAN, id=i, speed=picker_speed,
#                 current_location=start_location, pick_cart=_cart(),
#                 time_per_pick=time_per_pick, tour_setup_time=tour_setup_time,
#                 aisle_congestion_rate=aisle_congestion_rate,
#             )
#             for i in range(n_manual)
#         ]
#         cobots = [
#             CobotPicker(
#                 tpe=ResourceType.COBOT, id=i, speed=picker_speed,
#                 current_location=start_location, speed_follow_mode=speed_follow_mode,
#                 pick_cart=_cart(), time_per_pick=time_per_pick_cobot,
#                 tour_setup_time=tour_setup_time,
#                 aisle_congestion_rate=aisle_congestion_rate,
#             )
#             for i in range(n_manual, n_manual + n_cobots)
#         ]
#         return Resources(ResourceType.HUMAN, manual + cobots)
#
#     # ------------------------------------------------------------------
#     # Historic orders
#     # ------------------------------------------------------------------
#
#     def _build_historic_orders(self, picks: pd.DataFrame) -> OrdersDomain:
#         buffer_sec = 15 * 60
#         deadlines = (
#             picks.groupby([COL_CUSTOMER_ID, COL_DATE])[COL_END_SEC].max()
#                  .groupby(COL_CUSTOMER_ID).max()
#                  .add(buffer_sec)
#                  .to_dict()
#         )
#         order_list = []
#         for o_id in picks[COL_ORDER_ID].unique():
#             subset = picks[picks[COL_ORDER_ID] == o_id]
#             cust_id = subset[COL_CUSTOMER_ID].unique()[0]
#             positions = [
#                 OrderPosition(
#                     order_number=str(o_id),
#                     article_id=str(getattr(r, COL_ARTICLE_ID)),
#                     amount=int(getattr(r, COL_QUANTITY)),
#                 )
#                 for r in subset.itertuples(index=False)
#             ]
#             order_list.append(Order(
#                 order_id=str(o_id),
#                 order_date=0,
#                 due_date=deadlines[cust_id],
#                 order_positions=positions,
#             ))
#         return OrdersDomain(tpe=OrderType.STANDARD, orders=order_list)
#
#     def _build_jobs_replay(self, picks: pd.DataFrame):
#         releases = (
#             picks.groupby([COL_ORDER_ID, COL_DATE])[COL_START_SEC].min()
#                  .groupby(COL_ORDER_ID).min()
#                  .to_dict()
#         )
#         deadlines = (
#             picks.groupby([COL_ORDER_ID, COL_DATE])[COL_END_SEC].max()
#                  .groupby(COL_ORDER_ID).max()
#                  .to_dict()
#         )
#         jobs = []
#         for i, o_id in enumerate(picks[COL_ORDER_ID].unique()):
#             subset = picks[picks[COL_ORDER_ID] == o_id]
#             picker_id = subset[COL_PICKER_ID].unique()[0]
#             positions = [
#                 OrderPosition(
#                     order_number=str(o_id),
#                     article_id=str(getattr(r, COL_ARTICLE_ID)),
#                     amount=int(getattr(r, COL_QUANTITY)),
#                 )
#                 for r in subset.itertuples(index=False)
#             ]
#             order = Order(
#                 order_id=str(o_id), order_date=0,
#                 due_date=deadlines[o_id],
#                 order_positions=positions,
#             )
#             processing_time = deadlines[o_id] - releases[o_id]
#             job = Job(
#                 i, processing_time, releases[o_id], deadlines[o_id],
#                 len(positions), None,
#                 BatchObject(batch_id=i, orders=[order]),
#             )
#             jobs.append(ScheduledJob(job, picker_id, releases[o_id], deadlines[o_id]))
#         return jobs

from __future__ import annotations

import math
from collections import defaultdict
from functools import reduce
from pathlib import Path

import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall

from ware_ops_algos.algorithms import Job, BatchObject, ScheduledJob
from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import (
    Article, ArticleType, Articles,
    BoundingBox, CobotPicker, DimensionType,
    LayoutData, LayoutNetwork, LayoutParameters, LayoutType,
    Location, ManualPicker,
    OrdersDomain, OrderType, Order, OrderPosition,
    PickCart, ResourceType, Resources,
    StorageLocation, StorageLocations, StorageSlot, StorageType,
    WarehouseInfoType,
)
from ware_ops_algos.utils.io_helpers import dump_pickle, load_pickle

from casim.domain_objects.sim_domain import DynamicInfo, SimWarehouseDomain


# ---------------------------------------------------------------------
# Canonical order-stream schema
# ---------------------------------------------------------------------

COL_DATE = "date"
COL_ORDER_ID = "order_id"
COL_CUSTOMER_ID = "customer_id"
COL_PICKER_ID = "picker_id"
COL_ARTICLE_ID = "article_id"
COL_QUANTITY = "quantity"
COL_LINE_WEIGHT = "line_weight"
COL_KOLLI_VOLUME = "kolli_volume"
COL_AISLE = "aisle"
COL_HOUSE = "house"
COL_PICKLOCATION = "picklocation"
COL_START_SEC = "start_sec"
COL_END_SEC = "end_sec"
COL_ORDER_DATE = "order_date"
COL_DUE_DATE = "due_date"
COL_LENGTH = "length"
COL_WIDTH = "width"
COL_HEIGHT = "height"
COL_KOLLI_SIZE = "kolli_size"


class WarehousePickingLoader(DataLoader):
    """Load a warehouse picking domain from a canonical order-stream CSV.

    The order stream is produced upstream (anonymize -> calibrate -> generate)
    and is the same shape whether it originates from historic picks or
    synthetic generation, so the loader consumes it without knowing which.
    """

    CCG1_WIDTH: int = 800
    CCG1_LENGTH: int = 1200
    WIDTH_AISLE_SEGMENT: int = 800
    LENGTH_AISLE_SEGMENT: int = 12600
    AISLE_KEYS: list[int] = [-1, -2, -3, -4, -5, -6, -7, -8]

    def __init__(self, instances_dir: str | Path, cache_dir: str | Path = None, cfg=None):
        super().__init__(instances_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cfg = cfg
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.jobs = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def load(self, orders_path: str, layout_path: str, use_cache: bool = True, **kwargs) -> SimWarehouseDomain:
        cache_key = "grocery_retailer"
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
        orders = pd.read_csv(orders_path, dtype={COL_ARTICLE_ID: str, COL_ORDER_ID: str})
        layout_raw = pd.read_csv(layout_path, sep=";")
        return orders, layout_raw

    def _build_domain(self, orders_df: pd.DataFrame, layout_raw: pd.DataFrame) -> SimWarehouseDomain:
        articles = self._build_articles(orders_df)
        layout, storage_locations = self._build_layout(layout_raw)
        storage = self._build_storage(orders_df, storage_locations)
        resources = self._build_resources()
        orders = self._build_orders(orders_df)
        self.jobs = self._build_jobs_replay(orders_df)

        dynamic_info = DynamicInfo(
            tpe=WarehouseInfoType.OFFLINE, time=0.0, congestion_rate={},
            active_tours=[], current_picker=None, buffered_batches=None,
            done=False, n_staged_pallets=0,
        )
        return SimWarehouseDomain(
            problem_class=self.cfg.data_card.problem_type,
            objective="Distance",
            layout=layout, articles=articles, resources=resources,
            storage=storage, orders=orders, dynamic_warehouse_info=dynamic_info,
        )

    # ------------------------------------------------------------------
    # Articles
    # ------------------------------------------------------------------

    def _build_articles(self, orders: pd.DataFrame) -> Articles:
        df = orders.copy()
        df["weight_per_unit"] = df[COL_LINE_WEIGHT] / df[COL_QUANTITY]
        df["volume_per_unit"] = df[COL_KOLLI_VOLUME] / df[COL_KOLLI_SIZE]

        stats = df.groupby(COL_ARTICLE_ID).agg(
            weight_per_unit=("weight_per_unit", "median"),
            kolli_volume=(COL_KOLLI_VOLUME, "median"),
        ).reset_index()

        # kolli_size = (
        #     df.groupby(COL_ARTICLE_ID)[COL_QUANTITY]
        #       .apply(lambda s: int(reduce(math.gcd, s.astype(int).tolist())))
        #       .to_dict()
        # )
        # stats["kolli_size"] = stats[COL_ARTICLE_ID].map(kolli_size)
        # stats["volume_per_unit"] = stats["kolli_volume"] / stats["kolli_size"]

        have_dims = all(c in df.columns for c in (COL_LENGTH, COL_WIDTH, COL_HEIGHT, COL_KOLLI_VOLUME))
        if have_dims:
            dims = df.groupby(COL_ARTICLE_ID)[[COL_LENGTH, COL_WIDTH, COL_HEIGHT, "volume_per_unit", COL_KOLLI_SIZE]].first().reset_index()
            stats = stats.merge(dims, on=COL_ARTICLE_ID, how="left")

        article_list = []
        for _, row in stats.iterrows():
            article_list.append(Article(
                article_id=row[COL_ARTICLE_ID],
                weight=row["weight_per_unit"],
                volume=row["volume_per_unit"],
                length=(row[COL_LENGTH] if have_dims and pd.notna(row[COL_LENGTH]) else None),
                width=(row[COL_WIDTH] if have_dims and pd.notna(row[COL_WIDTH]) else None),
                height=(row[COL_HEIGHT] if have_dims and pd.notna(row[COL_HEIGHT]) else None),
                kolli_size=(row[COL_KOLLI_SIZE] if have_dims and pd.notna(row[COL_KOLLI_SIZE]) else None),
            ))
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
            graph=G, distance_matrix=dima, predecessor_matrix=predecessors,
            closest_node_to_start=(7500, 400),
            min_aisle_position=400, max_aisle_position=53600,
            start_node=start_location, end_node=end_location,
            node_list=nodes,
        )
        layout_params = LayoutParameters(
            n_aisles=8, n_pick_locations=60, n_blocks=2,
            dist_pick_locations=0.8, dist_aisle=15,
            dist_top_to_pick_location=0, dist_bottom_to_pick_location=0,
            dist_start=0, dist_end=0, dist_cross_aisle=2.4,
            start_location=start_location, end_location=end_location,
            start_connection_point=(0, 0), end_connection_point=(0, 0),
        )
        return LayoutData(
            tpe=LayoutType.CONVENTIONAL,
            graph_data=layout_params,
            layout_network=layout_network,
        ), storage_locations

    def _build_storage_locations(self, layout_np, rbs, rb_width):
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
                        val_left = layout_np[row, col - 1] if col > 0 else None
                        val_right = layout_np[row, col + 1] if col < layout_np.shape[1] - 1 else None
                        val_top = layout_np[row - 1, col] if row > 0 else None
                        val_bottom = layout_np[row + 1, col] if row < layout_np.shape[0] - 1 else None
                        adjacent_aisle = (
                            val_left if val_left in self.AISLE_KEYS else
                            val_right if val_right in self.AISLE_KEYS else
                            val_top if val_top in self.AISLE_KEYS else
                            val_bottom if val_bottom in self.AISLE_KEYS else None
                        )
                        house = (
                            str(val)[:2] if len(str(val)) == 4 else
                            str(val)[:1] if len(str(val)) == 3 else "0"
                        )
                        slots, bbox = None, None
                        for rb in rbs:
                            if val in rb:
                                slots = [StorageSlot(id=str(rb_id), level=0) for rb_id in rb]
                                bbox = BoundingBox(
                                    x_min=offset_horizontal, x_max=offset_horizontal + self.CCG1_LENGTH,
                                    y_min=offset_vertical, y_max=offset_vertical + rb_width,
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
                                x_min=offset_horizontal, x_max=offset_horizontal + self.CCG1_LENGTH,
                                y_min=offset_vertical, y_max=offset_vertical + self.CCG1_WIDTH,
                            )
                        aisle = str(adjacent_aisle * -1) if adjacent_aisle else "0"
                        storage_locations.append(StorageLocation(
                            id=f"{aisle}_{house}",
                            bbox=bbox,
                            aisle_id=adjacent_aisle,
                            slots=slots,
                        ))
                    offset_vertical += self.CCG1_WIDTH
                elif val in self.AISLE_KEYS:
                    max_y = self.LENGTH_AISLE_SEGMENT
                    offset_vertical += self.LENGTH_AISLE_SEGMENT
                else:
                    max_y = self.LENGTH_AISLE_SEGMENT
                    offset_vertical += self.WIDTH_AISLE_SEGMENT
        return storage_locations

    def _build_graph(self, layout_np, storage_locations):
        G = nx.Graph()
        vertical_aisle_x: dict[int, float] = {}
        cross_aisle_y: list[float] = []

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

        for aisle_id, x in vertical_aisle_x.items():
            for y in cross_aisle_y:
                G.add_node((x, y), pos=(x, y), node_type="intersection", aisle_id=aisle_id)

        x_sorted = sorted(vertical_aisle_x.values())
        for y in cross_aisle_y:
            for i in range(len(x_sorted) - 1):
                n1, n2 = (x_sorted[i], y), (x_sorted[i + 1], y)
                if G.has_node(n1) and G.has_node(n2):
                    G.add_edge(n1, n2, weight=x_sorted[i + 1] - x_sorted[i], edge_type="cross_aisle")

        for x in vertical_aisle_x.values():
            for i in range(len(cross_aisle_y) - 1):
                G.add_edge(
                    (x, cross_aisle_y[i]), (x, cross_aisle_y[i + 1]),
                    weight=cross_aisle_y[i + 1] - cross_aisle_y[i],
                    edge_type="vertical_aisle",
                )

        for slot in storage_locations:
            slot_y_center = (slot.bbox.y_min + slot.bbox.y_max) / 2
            aisle_x = vertical_aisle_x.get(slot.aisle_id)
            if aisle_x is not None:
                access_node = (aisle_x, slot_y_center)
                slot.pick_node = access_node
                G.add_node(access_node, pos=access_node, type="pick_node",
                           slot_id=slot.id, aisle_id=slot.aisle_id)

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

    def _build_storage(self, orders: pd.DataFrame, storage_locations: list[StorageLocation]) -> StorageLocations:
        locations = []
        seen = set()
        for row in orders.itertuples(index=False):
            article_id = getattr(row, COL_ARTICLE_ID)
            if article_id in seen:
                continue
            loc_key = f"{int(getattr(row, COL_AISLE))}_{int(getattr(row, COL_HOUSE))}"
            place = str(int(getattr(row, COL_PICKLOCATION)))
            matched = False
            for loc in storage_locations:
                if loc.id != loc_key:
                    continue
                for slot in loc.slots:
                    if slot.id == place and loc.pick_node is not None:
                        locations.append(Location(
                            x=loc.pick_node[0], y=loc.pick_node[1],
                            article_id=str(article_id), amount=9999,
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
        n_pickers = 19
        picker_speed = 2306
        speed_follow_mode = 1389
        time_per_pick = 19
        time_per_pick_cobot = 15.77
        tour_setup_time = 215
        aisle_congestion_rate = 0
        cobot_share = 0
        n_cobots = int(n_pickers * cobot_share)
        n_manual = n_pickers - n_cobots
        start_location = (0, -10000)

        def _cart() -> PickCart:
            return PickCart(n_dimension=1, n_boxes=2, capacities=[1], dimensions=[DimensionType.ORDERS])

        manual = [
            ManualPicker(
                tpe=ResourceType.HUMAN, id=i, speed=picker_speed,
                current_location=start_location, pick_cart=_cart(),
                time_per_pick=time_per_pick, tour_setup_time=tour_setup_time,
                aisle_congestion_rate=aisle_congestion_rate,
            )
            for i in range(n_manual)
        ]
        cobots = [
            CobotPicker(
                tpe=ResourceType.COBOT, id=i, speed=picker_speed,
                current_location=start_location, speed_follow_mode=speed_follow_mode,
                pick_cart=_cart(), time_per_pick=time_per_pick_cobot,
                tour_setup_time=tour_setup_time,
                aisle_congestion_rate=aisle_congestion_rate,
            )
            for i in range(n_manual, n_manual + n_cobots)
        ]
        return Resources(ResourceType.HUMAN, manual + cobots)

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
            order_list.append(Order(
                order_id=str(o_id),
                order_date=float(subset[COL_ORDER_DATE].iloc[0]),
                due_date=float(subset[COL_DUE_DATE].iloc[0]),
                order_positions=positions,
            ))
        return OrdersDomain(tpe=OrderType.STANDARD, orders=order_list)

    # ------------------------------------------------------------------
    # Historic replay jobs (no-ops when no picker/timing data, e.g. synthetic)
    # ------------------------------------------------------------------

    def _build_jobs_replay(self, orders_df: pd.DataFrame):
        if COL_PICKER_ID not in orders_df.columns or orders_df[COL_START_SEC].isna().all():
            return None

        releases = (
            orders_df.groupby([COL_ORDER_ID, COL_DATE])[COL_START_SEC].min()
                     .groupby(COL_ORDER_ID).min()
                     .to_dict()
        )
        deadlines = (
            orders_df.groupby([COL_ORDER_ID, COL_DATE])[COL_END_SEC].max()
                     .groupby(COL_ORDER_ID).max()
                     .to_dict()
        )
        jobs = []
        for i, o_id in enumerate(orders_df[COL_ORDER_ID].unique()):
            subset = orders_df[orders_df[COL_ORDER_ID] == o_id]
            picker_id = subset[COL_PICKER_ID].unique()[0]
            positions = [
                OrderPosition(
                    order_number=str(o_id),
                    article_id=str(getattr(r, COL_ARTICLE_ID)),
                    amount=int(getattr(r, COL_QUANTITY)),
                )
                for r in subset.itertuples(index=False)
            ]
            order = Order(
                order_id=str(o_id), order_date=0,
                due_date=deadlines[o_id],
                order_positions=positions,
            )
            processing_time = deadlines[o_id] - releases[o_id]
            job = Job(
                i, processing_time, releases[o_id], deadlines[o_id],
                len(positions), None,
                BatchObject(batch_id=i, orders=[order]),
            )
            jobs.append(ScheduledJob(job, picker_id, releases[o_id], deadlines[o_id]))
        return jobs