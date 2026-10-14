"""Loaders for the OOBP benchmark of Gil-Borrás et al. (2020), Task B.5 (L3).

Both loaders restore the instances from ``scripts/fetch_data.py oobp`` and build
a ``SimWarehouseDomain`` under the paper's protocol (Table 4: speed 48 LU/min,
6 items/min, setup 3 min; the arrival times are replayed from the archive CSVs).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall

from casim.data_loader import DataLoader
from casim.domain_objects.sim_domain import SimWarehouseDomain, initial_dynamic_info
from ware_ops_algos.domain_models import (
    Article, Articles, ArticleType, DimensionType, LayoutData, LayoutNetwork,
    LayoutParameters, LayoutType, Location, Order, OrderPosition, OrdersDomain,
    OrderType, PickCart, Resource, Resources, ResourceType, StorageLocations,
    StorageType, WarehouseInfo, WarehouseInfoType,
)

# Protocol of Gil-Borrás et al. (2020), Table 4.
SPEED_LU_PER_S = 48.0 / 60.0
TIME_PER_PICK_S = 60.0 / 6.0
TOUR_SETUP_S = 3.0 * 60.0


def _network(graph: nx.Graph, start_node, end_node, closest_node_to_start,
             min_aisle_position, max_aisle_position) -> LayoutNetwork:
    nodes = list(graph.nodes)
    matrix_array, predecessors = floyd_warshall(
        nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", dtype=float),
        directed=False, return_predecessors=True)
    return LayoutNetwork(
        graph=graph,
        distance_matrix=pd.DataFrame(matrix_array, index=nodes, columns=nodes),
        predecessor_matrix=predecessors,
        closest_node_to_start=closest_node_to_start,
        min_aisle_position=min_aisle_position,
        max_aisle_position=max_aisle_position,
        start_node=start_node,
        end_node=end_node,
        node_list=nodes,
    )


def _domain(layout: LayoutData, articles, orders, resources, storage) -> SimWarehouseDomain:
    return SimWarehouseDomain(
        problem_class="OBRSP",
        objective="Distance",
        layout=layout,
        articles=articles,
        orders=orders,
        resources=resources,
        storage=storage,
        dynamic_warehouse_info=initial_dynamic_info(),
        warehouse_info=WarehouseInfo(tpe=WarehouseInfoType.ONLINE),
    )


def _picker(capacity: float, dimension, start_location) -> Resource:
    cart = PickCart(n_dimension=1, n_boxes=1, capacities=[capacity],
                    dimensions=[dimension], box_can_mix_orders=True)
    return Resource(id=0, capacity=capacity, speed=SPEED_LU_PER_S,
                    time_per_pick=TIME_PER_PICK_S, tour_setup_time=TOUR_SETUP_S,
                    pick_cart=cart, current_location=start_location)


def _orders(positions: list[list[dict]], arrivals: dict[int, float], stem: str):
    missing = set(range(len(positions))) - set(arrivals)
    if missing:
        raise ValueError(f"{stem}: no arrival time for orders {sorted(missing)[:5]}")
    return OrdersDomain(tpe=OrderType.STANDARD, orders=[
        Order(order_id=index, order_date=arrivals[index],
              order_positions=[OrderPosition(order_number=index, **position)
                               for position in order])
        for index, order in enumerate(positions)
    ])


def _henn_arrival_file(directory: Path, stem: str) -> Path:
    """The archive names its Henn arrival files inconsistently: ``_`` or ``-``
    after the group, a missing trailing instance number, and ``s`` for ``l`` in
    ``ran2``. The match is on the separator-free core of the order-file stem.
    """
    core = re.sub(r"[-_]", "", stem)
    swapped = {re.sub("l", "s", text, count=1) for text in (core, core[:-1])} | \
              {re.sub("s", "l", text, count=1) for text in (core, core[:-1])}
    candidates = {core, core[:-1], *swapped}
    matches = [path for path in directory.glob("Arrival_Time_*.csv")
               if any(candidate in re.sub(r"[-_]", "", path.stem)
                      for candidate in candidates)]
    if len(matches) != 1:
        raise FileNotFoundError(f"one arrival CSV for {stem} in {directory}, "
                                f"found {[path.name for path in matches]}")
    return matches[0]


def _read_arrival_times(path: Path) -> dict[int, float]:
    """Arrival instants in ms from an archive CSV, keyed by 0-based order id."""
    times = {}
    for line in path.read_text(encoding="windows-1252").splitlines():
        parts = line.split(";")
        if len(parts) == 2 and parts[0].strip().isdigit():
            times[int(parts[0]) - 1] = float(parts[1]) / 1000.0
    return times


class OobpHennLoader(DataLoader):
    """The 64 Henn instances of the OOBP archive (warehouse W5).

    ``filepath`` is an order file such as ``abc1/29s-40-30-0.txt``; the layout
    comes from ``sett<number>.txt`` in the same directory. Geometry per the
    setting file: ``cell_lengt`` between consecutive locations and from each
    cross aisle to the first and last location, an aisle centre distance of
    ``2 * cell_width + aisle_widt``, and the depot ``dis_ais_wa`` below the
    front cross aisle in front of the leftmost aisle. With ``depot="centre"``
    the depot is placed on the front cross aisle between the two middle aisles
    (Gil-Borrás et al., Table 3 "Center"; Fig. 5a).
    """

    def __init__(self, instances_dir: str | Path, cfg: dict, cache_dir=None):
        super().__init__(instances_dir)
        self.cfg = cfg
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.depot = str(cfg.data_card.source.get("depot", "corner"))
        if self.depot not in {"corner", "centre"}:
            raise ValueError(f"depot must be corner or centre, got {self.depot!r}")

    def load(self, filepath: str, use_cache: bool = True) -> SimWarehouseDomain:
        path = Path(filepath)
        if not path.is_absolute():
            path = self.data_dir / path
        cache_path = None
        if use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"{path.stem}_{self.depot}_domain.pkl"
            if cache_path.exists():
                from ware_ops_algos.utils.io_helpers import load_pickle
                cached = load_pickle(str(cache_path))
                if isinstance(cached, SimWarehouseDomain):
                    return cached
        domain = self._build(*self._parse(path))
        if cache_path is not None:
            from ware_ops_algos.utils.io_helpers import dump_pickle
            dump_pickle(str(cache_path), domain)
        return domain

    def _parse(self, order_path: Path) -> tuple[dict, list[list[dict]], dict[int, float]]:
        number = re.match(r"(\d+)", order_path.stem).group(1)
        setting = self._parse_setting(order_path.parent / f"sett{number}.txt")
        arrivals = _read_arrival_times(_henn_arrival_file(order_path.parent, order_path.stem))
        setting["stem"] = order_path.stem
        orders: list[list[dict]] = []
        current: list[dict] = []
        for line in order_path.read_text(encoding="windows-1252").splitlines():
            parts = line.split()
            if line.startswith("Order"):
                if current:
                    orders.append(current)
                current = []
            elif parts and parts[0].isdigit():
                current.append({"aisle": int(parts[2]), "location": int(parts[4])})
        if current:
            orders.append(current)
        return setting, orders, arrivals

    @staticmethod
    def _parse_setting(path: Path) -> dict:
        values = {}
        for line in path.read_text(encoding="windows-1252").splitlines():
            key, separator, value = line.partition(":")
            key = key.strip()
            if separator and re.fullmatch(r"[a-z_]+", key) and value.strip():
                values[key.rstrip("_")] = value.strip()
        return values

    def _build(self, setting: dict, orders: list[list[dict]], arrivals: dict[int, float]):
        n_aisles = int(setting["no_aisles"])
        n_cells = int(setting["no_cells"])
        cell_length = float(setting["cell_lengt"])
        centre_distance = 2.0 * float(setting["cell_width"]) + float(setting["aisle_widt"])
        depot_stub = float(setting["dis_ais_wa"])
        capacity = float(setting["m_no_a_p_b"])

        from ware_ops_algos.domain_models import ShelfStorageGraphGenerator
        if self.depot == "corner":
            start_location = (1, -1)
            generator = ShelfStorageGraphGenerator(
                n_aisles=n_aisles, n_pick_locations=n_cells, dist_aisle=centre_distance,
                dist_pick_locations=cell_length, dist_aisle_location=cell_length,
                dist_start=depot_stub, dist_end=depot_stub,
                start_location=start_location, end_location=start_location,
                start_connection_point=(1, 0), end_connection_point=(1, 0))
            generator.populate_graph()
            graph = generator.G
            start_node = end_node = start_location
            closest = (1, 0)
        else:
            generator = ShelfStorageGraphGenerator(
                n_aisles=n_aisles, n_pick_locations=n_cells, dist_aisle=centre_distance,
                dist_pick_locations=cell_length, dist_aisle_location=cell_length,
                dist_start=0.0, dist_end=0.0,
                start_location=(1, 0), end_location=(1, 0),
                start_connection_point=(1, 0), end_connection_point=(1, 0))
            generator.populate_graph()
            graph = generator.G
            graph.remove_edge((1, 0), (1, 0))
            left, right = (n_aisles + 1) // 2, (n_aisles + 1) // 2 + 1
            middle = n_aisles / 2 + 0.5
            graph.remove_edge((left, 0), (right, 0))
            graph.add_node((middle, 0), pos=(middle, 0), type="depot_node")
            graph.add_edge((left, 0), (middle, 0), weight=centre_distance / 2.0)
            graph.add_edge((middle, 0), (right, 0), weight=centre_distance / 2.0)
            start_node = end_node = (middle, 0)
            closest = (middle, 0)
        start_location = start_node

        demand: dict[tuple[int, int, int], int] = defaultdict(int)
        for order in orders:
            for item in order:
                demand[(item["aisle"] // 2, item["aisle"] % 2, item["location"])] += 1
        article_of = {key: index for index, key in enumerate(sorted(demand))}
        positions = [[{"article_id": article_of[(item["aisle"] // 2, item["aisle"] % 2,
                                                  item["location"])], "amount": 1}
                      for item in order] for order in orders]
        articles = Articles(tpe=ArticleType.STANDARD, articles=[
            Article(article_id=index) for index in range(len(article_of))])
        storage = StorageLocations(tpe=StorageType.SCATTERED, locations=[
            Location(x=key[0] + 1, y=key[2], article_id=article_of[key], amount=amount)
            for key, amount in sorted(demand.items())])
        storage.build_article_location_mapping()

        layout_params = LayoutParameters(
            n_aisles=n_aisles, n_pick_locations=n_cells,
            dist_pick_locations=cell_length, dist_aisle=centre_distance,
            dist_top_to_pick_location=cell_length,
            dist_bottom_to_pick_location=cell_length,
            dist_start=depot_stub if self.depot == "corner" else 0.0,
            dist_end=depot_stub if self.depot == "corner" else 0.0,
            start_location=start_location, end_location=start_location,
            start_connection_point=closest, end_connection_point=closest,
            n_blocks=1, depot_location="bottom",
        )
        layout = LayoutData(tpe=LayoutType.CONVENTIONAL, graph_data=layout_params,
                            layout_network=_network(
                                graph, start_node, end_node, closest, 0, n_cells + 1))
        return _domain(layout, articles, _orders(positions, arrivals, setting["stem"]),
                       Resources(ResourceType.HUMAN,
                                 [_picker(capacity, DimensionType.ITEMS, start_location)]),
                       storage)


class AlbaredaLoader(DataLoader):
    """The 64 Albareda instances of the OOBP archive (warehouses W1 to W4).

    ``filepath`` is a ``wsrp_input_pedido_*.txt`` file; the layout is the
    ``wsrp_input_layout_*.txt`` file with the same suffix in the same directory.
    Adjacent aisles share their shelves, so the centre distance is the shelf
    width plus the aisle width; the depot from the layout file sits on the front
    cross aisle, at the leftmost aisle (0) or between the two middle aisles (1).
    """

    def __init__(self, instances_dir: str | Path, cfg: dict, cache_dir=None):
        super().__init__(instances_dir)
        self.cfg = cfg
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def load(self, filepath: str, use_cache: bool = True) -> SimWarehouseDomain:
        path = Path(filepath)
        if not path.is_absolute():
            path = self.data_dir / path
        cache_path = None
        if use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"{path.stem}_domain.pkl"
            if cache_path.exists():
                from ware_ops_algos.utils.io_helpers import load_pickle
                cached = load_pickle(str(cache_path))
                if isinstance(cached, SimWarehouseDomain):
                    return cached
        domain = self._build(*self._parse(path))
        if cache_path is not None:
            from ware_ops_algos.utils.io_helpers import dump_pickle
            dump_pickle(str(cache_path), domain)
        return domain

    def _parse(self, pedido_path: Path) -> tuple[dict, list[list[dict]], dict[int, float]]:
        match = re.fullmatch(r"wsrp_input_pedido_(\d{2})_(\d{3})\.txt", pedido_path.name)
        if not match:
            raise ValueError(f"unexpected pedido file name {pedido_path.name}")
        layout_path = pedido_path.parent / f"wsrp_input_layout_{match.group(1)}_{match.group(2)}.txt"
        layout_lines = layout_path.read_text(encoding="windows-1252").splitlines()

        def field(index: int) -> str:
            return layout_lines[index].strip()

        n_aisles = int(field(1).split()[0])
        setting = {
            "depot": int(field(3)),
            "storage": int(field(5)),
            "shelf_length": float(field(7).split()[0]),
            "shelf_width": float(field(7).split()[1]),
            "aisle_width": float(field(9)),
            "capacity": float(field(11)),
            "picking_time": float(field(13)),
            "turning": tuple(float(value) for value in field(15).split()),
            "aisles": {},
        }
        for line in layout_lines[17:]:
            parts = line.split()
            if len(parts) == 4 and parts[0].isdigit():
                setting["aisles"][int(parts[0])] = tuple(float(value) for value in parts[1:])
            elif parts and parts[0] == "9999":
                break
        if len(setting["aisles"]) != n_aisles:
            raise ValueError(f"{layout_path.name}: {len(setting['aisles'])} aisles, "
                             f"expected {n_aisles}")

        pedido_lines = pedido_path.read_text(encoding="windows-1252").splitlines()
        n_orders = int(pedido_lines[1])
        orders: list[list[dict]] = []
        index = 3
        while index < len(pedido_lines) and len(orders) < n_orders:
            n_items = int(pedido_lines[index].split()[1])
            orders.append([
                {"article_id": int(item[4]), "amount": 1, "weight": float(item[3]),
                 "aisle": int(item[0]), "position": float(item[2])}
                for item in (line.split() for line in pedido_lines[index + 1:index + 1 + n_items])
            ])
            index += n_items + 1

        arrival_path = pedido_path.parent / \
            f"Arrival_Time_W{int(match.group(1))}_{pedido_path.parent.name}_{match.group(2)}.csv"
        if not arrival_path.exists():
            raise FileNotFoundError(f"missing arrival CSV {arrival_path.name}")
        arrivals = _read_arrival_times(arrival_path)
        setting["stem"] = pedido_path.stem
        return setting, orders, arrivals

    def _build(self, setting: dict, orders: list[list[dict]], arrivals: dict[int, float]):
        n_aisles = len(setting["aisles"])
        shelf_length = setting["shelf_length"]
        spacing = setting["shelf_width"] + setting["aisle_width"]
        capacity = setting["capacity"]

        graph = nx.Graph()
        for aisle in range(n_aisles):
            x = aisle * spacing
            graph.add_node((x, 0.0), pos=(x, 0.0), type="change_aisle_node")
            graph.add_node((x, shelf_length), pos=(x, shelf_length),
                           type="change_aisle_node")
            if aisle:
                graph.add_edge(((aisle - 1) * spacing, 0.0), (x, 0.0), weight=spacing)
                graph.add_edge(((aisle - 1) * spacing, shelf_length),
                               (x, shelf_length), weight=spacing)
            positions = sorted({item["position"] for order in orders for item in order
                                if item["aisle"] == aisle})
            previous = 0.0
            for position in positions:
                graph.add_node((x, position), pos=(x, position), type="pick_node")
                graph.add_edge((x, previous), (x, position), weight=position - previous)
                previous = position
            graph.add_edge((x, previous), (x, shelf_length),
                           weight=shelf_length - previous)

        middle = n_aisles // 2
        if setting["depot"] == 0 or n_aisles % 2:
            depot = ((0.0 if setting["depot"] == 0 else middle * spacing), 0.0)
        else:
            depot = ((middle - 0.5) * spacing, 0.0)
            left, right = (middle - 1) * spacing, middle * spacing
            graph.remove_edge((left, 0.0), (right, 0.0))
            graph.add_edge((left, 0.0), (depot[0], 0.0), weight=spacing / 2.0)
            graph.add_edge((depot[0], 0.0), (right, 0.0), weight=spacing / 2.0)
        graph.add_node(depot, pos=depot, type="depot_node")

        articles = Articles(tpe=ArticleType.STANDARD, articles=[
            Article(article_id=item["article_id"], weight=item["weight"])
            for order in orders for item in order])
        positions = [[{"article_id": item["article_id"], "amount": 1}
                      for item in order] for order in orders]
        storage = StorageLocations(tpe=StorageType.SCATTERED, locations=[
            Location(x=item["aisle"] * spacing, y=item["position"],
                     article_id=item["article_id"], amount=1)
            for order in orders for item in order])
        storage.build_article_location_mapping()

        layout_params = LayoutParameters(
            n_aisles=n_aisles,
            n_pick_locations=max(len({item["position"] for order in orders
                                      for item in order if item["aisle"] == aisle})
                                 for aisle in range(n_aisles)),
            dist_pick_locations=1.0, dist_aisle=spacing,
            dist_top_to_pick_location=0.0, dist_bottom_to_pick_location=0.0,
            dist_start=0.0, dist_end=0.0,
            start_location=depot, end_location=depot,
            start_connection_point=depot, end_connection_point=depot,
            n_blocks=1, depot_location="bottom",
        )
        layout = LayoutData(tpe=LayoutType.CONVENTIONAL, graph_data=layout_params,
                            layout_network=_network(
                                graph, depot, depot, depot, 0.0, shelf_length))
        return _domain(layout, articles, _orders(positions, arrivals, setting["stem"]),
                       Resources(ResourceType.HUMAN,
                                 [_picker(capacity, DimensionType.WEIGHT, depot)]),
                       storage)
