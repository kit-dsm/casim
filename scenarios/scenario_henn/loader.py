from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall

from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.data_loaders.generators import ShelfStorageGraphGenerator
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


ORDER_RE = re.compile(
    r"Order\s+(\d+)\s+number\s+of\s+articles\s+(\d+)"
)
ITEM_RE = re.compile(
    r"(\d+)\s+Aisle\s+(\d+)\s+Location\s+(\d+)"
)


def load_manifest(path: str | Path) -> dict[str, list[str]]:
    """Return the checked-in exact source mapping."""
    manifest = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    entries = manifest.get("instances", {})
    if not isinstance(entries, dict) or len(entries) != 64:
        raise ValueError(
            f"{path}: expected 64 manifest entries, found {len(entries)}"
        )
    return entries


class HennDataLoader(DataLoader):
    """Load one raw Henn/Gil W5 instance into a CASIM domain."""

    def __init__(
        self,
        instances_dir: str | Path,
        aisle_end_offset: float = 1.5,
        depot_half_span: float | None = None,
        travel_speed: float = 0.8,
        time_per_pick: float = 10.0,
        tour_setup_time: float = 180.0,
    ):
        super().__init__(Path(instances_dir).resolve())
        self.aisle_end_offset = aisle_end_offset
        self.depot_half_span = depot_half_span
        self.travel_speed = travel_speed
        self.time_per_pick = time_per_pick
        self.tour_setup_time = tour_setup_time

    def load(
        self,
        manifest_path: str | Path,
        instance_id: str,
        **kwargs,
    ) -> SimWarehouseDomain:
        manifest_path = self._resolve_root_path(manifest_path)
        entries = load_manifest(manifest_path)
        if instance_id not in entries:
            raise ValueError(f"Unknown Henn instance id: {instance_id}")
        paths = entries[instance_id]
        if not isinstance(paths, list) or len(paths) != 3:
            raise ValueError(
                f"Manifest entry {instance_id} must contain three paths."
            )
        instance_path, settings_path, arrivals_path = [
            self.data_dir / "data" / value for value in paths
        ]
        for path in (instance_path, settings_path, arrivals_path):
            if not path.is_file():
                raise ValueError(f"Manifest source does not exist: {path}")

        settings = self._parse_settings(settings_path)
        raw_orders = self._parse_instance(instance_path)
        arrivals = self._parse_arrivals(arrivals_path)
        self._validate(instance_id, instance_path, settings, raw_orders, arrivals)
        return self._build_domain(
            instance_id,
            settings,
            raw_orders,
            arrivals,
            tuple(str(path) for path in (instance_path, settings_path, arrivals_path)),
        )

    def _resolve_root_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.data_dir / path

    def _parse_settings(self, path: Path) -> dict[str, str]:
        settings: dict[str, str] = {}
        for line in self._load_text(path):
            if ":" not in line:
                break
            key, value = line.split(":", 1)
            settings[key.strip()] = value.strip()
        required = {
            "no_aisles_",
            "no_cells__",
            "cell_lengt",
            "cell_width",
            "aisle_widt",
            "routing___",
            "no_orders_",
            "m_no_a_p_b",
        }
        missing = sorted(required - settings.keys())
        if missing:
            raise ValueError(f"{path}: missing settings {missing}")
        return settings

    def _parse_instance(
        self,
        path: Path,
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        lines = [line.strip() for line in self._load_text(path) if line.strip()]
        orders: list[tuple[int, list[tuple[int, int]]]] = []
        index = 0
        while index < len(lines):
            match = ORDER_RE.fullmatch(lines[index])
            if match is None:
                raise ValueError(
                    f"{path}:{index + 1}: invalid order header {lines[index]!r}"
                )
            order_id, count = map(int, match.groups())
            items: list[tuple[int, int]] = []
            for offset in range(1, count + 1):
                if index + offset >= len(lines):
                    raise ValueError(f"{path}: truncated order {order_id}")
                item_match = ITEM_RE.fullmatch(lines[index + offset])
                if item_match is None:
                    raise ValueError(
                        f"{path}:{index + offset + 1}: invalid item row"
                    )
                item_index, aisle_face, location = map(int, item_match.groups())
                if item_index != offset - 1:
                    raise ValueError(
                        f"{path}: order {order_id} item ids are not sequential"
                    )
                if not 0 <= aisle_face < 20 or not 0 <= location < 45:
                    raise ValueError(
                        f"{path}: out-of-range location "
                        f"({aisle_face}, {location})"
                    )
                items.append((aisle_face, location))
            orders.append((order_id, items))
            index += count + 1
        return orders

    def _parse_arrivals(self, path: Path) -> dict[int, float]:
        rows = self._load_text(path)
        if len(rows) < 4 or rows[2].strip() != "Order ID; Arrival Time":
            raise ValueError(f"{path}: unexpected arrival header")
        arrivals: dict[int, float] = {}
        previous = float("-inf")
        for line_number, row in enumerate(rows[3:], start=4):
            if not row.strip():
                continue
            try:
                raw_id, raw_time = row.split(";")
                order_id = int(raw_id) - 1
                arrival_s = int(raw_time) / 1000.0
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid arrival row"
                ) from exc
            if order_id in arrivals:
                raise ValueError(f"{path}: duplicate order id {order_id + 1}")
            if arrival_s < previous:
                raise ValueError(f"{path}: arrival times are not monotone")
            arrivals[order_id] = arrival_s
            previous = arrival_s
        return arrivals

    def _validate(
        self,
        instance_id: str,
        instance_path: Path,
        settings: dict[str, str],
        orders: list[tuple[int, list[tuple[int, int]]]],
        arrivals: dict[int, float],
    ) -> None:
        ids = [order_id for order_id, _ in orders]
        expected = list(range(len(orders)))
        if ids != expected:
            raise ValueError(f"{instance_path}: order ids are not sequential")
        if sorted(arrivals) != expected:
            raise ValueError(
                f"{instance_path}: orders and arrivals do not map one-to-one"
            )
        configured_count = int(settings["no_orders_"])
        if len(orders) != configured_count:
            raise ValueError(
                f"{instance_path}: parsed {len(orders)} orders, "
                f"settings declare {configured_count}"
            )
        id_parts = instance_id.split("_")
        if len(id_parts) != 4 or int(id_parts[2]) != len(orders):
            raise ValueError(
                f"{instance_id}: normalized id disagrees with source order count"
            )
        filename = re.fullmatch(
            r"(\d+)([sl])-(\d+)-(\d+)-0\.txt",
            instance_path.name,
        )
        if filename is None:
            raise ValueError(
                f"{instance_path}: invalid Henn instance filename"
            )
        setting, routing, order_count, capacity = filename.groups()
        if int(id_parts[3]) != int(setting):
            raise ValueError(
                f"{instance_id}: setting id disagrees with filename"
            )
        if int(order_count) != len(orders):
            raise ValueError(
                f"{instance_path}: filename order count disagrees with data"
            )
        if settings["routing___"].lower() != routing:
            raise ValueError(
                f"{instance_path}: routing code disagrees with settings"
            )
        if int(settings["m_no_a_p_b"]) != int(capacity):
            raise ValueError(
                f"{instance_path}: capacity disagrees with settings"
            )

    def _build_layout(self, settings: dict[str, str]) -> LayoutData:
        n_aisles = int(settings["no_aisles_"])
        n_cells = int(settings["no_cells__"])
        cell_length = float(settings["cell_lengt"])
        dist_aisle = (
            2 * float(settings["cell_width"])
            + float(settings["aisle_widt"])
        )
        end_offset = self.aisle_end_offset
        depot_half_span = (
            self.depot_half_span
            if self.depot_half_span is not None
            else dist_aisle / 2
        )
        left_centre = n_aisles // 2
        right_centre = left_centre + 1
        start_location = (-1, -1)
        end_location = (-2, -1)

        params = LayoutParameters(
            n_aisles=n_aisles,
            n_pick_locations=n_cells,
            n_blocks=1,
            dist_top_to_pick_location=end_offset,
            dist_bottom_to_pick_location=end_offset,
            dist_pick_locations=cell_length,
            dist_aisle=dist_aisle,
            dist_start=depot_half_span,
            dist_end=depot_half_span,
            start_location=start_location,
            end_location=end_location,
            start_connection_point=(left_centre, 0),
            end_connection_point=(right_centre, 0),
            depot_location="front_center",
        )
        generator = ShelfStorageGraphGenerator(
            n_aisles=n_aisles,
            n_pick_locations=n_cells,
            dist_aisle=dist_aisle,
            dist_pick_locations=cell_length,
            dist_aisle_location=end_offset,
            dist_start=depot_half_span,
            dist_end=depot_half_span,
            start_location=start_location,
            end_location=end_location,
            start_connection_point=(left_centre, 0),
            end_connection_point=(right_centre, 0),
        )
        generator.populate_graph()
        graph = generator.G
        graph.add_edge(
            start_location,
            (right_centre, 0),
            weight=depot_half_span,
        )
        graph.add_edge(
            end_location,
            (left_centre, 0),
            weight=depot_half_span,
        )
        graph.nodes[start_location]["pos"] = (n_aisles / 2 + 0.5, 0)
        graph.nodes[end_location]["pos"] = (n_aisles / 2 + 0.5, 0)

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
        network = LayoutNetwork(
            graph=graph,
            distance_matrix=pd.DataFrame(
                distances,
                index=nodes,
                columns=nodes,
            ),
            predecessor_matrix=predecessors,
            closest_node_to_start=(left_centre, 0),
            min_aisle_position=0,
            max_aisle_position=n_cells + 1,
            start_node=start_location,
            end_node=end_location,
            node_list=nodes,
        )
        return LayoutData(
            tpe=LayoutType.CONVENTIONAL,
            graph_data=params,
            layout_network=network,
        )

    def _build_domain(
        self,
        instance_id: str,
        settings: dict[str, str],
        raw_orders: list[tuple[int, list[tuple[int, int]]]],
        arrivals: dict[int, float],
        source_paths: tuple[str, str, str],
    ) -> SimWarehouseDomain:
        demand = Counter(
            aisle_face * 45 + location
            for _, items in raw_orders
            for aisle_face, location in items
        )
        article_list = [
            Article(article_id=article_id, weight=1.0)
            for article_id in range(900)
        ]
        articles = Articles(ArticleType.STANDARD, article_list)
        storage = StorageLocations(
            tpe=StorageType.DEDICATED,
            locations=[
                Location(
                    x=article_id // 45 // 2 + 1,
                    y=article_id % 45 + 1,
                    article_id=article_id,
                    amount=demand.get(article_id, 0),
                )
                for article_id in range(900)
            ],
        )
        storage.build_article_location_mapping()

        order_list = [
            Order(
                order_id=order_id,
                order_date=arrivals[order_id],
                due_date=None,
                order_positions=[
                    OrderPosition(
                        order_number=order_id,
                        article_id=aisle_face * 45 + location,
                        amount=1,
                    )
                    for aisle_face, location in items
                ],
            )
            for order_id, items in raw_orders
        ]
        orders = OrdersDomain(OrderType.STANDARD, order_list)

        capacity = int(settings["m_no_a_p_b"])
        cart = PickCart(
            n_dimension=1,
            capacities=[capacity],
            dimensions=[DimensionType.ITEMS],
            n_boxes=1,
            box_can_mix_orders=True,
        )
        layout = self._build_layout(settings)
        resources = Resources(
            ResourceType.HUMAN,
            [
                Resource(
                    id=0,
                    capacity=capacity,
                    speed=self.travel_speed,
                    time_per_pick=self.time_per_pick,
                    pick_cart=cart,
                    tour_setup_time=self.tour_setup_time,
                    available=True,
                    occupied=False,
                    current_location=layout.graph_data.start_location,
                )
            ],
        )
        dynamic = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=0.0,
            done=False,
        )
        domain = SimWarehouseDomain(
            problem_class="OBRP",
            objective="distance",
            layout=layout,
            articles=articles,
            orders=orders,
            resources=resources,
            storage=storage,
            dynamic_warehouse_info=dynamic,
        )
        domain.henn_metadata = {
            "instance_id": instance_id,
            "source_routing_code": settings["routing___"].lower(),
            "effective_routing": "s",
            "capacity": capacity,
            "source_paths": source_paths,
            "raw_settings": dict(settings),
        }
        return domain
