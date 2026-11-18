"""A warehouse drawn as a grid of codes in a CSV file (the IJPE grocery warehouse).

Each cell of the semicolon-separated file, without header, is one of:

- ``-1`` … ``-8``: an aisle (the number is the aisle);
- ``-90``: a cross aisle (a full row of ``-90`` is one);
- a positive number ``HHPP``: a storage place, house ``HH`` and place ``PP``
  (``HPP`` for houses below 10), facing the aisle next to it;
- ``-99``: a special pick place; ``0``: nothing.

The graph follows the submitted model's loader (``scenarios/scenario_ijpe/
grocery_retailer_loader_digraph.py``, ``WarehousePickingLoaderDigraph``).
Its 16.45 m aisle-column scale, 3.85 m aisle width and 1.2 m storage columns
give 15 m between neighboring aisle rails across the cross aisle, matching
the warehouse correspondence. The CSV cell boxes are schematic and their
edges are not measured rack faces. Each aisle has a
left and a right rail; pick nodes lie on the
rail next to their storage face; the two rails of an aisle are joined, and the
neighbouring rails connected, only at cross-aisle levels; levels of one place
share its pick node; places in ``rb_ranges`` form one location each. Lengths
are millimetres. The depot is two nodes, start and end, joined to the bottom of
aisle 1's right rail at zero length.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall
from ware_ops_algos.domain_models import (
    BoundingBox, LayoutData, LayoutNetwork, LayoutParameters, LayoutType, StorageLocation,
    StorageSlot,
)

SIDE_LEFT, SIDE_RIGHT = "L", "R"
CROSS_AISLE, SPECIAL_PICK = -90, -99


@dataclass(frozen=True)
class GridLayout:
    """Dimensions of the grid's cells and the codes that need them (mm)."""

    place_width: int = 800
    place_length: int = 1200
    cross_aisle_width: int = 800
    aisle_column_width: int = 16450
    aisle_width: int = 3850
    aisle_keys: tuple[int, ...] = (-1, -2, -3, -4, -5, -6, -7, -8)
    rb_ranges: tuple[range, ...] = (range(2210, 2284), range(2310, 2384), range(2010, 2084))
    # A cross aisle in the middle that the CSV does not draw as a full row.
    manual_cross_aisle_y: float | None = 28400
    start_aisle: int = -1
    start_node: tuple[float, float] = (0, -10000)
    end_node: tuple[float, float] = (-1, -10000)

    # ------------------------------------------------------------- reading

    @staticmethod
    def read(path: Path) -> "pd.DataFrame":
        raw = pd.read_csv(path, sep=";", header=None, encoding="utf-8-sig")
        return raw.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    def build(self, path: Path) -> tuple[LayoutData, list[StorageLocation]]:
        """The layout with its graph, and the storage locations with their pick nodes."""
        cells = self.read(path).to_numpy()
        locations = self._storage_locations(cells)
        graph, rails, levels = self._graph(cells, locations)
        bottom, top = min(levels), max(levels)
        closest = (rails[(self.start_aisle, SIDE_RIGHT)], bottom)
        graph.add_node(self.start_node, pos=self.start_node, node_type="start_node")
        graph.add_node(self.end_node, pos=self.end_node, node_type="end_node")
        graph.add_edge(self.start_node, closest, weight=0, edge_type="depot")
        graph.add_edge(self.end_node, closest, weight=0, edge_type="depot")
        nodes = list(graph.nodes())
        adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", dtype=float)
        distances, predecessors = floyd_warshall(adjacency, directed=False,
                                                 return_predecessors=True)
        network = LayoutNetwork(
            graph=graph, distance_matrix=pd.DataFrame(distances, index=nodes, columns=nodes),
            predecessor_matrix=predecessors, closest_node_to_start=closest,
            min_aisle_position=bottom, max_aisle_position=top,
            start_node=self.start_node, end_node=self.end_node, node_list=nodes)
        parameters = LayoutParameters(
            n_aisles=len({aisle for aisle, _ in rails}), n_pick_locations=len(locations),
            n_blocks=max(1, len(levels) - 1), dist_pick_locations=self.place_width / 1000,
            dist_aisle=self.aisle_width / 1000, dist_top_to_pick_location=0,
            dist_bottom_to_pick_location=0, dist_start=0, dist_end=0,
            dist_cross_aisle=self.cross_aisle_width / 1000,
            start_location=self.start_node, end_location=self.end_node,
            start_connection_point=closest, end_connection_point=closest)
        return (LayoutData(tpe=LayoutType.CONVENTIONAL, graph_data=parameters,
                           layout_network=network), locations)

    @staticmethod
    def pick_nodes(locations: list[StorageLocation]) -> dict[tuple[int, int, int], tuple]:
        """Pick node of every (aisle, house, place) the grid draws.

        A place is keyed by its slot ids: ``house * 100 + place``, both levels
        of a place, and every slot of a rack range.
        """
        nodes = {}
        for location in locations:
            if location.pick_node is None:
                continue
            aisle = -location.aisle_id
            for slot in location.slots:
                number = int(slot.id)
                nodes[(aisle, number // 100, number % 100)] = location.pick_node
        return nodes

    # ------------------------------------------------------------- cells

    def _is_aisle(self, value: int) -> bool:
        return value in self.aisle_keys

    @staticmethod
    def _is_storage(value: int) -> bool:
        return value > 0

    def _column_width(self, column) -> int:
        if any(self._is_aisle(int(v)) for v in column):
            return self.aisle_column_width
        if any(self._is_storage(int(v)) or int(v) == SPECIAL_PICK for v in column):
            return self.place_length
        return self.cross_aisle_width

    def _column_offsets(self, cells) -> dict[int, int]:
        offsets, x = {}, 0
        for col in range(cells.shape[1]):
            offsets[col] = x
            x += self._column_width(cells[:, col])
        return offsets

    def _row_centers(self, cells) -> dict[int, float]:
        centers, y = {}, 0
        for row in reversed(range(cells.shape[0])):
            centers[row] = y + self.place_width / 2
            y += self.place_width
        return centers

    def _aisle_centers(self, cells) -> dict[int, float]:
        offsets, centers = self._column_offsets(cells), {}
        for col in range(cells.shape[1]):
            aisles = [int(v) for v in cells[:, col] if self._is_aisle(int(v))]
            if aisles:
                aisle = max(set(aisles), key=aisles.count)
                centers[aisle] = offsets[col] + self._column_width(cells[:, col]) / 2
        return centers

    def _cross_aisle_levels(self, cells) -> list[float]:
        centers = self._row_centers(cells)
        levels = [centers[row] for row in range(cells.shape[0])
                  if all(int(v) == CROSS_AISLE for v in cells[row, :])]
        height = cells.shape[0] * self.place_width
        if self.manual_cross_aisle_y is not None and 0 < self.manual_cross_aisle_y < height:
            levels.append(float(self.manual_cross_aisle_y))
        return sorted(set(levels))

    def _adjacent_aisle(self, cells, row: int, col: int) -> tuple[int | None, str | None]:
        left = int(cells[row, col - 1]) if col > 0 else None
        right = int(cells[row, col + 1]) if col < cells.shape[1] - 1 else None
        if left in self.aisle_keys:
            return left, SIDE_RIGHT
        if right in self.aisle_keys:
            return right, SIDE_LEFT
        return None, None

    @staticmethod
    def _base_place(place: int) -> int:
        return place - 1 if place % 10 in (1, 3, 5) else place

    def _slots(self, place: int) -> list[StorageSlot]:
        base = self._base_place(place)
        if base % 10 in (0, 2, 4):
            return [StorageSlot(id=str(base), level=0), StorageSlot(id=str(base + 1), level=1)]
        return [StorageSlot(id=str(place), level=0)]

    @staticmethod
    def _house(place: int) -> str:
        text = str(place)
        return text[:2] if len(text) == 4 else text[:1] if len(text) == 3 else "0"

    def _rb_range(self, place: int) -> range | None:
        return next((rb for rb in self.rb_ranges if place in rb), None)

    # ------------------------------------------------------------- locations and graph

    def _storage_locations(self, cells) -> list[StorageLocation]:
        locations: list[StorageLocation] = []
        offsets, centers = self._column_offsets(cells), self._row_centers(cells)
        seen_places, seen_ranges = set(), set()
        for col in range(cells.shape[1]):
            for row in reversed(range(cells.shape[0])):
                value = int(cells[row, col])
                if not self._is_storage(value):
                    continue
                aisle, side = self._adjacent_aisle(cells, row, col)
                if aisle is None:
                    continue
                base = self._base_place(value)
                rb = self._rb_range(base)
                if rb is not None:
                    key = (aisle, side, rb.start, rb.stop)
                    if key in seen_ranges:
                        continue
                    seen_ranges.add(key)
                    slots = [StorageSlot(id=str(number), level=0) for number in rb]
                    half, place = 3 * self.place_width / 2, rb.start
                else:
                    key = (aisle, side, base)
                    if key in seen_places:
                        continue
                    seen_places.add(key)
                    slots, half, place = self._slots(base), self.place_width / 2, base
                location = StorageLocation(
                    id=f"{-aisle}_{self._house(place)}",
                    bbox=BoundingBox(x_min=offsets[col], x_max=offsets[col] + self.place_length,
                                     y_min=centers[row] - half, y_max=centers[row] + half),
                    aisle_id=aisle, slots=slots)
                location.side = side
                location.pick_node = None
                locations.append(location)
        return locations

    def _graph(self, cells, locations: list[StorageLocation]):
        graph = nx.Graph()
        levels = self._cross_aisle_levels(cells)
        rails: dict[tuple[int, str], float] = {}
        for aisle, x in self._aisle_centers(cells).items():
            rails[(aisle, SIDE_LEFT)] = x - self.aisle_width / 2
            rails[(aisle, SIDE_RIGHT)] = x + self.aisle_width / 2
        for (aisle, side), x in rails.items():
            for y in levels:
                graph.add_node((x, y), pos=(x, y), node_type="intersection",
                               aisle_id=aisle, side=side)
        for aisle in {aisle for aisle, _ in rails}:
            for y in levels:
                graph.add_edge((rails[(aisle, SIDE_LEFT)], y), (rails[(aisle, SIDE_RIGHT)], y),
                               weight=self.aisle_width, edge_type="rung")
        for y in levels:
            row = sorted(((x, y) for x in rails.values()), key=lambda node: node[0])
            for a, b in zip(row, row[1:]):
                if not graph.has_edge(a, b):
                    graph.add_edge(a, b, weight=abs(b[0] - a[0]), edge_type="cross_aisle")
        for location in locations:
            x = rails.get((location.aisle_id, location.side))
            if x is None:
                continue
            node = (x, (location.bbox.y_min + location.bbox.y_max) / 2)
            location.pick_node = node
            graph.add_node(node, pos=node, node_type="pick_node", slot_id=location.id,
                           aisle_id=location.aisle_id, side=location.side)
        on_rail: dict[tuple[int, str], list] = defaultdict(list)
        for node, data in graph.nodes(data=True):
            if data.get("aisle_id") is not None and data.get("side") is not None:
                on_rail[(data["aisle_id"], data["side"])].append(node)
        for nodes in on_rail.values():
            ordered = sorted(set(nodes), key=lambda node: node[1])
            for a, b in zip(ordered, ordered[1:]):
                if a != b:
                    graph.add_edge(a, b, weight=abs(b[1] - a[1]), edge_type="vertical_aisle")
        return graph, rails, levels
