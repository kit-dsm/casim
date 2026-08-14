from collections import defaultdict, deque
import copy

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall

from ware_ops_algos.domain_models import LayoutData


class LayoutManager:
    def __init__(self, layout: LayoutData):
        self.layout = layout
        self._dima = self.layout.layout_network.distance_matrix
        nodes = list(self._dima.index)
        self._node_to_idx = {n: i for i, n in enumerate(nodes)}
        self._dist = self._dima.to_numpy(dtype=float, copy=False)
        self._occupants = defaultdict(list)
        self._waiters = defaultdict(deque)
        self._held_by_token = {}
        graph = self.layout.layout_network.graph
        self._edge_constraints = {}
        for origin, destination, attributes in graph.edges(data=True):
            zone_id = attributes.get("zone_id")
            capacity = attributes.get("capacity")
            if zone_id is not None:
                constraint = ("zone", zone_id), int(capacity or 1)
            elif capacity is not None:
                constraint = (
                    "edge",
                    self._edge_id(graph, origin, destination),
                ), int(capacity)
            else:
                constraint = None, None
            self._edge_constraints[
                self._edge_id(graph, origin, destination)
            ] = constraint
        self._node_constraints = {
            node: (("pick", node), int(attributes["pick_capacity"]))
            for node, attributes in graph.nodes(data=True)
            if attributes.get("pick_capacity") is not None
        }
        self._congestion_keys = {
            key
            for key, _ in self._edge_constraints.values()
            if key is not None
        }
        self._planning_cache_key = None
        self._planning_cache = None

    def get_distance(self, a, b) -> float:
        return float(self._dist[self._node_to_idx[a.position], self._node_to_idx[b.position]])

    @staticmethod
    def _edge_id(graph, origin, destination):
        if graph.is_directed():
            return origin, destination
        return tuple(sorted((origin, destination), key=repr))

    def _edge_capacity(self, origin, destination):
        graph = self.layout.layout_network.graph
        return self._edge_constraints.get(
            self._edge_id(graph, origin, destination),
            (None, None),
        )

    def _node_capacity(self, node):
        return self._node_constraints.get(node, (None, None))

    def _request(self, key, capacity, token) -> bool:
        if key is None:
            return True
        if self._held_by_token.get(token) == key:
            return True
        if len(self._occupants[key]) < capacity:
            self._occupants[key].append(token)
            self._held_by_token[token] = key
            return True
        if token not in self._waiters[key]:
            self._waiters[key].append(token)
        return False

    def request_travel(self, origin, destination, token) -> bool:
        key, capacity = self._edge_capacity(origin, destination)
        return self._request(key, capacity, token)

    def request_pick(self, node, token) -> bool:
        key, capacity = self._node_capacity(node)
        return self._request(key, capacity, token)

    def _release(self, token):
        key = self._held_by_token.pop(token, None)
        if key is None:
            return []
        occupants = self._occupants[key]
        if token in occupants:
            occupants.remove(token)
        waiters = list(self._waiters.pop(key, ()))
        return waiters

    def release_travel(self, token):
        return self._release(token)

    def release_pick(self, token):
        return self._release(token)

    def replace_token(self, old_token, new_token) -> None:
        key = self._held_by_token.pop(old_token, None)
        if key is not None:
            occupants = self._occupants[key]
            occupants[occupants.index(old_token)] = new_token
            self._held_by_token[new_token] = key
        for waiters in self._waiters.values():
            try:
                waiters.remove(old_token)
            except ValueError:
                pass

    def release_all(self, token):
        waiters = self._release(token)
        for queued in self._waiters.values():
            try:
                queued.remove(token)
            except ValueError:
                pass
        return waiters

    def planning_layout(self, congestion_penalty: float = 0.0):
        """Return static layout or a detached occupied-edge cost snapshot."""
        if congestion_penalty <= 0:
            return self.layout
        occupied = tuple(
            sorted(
                (
                    (key, len(tokens))
                    for key, tokens in self._occupants.items()
                    if tokens and key in self._congestion_keys
                ),
                key=lambda item: repr(item[0]),
            )
        )
        if not occupied:
            return self.layout
        cache_key = float(congestion_penalty), occupied
        if cache_key == self._planning_cache_key:
            return self._planning_cache
        projected = copy.copy(self.layout)
        network = copy.copy(self.layout.layout_network)
        graph = network.graph.copy()
        for origin, destination, attributes in graph.edges(data=True):
            key, _ = self._edge_capacity(origin, destination)
            occupancy = len(self._occupants.get(key, ())) if key else 0
            if occupancy:
                attributes["weight"] = float(attributes["weight"]) + (
                    float(congestion_penalty) * occupancy
                )
        nodes = list(graph.nodes)
        adjacency = nx.to_scipy_sparse_array(
            graph,
            nodelist=nodes,
            weight="weight",
            dtype=float,
        )
        distances, predecessors = floyd_warshall(
            adjacency,
            directed=graph.is_directed(),
            return_predecessors=True,
        )
        network.graph = graph
        network.node_list = nodes
        network.distance_matrix = pd.DataFrame(
            distances,
            index=nodes,
            columns=nodes,
        )
        network.predecessor_matrix = np.asarray(predecessors)
        projected.layout_network = network
        self._planning_cache_key = cache_key
        self._planning_cache = projected
        return projected
