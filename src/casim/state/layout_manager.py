from collections import defaultdict, deque
import copy

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall

from ware_ops_algos.domain_models import LayoutData


class LayoutManager:
    def __init__(self, layout: LayoutData):
        self._layout = layout
        self._dima = self._layout.layout_network.distance_matrix
        nodes = list(self._dima.index)
        self._node_to_idx = {n: i for i, n in enumerate(nodes)}
        self._dist = self._dima.to_numpy(dtype=float, copy=False)
        self._occupants = defaultdict(list)
        self._waiters = defaultdict(deque)
        self._held_by_token = {}

    def get_layout(self):
        return self._layout

    def get_distance(self, a, b) -> float:
        return float(self._dist[self._node_to_idx[a.position], self._node_to_idx[b.position]])

    @staticmethod
    def _edge_id(graph, origin, destination):
        if graph.is_directed():
            return origin, destination
        return tuple(sorted((origin, destination), key=repr))

    def _edge_capacity(self, origin, destination):
        graph = self._layout.layout_network.graph
        if not graph.has_edge(origin, destination):
            return None, None
        attributes = graph.get_edge_data(origin, destination) or {}
        zone_id = attributes.get("zone_id")
        capacity = attributes.get("capacity")
        if zone_id is not None:
            return ("zone", zone_id), int(capacity or 1)
        if capacity is not None:
            return (
                "edge",
                self._edge_id(graph, origin, destination),
            ), int(capacity)
        return None, None

    def _node_capacity(self, node):
        graph = self._layout.layout_network.graph
        if node not in graph:
            return None, None
        capacity = graph.nodes[node].get("pick_capacity")
        if capacity is None:
            return None, None
        return ("pick", node), int(capacity)

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
            return self._layout
        projected = copy.copy(self._layout)
        network = copy.copy(self._layout.layout_network)
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
        return projected
