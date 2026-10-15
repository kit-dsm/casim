"""A conventional one-block layout with unit distances, built from its parameters."""

from __future__ import annotations

import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall
from ware_ops_algos.domain_models import (
    LayoutData, LayoutNetwork, LayoutParameters, LayoutType, ShelfStorageGraphGenerator,
)


def conventional_layout(
    n_aisles: int = 8,
    n_pick_locations: int = 16,
    *,
    start=(-1, -1),
    end=(-2, -1),
    start_connection_point=None,
    end_connection_point=None,
) -> LayoutData:
    start_connection_point = start_connection_point or (1, 0)
    end_connection_point = end_connection_point or (1, 0)
    params = LayoutParameters(
        n_aisles=n_aisles,
        n_pick_locations=n_pick_locations,
        n_blocks=1,
        dist_top_to_pick_location=1.0,
        dist_bottom_to_pick_location=1.0,
        dist_pick_locations=1.0,
        dist_aisle=1.0,
        dist_start=1.0,
        dist_end=1.0,
        start_location=start,
        end_location=end,
        start_connection_point=start_connection_point,
        end_connection_point=end_connection_point,
        depot_location="front_left",
    )
    generator = ShelfStorageGraphGenerator(
        n_aisles=n_aisles,
        n_pick_locations=n_pick_locations,
        dist_aisle=1.0,
        dist_pick_locations=1.0,
        dist_aisle_location=1.0,
        dist_start=1.0,
        dist_end=1.0,
        start_location=start,
        end_location=end,
        start_connection_point=start_connection_point,
        end_connection_point=end_connection_point,
    )
    generator.populate_graph()
    graph = generator.G
    graph.nodes[start]["pos"] = start
    graph.nodes[end]["pos"] = end
    nodes = list(graph.nodes)
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", dtype=float)
    distances, predecessors = floyd_warshall(adjacency, directed=False,
                                             return_predecessors=True)
    network = LayoutNetwork(
        graph=graph,
        distance_matrix=pd.DataFrame(distances, index=nodes, columns=nodes),
        predecessor_matrix=predecessors,
        closest_node_to_start=start_connection_point,
        min_aisle_position=0,
        max_aisle_position=n_pick_locations + 1,
        start_node=start,
        end_node=end,
        node_list=nodes,
    )
    return LayoutData(tpe=LayoutType.CONVENTIONAL, graph_data=params,
                      layout_network=network)
