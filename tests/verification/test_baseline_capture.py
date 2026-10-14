"""Hand-derived distances between positions off the layout graph.

Layout: two aisles at x = 1 and x = 2 with nodes at y = 0, 1, 2, 3, joined
by a front cross aisle at y = 0 and a rear cross aisle at y = 3. Every edge
has length 1. A position off the graph lies inside an aisle segment and is
reached through the node below or the node above it.

Two aisles, pa = (1, 1.4), pb = (2, 1.6):
    through the front: 1.4 down + 1 across + 1.6 up = 4.0
    through the rear:  1.6 up + 1 across + 1.4 down = 4.0
The nearest nodes alone, (1, 1) and (2, 2), give 0.4 + 4 + 0.4 = 4.8.

One aisle segment, pa = (1, 1.2), pb = (1, 1.7): 0.5 walked directly.
Through the nearest nodes (1, 1) and (1, 2) it would be 0.2 + 1 + 0.3 = 1.5.

One aisle, two segments, pa = (1, 0.5), pb = (1, 2.5): 0.5 + 1 + 0.5 = 2.0.
"""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pandas as pd
import pytest

from casim.baseline_capture import BaselineCapture


def _ladder():
    graph = nx.Graph()
    for x in (1, 2):
        for y in range(3):
            graph.add_edge((x, y), (x, y + 1), weight=1.0)
    graph.add_edge((1, 0), (2, 0), weight=1.0)
    graph.add_edge((1, 3), (2, 3), weight=1.0)
    nodes = sorted(graph.nodes)
    lengths = dict(nx.all_pairs_dijkstra_path_length(graph))
    matrix = pd.DataFrame([[lengths[a][b] for b in nodes] for a in nodes],
                          index=nodes, columns=nodes)
    network = SimpleNamespace(distance_matrix=matrix)
    return SimpleNamespace(
        get_layout=lambda: SimpleNamespace(layout_network=network))


@pytest.mark.parametrize(("pa", "pb", "expected"), [
    ((1, 1.4), (2, 1.6), 4.0),
    ((1.0 + 1e-12, 1.4), (2, 1.6), 4.0),
    ((1, 1.2), (1, 1.7), 0.5),
    ((1, 0.5), (1, 2.5), 2.0),
])
def test_position_distance_uses_both_aisle_neighbours(pa, pb, expected):
    distance, reason = BaselineCapture._position_distance(_ladder(), pa, pb)
    assert reason is None
    assert distance == pytest.approx(expected)
