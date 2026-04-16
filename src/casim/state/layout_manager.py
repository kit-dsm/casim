from ware_ops_algos.domain_models import LayoutData


class LayoutManager:
    def __init__(self, layout: LayoutData):
        self._layout = layout
        self._dima = self._layout.layout_network.distance_matrix
        nodes = list(self._dima.index)
        self._node_to_idx = {n: i for i, n in enumerate(nodes)}
        self._dist = self._dima.to_numpy(dtype=float, copy=False)

    def get_layout(self):
        return self._layout

    def get_distance(self, a, b) -> float:
        return float(self._dist[self._node_to_idx[a.position], self._node_to_idx[b.position]])
