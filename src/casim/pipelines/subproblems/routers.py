"""One way to build a router from a layout and the pickers it routes for."""

from ware_ops_algos.algorithms import RatliffRosenthalRouting
from ware_ops_algos.algorithms.routing.routing import Routing
from ware_ops_algos.domain_models import LayoutData


def router_arguments(router_class: type[Routing], layout: LayoutData, pickers,
                     *, gen_tour: bool = True, gen_item_sequence: bool = True,
                     **options) -> dict:
    """The constructor arguments of ``router_class`` for ``layout`` and ``pickers``.

    Every router gets the network's nodes, distance and predecessor matrices and
    node indices; Ratliff-Rosenthal also gets the aisle geometry of the graph
    parameters. ``options`` (a routing origin, a time limit) are passed on.
    """
    network = layout.layout_network
    nodes = list(network.graph.nodes)
    arguments = {
        "start_node": network.start_node,
        "end_node": network.end_node,
        "closest_node_to_start": network.closest_node_to_start,
        "min_aisle_position": network.min_aisle_position,
        "max_aisle_position": network.max_aisle_position,
        "distance_matrix": network.distance_matrix,
        "predecessor_matrix": network.predecessor_matrix,
        "picker": list(pickers),
        "gen_tour": gen_tour,
        "gen_item_sequence": gen_item_sequence,
        "node_list": network.node_list,
        "node_to_idx": {node: index for index, node in enumerate(nodes)},
        "idx_to_node": dict(enumerate(nodes)),
    }
    if issubclass(router_class, RatliffRosenthalRouting):
        graph = layout.graph_data
        arguments.update({
            "n_aisles": graph.n_aisles,
            "n_pick_locations": graph.n_pick_locations,
            "dist_aisle": graph.dist_aisle,
            "dist_pick_locations": graph.dist_pick_locations,
            "dist_aisle_location": graph.dist_bottom_to_pick_location,
            "dist_start": graph.dist_start,
            "dist_end": graph.dist_end,
        })
    arguments.update(options)
    return arguments


def build_router(router_class: type[Routing], layout: LayoutData, pickers,
                 **options) -> Routing:
    """A router of ``router_class`` for ``layout`` and ``pickers``; see ``router_arguments``."""
    return router_class(**router_arguments(router_class, layout, pickers, **options))
