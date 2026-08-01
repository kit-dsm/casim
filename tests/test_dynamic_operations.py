from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scenarios.scenario_dynamic_operations.experiment_dynamic_operations import (
    run,
)
from scenarios.scenario_dynamic_operations.loader import (
    DynamicOperationsLoader,
)
from casim.simulation_engine.state_adapter import ActiveTourRoutingAdapter
from casim.state.layout_manager import LayoutManager
from ware_ops_algos.algorithms import NodeType, RouteNode

ROOT = Path(__file__).parents[1]
SCENARIO = ROOT / "scenarios" / "scenario_dynamic_operations"


def _config(policy, output):
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(SCENARIO / "config"),
    ):
        cfg = compose(
            config_name="dynamic_operations_config",
            overrides=[f"engines={policy}", "simulation=smoke"],
        )
    OmegaConf.update(cfg, "project_root", str(ROOT), merge=False)
    OmegaConf.update(
        cfg,
        "instances_base",
        str(ROOT / "scenarios"),
        merge=False,
    )
    OmegaConf.update(cfg, "cache_base", str(output / "cache"), merge=False)
    OmegaConf.update(
        cfg,
        "experiment.output_dir",
        str(output),
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "experiment.working_dir",
        str(output / "work"),
        merge=False,
    )
    OmegaConf.update(cfg, "viz.record", False, merge=False)
    return cfg


def test_seeded_loader_is_repeatable_and_uses_independent_carts():
    loader = DynamicOperationsLoader(SCENARIO)
    first = loader.load(SCENARIO / "data" / "smoke.json")
    second = loader.load(SCENARIO / "data" / "smoke.json")
    signature = lambda domain: [
        (
            order.order_id,
            order.order_date,
            order.due_date,
            tuple(position.article_id for position in order.order_positions),
        )
        for order in domain.orders.orders
    ]
    assert signature(first) == signature(second)
    assert len(first.orders.orders) == 32
    assert len({id(value.pick_cart) for value in first.resources.resources}) == 3
    assert all(
        attributes.get("pick_capacity") == 1
        for _, attributes in first.layout.layout_network.graph.nodes(data=True)
        if attributes.get("type") == "pick_node"
    )


def test_layout_capacity_waits_fifo_and_directed_origin_continues_forward():
    domain = DynamicOperationsLoader(SCENARIO).load(
        SCENARIO / "data" / "smoke.json"
    )
    manager = LayoutManager(domain.layout)
    edge = next(
        (origin, destination)
        for origin, destination, attributes
        in domain.layout.layout_network.graph.edges(data=True)
        if attributes.get("capacity") == 2
    )
    assert manager.request_travel(*edge, (0, 1, 0))
    assert manager.request_travel(*edge, (1, 2, 0))
    assert not manager.request_travel(*edge, (2, 3, 0))
    assert manager.release_travel((0, 1, 0)) == [(2, 3, 0)]

    directed_layout = deepcopy(domain.layout)
    directed_layout.layout_network.graph = nx.DiGraph(
        directed_layout.layout_network.graph
    )
    origin_node, destination_node = edge
    synthetic = RouteNode((0.25, 0.25), NodeType.ROUTE)
    tour = SimpleNamespace(
        edge_origin=RouteNode(origin_node, NodeType.ROUTE),
        edge_destination=RouteNode(destination_node, NodeType.ROUTE),
        edge_distance=2.0,
    )
    projected = ActiveTourRoutingAdapter._layout_with_origin(
        directed_layout,
        tour,
        synthetic,
        0.5,
    )
    graph = projected.layout_network.graph
    assert graph.has_edge(synthetic.position, destination_node)
    assert not graph.has_edge(synthetic.position, origin_node)


@pytest.mark.parametrize(
    "policy",
    [
        "nightly_static",
        "periodic_rolling",
        "unstarted_reopt",
        "routing_intervention",
    ],
)
def test_reduced_dynamic_policies_are_deterministic(policy, tmp_path):
    results = [
        run(_config(policy, tmp_path / repetition))
        for repetition in ("first", "second")
    ]
    stable = deepcopy(results)
    for result in stable:
        result.pop("algorithm_runtime_s")
        result.pop("decision_elapsed_s")
    assert stable[0] == stable[1]
    assert results[0]["received_orders"] == 32
    assert results[0]["completion_reason"] == "horizon_complete"
    assert results[0]["route_replacements"] <= results[0]["interventions"]
