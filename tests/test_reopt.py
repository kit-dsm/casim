from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from ware_ops_algos.algorithms import (
    WarehouseOrder,
    exact_lorenz_makespan,
)
from ware_ops_algos.domain_models import DimensionType

from casim.pipelines.pipeline_runner import CoSySolver
from casim.setup import build_runtime
from ware_ops_algos.domain_models import load_and_flatten_data_card
from scenarios.scenario_reopt.experiment_reopt import run_experiment
from scenarios.scenario_reopt.loader import ReoptDataLoader
from scenarios.scenario_reopt.solver import (
    LorenzDPSolver,
    _resolved_orders,
)


ROOT = Path(__file__).parents[1]
SCENARIO = ROOT / "scenarios" / "scenario_reopt"


def _domain():
    return ReoptDataLoader(SCENARIO).load(
        SCENARIO / "data" / "paper_example.json"
    )


def _solver(release_mode: str) -> LorenzDPSolver:
    return LorenzDPSolver(
        batch_capacity_orders=2,
        max_states=2_000_000,
        max_runtime_s=60.0,
        release_mode=release_mode,
        problem_class="OBRSP",
    )


def _config(variant: str, output_dir: Path):
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str((SCENARIO / "config").resolve()),
    ):
        cfg = compose(
            config_name="reopt_config",
            overrides=[f"variant={variant}"],
        )
    with open_dict(cfg):
        cfg.project_root = str(ROOT.resolve())
        cfg.instances_base = str((ROOT / "scenarios").resolve())
        cfg.cache_base = str((output_dir / "cache").resolve())
        cfg.experiment.output_dir = str(output_dir.resolve())
        cfg.experiment.working_dir = str((output_dir / "work").resolve())
    return cfg


def _brute_static(orders, layout, picker, capacity):
    matrix = layout.layout_network.distance_matrix
    depot = layout.layout_network.start_node
    end = layout.layout_network.end_node
    best = float("inf")
    for order_permutation in itertools.permutations(orders):
        for separators in itertools.product(
            (False, True),
            repeat=len(orders) - 1,
        ):
            batches = []
            batch = [order_permutation[0]]
            for order, separator in zip(
                order_permutation[1:],
                separators,
            ):
                if separator:
                    batches.append(batch)
                    batch = []
                batch.append(order)
            batches.append(batch)
            if any(len(value) > capacity for value in batches):
                continue
            total = 0.0
            for value in batches:
                positions = [
                    position
                    for order in value
                    for position in order.pick_positions
                ]
                batch_best = float("inf")
                for sequence in itertools.permutations(positions):
                    distance = float(
                        matrix.at[depot, sequence[0].pick_node]
                    )
                    distance += sum(
                        float(matrix.at[left.pick_node, right.pick_node])
                        for left, right in zip(sequence, sequence[1:])
                    )
                    distance += float(
                        matrix.at[sequence[-1].pick_node, end]
                    )
                    batch_best = min(
                        batch_best,
                        distance / picker.speed
                        + len(sequence) * picker.time_per_pick,
                    )
                total += batch_best
            best = min(best, total)
    return best


def test_loader_builds_paper_semantic_domain():
    domain = _domain()
    assert domain.problem_class == "OBRSP"
    assert domain.objective == "makespan"
    assert [order.order_date for order in domain.orders.orders] == [
        10.0,
        42.0,
        50.0,
    ]
    picker = domain.resources.resources[0]
    assert picker.speed == 1.0
    assert picker.time_per_pick == 5.0
    assert picker.tour_setup_time == 0.0
    assert picker.pick_cart.capacities == [2]
    assert picker.pick_cart.dimensions == [DimensionType.ORDERS]


def test_published_example_cios():
    domain = _domain()
    solution, name, objective = _solver("actual").solve(domain)
    assert name == "LorenzMakespanDP"
    assert objective == 76.0
    assert [
        sorted(job.order_numbers) for job in solution.jobs
    ] == [[1, 2], [3]]
    turnovers = [
        solution.jobs[0].end_time - 10,
        solution.jobs[0].end_time - 42,
        solution.jobs[1].end_time - 50,
    ]
    assert sum(turnovers) / len(turnovers) == pytest.approx(
        37.333333333333336
    )
    assert solution.is_optimal is True


def test_dp_matches_exhaustive_static_paper_subset():
    domain = _domain()
    orders = [
        WarehouseOrder(
            order_id=order.order_id,
            parent_order_id=order.parent_order_id,
            due_date=order.due_date,
            order_date=0.0,
            pick_positions=order.pick_positions,
        )
        for order in _resolved_orders(domain, 0.0)[:2]
    ]
    solution = exact_lorenz_makespan(
        orders,
        domain.layout,
        domain.resources.resources[0],
        2,
    )
    assert solution.objective_value == pytest.approx(
        _brute_static(
            orders,
            domain.layout,
            domain.resources.resources[0],
            2,
        )
    )


def test_limits_fail_without_incumbent():
    domain = _domain()
    with pytest.raises(RuntimeError, match="no unproven incumbent"):
        exact_lorenz_makespan(
            _resolved_orders(domain, None),
            domain.layout,
            domain.resources.resources[0],
            2,
            max_states=1,
        )


def test_route_reconstruction_and_order_capacity():
    domain = _domain()
    solution, _, _ = _solver("actual").solve(domain)
    for job in solution.jobs:
        assert len(job.order_numbers) <= 2
        assert job.job.route.annotated_route
        assert len(job.job.route.item_sequence) == job.job.n_picks
        assert len(job.job.route.picking_times) == job.job.n_picks


def test_reopt_uses_standard_online_loop(tmp_path):
    result = run_experiment(_config("reopt", tmp_path))
    assert result["makespan"] == 117.0
    assert result["exactness"] == {
        "complete_information": False,
        "reoptimization_subproblems_optimal": True,
    }
    trace = [
        json.loads(line)
        for line in (tmp_path / "decision_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["algorithm_runtime_s"] >= 0 for row in trace)
    assert all(row["decision_elapsed_s"] >= 0 for row in trace)
    stable_trace = [
        {
            key: value
            for key, value in row.items()
            if key not in {"algorithm_runtime_s", "decision_elapsed_s"}
        }
        for row in trace
    ]
    assert stable_trace == [
        {
            "available_order_ids": [1],
            "committed_order_ids": [1],
            "decision_time": 10.0,
            "is_optimal": True,
            "objective_bound": None,
            "objective_value": 68.0,
            "optimality_gap": None,
            "pipeline_identifier": "LorenzMakespanDP",
            "planned_completion_time": 68.0,
            "solver": "LorenzDPSolver",
            "solver_status": "optimal",
        },
        {
            "available_order_ids": [2, 3],
            "committed_order_ids": [2, 3],
            "decision_time": 68.0,
            "is_optimal": True,
            "objective_bound": None,
            "objective_value": 117.0,
            "optimality_gap": None,
            "pipeline_identifier": "LorenzMakespanDP",
            "planned_completion_time": 117.0,
            "solver": "LorenzDPSolver",
            "solver_status": "optimal",
        },
    ]


def test_cios_run_writes_single_result(tmp_path):
    result = run_experiment(_config("cios", tmp_path))
    assert result["makespan"] == 76.0
    assert result["exactness"]["complete_instance_optimal"] is True
    assert (tmp_path / "result.json").is_file()
    assert not (tmp_path / "decision_trace.jsonl").exists()


def test_no_wait_config_builds_one_real_cosy_pipeline(tmp_path):
    cfg = _config("no_wait", tmp_path)
    data_card = load_and_flatten_data_card(cfg.data_card)
    _, decision_engine = build_runtime(cfg, data_card)
    solver = decision_engine.solver_for("OBRSP")
    assert isinstance(solver, CoSySolver)
    assert len(solver.pipelines) == 1
    component_names = {
        value.rsplit(".", 1)[-1]
        for value in cfg.variant.cosy_repo.components
    }
    assert component_names == {
        "InstanceLoader",
        "OrdersProvider",
        "GreedyIA",
        "FiFo",
        "TSPRouting",
        "SPTScheduler",
        "ResultAggregationScheduling",
    }


def test_optimizer_has_no_casim_hydra_or_scenario_imports():
    path = (
        ROOT.parent
        / "ware_ops_algos"
        / "src"
        / "ware_ops_algos"
        / "algorithms"
        / "lorenz_reoptimization.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    forbidden = ("casim", "hydra", "scenarios")
    assert not any(module.startswith(forbidden) for module in modules)
    assert not any(name.startswith(forbidden) for name in names)


def test_removed_fake_fallback_and_henn_policy():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in SCENARIO.rglob("*.py")
    )
    assert "prefer_cosy" not in source
    assert "cosy_exactness_diagnostics" not in source
    assert "scenario_henn" not in source
