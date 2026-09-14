"""Test that StochasticWaitingOptimizer works with casim domain objects."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ware_ops_algos.algorithms import CombinedRoutingSolution
from ware_ops_algos.algorithms.waiting import (
    StochasticWaitingOptimizer,
    WaitingAnalysisInput,
    WaitingSolution,
)
from ware_ops_algos.domain_models import load_and_flatten_data_card

from casim.setup import build_runtime
from scenarios.scenario_henn.scenario_specific_hooks import build_sim_hooks
from casim.events.operational_events import OrderArrival, FlushRemainingOrders


SCENARIO_ROOT = (
    Path(__file__).parents[1] / "scenarios" / "scenario_henn"
).resolve()


def _compose(*overrides):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(SCENARIO_ROOT / "config"),
    ):
        return compose(
            config_name="henn_config",
            overrides=list(overrides),
        )


def test_stochastic_waiting_optimizer_produces_decision(tmp_path):
    """Build WaitingAnalysisInput from a real simulation snapshot and
    run the stochastic waiting optimizer end-to-end."""
    cfg = _compose("batching=fcfs")
    project_root = Path(__file__).parents[1].resolve()
    OmegaConf.update(cfg, "project_root", str(project_root), merge=False)
    OmegaConf.update(cfg, "instances_base", str(project_root / "scenarios"), merge=False)
    OmegaConf.update(cfg, "cache_base", str(tmp_path / "cache"), merge=False)
    OmegaConf.update(cfg, "experiment.output_dir", str(tmp_path), merge=False)
    OmegaConf.update(cfg, "experiment.instance_name", "stochastic-test", merge=False)
    OmegaConf.update(cfg, "luigi.runtime", 1, merge=False)

    data_card = load_and_flatten_data_card(cfg.data_card)
    simulation, decision_engine = build_runtime(cfg, data_card)
    simulation.reset(hooks=build_sim_hooks(cfg))

    assert sum(isinstance(e, OrderArrival) for e in simulation.events) == 40

    route = None
    snapshot = None
    for _ in range(20):
        done, snapshot = simulation.run()
        if done:
            break
        result = decision_engine.solver_for("OBRP").solve(snapshot, action=None)
        if result is None:
            continue
        solution, _, _ = result
        if not isinstance(solution, CombinedRoutingSolution) or not solution.routes:
            continue
        if len(solution.routes[0].batch.orders) >= 2:
            route = solution.routes[0]
            break

    assert route is not None, "Could not find a batch with >= 2 orders"
    assert snapshot is not None

    batch_orders = route.batch.orders
    base_orders = batch_orders[:-1]
    insert_order = batch_orders[-1]
    picker = snapshot.resources.resources[0]

    analysis_input = WaitingAnalysisInput(
        layout=snapshot.layout,
        picker=picker,
        base_orders=base_orders,
        insert_order=insert_order,
        base_route=route,
        current_time=snapshot.dynamic_warehouse_info.time or 0.0,
        expected_interarrival=28.8,
    )

    optimizer = StochasticWaitingOptimizer(engine="discrete")
    result_solution = optimizer.solve(analysis_input)

    assert isinstance(result_solution, WaitingSolution)
    assert result_solution.algo_name == "StochasticWaiting"
    assert result_solution.execution_time > 0
    assert isinstance(result_solution.should_wait, bool)
    assert result_solution.engine == "discrete"
    assert result_solution.predicted_completion_time > 0
    assert result_solution.initial_completion_time > 0
    assert result_solution.expected_detour >= 0
