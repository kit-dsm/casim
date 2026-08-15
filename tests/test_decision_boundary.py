from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import casim.envs.order_batching as batching_env
from casim.trackers import ExperimentTracker
from learning.structured_batching.policy import (
    OrderScoreActor,
    decode_indices,
    features_tensor,
    make_observation,
)
from scenarios.scenario_henn_rl.data import (
    GeneratedHennDataLoader,
    generated_instance_splits,
)
from scenarios.scenario_henn_rl.runtime import build_environment


ROOT = Path(__file__).parents[1]


def _environment():
    spec = OmegaConf.to_container(
        OmegaConf.load(
            ROOT / "scenarios/scenario_henn_rl/config/data/henn_lorenz.yaml"
        ),
        resolve=True,
    )
    spec.update(train_instances=4, validation_instances=4, test_instances=4)
    instance_id = generated_instance_splits(spec)["train"][0]
    return (
        build_environment(
            data_loader=GeneratedHennDataLoader(spec)
        ),
        instance_id,
    )


def _observation(environment, *, route_aware):
    snapshot, orders = environment.current
    return make_observation(
        snapshot,
        orders,
        total_orders=environment.order_count,
        episode_horizon=environment.episode_horizon,
        route_aware=route_aware,
    )


def test_one_projection_and_assignment_per_exposed_decision(monkeypatch):
    environment, instance_id = _environment()
    adapter = environment.simulation.state_adapters["OBP"]
    original_transform = adapter.transform_state
    projections = 0

    def counted_transform(*args, **kwargs):
        nonlocal projections
        projections += 1
        return original_transform(*args, **kwargs)

    monkeypatch.setattr(adapter, "transform_state", counted_transform)
    original_assignment = batching_env.GreedyItemAssignment.solve
    assignments = 0

    def counted_assignment(self, *args, **kwargs):
        nonlocal assignments
        assignments += 1
        return original_assignment(self, *args, **kwargs)

    monkeypatch.setattr(
        batching_env.GreedyItemAssignment, "solve", counted_assignment
    )
    snapshot, orders = environment.reset(instance_id)
    assert snapshot.problem_class == "OBP"
    assert orders[0].pick_positions
    assert projections == assignments == 1

    observation = _observation(environment, route_aware=False)
    assert projections == assignments == 1
    actor = OrderScoreActor(feature_count=8, hidden=8)
    with torch.no_grad():
        scores = actor(features_tensor(observation))
    selected = decode_indices(scores, observation, "knapsack")
    current, _, done = environment.step(
        observation.order_ids[selected].tolist()
    )
    downstream = environment.decision_engine.solver_map["ORSP"]
    assert downstream.solve_calls == 1
    if not done:
        assert current is environment.current
        assert assignments == 2
        assert projections >= 2


def test_route_decoder_scores_candidates_and_only_commit_solves_route():
    environment, instance_id = _environment()
    environment.reset(instance_id)
    observation = _observation(environment, route_aware=True)
    snapshot, _ = environment.current
    downstream = environment.decision_engine.solver_map["ORSP"]
    router = downstream.router_for(snapshot)
    calls = {"score": 0, "solve": 0}
    original_score = router.score
    original_solve = router.solve

    def counted_score(values):
        calls["score"] += 1
        return original_score(values)

    def counted_solve(values):
        calls["solve"] += 1
        return original_solve(values)

    router.score = counted_score
    router.solve = counted_solve
    selected = decode_indices(
        np.ones(len(observation)),
        observation,
        "route_aware_greedy",
        router,
    )
    assert calls["score"] > 0
    assert calls["solve"] == 0
    environment.step(observation.order_ids[selected].tolist())
    assert calls["solve"] == 1


def test_flow_time_is_incremental_operational_kpi():
    tracker = ExperimentTracker(1)
    tracker.on_order_arrival(1, 2.0)
    tracker.on_order_arrival(2, 5.0)
    assert tracker.accrued_flow_time(8.0) == 9.0
    tracker.on_orders_completed([1], 8.0)
    assert tracker.accrued_flow_time(10.0) == 11.0
    tracker.on_orders_completed([2], 10.0)
    assert tracker.total_flow_time == 11.0
    assert tracker.order_completion_times == {1: 8.0, 2: 10.0}


def test_episode_reward_is_negative_flow_time_per_order():
    environment, instance_id = _environment()
    environment.reset(instance_id)
    episode_return = 0.0
    done = False
    while not done:
        observation = _observation(environment, route_aware=False)
        selected = decode_indices(
            np.ones(len(observation)), observation, "knapsack"
        )
        _, reward, done = environment.step(
            observation.order_ids[selected].tolist()
        )
        episode_return += reward
    tracker = environment.simulation.state.tracker
    total_flow = tracker.accrued_flow_time(
        environment.simulation.state.current_time
    )
    assert episode_return == -total_flow / environment.order_count


def test_required_dependency_direction():
    def imports_banned(path, banned):
        offenders = []
        for source in path.rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            if any(
                f"from {name}" in text or f"import {name}" in text
                for name in banned
            ):
                offenders.append(source.relative_to(ROOT).as_posix())
        return offenders

    assert imports_banned(ROOT / "src/casim", ("scenarios", "learning")) == []
    assert imports_banned(ROOT / "learning", ("scenarios",)) == []
    ware_ops = ROOT.parent / "ware_ops_algos" / "src"
    assert imports_banned(ware_ops, ("casim", "learning")) == []
