from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from learning.structured_batching.checkpoint import (
    assert_checkpoint_compatible,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from learning.structured_batching.evaluate import (
    cw_sav_choose_batch,
    evaluate_policy,
    fifo_choose_batch,
    make_actor_choose_batch,
)
from learning.structured_batching.learning import (
    complete_returns,
    fenchel_young_loss,
)
from learning.structured_batching.policy import (
    BatchingObservation,
    OrderScoreActor,
    PairwiseStructuredCritic,
    StructuredCritic,
    decode_action,
    greedy_route_aware_batch,
    knapsack_batch,
)
from learning.structured_batching.tracking import (
    log_evaluation,
    log_training,
    log_validation,
    start_tracking,
)
from learning.structured_batching.train import run_training
from scenarios.scenario_henn_rl.data import (
    GeneratedHennDataLoader,
    generated_instance_splits,
    generated_manifest,
)
from scenarios.scenario_henn_rl.runtime import build_environment


ROOT = Path(__file__).parents[1]


class Router:
    def __init__(self):
        self.score_calls = 0

    def score(self, positions):
        self.score_calls += 1
        values = sorted(float(position[1]) for position in positions)
        return 0.0 if not values else 2.0 * max(values)


class TrackingRun:
    def __init__(self):
        self.rows = []

    def log(self, values):
        self.rows.append(values)


def _flatten_positions(order_positions, selected_indices):
    return [
        position
        for index in selected_indices
        for position in order_positions[int(index)]
    ]


def brute_force_route_aware_batch(
    scores,
    demands,
    capacity,
    order_positions,
    router,
    route_cost_scale,
):
    """Enumerate the exact small route-aware optimum for tests."""
    scores = np.asarray(scores, dtype=float)
    demands = np.asarray(demands, dtype=int)
    scale = float(route_cost_scale) if route_cost_scale > 0 else 1.0
    best, best_value, best_cost = [], -np.inf, 0.0
    for size in range(1, len(scores) + 1):
        for selected in combinations(range(len(scores)), size):
            if demands[list(selected)].sum() > capacity:
                continue
            cost = float(router.score(_flatten_positions(order_positions, selected)))
            value = float(scores[list(selected)].sum()) - cost / scale
            if value > best_value + 1e-12:
                best, best_value, best_cost = list(selected), value, cost
    return np.asarray(best, dtype=int), float(best_value), float(best_cost)


def _observation():
    return BatchingObservation(
        order_ids=np.asarray([10, 11, 12]),
        features=np.asarray(
            [
                [0.1, 0.2, 0.1, 0.1, 0.2, 0.15, 0.3, 0.5],
                [0.2, 0.3, 0.1, 0.3, 0.4, 0.35, 0.3, 0.4],
                [0.3, 0.4, 0.2, 0.7, 0.8, 0.75, 0.3, 0.2],
            ],
            dtype=np.float32,
        ),
        demands=np.asarray([2, 3, 4]),
        capacity=5,
        order_positions=(((1, 1),), ((1, 4),), ((1, 8),)),
        route_cost_scale=10.0,
    )


def _environment_factory():
    spec = OmegaConf.to_container(
        OmegaConf.load(
            ROOT / "scenarios/scenario_henn_rl/config/data/henn_lorenz.yaml"
        ),
        resolve=True,
    )
    spec.update(train_instances=4, validation_instances=4, test_instances=4)
    splits = generated_instance_splits(spec)
    manifest = generated_manifest(spec)
    loader = GeneratedHennDataLoader(spec)

    def factory():
        return build_environment(data_loader=loader)

    return spec, splits, factory, manifest


def test_knapsack_is_exact_capacity_feasible_and_nonempty():
    selected = knapsack_batch(
        np.asarray([4.0, 5.0, 7.0]),
        np.asarray([2, 3, 4]),
        5,
    )
    assert selected.tolist() == [0, 1]
    assert knapsack_batch(
        np.asarray([-3.0, -1.0]), np.asarray([1, 1]), 1
    ).tolist() == [1]


def test_route_decoder_is_pure_greedy_approximation():
    state = _observation()
    router = Router()
    scores = np.asarray([1.0, 1.2, 2.0])
    greedy, value, cost = greedy_route_aware_batch(
        scores,
        state.demands,
        state.capacity,
        state.order_positions,
        router,
        state.route_cost_scale,
    )
    exact, exact_value, _ = brute_force_route_aware_batch(
        scores,
        state.demands,
        state.capacity,
        state.order_positions,
        router,
        state.route_cost_scale,
    )
    assert state.demands[greedy].sum() <= state.capacity
    assert value <= exact_value + 1e-12
    assert cost >= 0.0
    assert len(exact) > 0
    assert router.score_calls > 0


def test_actor_and_critics_are_permutation_consistent():
    torch.manual_seed(4)
    state = _observation()
    features = torch.tensor(state.features)
    permutation = torch.tensor([2, 0, 1])
    actor = OrderScoreActor(feature_count=8, hidden=8)
    assert torch.allclose(
        actor(features)[permutation], actor(features[permutation]), atol=1e-6
    )
    action = torch.tensor([1.0, 1.0, 0.0])
    for critic in (
        StructuredCritic(feature_count=8, hidden=8),
        PairwiseStructuredCritic(feature_count=8, hidden=8),
    ):
        assert torch.allclose(
            critic(features, action),
            critic(features[permutation], action[permutation]),
            atol=1e-6,
        )


def test_fenchel_young_loss_updates_actor_scores():
    actor = OrderScoreActor(feature_count=8, hidden=8)
    state = _observation()
    target = decode_action(np.asarray([2.0, 1.0, 0.0]), state, "knapsack")
    loss, mean_action = fenchel_young_loss(
        actor,
        state,
        target,
        "knapsack",
        None,
        sample_count=4,
        epsilon=0.01,
        generator=torch.Generator().manual_seed(3),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert mean_action.shape == target.shape
    assert any(parameter.grad is not None for parameter in actor.parameters())


def test_complete_returns_are_undiscounted():
    assert complete_returns([-0.2, -0.1, -0.4]) == pytest.approx(
        [-0.7, -0.5, -0.4]
    )


def test_tracking_records_useful_numeric_metrics_without_nested_episode_data(tmp_path):
    assert start_tracking({"enabled": False}, {}, tmp_path) is None
    run = TrackingRun()
    log_training(
        run,
        {
            "episode": 3,
            "instance_id": "train-3",
            "objective_per_order": 12.5,
            "mean_batch_fill": 0.8,
            "actor_time_s": 0.2,
        },
        [
            {"episode": 3, "actor_loss": 2.0, "critic_gradient_norm": 0.5},
            {"episode": 3, "actor_loss": 4.0, "critic_gradient_norm": 0.7},
        ],
    )
    log_validation(
        run,
        3,
        {
            "mean_objective_per_order": 11.0,
            "mean_decision_time_s": 0.1,
            "episodes": [{"flow_times": [1.0]}],
        },
    )
    log_evaluation(
        run,
        "fifo_first",
        {"mean_objective_per_order": 13.0, "episodes": [{"flow_times": [1.0]}]},
    )
    assert run.rows[0]["train/episode"] == 3
    assert run.rows[0]["train/objective_per_order"] == 12.5
    assert run.rows[0]["update/actor_loss"] == pytest.approx(3.0)
    assert "update/episode" not in run.rows[0]
    assert run.rows[1]["validation/mean_decision_time_s"] == 0.1
    assert "validation/episodes" not in run.rows[1]
    assert run.rows[2]["evaluation/fifo_first/mean_objective_per_order"] == 13.0


def test_checkpoint_round_trip_and_direct_compatibility(tmp_path):
    actor = OrderScoreActor(feature_count=8, hidden=8)
    critic = PairwiseStructuredCritic(feature_count=8, hidden=8)
    data = {"seed": 11, "order_counts": [40]}
    checkpoint = build_checkpoint(
        actor=actor,
        critic=critic,
        critic_kind="interaction",
        feature_count=8,
        actor_hidden=8,
        decoder_name="knapsack",
        data_spec=data,
        selected_episode=3,
    )
    path = tmp_path / "policy.pt"
    save_checkpoint(path, checkpoint)
    loaded_actor, loaded_critic, loaded = load_checkpoint(path)
    assert loaded["selected_episode"] == 3
    assert loaded["objective_scale"] == 1.0
    assert type(loaded_actor) is type(actor)
    assert type(loaded_critic) is type(critic)
    assert_checkpoint_compatible(
        loaded,
        decoder_name="knapsack",
        route_cost_scale=0.0,
    )
    with pytest.raises(ValueError, match="decoder"):
        assert_checkpoint_compatible(
            loaded,
            decoder_name="route_aware_greedy",
            route_cost_scale=1.0,
        )
    with pytest.raises(ValueError, match="scale"):
        assert_checkpoint_compatible(
            loaded,
            decoder_name="knapsack",
            route_cost_scale=0.0,
            objective_scale=2.0,
        )


def test_short_training_and_evaluation_use_same_casim_path(tmp_path):
    data, splits, factory, manifest = _environment_factory()
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "progress": False,
            "objective_scale": 1.0,
            "data": data,
            "decoder": {"name": "knapsack"},
            "model": {"hidden": 8, "critic": "interaction"},
            "learner": {
                "episodes": 1,
                "updates_per_episode": 1,
                "batch_size": 1,
                "replay_capacity": 20,
                "actor_learning_rate": 0.001,
                "critic_learning_rate": 0.002,
                "sigma_forward": 0.1,
                "sigma_target": 0.5,
                "temperature": 0.1,
                "candidate_count": 3,
                "epsilon": 0.01,
                "fy_samples": 2,
                "critic_target": "td",
                "normalize_candidate_advantages": False,
                "checkpoint_episodes": [0, 1],
            },
        }
    )
    result = run_training(
        cfg,
        tmp_path,
        splits=splits,
        dataset_manifest=manifest,
        environment_factory=factory,
    )
    assert result["status"] == "complete"
    assert result["max_reward_identity_error"] < 1e-9
    actor, _, checkpoint = load_checkpoint(result["selected_checkpoint"])
    assert checkpoint["objective_scale"] == 1.0
    learned = evaluate_policy(
        splits["test"][:1],
        choose_batch=make_actor_choose_batch(actor, "knapsack"),
        environment_factory=factory,
        objective_scale=1.0,
        decoder_name="knapsack",
    )
    fifo = evaluate_policy(
        splits["test"][:1],
        choose_batch=fifo_choose_batch,
        environment_factory=factory,
        objective_scale=1.0,
        decoder_name="knapsack",
    )
    assert learned["instances"] == fifo["instances"] == 1
    assert learned["max_reward_identity_error"] < 1e-9
    assert fifo["max_reward_identity_error"] < 1e-9
