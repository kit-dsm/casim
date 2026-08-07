from __future__ import annotations

import subprocess
import sys

import pytest


INSTANCE = "H_abc1_40_29"


def test_release_environment_is_direct_and_always_wait_terminates():
    pytest.importorskip("gymnasium")
    env_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    from scenarios.scenario_henn_rl.environment import ReleaseTimingEnv

    env = ReleaseTimingEnv([INSTANCE])
    env_checker.check_env(env, warn=True, skip_render_check=True)
    first, info = env.reset(seed=42, options={"instance_id": INSTANCE})
    second, _ = env.reset(seed=42, options={"instance_id": INSTANCE})
    assert env.observation_space.contains(first)
    assert (first == second).all()
    assert info["pipeline_controlled_by_agent"] is False
    terminated = False
    steps = 0
    while not terminated and steps < 100:
        observation, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(observation)
        assert reward <= 0
        assert not truncated
        steps += 1
    assert terminated
    assert info["forced_dispatches"] == 1
    assert steps < 100
    env.close()


def test_rl_import_path_does_not_load_luigi_or_cosy():
    code = """
import sys
import scenarios.scenario_henn_rl.environment
import scenarios.scenario_henn_rl.learning
import scenarios.scenario_henn_rl.experiment_henn_rl
blocked = [
    name for name in sys.modules
    if name == 'luigi' or name.startswith('luigi.')
    or name == 'cosy' or name.startswith('cosy.')
]
assert blocked == [], blocked
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_fixed_solver_does_not_mutate_live_orders():
    from scenarios.scenario_henn_rl.environment import ReleaseTimingEnv

    env = ReleaseTimingEnv([INSTANCE])
    env.reset(seed=13, options={"instance_id": INSTANCE})
    manager = env.simulation.state.order_manager

    def signature():
        return tuple(
            (
                order.order_id,
                order.order_date,
                tuple(
                    (position.article_id, position.amount)
                    for position in order.order_positions
                ),
            )
            for order in manager.get_order_buffer()
        )

    before = signature()
    snapshot = env.release_adapter.solver_snapshot(env.simulation.state, "OBRP")
    candidate = env._solve_fixed(snapshot)
    assert signature() == before
    assert candidate.routes
    assert all(
        planned is not live
        for route in candidate.routes
        for planned in route.batch.orders
        for live in manager.get_order_buffer()
    )
    env.close()


def test_instance_split_is_stratified_and_disjoint():
    from scenarios.scenario_henn_rl.learning import instance_splits

    splits = instance_splits()
    assert {name: len(values) for name, values in splits.items()} == {
        "train": 32,
        "validation": 16,
        "test": 16,
    }
    assert len(set().union(*map(set, splits.values()))) == 64


def test_counterfactual_labels_replay_from_identical_state():
    from scenarios.scenario_henn_rl.learning import generate_counterfactuals

    generated = generate_counterfactuals(
        [INSTANCE],
        fill_threshold=0.75,
        max_age_s=300.0,
        workers=1,
    )
    assert generated["labels"]
    assert all(
        row["q_wait"] != row["q_dispatch"]
        for row in generated["labels"]
    )
    assert generated["instances"][0]["replay_steps"] > 0


def test_short_ppo_uses_undiscounted_return():
    pytest.importorskip("stable_baselines3")
    from scenarios.scenario_henn_rl.learning import make_ppo

    model = make_ppo([INSTANCE], seed=42)
    model.learn(total_timesteps=256)
    assert model.num_timesteps == 256
    assert model.gamma == 1.0
    assert model.gae_lambda == 0.95
    model.get_env().close()

    model = make_ppo([INSTANCE], seed=42, gae_lambda=1.0)
    assert model.gae_lambda == 1.0
    assert model.rollout_buffer.gae_lambda == 1.0
    model.get_env().close()


def test_episode_reward_is_exact_normalized_negative_flow_time():
    from scenarios.scenario_henn_rl.environment import ReleaseTimingEnv

    env = ReleaseTimingEnv([INSTANCE])
    env.reset(seed=0, options={"instance_id": INSTANCE})
    normalizer = env.reward_normalizer
    terminated = False
    episode_return = 0.0
    while not terminated:
        _, reward, terminated, _, info = env.step(0)
        episode_return += reward
    assert episode_return == pytest.approx(
        -info["total_flow_time"] / normalizer, abs=1e-10
    )
    env.close()


def test_study_entry_point_returns_complete_result(monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    from scenarios.scenario_henn_rl import experiment_henn_rl as experiment

    splits = {"train": ["train"], "validation": ["val"], "test": ["test"]}
    monkeypatch.setattr(experiment, "instance_splits", lambda: splits)
    monkeypatch.setattr(
        experiment,
        "tune_heuristic",
        lambda *args, **kwargs: {
            "selected": {"fill_threshold": 0.75, "max_age_s": 300.0}
        },
    )
    monkeypatch.setattr(
        experiment,
        "evaluate_heuristic",
        lambda *args, **kwargs: {"mean_order_flow_time": 1.0},
    )
    monkeypatch.setattr(
        experiment,
        "generate_counterfactuals",
        lambda *args, **kwargs: {
            "labels": [
                {
                    "preferred_action": 1,
                    "advantage_dispatch": 0.5,
                }
            ]
        },
    )
    monkeypatch.setattr(
        experiment,
        "_train_arm",
        lambda **kwargs: {"arm": "stub", "seed": kwargs["seed"]},
    )
    cfg = OmegaConf.create(
        {
            "experiment": {"workers": 1, "labels_path": None},
            "heuristic": {
                "fill_thresholds": [0.75],
                "max_ages_s": [300.0],
            },
            "ppo": {"seeds": [11]},
        }
    )
    result = experiment.run_study(cfg, tmp_path)
    assert result["mode"] == "study"
    assert len(result["training_arms"]) == 2
    assert result["label_summary"]["count"] == 1


def test_structured_knapsack_is_exact_and_capacity_feasible():
    import numpy as np

    from scenarios.scenario_henn_rl.structured_environment import (
        knapsack_batch,
    )

    selected = knapsack_batch(
        np.asarray([4.0, 5.0, 7.0]),
        np.asarray([3, 4, 5]),
        7,
        allow_empty=True,
    )
    assert selected.tolist() == [0, 1]
    assert np.asarray([3, 4, 5])[selected].sum() <= 7
    assert knapsack_batch(
        np.asarray([-2.0, -1.0]),
        np.asarray([1, 1]),
        1,
        allow_empty=True,
    ).size == 0
    assert knapsack_batch(
        np.asarray([-2.0, -1.0]),
        np.asarray([1, 1]),
        1,
        allow_empty=False,
    ).tolist() == [1]


def test_structured_episode_actions_are_feasible_and_reward_is_exact():
    import numpy as np

    from scenarios.scenario_henn_rl.structured_environment import (
        StructuredBatchingEpisode,
    )

    episode = StructuredBatchingEpisode([INSTANCE])
    state = episode.reset(instance_id=INSTANCE)
    normalizer = episode.env.reward_normalizer
    episode_return = 0.0
    terminated = False
    info = {}
    while not terminated:
        selected = episode.oracle_action(
            np.ones(len(state["order_ids"])), state
        )
        assert state["demands"][selected].sum() <= state["capacity"]
        state, reward, terminated, truncated, info = episode.step(selected)
        assert not truncated
        episode_return += reward
    assert episode_return == pytest.approx(
        -info["total_flow_time"] / normalizer, abs=1e-10
    )
    assert info["oracle_time_s"] > 0.0
    episode.close()


def test_structured_knapsack_matches_exhaustive_small_cases():
    import itertools

    import numpy as np

    from scenarios.scenario_henn_rl.structured_environment import (
        knapsack_batch,
    )

    rng = np.random.default_rng(7)
    for size in range(1, 7):
        scores = rng.normal(size=size)
        demands = rng.integers(1, 5, size=size)
        capacity = 7
        feasible = [
            subset
            for length in range(size + 1)
            for subset in itertools.combinations(range(size), length)
            if demands[list(subset)].sum() <= capacity
        ]
        optimum = max(
            sum(scores[index] for index in subset) for subset in feasible
        )
        selected = knapsack_batch(
            scores, demands, capacity, allow_empty=True
        )
        assert scores[selected].sum() == pytest.approx(optimum)


def test_fenchel_young_gradient_matches_decoded_mean_minus_target():
    torch = pytest.importorskip("torch")
    import numpy as np

    from scenarios.scenario_henn_rl.structured_policy import (
        fenchel_young_loss,
    )

    class DirectActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.3]))

        def forward(self, features):
            return self.scores

    state = {
        "features": np.zeros((3, 7), dtype=np.float32),
        "demands": np.asarray([1, 1, 1]),
        "capacity": 2,
        "input_closed": False,
    }
    actor = DirectActor()
    target = torch.tensor([1.0, 0.0, 1.0])
    loss, decoded_mean = fenchel_young_loss(
        actor,
        state,
        target,
        sample_count=20,
        epsilon=0.1,
        generator=torch.Generator().manual_seed(3),
    )
    loss.backward()
    assert torch.allclose(
        actor.scores.grad, decoded_mean - target, atol=1e-6
    )


def test_existing_cw_structured_action_is_capacity_feasible():
    from scenarios.scenario_henn_rl.structured_environment import (
        StructuredBatchingEpisode,
    )

    episode = StructuredBatchingEpisode([INSTANCE])
    state = episode.reset(instance_id=INSTANCE)
    indices = episode.existing_policy_action(state, batching="cw")
    if len(indices):
        assert state["demands"][indices].sum() <= state["capacity"]
    episode.close()


def test_short_structured_rl_training_completes():
    pytest.importorskip("torch")
    from scenarios.scenario_henn_rl.structured_policy import OrderScoreActor
    from scenarios.scenario_henn_rl.structured_training import train_structured_rl

    actor = OrderScoreActor(hidden=8)
    _, result = train_structured_rl(
        actor,
        [INSTANCE],
        episodes=1,
        updates_per_episode=1,
        batch_size=1,
        replay_capacity=20,
        actor_learning_rate=0.001,
        critic_learning_rate=0.002,
        sigma_forward=0.1,
        sigma_target=0.2,
        temperature=1.0,
        candidate_count=3,
        epsilon=0.01,
        fy_samples=2,
        gamma=1.0,
        seed=5,
        validation_ids=[INSTANCE],
        checkpoint_every=1,
    )
    assert len(result["episodes"]) == 1
    assert result["gamma"] == 1.0
    assert result["candidate_count"] == 3


def test_srl_soft_target_is_fractional_but_capacity_feasible_in_expectation():
    torch = pytest.importorskip("torch")
    import numpy as np

    from scenarios.scenario_henn_rl.structured_policy import (
        critic_soft_target,
    )

    class Actor(torch.nn.Module):
        def forward(self, features):
            return features[:, 0]

    class Critic(torch.nn.Module):
        def forward(self, features, action):
            return (features[:, 1] * action).sum()

    state = {
        "features": np.asarray(
            [
                [0.1, 1.0, 0, 0, 0, 0, 0],
                [0.2, 2.0, 0, 0, 0, 0, 0],
                [0.3, 3.0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
        "demands": np.asarray([2, 3, 4]),
        "capacity": 5,
        "input_closed": False,
    }
    target, diagnostics = critic_soft_target(
        Actor(),
        Critic(),
        state,
        candidate_count=40,
        sigma=1.0,
        temperature=0.1,
        generator=torch.Generator().manual_seed(9),
    )
    assert diagnostics["candidate_count"] == 40
    assert 1 <= diagnostics["unique_candidates"] <= 40
    assert torch.all((0 <= target) & (target <= 1))
    assert float(target @ torch.tensor([2.0, 3.0, 4.0])) <= 5.0 + 1e-6


def test_complete_return_to_go_matches_undiscounted_episode_return():
    from scenarios.scenario_henn_rl.structured_policy import (
        discounted_returns,
    )

    rewards = [-0.2, -0.1, -0.4]
    returns = discounted_returns(rewards, gamma=1.0)
    assert returns == pytest.approx([-0.7, -0.5, -0.4])
    assert returns[0] == pytest.approx(sum(rewards))
    assert discounted_returns(rewards, gamma=0.5) == pytest.approx(
        [-0.35, -0.3, -0.4]
    )


def test_normalized_candidate_weights_are_affine_invariant_and_finite():
    torch = pytest.importorskip("torch")

    from scenarios.scenario_henn_rl.structured_policy import (
        candidate_weights,
    )

    values = torch.tensor([-0.4, 0.1, 0.8])
    expected = candidate_weights(
        values,
        temperature=1.0,
        normalize_advantages=True,
    )
    transformed = candidate_weights(
        7.0 * values + 19.0,
        temperature=1.0,
        normalize_advantages=True,
    )
    assert torch.allclose(expected, transformed, atol=1e-6)
    uniform = candidate_weights(
        torch.ones(4),
        temperature=1.0,
        normalize_advantages=True,
    )
    assert torch.isfinite(uniform).all()
    assert torch.allclose(uniform, torch.full((4,), 0.25))
    legacy = candidate_weights(
        values,
        temperature=0.2,
        normalize_advantages=False,
    )
    assert torch.allclose(legacy, torch.softmax(values / 0.2, dim=0))


def test_frozen_critic_transfer_gate_excludes_actor_regret():
    from types import SimpleNamespace

    from scenarios.scenario_henn_rl.experiment_support import (
        _critic_transfer_gate,
    )

    summary = {
        "mean_spearman": 0.7,
        "mean_critic_regret_fraction": 0.1,
        "mean_actor_regret_fraction": 0.9,
        "soft_target_capture_fraction": 0.4,
    }
    settings = SimpleNamespace(
        min_spearman=0.6,
        max_critic_regret_fraction=0.12,
        max_actor_regret_fraction=0.25,
        min_soft_target_capture_fraction=0.3,
    )
    gate = _critic_transfer_gate(summary, settings)
    assert gate["passed"]
    assert "actor_regret_fraction" not in gate["checks"]


def test_short_structured_training_accepts_return_to_go_critic():
    torch = pytest.importorskip("torch")

    from scenarios.scenario_henn_rl.structured_policy import OrderScoreActor
    from scenarios.scenario_henn_rl.structured_training import train_structured_rl

    _, result = train_structured_rl(
        OrderScoreActor(hidden=8),
        [INSTANCE],
        episodes=1,
        updates_per_episode=1,
        batch_size=1,
        replay_capacity=20,
        actor_learning_rate=0.001,
        critic_learning_rate=0.002,
        sigma_forward=0.1,
        sigma_target=0.2,
        temperature=1.0,
        candidate_count=3,
        epsilon=0.01,
        fy_samples=2,
        gamma=1.0,
        seed=5,
        validation_ids=[INSTANCE],
        checkpoint_every=1,
        reward_power=2.0,
        critic_target="return_to_go",
        normalize_candidate_advantages=True,
    )
    assert result["critic_target"] == "return_to_go"
    assert result["normalize_candidate_advantages"] is True
    assert result["target_copy"] == "unused"
    assert result["episodes"][0]["wait_actions"] == 0
    assert result["max_reward_identity_error"] < 1e-10


def test_counterfactual_calibration_uses_exact_candidate_returns(monkeypatch):
    torch = pytest.importorskip("torch")
    import numpy as np

    import scenarios.scenario_henn_rl.counterfactuals as counterfactuals
    import scenarios.scenario_henn_rl.critic_fitting as critic_fitting
    import scenarios.scenario_henn_rl.structured_policy as policy

    state = {
        "order_ids": np.asarray([10, 11, 12]),
        "features": np.asarray(
            [
                [0.1, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0],
                [0.2, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.3, 0.8, 0.0, 0.9, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "demands": np.asarray([1, 1, 1]),
        "capacity": 2,
        "input_closed": True,
    }
    monkeypatch.setattr(
        counterfactuals,
        "_reference_trajectory",
        lambda actor, instance_id: [{"state": state}],
    )
    audit = {
        "reward_power": 2.0,
        "states": [
            {
                "instance_id": "training_instance",
                "decision_index": 0,
                "candidate_rows": [
                    {"order_ids": [10, 11], "true_q": {"2.0": -3.0}},
                    {"order_ids": [11, 12], "true_q": {"2.0": -1.0}},
                    {"order_ids": [10, 12], "true_q": {"2.0": -2.0}},
                ],
            }
        ],
    }
    actor = policy.OrderScoreActor(hidden=8)
    critic = policy.StructuredCritic(hidden=8)
    result = critic_fitting.calibrate_from_counterfactual_audit(
        actor,
        critic,
        audit,
        critic_epochs=30,
        actor_epochs=2,
        critic_learning_rate=0.01,
        actor_learning_rate=0.001,
        epsilon=0.01,
        fy_samples=2,
        seed=5,
    )
    assert result["states"] == 1
    assert result["candidates"] == 3
    assert result["target"] == "state_standardized_exact_counterfactual_return"
    assert result["critic_after"]["mean_loss"] < result["critic_before"]["mean_loss"]
    assert np.isfinite(result["final_actor_loss"])

    for loss_kind in ["regression", "pairwise_ranking", "pairwise_top"]:
        _, memorization = critic_fitting.overfit_counterfactual_critic(
            actor,
            audit,
            epochs=5,
            learning_rate=0.01,
            loss_kind=loss_kind,
            seed=7,
        )
        assert memorization["loss"] == loss_kind
        assert [row["epoch"] for row in memorization["history"]][0] == 0
        assert memorization["history"][-1]["epoch"] == 5
        assert 0.0 <= memorization["final"]["pairwise_accuracy"] <= 1.0

    _, interaction = critic_fitting.overfit_counterfactual_critic(
        actor,
        audit,
        epochs=5,
        learning_rate=0.01,
        loss_kind="pairwise_ranking",
        seed=7,
        interaction_aware=True,
    )
    assert interaction["critic"] == "pairwise_interaction"

    with pytest.raises(ValueError, match="memorization loss"):
        critic_fitting.overfit_counterfactual_critic(
            actor,
            audit,
            epochs=1,
            learning_rate=0.01,
            loss_kind="unknown",
            seed=7,
        )


def test_pairwise_critic_is_permutation_invariant_and_action_sensitive():
    torch = pytest.importorskip("torch")

    from scenarios.scenario_henn_rl.structured_policy import (
        PairwiseStructuredCritic,
    )

    torch.manual_seed(3)
    critic = PairwiseStructuredCritic(hidden=8)
    features = torch.randn(4, 7)
    action = torch.tensor([1.0, 0.0, 1.0, 0.0])
    permutation = torch.tensor([2, 0, 3, 1])
    value = critic(features, action)
    permuted = critic(features[permutation], action[permutation])
    alternative = critic(features, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(value, permuted, atol=1e-6)
    assert not torch.allclose(value, alternative)
    assert torch.isfinite(critic(features, torch.tensor([1.0, 0.0, 0.0, 0.0])))


def test_structured_candidate_generation_preserves_training_multiplicity():
    torch = pytest.importorskip("torch")
    import numpy as np

    from scenarios.scenario_henn_rl.structured_policy import (
        structured_candidates,
    )

    class Actor(torch.nn.Module):
        def forward(self, features):
            return torch.ones(len(features))

    state = {
        "features": np.zeros((3, 7), dtype=np.float32),
        "demands": np.asarray([1, 1, 1]),
        "capacity": 2,
        "input_closed": False,
    }
    candidates = structured_candidates(
        Actor(),
        state,
        candidate_count=4,
        sigma=0.0,
        generator=torch.Generator().manual_seed(1),
    )
    distinct = structured_candidates(
        Actor(),
        state,
        candidate_count=4,
        sigma=0.0,
        generator=torch.Generator().manual_seed(1),
        deduplicate=True,
    )
    assert len(candidates) == 4
    assert len(distinct) == 1


def test_candidate_feature_ridge_and_metrics_rank_within_state():
    import numpy as np

    from scenarios.scenario_henn_rl.candidate_feature_experiment import (
        _ridge_fit,
        _ridge_predict,
        _state_metrics,
    )

    x = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    y = np.asarray([-1.0, -2.0, -3.0, -4.0])
    model = _ridge_fit(x, y, alpha=1e-6)
    predicted = _ridge_predict(model, x)
    rows = [
        {
            "instance_id": "a",
            "decision_index": decision,
            "true_q": float(value),
        }
        for decision, value in [(0, -1.0), (0, -2.0), (1, -3.0), (1, -4.0)]
    ]
    metrics = _state_metrics(rows, predicted)
    assert metrics["mean_spearman"] > 0.99
    assert metrics["mean_critic_regret_fraction"] == 0.0
    assert metrics["critic_top_accuracy"] == 1.0
    assert np.all(np.isfinite(predicted))


def test_structured_critic_audit_replays_counterfactual_candidates():
    torch = pytest.importorskip("torch")

    from scenarios.scenario_henn_rl.counterfactuals import (
        audit_structured_critic,
        score_counterfactual_audit,
    )
    from scenarios.scenario_henn_rl.structured_policy import (
        OrderScoreActor,
        StructuredCritic,
    )

    actor = OrderScoreActor(hidden=8)
    critic = StructuredCritic(hidden=8)
    result = audit_structured_critic(
        actor,
        critic,
        [INSTANCE],
        reward_power=1.0,
        comparison_powers=[1.0, 2.0],
        state_quantiles=[0.5],
        candidate_count=3,
        sigma=1.0,
        temperature=0.001,
        seed=7,
    )
    assert result["summary"]["state_count"] == 1
    state = result["states"][0]
    assert state["unique_candidates"] >= 1
    assert len(state["candidate_rows"]) == state["unique_candidates"]
    assert all(
        set(row["true_q"]) == {"1.0", "2.0"}
        for row in state["candidate_rows"]
    )
    assert state["critic_top_regret"] >= 0.0
    assert state["actor_regret"] >= 0.0
    rescored = score_counterfactual_audit(
        actor,
        critic,
        result,
        temperature=0.001,
        normalize_advantages=False,
    )
    for metric in [
        "mean_spearman",
        "critic_top_accuracy",
        "mean_critic_regret_fraction",
        "mean_actor_regret_fraction",
        "soft_target_capture_fraction",
    ]:
        assert rescored["summary"][metric] == pytest.approx(
            result["summary"][metric], rel=1e-5, abs=1e-8
        )


@pytest.mark.parametrize("power", [1.0, 1.5, 2.0])
def test_structured_no_wait_reward_identity_and_raw_flow(power):
    import numpy as np

    from scenarios.scenario_henn_rl.structured_environment import (
        StructuredBatchingEpisode,
    )

    episode = StructuredBatchingEpisode([INSTANCE], reward_power=power)
    state = episode.reset(instance_id=INSTANCE)
    episode_return = 0.0
    done = False
    while not done:
        selected = episode.oracle_action(
            np.full(len(state["order_ids"]), -1.0), state
        )
        assert selected.size > 0
        state, reward, done, truncated, info = episode.step(selected)
        assert not truncated
        episode_return += reward
    completions = episode.env.completion_times()
    raw_flow = sum(
        completions[order_id] - arrival
        for order_id, arrival in episode.env.arrivals.items()
    )
    assert info["wait_actions"] == 0
    assert info["total_flow_time"] == pytest.approx(raw_flow)
    assert info["objective_cost"] == pytest.approx(
        sum(
            (completions[order_id] - arrival) ** power
            for order_id, arrival in episode.env.arrivals.items()
        )
    )
    assert episode_return == pytest.approx(
        -info["objective_cost"] / episode.env.reward_normalizer,
        rel=1e-12,
    )
    episode.close()


def test_structured_decoder_is_nonempty_before_input_closure():
    torch = pytest.importorskip("torch")
    import numpy as np

    from scenarios.scenario_henn_rl.structured_policy import decode_scores

    state = {
        "demands": np.asarray([2, 3]),
        "capacity": 3,
        "input_closed": False,
    }
    decoded = decode_scores(torch.tensor([-5.0, -2.0]), state)
    assert decoded.tolist() == [0.0, 1.0]


def test_objective_arms_share_initialization_and_training_order():
    torch = pytest.importorskip("torch")

    from scenarios.scenario_henn_rl.experiment_support import (
        _state_digest,
        _training_order,
    )
    from scenarios.scenario_henn_rl.structured_policy import OrderScoreActor

    ids = [f"instance_{index}" for index in range(32)]
    order = _training_order(ids, seed=11, passes=4)
    assert len(order) == 128
    assert all(
        sorted(order[start:start + 32]) == sorted(ids)
        for start in range(0, 128, 32)
    )
    assert order == _training_order(ids, seed=11, passes=4)
    digests = []
    for _ in (1.0, 1.5, 2.0):
        torch.manual_seed(11)
        digests.append(_state_digest(OrderScoreActor()))
    assert len(set(digests)) == 1


def test_nonuniform_checkpoints_are_evaluated_and_saved(tmp_path):
    pytest.importorskip("torch")
    from scenarios.scenario_henn_rl.structured_policy import OrderScoreActor
    from scenarios.scenario_henn_rl.structured_training import train_structured_rl

    _, result = train_structured_rl(
        OrderScoreActor(hidden=8),
        [INSTANCE, INSTANCE, INSTANCE],
        episodes=3,
        updates_per_episode=0,
        batch_size=1,
        replay_capacity=20,
        actor_learning_rate=0.001,
        critic_learning_rate=0.002,
        sigma_forward=0.1,
        sigma_target=0.2,
        temperature=0.001,
        candidate_count=3,
        epsilon=0.01,
        fy_samples=2,
        gamma=1.0,
        seed=11,
        validation_ids=[INSTANCE],
        checkpoint_episodes=[0, 1, 3],
        checkpoint_dir=tmp_path,
        reward_power=1.5,
    )
    assert [row["episode"] for row in result["validation_checkpoints"]] == [0, 1, 3]
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == [
        "episode_000.pt", "episode_001.pt", "episode_003.pt"
    ]
    assert result["max_reward_identity_error"] < 1e-10


def test_baseline_selection_uses_only_supplied_validation_rows():
    from scenarios.scenario_henn_rl.experiment_support import (
        _select_objective_baselines,
    )

    rows = [
        {
            "batching": "fifo",
            "selector": "first",
            "evaluation": {"episodes": [{"flow_times": [1.0, 9.0]}]},
        },
        {
            "batching": "cw",
            "selector": "sav",
            "evaluation": {"episodes": [{"flow_times": [5.0, 5.0]}]},
        },
    ]
    selected = _select_objective_baselines(rows, [1.0, 2.0])
    assert selected["1.0"] is rows[0]
    assert selected["2.0"] is rows[1]
