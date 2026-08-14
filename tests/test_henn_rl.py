from __future__ import annotations

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


def test_structured_knapsack_is_exact_and_capacity_feasible():
    import numpy as np

    from scenarios.scenario_henn_rl.structured.decoders import (
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

    from scenarios.scenario_henn_rl.structured.environment import (
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
            np.ones(len(state.order_ids)), state
        )
        assert state.demands[selected].sum() <= state.capacity
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

    from scenarios.scenario_henn_rl.structured.decoders import (
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

    from scenarios.scenario_henn_rl.structured.decoders import make_decoder
    from scenarios.scenario_henn_rl.structured.models import (
        fenchel_young_loss,
    )
    from scenarios.scenario_henn_rl.structured.policy import StructuredPolicy
    from scenarios.scenario_henn_rl.structured.state import BatchingState

    class DirectActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.3]))

        def forward(self, features):
            return self.scores

    state = BatchingState(
        order_ids=np.asarray([0, 1, 2], dtype=np.int64),
        features=np.zeros((3, 7), dtype=np.float32),
        demands=np.asarray([1, 1, 1]),
        capacity=2,
        order_positions=[[(0, 1)], [(0, 2)], [(1, 3)]],
        input_closed=False,
        feature_schema="legacy_v1",
    )
    actor = DirectActor()
    policy = StructuredPolicy(actor, make_decoder("knapsack"))
    target = torch.tensor([1.0, 0.0, 1.0])
    loss, decoded_mean = fenchel_young_loss(
        policy,
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
    from scenarios.scenario_henn_rl.structured.environment import (
        StructuredBatchingEpisode,
    )

    episode = StructuredBatchingEpisode([INSTANCE])
    state = episode.reset(instance_id=INSTANCE)
    indices = episode.existing_policy_action(state, batching="cw")
    if len(indices):
        assert state.demands[indices].sum() <= state.capacity
    episode.close()


def test_short_structured_rl_training_completes():
    pytest.importorskip("torch")
    from scenarios.scenario_henn_rl.structured.models import OrderScoreActor
    from scenarios.scenario_henn_rl.structured.training import train_structured_rl

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

    from scenarios.scenario_henn_rl.structured.decoders import make_decoder
    from scenarios.scenario_henn_rl.structured.models import (
        critic_soft_target,
    )
    from scenarios.scenario_henn_rl.structured.policy import StructuredPolicy
    from scenarios.scenario_henn_rl.structured.state import BatchingState

    class Actor(torch.nn.Module):
        def forward(self, features):
            return features[:, 0]

    class Critic(torch.nn.Module):
        def forward(self, features, action):
            return (features[:, 1] * action).sum()

    state = BatchingState(
        order_ids=np.asarray([0, 1, 2], dtype=np.int64),
        features=np.asarray(
            [
                [0.1, 1.0, 0, 0, 0, 0, 0],
                [0.2, 2.0, 0, 0, 0, 0, 0],
                [0.3, 3.0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
        demands=np.asarray([2, 3, 4]),
        capacity=5,
        order_positions=[[(0, 1)], [(0, 2)], [(1, 3)]],
        input_closed=False,
        feature_schema="legacy_v1",
    )
    policy = StructuredPolicy(Actor(), make_decoder("knapsack"))
    target, diagnostics = critic_soft_target(
        policy,
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
    from scenarios.scenario_henn_rl.structured.models import (
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

    from scenarios.scenario_henn_rl.structured.models import (
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


def test_short_structured_training_accepts_return_to_go_critic():
    torch = pytest.importorskip("torch")

    from scenarios.scenario_henn_rl.structured.models import OrderScoreActor
    from scenarios.scenario_henn_rl.structured.training import train_structured_rl

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


def test_pairwise_critic_is_permutation_invariant_and_action_sensitive():
    torch = pytest.importorskip("torch")

    from scenarios.scenario_henn_rl.structured.models import (
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

    from scenarios.scenario_henn_rl.structured.decoders import make_decoder
    from scenarios.scenario_henn_rl.structured.models import (
        structured_candidates,
    )
    from scenarios.scenario_henn_rl.structured.policy import StructuredPolicy
    from scenarios.scenario_henn_rl.structured.state import BatchingState

    class Actor(torch.nn.Module):
        def forward(self, features):
            return torch.ones(len(features))

    state = BatchingState(
        order_ids=np.asarray([0, 1, 2], dtype=np.int64),
        features=np.zeros((3, 7), dtype=np.float32),
        demands=np.asarray([1, 1, 1]),
        capacity=2,
        order_positions=[[(0, 1)], [(0, 2)], [(1, 3)]],
        input_closed=False,
        feature_schema="legacy_v1",
    )
    policy = StructuredPolicy(Actor(), make_decoder("knapsack"))
    candidates = structured_candidates(
        policy,
        state,
        candidate_count=4,
        sigma=0.0,
        generator=torch.Generator().manual_seed(1),
    )
    distinct = structured_candidates(
        policy,
        state,
        candidate_count=4,
        sigma=0.0,
        generator=torch.Generator().manual_seed(1),
        deduplicate=True,
    )
    assert len(candidates) == 4
    assert len(distinct) == 1


def test_structured_critic_audit_replays_counterfactual_candidates():
    torch = pytest.importorskip("torch")

    from scenarios.scenario_henn_rl.structured.audit import (
        audit_structured_critic,
        score_counterfactual_audit,
    )
    from scenarios.scenario_henn_rl.structured.models import (
        OrderScoreActor,
        StructuredCritic,
    )
    from scenarios.scenario_henn_rl.structured.policy import StructuredPolicy

    actor = OrderScoreActor(hidden=8)
    episode_kwargs = {}
    episode = __import__(
        "scenarios.scenario_henn_rl.structured.environment",
        fromlist=["StructuredBatchingEpisode"],
    ).StructuredBatchingEpisode([INSTANCE], **episode_kwargs)
    policy = StructuredPolicy(actor, episode.decoder)
    episode.close()
    critic = StructuredCritic(hidden=8)
    result = audit_structured_critic(
        policy,
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
        policy,
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

    from scenarios.scenario_henn_rl.structured.environment import (
        StructuredBatchingEpisode,
    )

    episode = StructuredBatchingEpisode([INSTANCE], reward_power=power)
    state = episode.reset(instance_id=INSTANCE)
    episode_return = 0.0
    done = False
    while not done:
        selected = episode.oracle_action(
            np.full(len(state.order_ids), -1.0), state
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

    from scenarios.scenario_henn_rl.structured.decoders import make_decoder
    from scenarios.scenario_henn_rl.structured.policy import StructuredPolicy
    from scenarios.scenario_henn_rl.structured.state import BatchingState

    state = BatchingState(
        order_ids=np.asarray([0, 1], dtype=np.int64),
        features=np.zeros((2, 7), dtype=np.float32),
        demands=np.asarray([2, 3]),
        capacity=3,
        order_positions=[[(0, 1)], [(0, 2)]],
        input_closed=False,
        feature_schema="legacy_v1",
    )
    policy = StructuredPolicy(
        torch.nn.Linear(7, 2), make_decoder("knapsack")
    )
    decoded = policy.decode(torch.tensor([-5.0, -2.0]), state)
    assert decoded.tolist() == [0.0, 1.0]


def test_nonuniform_checkpoints_are_evaluated_and_saved(tmp_path):
    pytest.importorskip("torch")
    from scenarios.scenario_henn_rl.structured.models import OrderScoreActor
    from scenarios.scenario_henn_rl.structured.training import train_structured_rl

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


def test_srl_diagnosis_reward_identity_is_exact():
    from scenarios.scenario_henn_rl.studies.srl_failure_diagnosis import (
        run_reward_identity,
    )

    result = run_reward_identity([INSTANCE])
    assert result["summary"]["instances"] == 1
    assert result["summary"]["max_reward_identity_error"] < 1e-9
    assert result["rows"][0]["decisions"] > 0


def test_srl_diagnosis_representability_covers_additive_limits():
    from scenarios.scenario_henn_rl.studies.srl_failure_diagnosis import (
        run_representability,
    )

    def record(instance, decision, demands, capacity, action, true_q):
        return {
            "instance_id": instance,
            "decision_index": decision,
            "order_ids": list(range(len(demands))),
            "demands": demands,
            "capacity": capacity,
            "action": action,
            "true_q": true_q,
        }

    demands = [10, 10, 10, 10]
    capacity = 20
    actions = [
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    true_q = [10.0, 8.0, 7.0, 5.0, 4.0, 3.0]
    dataset = {
        "collect_q": True,
        "records": [
            record("a", 0, demands, capacity, list(map(float, action)), value)
            for action, value in zip(actions, true_q)
        ],
    }
    result = run_representability(dataset)
    state = result["states"][0]
    assert result["summary"]["states"] == 1
    assert state["best_batch"]["status"] == "positive_margin"
    assert state["best_batch"]["decoder_selects"] is True
    assert state["best_batch"]["margin"] >= 1.0
    assert state["preferences"]["perfect_pairwise"] is True
    assert state["preferences"]["lsq_top_match"] is True
    assert result["summary"]["preferences"]["top_match_fraction"] == 1.0

    dataset["collect_q"] = False
    try:
        run_representability(dataset)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError without --collect-q data")


def test_release_env_propagates_threshold_and_scale():
    from scenarios.scenario_henn_rl.environment import ReleaseTimingEnv

    env = ReleaseTimingEnv(
        [INSTANCE], sla_threshold_s=50.0, objective_scale=4.0
    )
    env.reset(seed=0, options={"instance_id": INSTANCE})
    assert env.reward_model.thresholds_s == {
        order_id: 50.0 for order_id in env.arrivals
    }
    assert env.reward_normalizer == pytest.approx(
        max(1.0, len(env.arrivals)) * 4.0
    )
    env.close()


def test_structured_sla_objective_and_return_identity():
    import numpy as np

    from scenarios.scenario_henn_rl.structured.environment import (
        StructuredBatchingEpisode,
    )

    episode = StructuredBatchingEpisode(
        [INSTANCE], sla_threshold_s=50.0, objective_scale=2.5
    )
    state = episode.reset(instance_id=INSTANCE)
    normalizer = episode.env.reward_normalizer
    episode_return = 0.0
    done = False
    while not done:
        selected = episode.oracle_action(
            np.ones(len(state.order_ids)), state
        )
        assert selected.size > 0
        state, reward, done, _, info = episode.step(selected)
        episode_return += reward
    completions = episode.env.completion_times()
    flows = [
        completions[order_id] - arrival
        for order_id, arrival in episode.env.arrivals.items()
    ]
    tardiness = sum(max(0.0, flow - 50.0) for flow in flows)
    assert info["objective_cost"] == pytest.approx(tardiness, abs=1e-6)
    assert episode_return == pytest.approx(
        -info["objective_cost"] / normalizer, abs=1e-9
    )
    assert info["wait_actions"] == 0
    episode.close()


def test_thresholded_actor_evaluation_identity_is_exact():
    from scenarios.scenario_henn_rl.structured.evaluation import (
        evaluate_structured_actor,
    )
    from scenarios.scenario_henn_rl.structured.models import OrderScoreActor

    evaluation = evaluate_structured_actor(
        OrderScoreActor(hidden=8),
        [INSTANCE],
        sla_threshold_s=100.0,
        objective_scale=3.0,
    )
    assert evaluation["max_reward_identity_error"] < 1e-9
    assert evaluation["episodes"][0]["objective_cost"] > 0.0

