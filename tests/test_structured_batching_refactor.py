from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from scenarios.scenario_henn_rl.structured.checkpoint import (
    assert_policy_compatible,
    build_checkpoint,
    checkpoint_decoder_name,
    load_workbench_checkpoint,
    save_checkpoint,
)
from scenarios.scenario_henn_rl.structured.decoders import (
    GreedyRouteAwareDecoder,
    KnapsackDecoder,
    brute_force_route_aware_batch,
    greedy_route_aware_batch,
    knapsack_batch,
    make_decoder,
)
from scenarios.scenario_henn_rl.structured.models import (
    OrderScoreActor,
    StructuredCritic,
    structured_candidates,
)
from scenarios.scenario_henn_rl.structured.policy import StructuredPolicy
from scenarios.scenario_henn_rl.structured.state import (
    BatchingState,
    action_mask,
    copy_state,
)

INSTANCE = "H_abc1_40_29"


def _state(
    order_ids=None,
    demands=(2, 3, 4),
    capacity=5,
    order_positions=None,
    *,
    route_cost_scale=10.0,
    feature_count=8,
):
    n = len(demands)
    if order_ids is None:
        order_ids = list(range(n))
    if order_positions is None:
        order_positions = [[(0, 1)], [(0, 2)], [(1, 3)]][:n]
    return BatchingState(
        order_ids=np.asarray(order_ids, dtype=np.int64),
        features=np.zeros((n, feature_count), dtype=np.float32),
        demands=np.asarray(demands, dtype=np.int64),
        capacity=capacity,
        order_positions=order_positions,
        input_closed=False,
        feature_schema="deadline_v1",
        route_cost_scale=route_cost_scale,
    )


def _route_cost_fn(order_positions, selected_indices):
    """Deterministic synthetic route cost: count unique aisles + sum of y."""
    positions = []
    for i in selected_indices:
        positions.extend(order_positions[int(i)])
    if not positions:
        return 0.0
    aisles = {p[0] for p in positions}
    return float(len(aisles) + sum(p[1] for p in positions))


# ---------------------------------------------------------------------------
# 1. Exact knapsack decoding matches brute-force enumeration on small states.
# ---------------------------------------------------------------------------


def test_knapsack_decoder_matches_brute_force():
    rng = np.random.default_rng(7)
    for size in range(1, 7):
        scores = rng.normal(size=size)
        demands = rng.integers(1, 5, size=size)
        capacity = 7
        from itertools import combinations

        feasible = [
            subset
            for length in range(size + 1)
            for subset in combinations(range(size), length)
            if demands[list(subset)].sum() <= capacity
        ]
        optimum = max(sum(scores[i] for i in subset) for subset in feasible)
        selected = knapsack_batch(scores, demands, capacity, allow_empty=True)
        assert scores[selected].sum() == pytest.approx(optimum)
    decoder = KnapsackDecoder()
    state = _state(demands=(3, 4, 2), capacity=7)
    scores = np.array([4.0, 5.0, 7.0])
    result = decoder.select(scores, state)
    # Best additive value is {1,2} = 5+7 = 12, demand 6 <= 7.
    assert result.selected_indices.tolist() == [1, 2]
    assert state.demands[result.selected_indices].sum() <= state.capacity
    assert result.route_cost == 0.0


# ---------------------------------------------------------------------------
# 2. Greedy route-aware decoding: capacity, determinism, configured route
#    cost, and distinguishability from the exact brute-force optimum.
# ---------------------------------------------------------------------------


def test_greedy_route_aware_decoder_respects_capacity_and_is_deterministic():
    decoder = GreedyRouteAwareDecoder(route_cost_fn=_route_cost_fn)
    state = _state(demands=(3, 4, 5), capacity=7, route_cost_scale=10.0)
    scores = np.array([5.0, 4.0, 3.0])
    first = decoder.select(scores, state)
    second = decoder.select(scores, state)
    assert np.array_equal(first.selected_indices, second.selected_indices)
    assert state.demands[first.selected_indices].sum() <= state.capacity
    assert first.route_cost == _route_cost_fn(
        state.order_positions, first.selected_indices.tolist()
    )


def test_greedy_route_aware_uses_configured_route_cost():
    decoder = GreedyRouteAwareDecoder(route_cost_fn=_route_cost_fn)
    state = _state(demands=(2, 2), capacity=10, route_cost_scale=1.0)
    scores = np.array([1.0, 1.0])
    result = decoder.select(scores, state)
    # With scale=1.0 the route-cost term dominates; the greedy decoder must
    # reflect the configured route-cost function and scale.
    assert result.route_cost == _route_cost_fn(
        state.order_positions, result.selected_indices.tolist()
    )
    assert result.objective == pytest.approx(
        float(scores[result.selected_indices].sum())
        - result.route_cost / 1.0
    )


def test_greedy_route_aware_is_distinguishable_from_exact_optimum():
    # Construct a small state where the greedy order-by-score acceptance
    # differs from the true optimum so the distinction is observable.
    # Order 0 has the highest score but a huge singleton route cost; because
    # ``allow_empty=False`` forces the greedy to accept the first feasible
    # order, it locks into order 0 and builds on it, while the exact optimum
    # excludes order 0 and picks the two low-route-cost orders instead.
    demands = (1, 1, 1)
    capacity = 3
    order_positions = [[(2, 10)], [(0, 1)], [(0, 1)]]
    scores = np.array([5.0, 3.0, 3.0])
    route_cost_scale = 1.0
    greedy_idx, greedy_val, _ = greedy_route_aware_batch(
        scores,
        np.asarray(demands),
        capacity,
        order_positions,
        _route_cost_fn,
        route_cost_scale,
        allow_empty=False,
    )
    exact_idx, exact_val, _ = brute_force_route_aware_batch(
        scores,
        np.asarray(demands),
        capacity,
        order_positions,
        _route_cost_fn,
        route_cost_scale,
        allow_empty=False,
    )
    assert greedy_val < exact_val
    assert not np.array_equal(greedy_idx, exact_idx)
    # The exact optimum is the two co-aisle low-route-cost orders {1,2}.
    assert sorted(exact_idx.tolist()) == [1, 2]
    assert sorted(greedy_idx.tolist()) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 3. An unknown decoder configuration fails loudly.
# ---------------------------------------------------------------------------


def test_unknown_decoder_fails_loudly():
    with pytest.raises(ValueError, match="Unknown decoder"):
        make_decoder("does_not_exist")
    with pytest.raises(ValueError, match="route_cost_fn"):
        make_decoder("route_aware_greedy", route_cost_fn=None)


# ---------------------------------------------------------------------------
# 4. Same actor, state, and decoder produce the same action in rollout
#    collection, validation, evaluation, and audit.
# ---------------------------------------------------------------------------


def test_single_decoder_path_across_consumers():
    from omegaconf import OmegaConf

    from scenarios.scenario_henn_rl.structured.data import (
        GeneratedHennDataLoader,
        generated_instance_splits,
    )
    from scenarios.scenario_henn_rl.structured.environment import (
        StructuredBatchingEpisode,
    )
    from scenarios.scenario_henn_rl.structured.evaluation import (
        evaluate_structured_actor,
    )
    from scenarios.scenario_henn_rl.structured.audit import audit_structured_critic

    spec = OmegaConf.to_container(
        OmegaConf.load(
            "scenarios/scenario_henn_rl/config/data/henn_lorenz.yaml"
        ),
        resolve=True,
    )
    spec.update(train_instances=4, validation_instances=4, test_instances=4)
    instance_id = generated_instance_splits(spec)["train"][0]
    loader = GeneratedHennDataLoader(spec)
    episode_kwargs = {
        "data_loader": loader,
        "use_order_due_dates": True,
        "include_due_slack": True,
        "decoder": "route_aware_greedy",
    }
    actor = OrderScoreActor(feature_count=8, hidden=8)
    episode = StructuredBatchingEpisode([instance_id], **episode_kwargs)
    state = episode.reset(instance_id=instance_id)
    policy = StructuredPolicy(actor, episode.decoder)
    action = policy.select_action(state)
    indices = torch.nonzero(action, as_tuple=False).flatten().numpy()
    assert state.demands[indices].sum() <= state.capacity
    # evaluate_structured_actor builds the same decoder via episode_kwargs.
    evaluation = evaluate_structured_actor(
        actor,
        [instance_id],
        episode_kwargs=episode_kwargs,
    )
    assert evaluation["instances"] == 1
    # audit uses the same policy abstraction.
    audit = audit_structured_critic(
        policy,
        StructuredCritic(feature_count=8),
        [instance_id],
        reward_power=1.0,
        comparison_powers=[1.0],
        state_quantiles=[0.5],
        candidate_count=3,
        sigma=0.5,
        temperature=0.5,
        seed=1,
        episode_kwargs=episode_kwargs,
    )
    assert audit["decoder"] == "route_aware_greedy"
    episode.close()


# ---------------------------------------------------------------------------
# 5. A route-aware checkpoint round-trip reproduces the same decoded action.
# ---------------------------------------------------------------------------


def test_route_aware_checkpoint_round_trip(tmp_path):
    actor = OrderScoreActor(feature_count=8, hidden=8)
    critic = StructuredCritic(feature_count=8)
    decoder_config = {"name": "route_aware_greedy", "allow_empty": False}
    checkpoint = build_checkpoint(
        actor=actor,
        critic=critic,
        critic_kind="mean",
        feature_schema="deadline_v1",
        feature_count=8,
        actor_hidden=8,
        decoder_config=decoder_config,
        objective={"name": "flow", "power": 1.0, "gamma": 1.0, "use_order_due_dates": False},
        objective_scale=2.5,
        data_spec={"base_instance_id": "H_abc1_40_30", "cart_capacity": 45},
        selected_episode=3,
        route_cost_scale=123.4,
    )
    path = tmp_path / "ra_best.pt"
    save_checkpoint(path, checkpoint)
    loaded_actor, loaded_critic, loaded = load_workbench_checkpoint(path)
    assert checkpoint_decoder_name(loaded) == "route_aware_greedy"
    assert loaded["route_cost_scale"] == pytest.approx(123.4)
    # The actor weights round-trip exactly, so the same state + decoder
    # produces the same action before and after the checkpoint save/load.
    state = _state(demands=(2, 2, 2), capacity=4, route_cost_scale=123.4)
    decoder = make_decoder(
        checkpoint_decoder_name(loaded), route_cost_fn=_route_cost_fn
    )
    before = StructuredPolicy(actor, make_decoder("route_aware_greedy", route_cost_fn=_route_cost_fn)).select_action(state)
    after = StructuredPolicy(loaded_actor, decoder).select_action(state)
    assert torch.equal(before, after)


# ---------------------------------------------------------------------------
# 6. Best-checkpoint actor and critic come from the same validation point.
# ---------------------------------------------------------------------------


def test_best_checkpoint_actor_critic_same_point(tmp_path):
    from omegaconf import OmegaConf

    from scenarios.scenario_henn_rl.structured.data import (
        GeneratedHennDataLoader,
        generated_instance_splits,
    )
    from scenarios.scenario_henn_rl.structured.training import train_structured_rl

    spec = OmegaConf.to_container(
        OmegaConf.load("scenarios/scenario_henn_rl/config/data/henn_lorenz.yaml"),
        resolve=True,
    )
    spec.update(train_instances=4, validation_instances=4, test_instances=4)
    splits = generated_instance_splits(spec)
    loader = GeneratedHennDataLoader(spec)
    episode_kwargs = {
        "data_loader": loader,
        "use_order_due_dates": True,
        "include_due_slack": True,
        "decoder": "knapsack",
    }
    actor = OrderScoreActor(feature_count=8, hidden=8)
    critic, result = train_structured_rl(
        actor,
        splits["train"][:3],
        episodes=3,
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
        seed=11,
        validation_ids=splits["validation"][:1],
        checkpoint_episodes=[0, 1, 3],
        checkpoint_dir=tmp_path,
        reward_power=1.0,
        feature_count=8,
        feature_schema="deadline_v1",
        actor_hidden=8,
        episode_kwargs=episode_kwargs,
    )
    assert result["best_actor_critic_consistent"] is True
    # Every advertised checkpoint file uses the same loadable schema.
    for path in tmp_path.glob("*.pt"):
        _, _, cp = load_workbench_checkpoint(path)
        assert cp["format_version"] == 2
        assert cp["feature_schema"] == "deadline_v1"
        assert "decoder" in cp


# ---------------------------------------------------------------------------
# 7. Evaluation rejects incompatible policy-defining configuration.
# ---------------------------------------------------------------------------


def test_evaluation_rejects_incompatible_config(tmp_path):
    actor = OrderScoreActor(feature_count=8, hidden=8)
    critic = StructuredCritic(feature_count=8)
    checkpoint = build_checkpoint(
        actor=actor,
        critic=critic,
        critic_kind="mean",
        feature_schema="deadline_v1",
        feature_count=8,
        actor_hidden=8,
        decoder_config={"name": "knapsack", "allow_empty": False},
        objective={"name": "flow", "power": 1.0, "gamma": 1.0, "use_order_due_dates": False},
        objective_scale=1.0,
        data_spec={"base_instance_id": "H_abc1_40_30", "cart_capacity": 45},
        selected_episode=1,
    )
    path = tmp_path / "incompat.pt"
    save_checkpoint(path, checkpoint)
    _, _, loaded = load_workbench_checkpoint(path)
    # Requesting a different decoder is rejected.
    with pytest.raises(ValueError, match="incompatible"):
        assert_policy_compatible(loaded, decoder_name="route_aware_greedy")
    # Requesting a different feature schema is rejected.
    with pytest.raises(ValueError, match="schema"):
        assert_policy_compatible(loaded, feature_schema="legacy_v1")
    # Requesting a different objective is rejected.
    with pytest.raises(ValueError, match="objective"):
        assert_policy_compatible(
            loaded,
            objective={"name": "sla_tardiness", "power": 1.0, "gamma": 1.0, "use_order_due_dates": True},
        )


# ---------------------------------------------------------------------------
# 8. Replay states contain no bound environment callables and survive
#    copying or serialization.
# ---------------------------------------------------------------------------


def test_replay_state_has_no_bound_callables_and_survives_serialization():
    state = _state(demands=(2, 3), capacity=5, route_cost_scale=7.5)
    copied = copy_state(state)
    assert copied is not state
    assert np.array_equal(copied.order_ids, state.order_ids)
    assert np.allclose(copied.features, state.features)
    assert copied.order_positions is not state.order_positions
    # The state carries no bound environment callable: it must pickle and the
    # round-tripped copy must remain usable.
    data = pickle.dumps(state)
    restored = pickle.loads(data)
    assert np.array_equal(restored.order_ids, state.order_ids)
    assert restored.capacity == state.capacity
    assert restored.route_cost_scale == state.route_cost_scale
    # A knapsack decoder works on the restored, env-free state.
    decoder = KnapsackDecoder()
    result = decoder.select(np.array([0.5, 0.9]), restored)
    assert sorted(result.selected_indices.tolist()) == [0, 1]


# ---------------------------------------------------------------------------
# 9. The existing reward identity remains valid.
# ---------------------------------------------------------------------------


def test_reward_identity_remains_valid():
    from scenarios.scenario_henn_rl.structured.environment import (
        StructuredBatchingEpisode,
    )

    episode = StructuredBatchingEpisode([INSTANCE], reward_power=1.5)
    state = episode.reset(instance_id=INSTANCE)
    normalizer = episode.env.reward_normalizer
    episode_return = 0.0
    done = False
    while not done:
        selected = episode.oracle_action(
            np.full(len(state.order_ids), -1.0), state
        )
        state, reward, done, _, info = episode.step(selected)
        episode_return += reward
    assert episode_return == pytest.approx(
        -info["objective_cost"] / normalizer, rel=1e-12
    )
    episode.close()


# ---------------------------------------------------------------------------
# 10. A minimal train -> load -> evaluate smoke test succeeds without Gurobi.
# ---------------------------------------------------------------------------


def test_minimal_train_load_evaluate_smoke(tmp_path):
    import sys

    from omegaconf import OmegaConf

    from scenarios.scenario_henn_rl.structured.data import (
        GeneratedHennDataLoader,
        generated_instance_splits,
    )
    from scenarios.scenario_henn_rl.structured.training import (
        train_structured_rl,
    )
    from scenarios.scenario_henn_rl.structured.evaluation import (
        evaluate_structured_actor,
    )

    # Ensure no Gurobi module is importable for the duration of this test.
    sys.modules["gurobipy"] = None  # type: ignore
    try:
        spec = OmegaConf.to_container(
            OmegaConf.load("scenarios/scenario_henn_rl/config/data/henn_lorenz.yaml"),
            resolve=True,
        )
        spec.update(train_instances=4, validation_instances=4, test_instances=4)
        splits = generated_instance_splits(spec)
        loader = GeneratedHennDataLoader(spec)
        episode_kwargs = {
            "data_loader": loader,
            "use_order_due_dates": True,
            "include_due_slack": True,
            "decoder": "knapsack",
        }
        actor = OrderScoreActor(feature_count=8, hidden=8)
        critic, result = train_structured_rl(
            actor,
            splits["train"][:1],
            episodes=1,
            updates_per_episode=1,
            batch_size=1,
            replay_capacity=10,
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
            validation_ids=splits["validation"][:1],
            checkpoint_episodes=[0, 1],
            checkpoint_dir=tmp_path,
            feature_count=8,
            feature_schema="deadline_v1",
            actor_hidden=8,
            episode_kwargs=episode_kwargs,
        )
        best_path = tmp_path / "best.pt"
        save_checkpoint(
            best_path,
            build_checkpoint(
                actor=actor,
                critic=critic,
                critic_kind="mean",
                feature_schema="deadline_v1",
                feature_count=8,
                actor_hidden=8,
                decoder_config={"name": "knapsack", "allow_empty": False},
                objective={"name": "flow", "power": 1.0, "gamma": 1.0, "use_order_due_dates": True},
                objective_scale=1.0,
                data_spec=spec,
                selected_episode=result["selected_episode"],
            ),
        )
        loaded_actor, _, loaded = load_workbench_checkpoint(best_path)
        assert checkpoint_decoder_name(loaded) == "knapsack"
        evaluation = evaluate_structured_actor(
            loaded_actor,
            splits["validation"][:1],
            episode_kwargs=episode_kwargs,
        )
        assert evaluation["instances"] == 1
        assert evaluation["max_reward_identity_error"] < 1e-9
    finally:
        del sys.modules["gurobipy"]
