"""Causal SRL-failure diagnostics for structured order batching.

Compact reproduction of the four decisive evidence points:

- ``reward-identity``: the undiscounted sum of per-decision rewards equals the
  normalized final flow-time objective (``OrderCostReward`` is exact), so the
  learning signal is not corrupted by the reward model.
- ``collect``: builds a per-candidate exact-Q dataset from a checkpoint actor
  (state features, demands, capacity, actions, and exact counterfactual
  continuation values under the frozen continuation policy).
- ``actor-update``: applies exactly one Fenchel-Young gradient step to a fresh
  copy of the checkpoint actor per state and measures the change in exact Q of
  the chosen action, comparing the learned-critic target with targets weighted
  by exact rollout Q (normalized/softer variants included).
- ``representability``: asks whether the additive per-order-score action space
  (the production knapsack decoder) can represent the exact-Q evidence: it can
  select the exact-Q-best candidate over the complete feasible batch space in
  every audited state, but frequently cannot reproduce the exact-Q candidate
  ordering.

The encoder/representation study is supporting context only; its numbers are
retained in ``docs/structured_batching_srl_diagnosis.md``, not reproduced here.

Usage:
    uv run python -m scenarios.scenario_henn_rl.studies.srl_failure_diagnosis \
        --mode reward-identity --output <dir> --instances 4
    uv run python -m scenarios.scenario_henn_rl.studies.srl_failure_diagnosis \
        --mode collect --checkpoint <best.pt> --output <dir> \
        --split validation --instances 8 --state-quantiles 0.33,0.67 \
        --candidate-count 24 --collect-q
    uv run python -m scenarios.scenario_henn_rl.studies.srl_failure_diagnosis \
        --mode actor-update --checkpoint <best.pt> --output <dir> \
        --split validation --instances 8 --state-quantiles 0.33,0.67
    uv run python -m scenarios.scenario_henn_rl.studies.srl_failure_diagnosis \
        --mode representability --dataset <dir>/dataset.json --output <dir>
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog, lsq_linear
from scipy.stats import spearmanr

from scenarios.scenario_henn_rl.structured.audit import _candidate_rollout
from scenarios.scenario_henn_rl.structured.data import (
    GeneratedHennDataLoader,
    generated_instance_splits,
)
from scenarios.scenario_henn_rl.structured.environment import (
    StructuredBatchingEpisode,
    knapsack_batch,
)
from scenarios.scenario_henn_rl.structured.models import (
    _copy_state,
    _features,
    critic_soft_target,
    decode_scores,
    fenchel_young_loss,
    load_workbench_checkpoint,
    structured_candidates,
)
from scenarios.scenario_henn_rl.structured.results import write_json

DATA_SPEC = {
    "base_instance_id": "H_abc1_40_30",
    "cart_capacity": 45,
    "cutoff_cadence_hours": 1,
    "location_policy": "uniform",
    "max_order_items": 24,
    "min_order_items": 4,
    "minimum_lead_hours": 4,
    "order_counts": [40, 60, 80, 100],
    "orders_per_four_hours": 90,
    "seed": 11,
    "test_instances": 32,
    "time_per_pick": 10.0,
    "tour_setup_time": 180.0,
    "train_instances": 128,
    "travel_speed": 0.8,
    "validation_instances": 32,
}


def _episode_kwargs():
    loader = GeneratedHennDataLoader(DATA_SPEC)
    return {
        "data_loader": loader,
        "use_order_due_dates": False,
        "include_due_slack": True,
    }


def _trajectory(actor, instance_id: str, episode_kwargs: dict | None = None):
    """Actor-decided reference trajectory (states and chosen order IDs)."""
    episode = StructuredBatchingEpisode([instance_id], **(episode_kwargs or {}))
    state = episode.reset(instance_id=instance_id)
    trajectory = []
    done = False
    while not done:
        with torch.no_grad():
            action = decode_scores(actor(_features(state)), state)
        indices = torch.nonzero(action, as_tuple=False).flatten().cpu().numpy()
        selected = [int(state["order_ids"][index]) for index in indices]
        trajectory.append(
            {
                "state": _copy_state(state),
                "actor_order_ids": selected,
            }
        )
        state, _, done, _, _ = episode.step(indices)
    episode.close()
    return trajectory


def _indices_for_orders(state: dict[str, object], order_ids) -> np.ndarray:
    index_by_order = {
        int(order_id): index for index, order_id in enumerate(state["order_ids"])
    }
    return np.asarray(
        [index_by_order[int(order_id)] for order_id in order_ids], dtype=int
    )


def collect_dataset(
    actor,
    instance_ids: list[str],
    *,
    state_quantiles: list[float],
    candidate_count: int,
    sigma: float,
    seed: int,
    collect_q: bool,
    objective_scale: float | None = None,
    episode_kwargs: dict | None = None,
    reward_power: float = 1.0,
) -> dict[str, object]:
    """Collect feasible candidates with exact continuation Q per audited state."""
    generator = torch.Generator().manual_seed(seed)
    records = []
    started = time.perf_counter()
    for instance_id in instance_ids:
        trajectory = _trajectory(actor, instance_id, episode_kwargs)
        indices = sorted(
            {
                min(
                    len(trajectory) - 1,
                    max(0, int(round(q * (len(trajectory) - 1)))),
                )
                for q in state_quantiles
            }
        )
        episode = StructuredBatchingEpisode([instance_id], **(episode_kwargs or {}))
        state = episode.reset(instance_id=instance_id)
        wanted = set(indices)
        for decision_index, reference in enumerate(trajectory):
            current_state = episode.state()
            if not np.array_equal(
                current_state["order_ids"], reference["state"]["order_ids"]
            ):
                raise RuntimeError("Replay diverged from reference order buffer")
            state = current_state
            if decision_index in wanted:
                prefix_order_ids = [
                    trajectory[position]["actor_order_ids"]
                    for position in range(decision_index)
                ]
                candidates = structured_candidates(
                    actor,
                    state,
                    candidate_count=candidate_count,
                    sigma=sigma,
                    generator=generator,
                    deduplicate=True,
                )
                for action in candidates:
                    selected_indices = torch.nonzero(
                        action, as_tuple=False
                    ).flatten().cpu().numpy()
                    order_ids = [
                        int(state["order_ids"][index]) for index in selected_indices
                    ]
                    record = {
                        "instance_id": instance_id,
                        "decision_index": decision_index,
                        "order_ids": state["order_ids"].tolist(),
                        "features": state["features"].tolist(),
                        "demands": state["demands"].tolist(),
                        "capacity": int(state["capacity"]),
                        "action": action.tolist(),
                    }
                    if collect_q:
                        rollout = _candidate_rollout(
                            actor,
                            instance_id,
                            prefix_order_ids,
                            state,
                            order_ids,
                            [reward_power],
                            sla_threshold_s=0.0,
                            objective_scale=objective_scale,
                            episode_kwargs=episode_kwargs,
                        )
                        record["true_q"] = float(
                            rollout["true_q"][str(reward_power)]
                        )
                    records.append(record)
            state, _, done, _, _ = episode.step(
                _indices_for_orders(state, reference["actor_order_ids"])
            )
            if done:
                break
        episode.close()
    return {
        "actor_checkpoint": getattr(actor, "checkpoint_path", None),
        "reward_power": reward_power,
        "state_quantiles": state_quantiles,
        "candidate_count": candidate_count,
        "sigma": sigma,
        "seed": seed,
        "collect_q": bool(collect_q),
        "records": records,
        "wall_time_s": time.perf_counter() - started,
    }


def _full_episode_q(
    policy_actor,
    instance_id: str,
    powers: list[float],
    sla_threshold_s: float,
    objective_scale: float | None,
    episode_kwargs: dict | None,
) -> dict[str, float]:
    """Exact total Q of a policy over one whole episode from the initial state."""
    probe = StructuredBatchingEpisode([instance_id], **(episode_kwargs or {}))
    initial_state = probe.reset(instance_id=instance_id)
    with torch.no_grad():
        action = decode_scores(policy_actor(_features(initial_state)), initial_state)
    indices = torch.nonzero(action, as_tuple=False).flatten().cpu().numpy()
    initial_order_ids = [
        int(initial_state["order_ids"][index]) for index in indices
    ]
    probe.close()
    rollout = _candidate_rollout(
        policy_actor,
        instance_id,
        [],
        initial_state,
        initial_order_ids,
        powers,
        sla_threshold_s=sla_threshold_s,
        objective_scale=objective_scale,
        episode_kwargs=episode_kwargs,
    )
    return {str(power): float(rollout["true_q"][str(power)]) for power in powers}


def run_actor_update_audit(
    actor,
    critic,
    instance_ids: list[str],
    *,
    state_quantiles: list[float],
    candidate_count: int,
    sigma: float,
    temperature: float,
    fy_samples: int,
    epsilon: float,
    learning_rate: float,
    seed: int,
    reward_power: float,
    objective_scale: float | None,
    episode_kwargs: dict | None,
    measure_full_policy: bool = True,
    targets: list[dict] | None = None,
) -> dict[str, object]:
    """Isolated single-step actor-update audit, optionally comparing targets.

    For each selected state a fresh copy of the frozen checkpoint actor is
    created and exactly one Fenchel-Young gradient step is applied to it.  The
    continuation policy used to evaluate exact Q is always the *frozen*
    checkpoint actor, so ``delta_q`` isolates the value change of the chosen
    action at this state (per-step policy-improvement check).  When
    ``measure_full_policy`` is set, whole-episode exact Q of the frozen and of
    the updated policy is also reported.

    ``targets`` (default: the raw-Q target at ``temperature``) is a list of
    target configurations ``{"name", "temperature", "normalize_advantages",
    "values_source"}`` where ``values_source`` is ``"critic"`` (default;
    candidate quality comes from the learned critic) or ``"exact"`` (candidate
    quality comes from exact counterfactual rollout Q under the frozen
    continuation policy).  All targets are evaluated on the *same* candidate
    set per state, and each target applies its single update to its own fresh
    actor copy with the same Monte-Carlo Fenchel-Young noise, so the rows
    differ only through the target weighting.

    This deliberately differs from real training dynamics, where a single
    shared actor is updated sequentially and later decisions are reached under
    the updated actor: the audit measures the per-state action improvement in
    isolation and must not be read as a training-loss trend.
    """
    if targets is None:
        targets = [
            {
                "name": "current",
                "temperature": temperature,
                "normalize_advantages": False,
            }
        ]
    powers = [reward_power]
    power = str(reward_power)
    generator = torch.Generator().manual_seed(seed)
    fy_generator = torch.Generator().manual_seed(seed + 1)
    rows = []
    for instance_id in instance_ids:
        trajectory = _trajectory(actor, instance_id, episode_kwargs)
        indices = sorted(
            {
                min(
                    len(trajectory) - 1,
                    max(0, int(round(q * (len(trajectory) - 1)))),
                )
                for q in state_quantiles
            }
        )
        episode = StructuredBatchingEpisode([instance_id], **(episode_kwargs or {}))
        state = episode.reset(instance_id=instance_id)
        for decision_index, reference in enumerate(trajectory):
            current_state = episode.state()
            if not np.array_equal(
                current_state["order_ids"], reference["state"]["order_ids"]
            ):
                raise RuntimeError("Replay diverged from reference order buffer")
            state = current_state
            if decision_index not in indices:
                state, _, done, _, _ = episode.step(
                    _indices_for_orders(state, reference["actor_order_ids"])
                )
                if done:
                    break
                continue
            prefix_order_ids = [
                trajectory[position]["actor_order_ids"]
                for position in range(decision_index)
            ]
            candidates = structured_candidates(
                actor,
                state,
                candidate_count=candidate_count,
                sigma=sigma,
                generator=generator,
                deduplicate=False,
            )

            with torch.no_grad():
                before_action = decode_scores(actor(_features(state)), state)
            before_indices = torch.nonzero(
                before_action, as_tuple=False
            ).flatten().cpu().numpy()
            before_order_ids = [
                int(state["order_ids"][index]) for index in before_indices
            ]
            before_q = _candidate_rollout(
                actor,
                instance_id,
                prefix_order_ids,
                state,
                before_order_ids,
                powers,
                sla_threshold_s=0.0,
                objective_scale=objective_scale,
                episode_kwargs=episode_kwargs,
            )["true_q"][power]
            if measure_full_policy:
                full_before = _full_episode_q(
                    actor,
                    instance_id,
                    powers,
                    sla_threshold_s=0.0,
                    objective_scale=objective_scale,
                    episode_kwargs=episode_kwargs,
                )[power]
            fy_state = fy_generator.get_state()
            needs_exact = any(
                target_spec.get("values_source") == "exact"
                for target_spec in targets
            )
            exact_values = None
            if needs_exact:
                exact_values = torch.stack(
                    [
                        torch.as_tensor(
                            _candidate_rollout(
                                actor,
                                instance_id,
                                prefix_order_ids,
                                state,
                                [
                                    int(state["order_ids"][index])
                                    for index in torch.nonzero(
                                        action, as_tuple=False
                                    )
                                    .flatten()
                                    .cpu()
                                    .numpy()
                                ],
                                powers,
                                sla_threshold_s=0.0,
                                objective_scale=objective_scale,
                                episode_kwargs=episode_kwargs,
                            )["true_q"][power]
                        )
                        for action in candidates
                    ]
                )
            for target_spec in targets:
                updated_actor = copy.deepcopy(actor)

                target, diagnostic = critic_soft_target(
                    actor,
                    critic,
                    state,
                    candidate_count=candidate_count,
                    sigma=sigma,
                    temperature=float(target_spec["temperature"]),
                    generator=generator,
                    location_features=True,
                    normalize_advantages=bool(
                        target_spec.get("normalize_advantages", False)
                    ),
                    actions=candidates,
                    values=(
                        exact_values
                        if target_spec.get("values_source") == "exact"
                        else None
                    ),
                )
                fy_generator.set_state(fy_state)
                actor_loss, _ = fenchel_young_loss(
                    updated_actor,
                    state,
                    target,
                    sample_count=fy_samples,
                    epsilon=epsilon,
                    generator=fy_generator,
                    location_features=True,
                )
                optimizer = torch.optim.Adam(
                    updated_actor.parameters(), lr=learning_rate
                )
                optimizer.zero_grad()
                actor_loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(updated_actor.parameters(), 1.0)
                )
                optimizer.step()

                with torch.no_grad():
                    after_action = decode_scores(
                        updated_actor(_features(state)), state
                    )
                after_indices = torch.nonzero(
                    after_action, as_tuple=False
                ).flatten().cpu().numpy()
                after_order_ids = [
                    int(state["order_ids"][index]) for index in after_indices
                ]
                after_q = _candidate_rollout(
                    actor,
                    instance_id,
                    prefix_order_ids,
                    state,
                    after_order_ids,
                    powers,
                    sla_threshold_s=0.0,
                    objective_scale=objective_scale,
                    episode_kwargs=episode_kwargs,
                )["true_q"][power]
                row = {
                    "target": target_spec["name"],
                    "instance_id": instance_id,
                    "decision_index": decision_index,
                    "visible_orders": len(state["order_ids"]),
                    "q_before": float(before_q),
                    "q_after": float(after_q),
                    "delta_q": float(after_q - before_q),
                    "target_entropy": float(diagnostic["target_entropy"]),
                    "max_weight": float(diagnostic["max_weight"]),
                    "effective_count": float(diagnostic["effective_count"]),
                    "unique_candidates": int(diagnostic["unique_candidates"]),
                    "candidate_q_spread": float(diagnostic["candidate_q_spread"]),
                    "actor_loss": float(actor_loss.detach()),
                    "grad_norm": grad_norm,
                    "changed": bool(
                        not np.array_equal(before_order_ids, after_order_ids)
                    ),
                }
                if measure_full_policy:
                    full_after = _full_episode_q(
                        updated_actor,
                        instance_id,
                        powers,
                        sla_threshold_s=0.0,
                        objective_scale=objective_scale,
                        episode_kwargs=episode_kwargs,
                    )[power]
                    row["full_policy_before_q"] = float(full_before)
                    row["full_policy_after_q"] = float(full_after)
                    row["full_policy_delta_q"] = float(full_after - full_before)
                rows.append(row)

            state, _, done, _, _ = episode.step(
                _indices_for_orders(state, reference["actor_order_ids"])
            )
            if done:
                break
        episode.close()
    summary = {}
    for target_spec in targets:
        name = target_spec["name"]
        target_rows = [row for row in rows if row["target"] == name]
        deltas = np.asarray([row["delta_q"] for row in target_rows])
        item = {
            "states": len(target_rows),
            "fraction_improving": float(np.mean(deltas > 0)),
            "fraction_worsening": float(np.mean(deltas < 0)),
            "fraction_changed": float(
                np.mean([row["changed"] for row in target_rows])
            ),
            "mean_delta_q": float(np.mean(deltas)),
            "median_delta_q": float(np.median(deltas)),
            "std_delta_q": float(np.std(deltas)),
            "mean_target_entropy": float(
                np.mean([row["target_entropy"] for row in target_rows])
            ),
            "mean_max_weight": float(
                np.mean([row["max_weight"] for row in target_rows])
            ),
            "mean_effective_count": float(
                np.mean([row["effective_count"] for row in target_rows])
            ),
            "mean_candidate_q_spread": float(
                np.mean([row["candidate_q_spread"] for row in target_rows])
            ),
            "mean_actor_loss": float(
                np.mean([row["actor_loss"] for row in target_rows])
            ),
            "mean_grad_norm": float(
                np.mean([row["grad_norm"] for row in target_rows])
            ),
        }
        if measure_full_policy:
            full_deltas = np.asarray(
                [row["full_policy_delta_q"] for row in target_rows]
            )
            item["mean_full_policy_delta_q"] = float(np.mean(full_deltas))
            item["median_full_policy_delta_q"] = float(np.median(full_deltas))
            item["fraction_full_policy_improving"] = float(
                np.mean(full_deltas > 0)
            )
        summary[name] = item
    return {
        "rows": rows,
        "summary": summary,
        "targets": [
            {
                "name": target_spec["name"],
                "temperature": float(target_spec["temperature"]),
                "normalize_advantages": bool(
                    target_spec.get("normalize_advantages", False)
                ),
                "values_source": target_spec.get("values_source", "critic"),
            }
            for target_spec in targets
        ],
        "method": {
            "continuation_policy": "frozen_checkpoint_actor",
            "actor_per_state": "deepcopy_from_checkpoint",
            "candidate_set": "shared_across_targets",
            "isolates": "single_step_action_improvement",
            "full_policy_measured": measure_full_policy,
        },
    }


def _feasible_batches(
    n_orders: int, demands: np.ndarray, capacity: int
) -> list[np.ndarray]:
    """Enumerate all non-empty capacity-feasible order subsets as indicator vectors."""
    from itertools import combinations

    batches = []
    for size in range(1, n_orders + 1):
        for combo in combinations(range(n_orders), size):
            if sum(demands[index] for index in combo) <= capacity:
                indicator = np.zeros(n_orders, dtype=float)
                indicator[list(combo)] = 1.0
                batches.append(indicator)
    return batches


def _max_margin_lp(
    n_orders: int, rows: np.ndarray
) -> tuple[np.ndarray, float]:
    """Maximize ``m`` over ``(x - x_star) @ theta >= m`` for each constraint row.

    Variables are ``[theta; m]`` with ``theta`` in ``[-1, 1]``; positive scaling
    preserves the argmax of a linear score, so the box bound loses no strict
    representability.  Returns the optimal ``theta`` and the margin ``m``.
    """
    result = linprog(
        np.concatenate([np.zeros(n_orders), [-1.0]]),
        A_ub=rows,
        b_ub=np.zeros(len(rows)),
        bounds=[(-1.0, 1.0)] * n_orders + [(-2 * n_orders - 1, 2 * n_orders + 1)],
        method="highs",
    )
    if result.status != 0:
        raise RuntimeError(f"Representability LP failed: {result.message}")
    return result.x[:n_orders], float(-result.fun)


def run_representability(dataset_dict: dict[str, object]) -> dict[str, object]:
    """Test whether additive per-order scores can represent the exact-Q evidence.

    The actor's action space is the family of capacity-feasible order subsets
    selected by argmax over per-order additive scores (the production knapsack
    decoder).  This probe asks, per collected state, whether *any* score vector
    ``theta`` can reproduce the exact-Q evidence:

    - ``best-batch``: can some ``theta`` make the production knapsack decoder
      select the exact-Q-best observed candidate ``B*`` over the *complete*
      feasible batch space, and with what margin (LP over the full feasible
      set, then verified through the production decoder)?
    - ``preferences``: how much of the exact-Q candidate ordering over the
      observed candidates can an additive utility ``U(B) = sum_i theta_i``
      reproduce (maximum pairwise margin, and LSQ-fit Spearman / pairwise
      agreement / top-1 match)?

    No actor or critic is involved: the answer is purely about the additive
    score parameterization, not about whether the network can emit it.
    """
    records = dataset_dict["records"]
    if not dataset_dict.get("collect_q"):
        raise ValueError(
            "representability requires a dataset collected with --collect-q"
        )
    grouped: dict[tuple[str, int], list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(
            (record["instance_id"], record["decision_index"]), []
        ).append(index)
    tol = 1e-6
    states = []
    for (instance_id, decision_index), indices in grouped.items():
        state_records = [records[index] for index in indices]
        n_orders = len(state_records[0]["order_ids"])
        demands = np.asarray(state_records[0]["demands"], dtype=int)
        capacity = int(state_records[0]["capacity"])
        seen: dict[tuple, tuple[np.ndarray, float]] = {}
        for record in state_records:
            key = tuple(record["action"])
            if key not in seen or record["true_q"] > seen[key][1]:
                seen[key] = (
                    np.asarray(record["action"], dtype=float),
                    float(record["true_q"]),
                )
        actions = list(seen.values())
        X = np.stack([action for action, _ in actions])
        q = np.asarray([value for _, value in actions], dtype=float)
        m_candidates = X.shape[0]
        best_index = int(np.argmax(q))
        x_star = X[best_index]
        q_star = float(q[best_index])
        actor_action = X[0]
        actor_is_best = bool(np.array_equal(actor_action, x_star))
        spread = max(float(np.ptp(q)), 1e-12)
        actor_regret_fraction = float((q_star - q[0]) / spread)

        feasible = _feasible_batches(n_orders, demands, capacity)
        rows = [x - x_star for x in feasible if not np.array_equal(x, x_star)]
        if rows:
            theta_star, margin = _max_margin_lp(
                n_orders,
                np.asarray([np.concatenate([row, [1.0]]) for row in rows]),
            )
        else:
            theta_star, margin = np.zeros(n_orders), 0.0
        decoder_indices = knapsack_batch(
            theta_star, demands, capacity, allow_empty=False
        )
        decoder_selects = bool(
            set(decoder_indices.tolist())
            == set(np.flatnonzero(x_star).tolist())
        )
        if margin > tol:
            best_batch_status = "positive_margin"
        elif decoder_selects:
            best_batch_status = "tie_only"
        else:
            best_batch_status = "not_strictly_representable"

        pairs = [
            (i, j)
            for i in range(m_candidates)
            for j in range(m_candidates)
            if q[i] > q[j] + 1e-9
        ]
        pair_count = len(pairs)
        if pair_count:
            pair_rows = np.asarray(
                [
                    np.concatenate([X[j] - X[i], [1.0]])
                    for i, j in pairs
                ]
            )
            theta_pair, pair_margin = _max_margin_lp(n_orders, pair_rows)
            pred_lp = X @ theta_pair
            lp_agreement = float(
                np.mean([pred_lp[i] > pred_lp[j] for i, j in pairs])
            )
        else:
            pair_margin = 0.0
            pred_lp = np.zeros(m_candidates)
            lp_agreement = 1.0
        fit = lsq_linear(X, q, bounds=(-1.0, 1.0))
        pred_lsq = X @ fit.x
        rho = spearmanr(pred_lsq, q).statistic
        if np.isnan(rho):
            rho = None
        lsq_agreement = (
            float(np.mean([pred_lsq[i] > pred_lsq[j] for i, j in pairs]))
            if pair_count
            else 1.0
        )
        states.append(
            {
                "instance_id": instance_id,
                "decision_index": decision_index,
                "visible_orders": n_orders,
                "candidates": m_candidates,
                "feasible_batches": len(feasible),
                "capacity": capacity,
                "q_star": q_star,
                "actor_q": float(q[0]),
                "actor_is_best": actor_is_best,
                "actor_regret_fraction": actor_regret_fraction,
                "best_batch": {
                    "margin": margin,
                    "status": best_batch_status,
                    "decoder_selects": decoder_selects,
                },
                "preferences": {
                    "pair_count": pair_count,
                    "max_margin": pair_margin,
                    "perfect_pairwise": bool(pair_margin > tol),
                    "lp_pairwise_agreement": lp_agreement,
                    "lsq_spearman": rho,
                    "lsq_pairwise_agreement": lsq_agreement,
                    "lsq_top_match": bool(int(np.argmax(pred_lsq)) == best_index),
                },
            }
        )

    best_batch_rows = [state["best_batch"] for state in states]
    preference_rows = [state["preferences"] for state in states]
    margins = np.asarray([row["margin"] for row in best_batch_rows])
    non_positive = [
        {
            "instance_id": state["instance_id"],
            "decision_index": state["decision_index"],
            "margin": state["best_batch"]["margin"],
            "decoder_selects": state["best_batch"]["decoder_selects"],
        }
        for state in states
        if state["best_batch"]["status"] != "positive_margin"
    ]
    rho_values = [
        float(row["lsq_spearman"])
        for row in preference_rows
        if row["lsq_spearman"] is not None
    ]
    lowest_spearman = sorted(
        (
            {
                "instance_id": state["instance_id"],
                "decision_index": state["decision_index"],
                "spearman": state["preferences"]["lsq_spearman"],
                "agreement": state["preferences"]["lsq_pairwise_agreement"],
            }
            for state in states
            if state["preferences"]["lsq_spearman"] is not None
        ),
        key=lambda row: row["spearman"],
    )[:5]
    summary: dict[str, object] = {
        "states": len(states),
        "best_batch": {
            "fraction_positive_margin": float(
                np.mean([row["status"] == "positive_margin" for row in best_batch_rows])
            ),
            "fraction_tie_only": float(
                np.mean([row["status"] == "tie_only" for row in best_batch_rows])
            ),
            "fraction_not_strictly_representable": float(
                np.mean(
                    [
                        row["status"] == "not_strictly_representable"
                        for row in best_batch_rows
                    ]
                )
            ),
            "fraction_decoder_selects": float(
                np.mean([row["decoder_selects"] for row in best_batch_rows])
            ),
            "mean_margin": float(np.mean(margins)),
            "min_margin": float(np.min(margins)),
            "non_positive_states": non_positive,
        },
        "preferences": {
            "fraction_perfect_pairwise": float(
                np.mean([row["perfect_pairwise"] for row in preference_rows])
            ),
            "mean_max_pair_margin": float(
                np.mean([row["max_margin"] for row in preference_rows])
            ),
            "mean_lp_pairwise_agreement": float(
                np.mean([row["lp_pairwise_agreement"] for row in preference_rows])
            ),
            "mean_lsq_spearman": (
                float(np.mean(rho_values)) if rho_values else None
            ),
            "mean_lsq_pairwise_agreement": float(
                np.mean([row["lsq_pairwise_agreement"] for row in preference_rows])
            ),
            "top_match_fraction": float(
                np.mean([row["lsq_top_match"] for row in preference_rows])
            ),
            "lowest_spearman_states": lowest_spearman,
        },
        "context": {
            "mean_visible_orders": float(
                np.mean([state["visible_orders"] for state in states])
            ),
            "mean_candidates": float(
                np.mean([state["candidates"] for state in states])
            ),
            "mean_feasible_batches": float(
                np.mean([state["feasible_batches"] for state in states])
            ),
            "fraction_actor_is_best": float(
                np.mean([state["actor_is_best"] for state in states])
            ),
            "mean_actor_regret_fraction": float(
                np.mean([state["actor_regret_fraction"] for state in states])
            ),
        },
    }
    return {"states": states, "summary": summary}


def run_reward_identity(
    instance_ids: list[str],
    *,
    reward_power: float = 1.0,
    episode_kwargs: dict | None = None,
) -> dict[str, object]:
    """Verify the exact reward identity on fresh episodes.

    For each instance the episode is rolled out with the production knapsack
    oracle (equal scores, so the first capacity-feasible batch is taken at
    every decision).  The identity ``sum(rewards) == -objective_cost / scale``
    holds exactly for ``OrderCostReward``; with power one and zero threshold the
    objective is the total flow time, so the normalized sum of rewards equals
    minus the average order flow time divided by the per-order objective scale.
    """
    rows = []
    for instance_id in instance_ids:
        episode = StructuredBatchingEpisode(
            [instance_id],
            reward_power=reward_power,
            **(episode_kwargs or {}),
        )
        state = episode.reset(instance_id=instance_id)
        normalizer = episode.env.reward_normalizer
        episode_return = 0.0
        decisions = 0
        done = False
        while not done:
            selected = episode.oracle_action(
                np.full(len(state["order_ids"]), -1.0), state
            )
            state, reward, done, _, info = episode.step(selected)
            episode_return += reward
            decisions += 1
        flow_times = [
            completion - episode.env.arrivals[order_id]
            for order_id, completion in episode.env.completion_times().items()
        ]
        rows.append(
            {
                "instance_id": instance_id,
                "return": float(episode_return),
                "total_flow_time": float(info["total_flow_time"]),
                "objective_cost": float(info["objective_cost"]),
                "normalizer": float(normalizer),
                "mean_flow_time": float(np.mean(flow_times)),
                "decisions": decisions,
                "reward_identity_error": abs(
                    episode_return + info["objective_cost"] / normalizer
                ),
            }
        )
        episode.close()
    return {
        "reward_power": reward_power,
        "rows": rows,
        "summary": {
            "instances": len(rows),
            "max_reward_identity_error": max(
                row["reward_identity_error"] for row in rows
            ),
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "mean_mean_flow_time": float(
                np.mean([row["mean_flow_time"] for row in rows])
            ),
        },
    }


def _load_actor(checkpoint: str):
    actor, critic, checkpoint_dict = load_workbench_checkpoint(checkpoint)
    actor.checkpoint_path = checkpoint
    actor.eval()
    critic.eval()
    return actor, critic, checkpoint_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["reward-identity", "collect", "actor-update", "representability"],
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="outputs/henn_rl/final_diagnosis")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--instances", type=int, default=6)
    parser.add_argument("--state-quantiles", default="0.33,0.67")
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--collect-q", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument(
        "--normalize-temperature", type=float, default=None
    )
    parser.add_argument(
        "--exact-q",
        action="store_true",
        help="compare the current critic target with a soft target weighted by "
        "exact rollout Q (frozen continuation policy)",
    )
    parser.add_argument("--no-full-policy", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    quantiles = [float(value) for value in args.state_quantiles.split(",")]

    if args.mode == "reward-identity":
        split_ids = generated_instance_splits(DATA_SPEC)[args.split]
        result = run_reward_identity(
            split_ids[: args.instances],
            reward_power=1.0,
            episode_kwargs=_episode_kwargs(),
        )
        result["mode"] = "reward-identity"
        result["status"] = "complete"
        path = output_dir / "reward_identity.json"
        write_json(path, result)
        print(json.dumps(result["summary"], indent=2))
        return

    if args.mode == "collect":
        if not args.checkpoint:
            raise SystemExit("collect requires --checkpoint")
        actor, _, checkpoint = _load_actor(args.checkpoint)
        split_ids = generated_instance_splits(DATA_SPEC)[args.split]
        instance_ids = split_ids[: args.instances]
        dataset = collect_dataset(
            actor,
            instance_ids,
            state_quantiles=quantiles,
            candidate_count=args.candidate_count,
            sigma=args.sigma,
            seed=args.seed,
            collect_q=args.collect_q,
            objective_scale=checkpoint.get("objective_scale"),
            episode_kwargs=_episode_kwargs(),
            reward_power=1.0,
        )
        path = output_dir / "dataset.json"
        write_json(path, dataset)
        print(f"collected {len(dataset['records'])} candidates -> {path}")
        return

    if args.mode == "actor-update":
        if not args.checkpoint:
            raise SystemExit("actor-update requires --checkpoint")
        actor, critic, checkpoint = _load_actor(args.checkpoint)
        split_ids = generated_instance_splits(DATA_SPEC)[args.split]
        instance_ids = split_ids[: args.instances]
        if args.exact_q:
            targets = [
                {
                    "name": "current",
                    "temperature": args.temperature,
                    "normalize_advantages": False,
                    "values_source": "critic",
                },
                {
                    "name": "exact_q",
                    "temperature": 0.1,
                    "normalize_advantages": True,
                    "values_source": "exact",
                },
            ]
        elif args.normalize_temperature is None:
            targets = None
        else:
            targets = [
                {
                    "name": "current",
                    "temperature": args.temperature,
                    "normalize_advantages": False,
                },
                {
                    "name": "normalized",
                    "temperature": args.normalize_temperature,
                    "normalize_advantages": True,
                },
            ]
        result = run_actor_update_audit(
            actor,
            critic,
            instance_ids,
            state_quantiles=quantiles,
            candidate_count=args.candidate_count,
            sigma=args.sigma,
            temperature=args.temperature,
            fy_samples=20,
            epsilon=0.01,
            learning_rate=0.001,
            seed=args.seed,
            reward_power=1.0,
            objective_scale=checkpoint.get("objective_scale"),
            episode_kwargs=_episode_kwargs(),
            measure_full_policy=not args.no_full_policy,
            targets=targets,
        )
        result["mode"] = "actor-update"
        result["status"] = "complete"
        result["checkpoint"] = args.checkpoint
        path = output_dir / "actor_update.json"
        write_json(path, result)
        print(json.dumps(result["summary"], indent=2))
        return

    if args.mode == "representability":
        dataset_path = (
            Path(args.dataset) if args.dataset else output_dir / "dataset.json"
        )
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        result = run_representability(dataset)
        result["mode"] = "representability"
        result["status"] = "complete"
        path = output_dir / "representability.json"
        write_json(path, result)
        print(json.dumps(result["summary"], indent=2))
        return


if __name__ == "__main__":
    main()
