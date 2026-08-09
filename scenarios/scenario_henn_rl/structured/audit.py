from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr

from scenarios.scenario_henn_rl.rewards import OrderCostReward
from scenarios.scenario_henn_rl.structured.environment import StructuredBatchingEpisode
from scenarios.scenario_henn_rl.structured.models import (
    PairwiseStructuredCritic,
    StructuredCritic,
    _copy_state,
    _features,
    _mask,
    candidate_weights,
    decode_scores,
    fenchel_young_loss,
    structured_candidates,
    load_workbench_checkpoint,
)
from scenarios.scenario_henn_rl.structured.data import (
    GeneratedHennDataLoader,
    generated_instance_splits,
    generated_manifest,
)
from scenarios.scenario_henn_rl.structured.results import write_json
from scenarios.scenario_henn_rl.structured.tracking import log_artifact


def _indices_for_orders(state: dict[str, object], order_ids) -> np.ndarray:
    index_by_order = {
        int(order_id): index
        for index, order_id in enumerate(state["order_ids"])
    }
    missing = set(map(int, order_ids)) - set(index_by_order)
    if missing:
        raise RuntimeError(f"Counterfactual replay is missing orders {missing}")
    return np.asarray(
        [index_by_order[int(order_id)] for order_id in order_ids], dtype=int
    )


def _reference_trajectory(
    actor, instance_id: str, episode_kwargs: dict | None = None
) -> list[dict[str, object]]:
    episode = StructuredBatchingEpisode([instance_id], **(episode_kwargs or {}))
    state = episode.reset(instance_id=instance_id)
    prefix: list[list[int]] = []
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
                "prefix": [list(order_ids) for order_ids in prefix],
                "actor_order_ids": selected,
            }
        )
        prefix.append(selected)
        state, _, done, _, _ = episode.step(indices)
    episode.close()
    return trajectory


def _candidate_rollout(
    actor,
    instance_id: str,
    prefix: list[list[int]],
    expected_state: dict[str, object],
    candidate_order_ids: list[int],
    powers: list[float],
    sla_threshold_s: float = 0.0,
    objective_scale: float | None = None,
    episode_kwargs: dict | None = None,
) -> dict[str, object]:
    episode = StructuredBatchingEpisode([instance_id], **(episode_kwargs or {}))
    state = episode.reset(instance_id=instance_id)
    done = False
    for order_ids in prefix:
        state, _, done, _, _ = episode.step(
            _indices_for_orders(state, order_ids)
        )
        if done:
            raise RuntimeError("Counterfactual prefix terminated too early")
    if not np.array_equal(state["order_ids"], expected_state["order_ids"]):
        raise RuntimeError("Counterfactual replay reached a different buffer")
    if not np.allclose(state["features"], expected_state["features"]):
        raise RuntimeError("Counterfactual replay reached different features")

    now = float(episode.env.simulation.state.current_time)
    completions_before = episode.env.completion_times()
    use_due_dates = bool((episode_kwargs or {}).get("use_order_due_dates"))
    models = {
        power: OrderCostReward(
            episode.env.arrivals,
            power=power,
            thresholds_s=None if use_due_dates else sla_threshold_s,
            due_times=episode.env.due_times if use_due_dates else None,
            normalizer=(
                max(1.0, len(episode.env.arrivals)) * objective_scale
                if objective_scale is not None
                else max(
                    1.0,
                    len(episode.env.arrivals)
                    * episode.env.episode_horizon**power,
                )
            ),
        )
        for power in powers
    }
    accrued = {
        power: model.accrued_cost(now, completions_before)
        for power, model in models.items()
    }
    state, _, done, _, _ = episode.step(
        _indices_for_orders(state, candidate_order_ids)
    )
    while not done:
        with torch.no_grad():
            action = decode_scores(actor(_features(state)), state)
        indices = torch.nonzero(action, as_tuple=False).flatten().cpu().numpy()
        state, _, done, _, _ = episode.step(indices)
    completions = episode.env.completion_times()
    true_q = {
        str(power): -(
            model.objective(completions) - accrued[power]
        ) / model.normalizer
        for power, model in models.items()
    }
    episode.close()
    return {"true_q": true_q}


def audit_structured_critic(
    actor,
    critic,
    instance_ids: list[str],
    *,
    reward_power: float,
    comparison_powers: list[float],
    state_quantiles: list[float],
    candidate_count: int,
    sigma: float,
    temperature: float,
    seed: int,
    normalize_advantages: bool = False,
    sla_threshold_s: float = 0.0,
    objective_scale: float | None = None,
    episode_kwargs: dict | None = None,
) -> dict[str, object]:
    """Compare critic rankings with complete counterfactual continuations."""
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for instance_id in instance_ids:
        trajectory = _reference_trajectory(actor, instance_id, episode_kwargs)
        indices = sorted(
            {
                min(
                    len(trajectory) - 1,
                    max(0, int(round(q * (len(trajectory) - 1)))),
                )
                for q in state_quantiles
            }
        )
        for decision_index in indices:
            reference = trajectory[decision_index]
            state = reference["state"]
            candidates = structured_candidates(
                actor,
                state,
                candidate_count=candidate_count,
                sigma=sigma,
                generator=generator,
                deduplicate=True,
            )
            features = _features(state)
            with torch.no_grad():
                predicted = np.asarray(
                    [float(critic(features, action)) for action in candidates]
                )
            candidate_rows = []
            for action, predicted_q in zip(candidates, predicted):
                selected_indices = torch.nonzero(
                    action, as_tuple=False
                ).flatten().cpu().numpy()
                order_ids = [
                    int(state["order_ids"][index]) for index in selected_indices
                ]
                rollout = _candidate_rollout(
                    actor,
                    instance_id,
                    reference["prefix"],
                    state,
                    order_ids,
                    comparison_powers,
                    sla_threshold_s=sla_threshold_s,
                    objective_scale=objective_scale,
                    episode_kwargs=episode_kwargs,
                )
                candidate_rows.append(
                    {
                        "order_ids": order_ids,
                        "predicted_q": float(predicted_q),
                        **rollout,
                    }
                )
            true_values = np.asarray(
                [row["true_q"][str(reward_power)] for row in candidate_rows]
            )
            correlation = spearmanr(predicted, true_values).statistic
            critic_index = int(np.argmax(predicted))
            true_index = int(np.argmax(true_values))
            true_spread = float(np.ptp(true_values))
            weights = candidate_weights(
                torch.as_tensor(predicted),
                temperature=temperature,
                normalize_advantages=normalize_advantages,
            ).numpy()
            entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))))
            soft_target_q = float(weights @ true_values)
            actor_q = float(true_values[0])
            best_by_power = {
                str(power): int(
                    np.argmax([row["true_q"][str(power)] for row in candidate_rows])
                )
                for power in comparison_powers
            }
            rows.append(
                {
                    "instance_id": instance_id,
                    "decision_index": decision_index,
                    "visible_orders": len(state["order_ids"]),
                    "unique_candidates": len(candidate_rows),
                    "spearman": None if np.isnan(correlation) else float(correlation),
                    "predicted_q_spread": float(np.ptp(predicted)),
                    "true_q_spread": true_spread,
                    "critic_top_index": critic_index,
                    "actor_index": 0,
                    "true_top_index": true_index,
                    "critic_top_regret": float(
                        true_values[true_index] - true_values[critic_index]
                    ),
                    "actor_regret": float(
                        true_values[true_index] - true_values[0]
                    ),
                    "soft_target_true_q": soft_target_q,
                    "soft_target_advantage_over_actor": soft_target_q - actor_q,
                    "soft_target_regret": float(
                        true_values[true_index] - soft_target_q
                    ),
                    "critic_top_regret_fraction": float(
                        (true_values[true_index] - true_values[critic_index])
                        / max(true_spread, 1e-12)
                    ),
                    "actor_regret_fraction": float(
                        (true_values[true_index] - true_values[0])
                        / max(true_spread, 1e-12)
                    ),
                    "softmax_entropy": entropy,
                    "normalized_advantages": normalize_advantages,
                    "best_index_by_power": best_by_power,
                    "candidate_rows": candidate_rows,
                }
            )
    correlations = [row["spearman"] for row in rows if row["spearman"] is not None]
    actor_regret = sum(row["actor_regret"] for row in rows)
    soft_advantage = sum(
        row["soft_target_advantage_over_actor"] for row in rows
    )
    return {
        "reward_power": reward_power,
        "states": rows,
        "summary": {
            "state_count": len(rows),
            "mean_unique_candidates": float(
                np.mean([row["unique_candidates"] for row in rows])
            ),
            "mean_spearman": float(np.mean(correlations)) if correlations else None,
            "positive_spearman_fraction": float(
                np.mean([value > 0.0 for value in correlations])
            ) if correlations else None,
            "critic_top_accuracy": float(
                np.mean(
                    [row["critic_top_index"] == row["true_top_index"] for row in rows]
                )
            ),
            "actor_top_accuracy": float(
                np.mean([row["true_top_index"] == 0 for row in rows])
            ),
            "mean_critic_regret_fraction": float(
                np.mean([row["critic_top_regret_fraction"] for row in rows])
            ),
            "mean_actor_regret_fraction": float(
                np.mean([row["actor_regret_fraction"] for row in rows])
            ),
            "mean_soft_target_advantage_over_actor": float(
                np.mean(
                    [row["soft_target_advantage_over_actor"] for row in rows]
                )
            ),
            "mean_soft_target_regret": float(
                np.mean([row["soft_target_regret"] for row in rows])
            ),
            "soft_target_capture_fraction": float(
                soft_advantage / max(actor_regret, 1e-12)
            ),
            "mean_predicted_q_spread": float(
                np.mean([row["predicted_q_spread"] for row in rows])
            ),
            "mean_true_q_spread": float(
                np.mean([row["true_q_spread"] for row in rows])
            ),
            "mean_softmax_entropy": float(
                np.mean([row["softmax_entropy"] for row in rows])
            ),
            "objective_preference_difference_fraction": float(
                np.mean(
                    [
                        len(set(row["best_index_by_power"].values())) > 1
                        for row in rows
                    ]
                )
            ),
        },
    }

def _counterfactual_examples(
    actor, audit: dict[str, object], *, standardize_targets: bool = True
):
    power = str(audit["reward_power"])
    trajectories = {}
    examples = []
    for row in audit["states"]:
        instance_id = row["instance_id"]
        if instance_id not in trajectories:
            trajectories[instance_id] = _reference_trajectory(actor, instance_id)
        state = trajectories[instance_id][int(row["decision_index"])]["state"]
        actions = []
        targets = []
        for candidate in row["candidate_rows"]:
            indices = _indices_for_orders(state, candidate["order_ids"])
            actions.append(_mask(indices, len(state["order_ids"])))
            targets.append(float(candidate["true_q"][power]))
        target = torch.as_tensor(targets, dtype=torch.float32)
        if standardize_targets:
            target = (target - target.mean()) / target.std(
                unbiased=False
            ).clamp_min(1e-6)
        examples.append((_copy_state(state), actions, target))
    return examples


def score_counterfactual_audit(
    actor,
    critic,
    audit: dict[str, object],
    *,
    temperature: float,
    normalize_advantages: bool,
) -> dict[str, object]:
    """Score a frozen critic on previously generated exact branches."""
    examples = _counterfactual_examples(
        actor, audit, standardize_targets=False
    )
    rows = []
    for (state, actions, true_values), source_row in zip(
        examples, audit["states"]
    ):
        features = _features(state)
        with torch.no_grad():
            predicted = torch.stack(
                [critic(features, action) for action in actions]
            )
            weights = candidate_weights(
                predicted,
                temperature=temperature,
                normalize_advantages=normalize_advantages,
            )
        predicted_values = predicted.numpy()
        true = true_values.numpy()
        correlation = spearmanr(predicted_values, true).statistic
        critic_index = int(np.argmax(predicted_values))
        true_index = int(np.argmax(true))
        spread = float(np.ptp(true))
        actor_regret = float(true[true_index] - true[0])
        soft_q = float(weights.numpy() @ true)
        rows.append(
            {
                "instance_id": source_row["instance_id"],
                "decision_index": source_row["decision_index"],
                "spearman": None if np.isnan(correlation) else float(correlation),
                "critic_top_accuracy": int(critic_index == true_index),
                "critic_regret_fraction": float(
                    (true[true_index] - true[critic_index])
                    / max(spread, 1e-12)
                ),
                "actor_regret_fraction": float(
                    actor_regret / max(spread, 1e-12)
                ),
                "actor_regret": actor_regret,
                "soft_target_advantage_over_actor": soft_q - float(true[0]),
            }
        )
    correlations = [row["spearman"] for row in rows if row["spearman"] is not None]
    actor_regret = sum(row["actor_regret"] for row in rows)
    soft_advantage = sum(
        row["soft_target_advantage_over_actor"] for row in rows
    )
    return {
        "reward_power": audit["reward_power"],
        "states": rows,
        "summary": {
            "state_count": len(rows),
            "mean_spearman": float(np.mean(correlations)) if correlations else None,
            "positive_spearman_fraction": float(
                np.mean([value > 0.0 for value in correlations])
            ) if correlations else None,
            "critic_top_accuracy": float(
                np.mean([row["critic_top_accuracy"] for row in rows])
            ),
            "mean_critic_regret_fraction": float(
                np.mean([row["critic_regret_fraction"] for row in rows])
            ),
            "mean_actor_regret_fraction": float(
                np.mean([row["actor_regret_fraction"] for row in rows])
            ),
            "soft_target_capture_fraction": float(
                soft_advantage / max(actor_regret, 1e-12)
            ),
        },
    }


def counterfactual_examples(
    actor,
    instance_ids: list[str],
    *,
    reward_power: float,
    state_quantiles: list[float],
    candidate_count: int,
    sigma: float,
    seed: int,
    sla_threshold_s: float = 0.0,
    objective_scale: float | None = None,
    episode_kwargs: dict | None = None,
) -> list[tuple[dict[str, object], list[torch.Tensor], torch.Tensor]]:
    """Exact per-candidate Q labels for a sample of training states."""
    generator = torch.Generator().manual_seed(seed)
    examples = []
    for instance_id in instance_ids:
        trajectory = _reference_trajectory(actor, instance_id, episode_kwargs)
        indices = sorted(
            {
                min(
                    len(trajectory) - 1,
                    max(0, int(round(q * (len(trajectory) - 1)))),
                )
                for q in state_quantiles
            }
        )
        for decision_index in indices:
            reference = trajectory[decision_index]
            state = reference["state"]
            candidates = structured_candidates(
                actor,
                state,
                candidate_count=candidate_count,
                sigma=sigma,
                generator=generator,
                deduplicate=True,
            )
            actions = []
            true_values = []
            for action in candidates:
                selected_indices = torch.nonzero(
                    action, as_tuple=False
                ).flatten().cpu().numpy()
                order_ids = [
                    int(state["order_ids"][index])
                    for index in selected_indices
                ]
                rollout = _candidate_rollout(
                    actor,
                    instance_id,
                    reference["prefix"],
                    state,
                    order_ids,
                    [reward_power],
                    sla_threshold_s=sla_threshold_s,
                    objective_scale=objective_scale,
                    episode_kwargs=episode_kwargs,
                )
                actions.append(action)
                true_values.append(float(rollout["true_q"][str(reward_power)]))
            target = torch.as_tensor(true_values, dtype=torch.float32)
            target = (target - target.mean()) / target.std(
                unbiased=False
            ).clamp_min(1e-6)
            examples.append((_copy_state(state), actions, target))
    return examples


def run_audit(cfg: dict, output_dir: Path, tracking_run=None) -> dict[str, object]:
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("Audit requires checkpoint=<path-to-best.pt>")
    actor, critic, checkpoint = load_workbench_checkpoint(checkpoint_path)
    objective_scale = checkpoint.get("objective_scale")
    data_spec = dict(cfg["data"])
    splits = generated_instance_splits(data_spec)
    settings = cfg["experiment"]
    split = str(settings["split"])
    count = int(settings["instances"])
    instance_ids = splits[split][:count]
    loader = GeneratedHennDataLoader(data_spec)
    objective = cfg["objective"]
    episode_kwargs = {
        "data_loader": loader,
        "use_order_due_dates": bool(objective["use_order_due_dates"]),
        "include_due_slack": True,
    }
    audit = audit_structured_critic(
        actor,
        critic,
        instance_ids,
        reward_power=float(objective["power"]),
        comparison_powers=[float(objective["power"])],
        state_quantiles=[float(value) for value in settings["state_quantiles"]],
        candidate_count=int(settings["candidate_count"]),
        sigma=float(settings["sigma"]),
        temperature=float(settings["temperature"]),
        seed=int(cfg["seed"]),
        objective_scale=objective_scale,
        episode_kwargs=episode_kwargs,
    )
    result = {
        "mode": "audit",
        "status": "complete",
        "split": split,
        "checkpoint": str(checkpoint_path),
        "audit": audit,
    }
    result_path = output_dir / "result.json"
    manifest_path = output_dir / "dataset_manifest.json"
    write_json(result_path, result)
    write_json(manifest_path, generated_manifest(data_spec))
    if tracking_run is not None:
        tracking_run.log(
            {
                f"audit/{key}": value
                for key, value in audit["summary"].items()
                if isinstance(value, (int, float))
            }
        )
    log_artifact(
        tracking_run,
        output_dir,
        [result_path, manifest_path, output_dir / "resolved_config.json"],
    )
    return result
