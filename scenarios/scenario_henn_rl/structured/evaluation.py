from __future__ import annotations

import time

import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm

from scenarios.scenario_henn_rl.structured.environment import StructuredBatchingEpisode
from scenarios.scenario_henn_rl.structured.models import _features
from scenarios.scenario_henn_rl.structured.data import (
    GeneratedHennDataLoader,
    generated_instance_splits,
    generated_manifest,
)
from scenarios.scenario_henn_rl.structured.models import load_workbench_checkpoint
from scenarios.scenario_henn_rl.structured.results import write_json
from scenarios.scenario_henn_rl.structured.tracking import log_artifact, log_evaluation


def evaluate_structured_actor(
    actor,
    instance_ids: list[str],
    *,
    location_features: bool = True,
    reward_power: float = 1.0,
    sla_threshold_s: float = 0.0,
    objective_scale: float | None = None,
    episode_kwargs: dict | None = None,
    show_progress: bool = False,
    progress_desc: str = "Evaluating actor",
) -> dict[str, object]:
    episode = StructuredBatchingEpisode(
        instance_ids,
        reward_power=reward_power,
        sla_threshold_s=sla_threshold_s,
        objective_scale=objective_scale,
        **(episode_kwargs or {}),
    )
    rows = []
    for instance_id in tqdm(
        instance_ids,
        desc=progress_desc,
        unit="instance",
        leave=False,
        disable=not show_progress,
    ):
        state = episode.reset(instance_id=instance_id)
        normalizer = episode.env.reward_normalizer
        done = False
        episode_return = 0.0
        fills = []
        batch_orders = []
        inference_time = 0.0
        decisions = 0
        while not done:
            started = time.perf_counter()
            with torch.no_grad():
                scores = actor(
                    _features(state, location_features=location_features)
                )
            indices = episode.oracle_action(scores.cpu().numpy(), state)
            inference_time += time.perf_counter() - started
            decisions += 1
            if len(indices):
                fills.append(
                    float(state["demands"][indices].sum())
                    / int(state["capacity"])
                )
                batch_orders.append(int(len(indices)))
            state, reward, done, _, info = episode.step(indices)
            episode_return += reward
        flow_times = [
            completion - episode.env.arrivals[order_id]
            for order_id, completion in episode.env.completion_times().items()
        ]
        tardiness = list(info["tardiness_by_order"].values())
        rows.append(
            {
                **info,
                "return": episode_return,
                "reward_identity_error": abs(
                    episode_return + info["objective_cost"] / normalizer
                ),
                "mean_flow_time": float(np.mean(flow_times)),
                "median_flow_time": float(np.median(flow_times)),
                "p90_flow_time": float(np.quantile(flow_times, 0.90)),
                "p95_flow_time": float(np.quantile(flow_times, 0.95)),
                "max_flow_time": float(np.max(flow_times)),
                "mean_flow_power_1_5": float(
                    np.mean(np.asarray(flow_times) ** 1.5)
                ),
                "mean_flow_power_2": float(
                    np.mean(np.asarray(flow_times) ** 2.0)
                ),
                "generalized_mean_1": float(np.mean(flow_times)),
                "generalized_mean_1_5": float(
                    np.mean(np.asarray(flow_times) ** 1.5) ** (1.0 / 1.5)
                ),
                "generalized_mean_2": float(
                    np.mean(np.asarray(flow_times) ** 2.0) ** 0.5
                ),
                "objective_per_order": float(info["objective_cost"])
                / len(flow_times),
                "mean_batch_fill": float(np.mean(fills)),
                "mean_batch_orders": float(np.mean(batch_orders)),
                "orders_per_sim_hour": _orders_per_sim_hour(episode),
                "inference_time_s": inference_time,
                "mean_decision_latency_s": inference_time / max(1, decisions),
                "flow_times": flow_times,
                "mean_tardiness": float(np.mean(tardiness)) if tardiness else 0.0,
                "max_tardiness": float(np.max(tardiness)) if tardiness else 0.0,
                "violation_fraction": (
                    float(np.count_nonzero(tardiness)) / len(tardiness)
                    if tardiness
                    else 0.0
                ),
            }
        )
    episode.close()
    return _summarize(rows)


def reference_objective_scale(
    instance_ids: list[str],
    *,
    reward_power: float,
    episode_kwargs: dict | None = None,
    show_progress: bool = False,
    progress_desc: str = "Reference objective scale evaluation",
) -> float:
    """Return a reward scale from the reference policy's per-order objective.

    The scale is the mean per-order objective value of a fixed existing
    policy, so the total normalized episode reward is near minus one under
    the configured objective (flow time or tardiness) regardless of instance
    size. This keeps critic Q-values on a scale that the configured candidate
    temperature can resolve into informative soft targets.
    """
    reference = evaluate_existing_policy(
        instance_ids,
        batching="cw",
        waiting_policy="no_wait",
        selector="sav",
        time_limit_s=0.25,
        reward_power=reward_power,
        episode_kwargs=episode_kwargs,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    per_order = np.asarray(
        [row["objective_per_order"] for row in reference["episodes"]],
        dtype=float,
    )
    return max(1.0, float(np.mean(per_order)))


def evaluate_existing_policy(
    instance_ids: list[str],
    *,
    batching: str,
    waiting_policy: str = "fill_or_age",
    selector: str = "short",
    fill_threshold: float = 0.75,
    max_age_s: float = 300.0,
    time_limit_s: float = 1.0,
    reward_power: float = 1.0,
    sla_threshold_s: float = 0.0,
    objective_scale: float | None = None,
    episode_kwargs: dict | None = None,
    show_progress: bool = False,
    progress_desc: str = "Evaluating baseline",
) -> dict[str, object]:
    episode = StructuredBatchingEpisode(
        instance_ids,
        reward_power=reward_power,
        sla_threshold_s=sla_threshold_s,
        objective_scale=objective_scale,
        **(episode_kwargs or {}),
    )
    rows = []
    for instance_id in tqdm(
        instance_ids,
        desc=progress_desc,
        unit="instance",
        leave=False,
        disable=not show_progress,
    ):
        state = episode.reset(instance_id=instance_id)
        normalizer = episode.env.reward_normalizer
        done = False
        episode_return = 0.0
        fills = []
        batch_orders = []
        decision_time = 0.0
        decisions = 0
        while not done:
            started = time.perf_counter()
            decision = episode.existing_policy_decision(
                batching=batching,
                waiting_policy=waiting_policy,
                selector=selector,
                fill_threshold=fill_threshold,
                max_age_s=max_age_s,
                time_limit_s=time_limit_s,
            )
            decision_time += time.perf_counter() - started
            decisions += 1
            if decision.action == "dispatch":
                selected = decision.details["selected_order_ids"]
                groups = (
                    selected
                    if selected and isinstance(selected[0], list)
                    else [selected]
                )
                demand_by_order = dict(
                    zip(state["order_ids"], state["demands"])
                )
                for group in groups:
                    fills.append(
                        sum(demand_by_order[int(order_id)] for order_id in group)
                        / int(state["capacity"])
                    )
                    batch_orders.append(len(group))
            state, reward, done, _, info, _ = episode.step_existing_policy(
                decision
            )
            episode_return += reward
        flow_times = [
            completion - episode.env.arrivals[order_id]
            for order_id, completion in episode.env.completion_times().items()
        ]
        tardiness = list(info["tardiness_by_order"].values())
        rows.append(
            {
                **info,
                "return": episode_return,
                "reward_identity_error": abs(
                    episode_return + info["objective_cost"] / normalizer
                ),
                "mean_flow_time": float(np.mean(flow_times)),
                "median_flow_time": float(np.median(flow_times)),
                "p90_flow_time": float(np.quantile(flow_times, 0.90)),
                "p95_flow_time": float(np.quantile(flow_times, 0.95)),
                "max_flow_time": float(np.max(flow_times)),
                "mean_flow_power_1_5": float(
                    np.mean(np.asarray(flow_times) ** 1.5)
                ),
                "mean_flow_power_2": float(
                    np.mean(np.asarray(flow_times) ** 2.0)
                ),
                "generalized_mean_1": float(np.mean(flow_times)),
                "generalized_mean_1_5": float(
                    np.mean(np.asarray(flow_times) ** 1.5) ** (1.0 / 1.5)
                ),
                "generalized_mean_2": float(
                    np.mean(np.asarray(flow_times) ** 2.0) ** 0.5
                ),
                "objective_per_order": float(info["objective_cost"])
                / len(flow_times),
                "mean_batch_fill": float(np.mean(fills)),
                "mean_batch_orders": float(np.mean(batch_orders)),
                "orders_per_sim_hour": _orders_per_sim_hour(episode),
                "inference_time_s": decision_time,
                "mean_decision_latency_s": decision_time / max(1, decisions),
                "flow_times": flow_times,
                "mean_tardiness": float(np.mean(tardiness)) if tardiness else 0.0,
                "max_tardiness": float(np.max(tardiness)) if tardiness else 0.0,
                "violation_fraction": (
                    float(np.count_nonzero(tardiness)) / len(tardiness)
                    if tardiness
                    else 0.0
                ),
            }
        )
    episode.close()
    return _summarize(rows)


def _orders_per_sim_hour(episode: StructuredBatchingEpisode) -> float:
    completions = episode.env.completion_times()
    if not completions:
        return 0.0
    elapsed = max(completions.values()) - min(episode.env.arrivals.values())
    return len(completions) * 3600.0 / max(1.0, elapsed)


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    flow_times = np.asarray(
        [value for row in rows for value in row["flow_times"]], dtype=float
    )
    return {
        "instances": len(rows),
        "mean_order_flow_time": float(
            np.mean([row["mean_flow_time"] for row in rows])
        ),
        "median_order_flow_time": float(
            np.median([row["mean_flow_time"] for row in rows])
        ),
        "mean_p95_order_flow_time": float(
            np.mean([row["p95_flow_time"] for row in rows])
        ),
        "order_flow_time": {
            "mean": float(np.mean(flow_times)),
            "median": float(np.median(flow_times)),
            "p90": float(np.quantile(flow_times, 0.90)),
            "p95": float(np.quantile(flow_times, 0.95)),
            "maximum": float(np.max(flow_times)),
        },
        "cross_objectives": {
            "mean_flow_power_1": float(np.mean(flow_times)),
            "mean_flow_power_1_5": float(np.mean(flow_times**1.5)),
            "mean_flow_power_2": float(np.mean(flow_times**2.0)),
            "generalized_mean_1": float(np.mean(flow_times)),
            "generalized_mean_1_5": float(
                np.mean(flow_times**1.5) ** (1.0 / 1.5)
            ),
            "generalized_mean_2": float(np.mean(flow_times**2.0) ** 0.5),
        },
        "mean_objective_per_order": float(
            np.mean([row["objective_per_order"] for row in rows])
        ),
        "mean_distance": float(np.mean([row["total_distance"] for row in rows])),
        "mean_tours": float(np.mean([row["completed_tours"] for row in rows])),
        "mean_batch_fill": float(
            np.mean([row["mean_batch_fill"] for row in rows])
        ),
        "mean_batch_orders": float(
            np.mean([row["mean_batch_orders"] for row in rows])
        ),
        "mean_orders_per_sim_hour": float(
            np.mean([row["orders_per_sim_hour"] for row in rows])
        ),
        "mean_tardiness": float(np.mean([row["mean_tardiness"] for row in rows])),
        "max_tardiness": float(np.max([row["max_tardiness"] for row in rows])),
        "mean_violation_fraction": float(
            np.mean([row["violation_fraction"] for row in rows])
        ),
        "mean_wait_time_s": float(
            np.mean([row.get("wait_time_s", 0.0) for row in rows])
        ),
        "wait_fraction": sum(row["wait_actions"] for row in rows)
        / max(
            1,
            sum(
                row["wait_actions"] + row["dispatch_actions"] for row in rows
            ),
        ),
        "max_reward_identity_error": max(
            row["reward_identity_error"] for row in rows
        ),
        "mean_inference_time_s": float(
            np.mean([row.get("inference_time_s", 0.0) for row in rows])
        ),
        "mean_decision_latency_s": float(
            np.mean([row.get("mean_decision_latency_s", 0.0) for row in rows])
        ),
        "episodes": rows,
    }


def run_evaluation(cfg: dict, output_dir: Path, tracking_run=None) -> dict[str, object]:
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("Evaluation requires checkpoint=<path-to-best.pt>")
    actor, _, checkpoint = load_workbench_checkpoint(checkpoint_path)
    objective_scale = checkpoint.get("objective_scale")
    data_spec = dict(cfg["data"])
    splits = generated_instance_splits(data_spec)
    split = str(cfg["experiment"]["split"])
    if split not in splits:
        raise ValueError(f"Unknown evaluation split: {split}")
    write_json(output_dir / "dataset_manifest.json", generated_manifest(data_spec))
    loader = GeneratedHennDataLoader(data_spec)
    objective = cfg["objective"]
    episode_kwargs = {
        "data_loader": loader,
        "use_order_due_dates": bool(objective["use_order_due_dates"]),
        "include_due_slack": True,
        "decoder": str(cfg["model"].get("decoder", "knapsack")),
    }
    learned = evaluate_structured_actor(
        actor,
        splits[split],
        reward_power=float(objective["power"]),
        objective_scale=objective_scale,
        episode_kwargs=episode_kwargs,
        show_progress=bool(cfg.get("progress", True)),
        progress_desc=f"Actor evaluation ({split})",
    )
    baselines = {}
    for baseline in cfg["experiment"].get("baselines", []):
        name = f"{baseline['batching']}_{baseline['selector']}"
        baselines[name] = evaluate_existing_policy(
            splits[split],
            batching=str(baseline["batching"]),
            waiting_policy="no_wait",
            selector=str(baseline["selector"]),
            time_limit_s=float(baseline.get("time_limit_s", 0.25)),
            reward_power=float(objective["power"]),
            objective_scale=objective_scale,
            episode_kwargs=episode_kwargs,
            show_progress=bool(cfg.get("progress", True)),
            progress_desc=f"Baseline {name} ({split})",
        )
    result = {
        "mode": "evaluate",
        "status": "complete",
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_selected_episode": checkpoint["selected_episode"],
        "learned": learned,
        "baselines": baselines,
    }
    result_path = output_dir / "result.json"
    write_json(result_path, result)
    log_evaluation(tracking_run, "evaluation/learned", learned)
    for name, evaluation in baselines.items():
        log_evaluation(tracking_run, f"evaluation/{name}", evaluation)
    log_artifact(
        tracking_run,
        output_dir,
        [
            result_path,
            output_dir / "dataset_manifest.json",
            output_dir / "resolved_config.json",
        ],
    )
    return result
