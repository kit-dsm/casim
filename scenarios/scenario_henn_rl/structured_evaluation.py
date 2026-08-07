from __future__ import annotations

import time

import numpy as np
import torch

from scenarios.scenario_henn_rl.structured_environment import StructuredBatchingEpisode
from scenarios.scenario_henn_rl.structured_policy import _features


def evaluate_structured_actor(
    actor,
    instance_ids: list[str],
    *,
    location_features: bool = True,
    reward_power: float = 1.0,
) -> dict[str, object]:
    episode = StructuredBatchingEpisode(
        instance_ids, reward_power=reward_power
    )
    rows = []
    for instance_id in instance_ids:
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
            }
        )
    episode.close()
    return _summarize(rows)


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
) -> dict[str, object]:
    episode = StructuredBatchingEpisode(
        instance_ids, reward_power=reward_power
    )
    rows = []
    for instance_id in instance_ids:
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
