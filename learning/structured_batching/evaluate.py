"""Evaluate learned and established batching policies in the same CASIM loop."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from ware_ops_algos.algorithms import ClarkAndWrightBatching, SShapeRouting
from ware_ops_algos.algorithms.batching.batching import FifoBatching

from casim.io_helpers import dump_json, dump_jsonl
from learning.structured_batching.checkpoint import (
    assert_checkpoint_compatible,
    load_checkpoint,
)
from learning.structured_batching.policy import (
    decode_indices,
    features_tensor,
    make_observation,
)
from learning.structured_batching.tracking import log_artifact, log_evaluation


def _flatten_episodes(learned, learned_name, baselines):
    rows = []
    for row in learned["episodes"]:
        rows.append({**row, "policy": learned_name})
    for name, evaluation in baselines.items():
        for row in evaluation["episodes"]:
            rows.append({**row, "policy": name})
    return rows


def _observation(environment, decoder_name):
    snapshot, orders = environment.current
    return make_observation(
        snapshot,
        orders,
        total_orders=environment.order_count,
        episode_horizon=environment.episode_horizon,
        route_aware=decoder_name == "route_aware_greedy",
    )


def _router(environment, decoder_name):
    if decoder_name != "route_aware_greedy":
        return None
    snapshot, _ = environment.current
    return environment.decision_engine.solver_map["ORSP"].router_for(snapshot)


def _routing_kwargs(snapshot):
    network = snapshot.layout.layout_network
    nodes = list(network.graph.nodes)
    return {
        "start_node": network.start_node,
        "end_node": network.end_node,
        "closest_node_to_start": network.closest_node_to_start,
        "min_aisle_position": network.min_aisle_position,
        "max_aisle_position": network.max_aisle_position,
        "distance_matrix": network.distance_matrix,
        "predecessor_matrix": network.predecessor_matrix,
        "picker": snapshot.resources.resources,
        "gen_tour": True,
        "gen_item_sequence": True,
        "node_list": network.node_list,
        "node_to_idx": {node: index for index, node in enumerate(nodes)},
        "idx_to_node": {index: node for index, node in enumerate(nodes)},
    }


def _episode_metrics(
    environment,
    *,
    objective_scale,
    episode_return,
    fills,
    batch_sizes,
    decision_time,
    decisions,
):
    state = environment.simulation.state
    tracker = state.tracker
    flow_times = [
        tracker.order_completion_times[order_id] - arrival
        for order_id, arrival in tracker.order_arrival_times.items()
    ]
    total_flow_time = tracker.accrued_flow_time(state.current_time)
    elapsed = max(tracker.order_completion_times.values()) - min(
        tracker.order_arrival_times.values()
    )
    return {
        "total_flow_time": float(total_flow_time),
        "objective_per_order": float(total_flow_time) / max(1, environment.order_count),
        "return": float(episode_return),
        "reward_identity_error": abs(
            float(episode_return)
            + float(total_flow_time)
            / (max(1, environment.order_count) * objective_scale)
        ),
        "mean_flow_time": float(np.mean(flow_times)),
        "p95_flow_time": float(np.quantile(flow_times, 0.95)),
        "flow_times": flow_times,
        "total_distance": float(sum(tracker.distance_by_picker.values())),
        "completed_tours": len(tracker.completed_tours),
        "decisions": int(decisions),
        "mean_batch_fill": float(np.mean(fills)) if fills else 0.0,
        "mean_batch_orders": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        "orders_per_sim_hour": len(flow_times) * 3600.0 / max(1.0, elapsed),
        "decision_time_s": float(decision_time),
        "mean_decision_latency_s": float(decision_time) / max(1, decisions),
    }


def _summarize(rows):
    flow_times = np.asarray(
        [value for row in rows for value in row["flow_times"]], dtype=float
    )
    return {
        "instances": len(rows),
        "mean_objective_per_order": float(
            np.mean([row["objective_per_order"] for row in rows])
        ),
        "mean_order_flow_time": float(np.mean(flow_times)),
        "p95_order_flow_time": float(np.quantile(flow_times, 0.95)),
        "mean_distance": float(np.mean([row["total_distance"] for row in rows])),
        "mean_tours": float(np.mean([row["completed_tours"] for row in rows])),
        "mean_decisions": float(np.mean([row["decisions"] for row in rows])),
        "mean_batch_fill": float(np.mean([row["mean_batch_fill"] for row in rows])),
        "mean_batch_orders": float(np.mean([row["mean_batch_orders"] for row in rows])),
        "mean_orders_per_sim_hour": float(
            np.mean([row["orders_per_sim_hour"] for row in rows])
        ),
        "max_reward_identity_error": max(row["reward_identity_error"] for row in rows),
        "mean_decision_time_s": float(np.mean([row["decision_time_s"] for row in rows])),
        "episodes": rows,
    }


def evaluate_policy(
    instance_ids,
    *,
    choose_batch,
    environment_factory,
    objective_scale,
    decoder_name="knapsack",
    location_features=True,
    show_progress=False,
    progress_desc="Evaluating",
):
    """Run one batch-selection function through the controlled CASIM loop."""
    environment = environment_factory()
    environment.objective_scale = objective_scale
    rows = []
    for instance_id in tqdm(
        instance_ids,
        desc=progress_desc,
        unit="instance",
        leave=False,
        disable=not show_progress,
    ):
        environment.reset(instance_id)
        router = _router(environment, decoder_name)
        observation = _observation(environment, decoder_name)
        done = False
        episode_return = 0.0
        fills, batch_sizes = [], []
        decision_time = 0.0
        decisions = 0
        while not done:
            started = time.perf_counter()
            order_ids = choose_batch(observation, environment, router)
            decision_time += time.perf_counter() - started
            decisions += 1
            demand_map = dict(
                zip(observation.order_ids.tolist(), observation.demands.tolist())
            )
            fills.append(
                sum(demand_map[int(oid)] for oid in order_ids) / observation.capacity
            )
            batch_sizes.append(len(order_ids))
            _, reward, done = environment.step(order_ids)
            episode_return += reward
            if not done:
                observation = _observation(environment, decoder_name)
        rows.append(
            _episode_metrics(
                environment,
                objective_scale=objective_scale,
                episode_return=episode_return,
                fills=fills,
                batch_sizes=batch_sizes,
                decision_time=decision_time,
                decisions=decisions,
            )
        )
    return _summarize(rows)


def make_actor_choose_batch(actor, decoder_name, *, location_features=True):
    """Return a choose_batch function for the learned actor."""

    def choose_batch(observation, environment, router):
        with torch.no_grad():
            scores = actor(
                features_tensor(observation, location_features=location_features)
            )
        indices = decode_indices(scores, observation, decoder_name, router)
        return observation.order_ids[indices].tolist()

    return choose_batch


def fifo_choose_batch(observation, environment, router):
    snapshot, orders = environment.current
    picker = snapshot.resources.resources[0]
    batcher = FifoBatching(pick_cart=picker.pick_cart, articles=snapshot.articles)
    candidate = batcher.solve(list(orders))
    batch = min(candidate.batches, key=lambda value: int(value.batch_id))
    return list(batch.order_numbers)


def cw_sav_choose_batch(observation, environment, router):
    snapshot, orders = environment.current
    picker = snapshot.resources.resources[0]
    batcher = ClarkAndWrightBatching(
        pick_cart=picker.pick_cart,
        articles=snapshot.articles,
        routing_class=SShapeRouting,
        routing_class_kwargs=_routing_kwargs(snapshot),
    )
    candidate = batcher.solve(list(orders))
    sav_router = environment.decision_engine.solver_map["ORSP"].router_for(snapshot)

    def service(value):
        distance = float(sav_router.score(value.pick_positions))
        return (
            float(picker.tour_setup_time or 0.0)
            + distance / float(picker.speed)
            + len(value.pick_positions) * float(picker.time_per_pick)
        )

    def saving(value):
        singles = sum(
            service(type(value)(order.order_id, [order]))
            for order in value.orders
        )
        return singles - service(value)

    batch = max(candidate.batches, key=lambda value: (saving(value), -int(value.batch_id)))
    return list(batch.order_numbers)


def reference_objective_scale(
    environment_factory,
    instance_ids,
    *,
    show_progress=False,
):
    """Return the mean per-order flow time of the C&W/SAV reference policy.

    The scale makes the total normalized episode return near minus one under
    the reference policy, regardless of instance size.
    """
    reference = evaluate_policy(
        instance_ids,
        choose_batch=cw_sav_choose_batch,
        environment_factory=environment_factory,
        objective_scale=1.0,
        decoder_name="knapsack",
        show_progress=show_progress,
        progress_desc="Reference objective scale evaluation",
    )
    return max(1.0, float(reference["mean_objective_per_order"]))


def run_evaluation(
    cfg,
    output_dir,
    *,
    splits,
    dataset_manifest,
    environment_factory,
    tracking_run=None,
):
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("Evaluation requires checkpoint=<path-to-best.pt>")
    actor, _, checkpoint = load_checkpoint(checkpoint_path)
    split = str(cfg["experiment"]["split"])
    if split not in splits:
        raise ValueError(f"Unknown evaluation split: {split}")
    decoder_name = str(cfg["decoder"]["name"])
    objective_scale = float(checkpoint.get("objective_scale", 1.0))
    probe = environment_factory()
    probe.reset(splits[split][0])
    probe_obs = _observation(probe, decoder_name)
    route_scale = probe_obs.route_cost_scale
    feature_count = int(probe_obs.features.shape[1])
    assert_checkpoint_compatible(
        checkpoint,
        decoder_name=decoder_name,
        route_cost_scale=route_scale,
        feature_count=feature_count,
        objective_scale=objective_scale,
    )
    progress = bool(cfg.get("progress", True))
    learned = evaluate_policy(
        splits[split],
        choose_batch=make_actor_choose_batch(
            actor,
            decoder_name,
            location_features=True,
        ),
        environment_factory=environment_factory,
        objective_scale=objective_scale,
        decoder_name=decoder_name,
        show_progress=progress,
        progress_desc=f"Actor evaluation ({split})",
    )
    log_evaluation(tracking_run, "learned", learned)
    baselines = {}
    for baseline in cfg["experiment"].get("baselines", []):
        name = f"{baseline['batching']}_{baseline['selector']}"
        choose = (
            cw_sav_choose_batch
            if baseline["selector"] == "sav"
            else fifo_choose_batch
        )
        baselines[name] = evaluate_policy(
            splits[split],
            choose_batch=choose,
            environment_factory=environment_factory,
            objective_scale=objective_scale,
            decoder_name="knapsack",
            show_progress=progress,
            progress_desc=f"Baseline {name} ({split})",
        )
        log_evaluation(tracking_run, name, baselines[name])
    result = {
        "mode": "evaluate",
        "status": "complete",
        "decoder": decoder_name,
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_selected_episode": checkpoint["selected_episode"],
        "objective_scale": objective_scale,
        "learned": {
            "mean_objective_per_order": learned["mean_objective_per_order"],
            "mean_order_flow_time": learned["mean_order_flow_time"],
            "p95_order_flow_time": learned["p95_order_flow_time"],
            "mean_batch_fill": learned["mean_batch_fill"],
            "mean_batch_orders": learned["mean_batch_orders"],
        },
        "baselines": {
            name: {
                "mean_objective_per_order": val["mean_objective_per_order"],
                "mean_order_flow_time": val["mean_order_flow_time"],
                "p95_order_flow_time": val["p95_order_flow_time"],
                "mean_batch_fill": val["mean_batch_fill"],
                "mean_batch_orders": val["mean_batch_orders"],
            }
            for name, val in baselines.items()
        },
    }
    dump_json(Path(output_dir) / "dataset_manifest.json", dataset_manifest)
    dump_json(Path(output_dir) / "result.json", result)
    dump_jsonl(Path(output_dir) / "evaluation_episodes.jsonl", _flatten_episodes(learned, "learned", baselines))
    log_artifact(
        tracking_run,
        Path(output_dir),
        [
            Path(output_dir) / "resolved_config.json",
            Path(output_dir) / "dataset_manifest.json",
            Path(output_dir) / "result.json",
            Path(checkpoint_path),
        ],
    )
    return result

