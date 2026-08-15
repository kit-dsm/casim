"""Rollout, replay, Structured-RL updates, and training orchestration."""

import copy
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from casim.io_helpers import dump_json, dump_jsonl
from learning.structured_batching.checkpoint import (
    build_checkpoint,
    save_checkpoint,
)
from learning.structured_batching.evaluate import (
    _observation,
    _router,
    evaluate_policy,
    make_actor_choose_batch,
    reference_objective_scale,
)
from learning.structured_batching.learning import (
    complete_returns,
    critic_soft_target,
    fenchel_young_loss,
)
from learning.structured_batching.policy import (
    OrderScoreActor,
    decode_action,
    features_tensor,
    make_critic,
    make_observation,
)
from learning.structured_batching.tracking import (
    log_artifact,
    log_training,
    log_validation,
)


def collect_rollout(
    actor,
    environment,
    observation,
    *,
    decoder_name,
    router,
    sigma_forward,
    generator,
    location_features=True,
):
    """Run one complete simulation episode and collect learner transitions."""
    transitions = []
    episode_return = 0.0
    actor_time = 0.0
    decoder_time = 0.0
    fills = []
    batch_sizes = []
    done = False
    while not done:
        started = time.perf_counter()
        with torch.no_grad():
            scores = actor(
                features_tensor(observation, location_features=location_features)
            )
            noise = torch.randn(scores.shape, generator=generator)
        actor_time += time.perf_counter() - started
        started = time.perf_counter()
        action = decode_action(
            scores + sigma_forward * noise,
            observation,
            decoder_name,
            router,
        )
        decoder_time += time.perf_counter() - started
        indices = torch.nonzero(action, as_tuple=False).flatten().cpu().numpy()
        fills.append(float(observation.demands[indices].sum()) / observation.capacity)
        batch_sizes.append(len(indices))
        current, reward, done = environment.step(
            observation.order_ids[indices].tolist()
        )
        next_observation = None if done else _observation(environment, decoder_name)
        transitions.append(
            {
                "state": observation,
                "action": action.clone(),
                "reward": float(reward),
                "next_state": next_observation,
                "done": bool(done),
                "return_to_go": 0.0,
            }
        )
        episode_return += reward
        observation = next_observation
    return transitions, episode_return, {
        "actor_time_s": actor_time,
        "decoder_time_s": decoder_time,
        "fills": fills,
        "batch_sizes": batch_sizes,
    }


def run_critic_update(
    critic,
    target_critic,
    actor,
    samples,
    *,
    decoder_name,
    router,
    critic_target,
    optimizer,
    location_features=True,
):
    losses = []
    for transition in samples:
        features = features_tensor(
            transition["state"], location_features=location_features
        )
        prediction = critic(features, transition["action"])
        with torch.no_grad():
            target = torch.tensor(
                transition["return_to_go"]
                if critic_target == "return_to_go"
                else transition["reward"]
            )
            if critic_target == "td" and not transition["done"]:
                next_features = features_tensor(
                    transition["next_state"], location_features=location_features
                )
                next_scores = actor(next_features)
                next_action = decode_action(
                    next_scores,
                    transition["next_state"],
                    decoder_name,
                    router,
                )
                target = target + target_critic(next_features, next_action)
        losses.append(torch.nn.functional.smooth_l1_loss(prediction, target))
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    gradient_norm = float(
        sum(
            parameter.grad.detach().pow(2).sum()
            for parameter in critic.parameters()
            if parameter.grad is not None
        ).sqrt()
    )
    optimizer.step()
    return float(loss.detach()), gradient_norm


def run_actor_update(
    actor,
    critic,
    samples,
    *,
    decoder_name,
    router,
    candidate_count,
    sigma_target,
    temperature,
    target_generator,
    fy_generator,
    epsilon,
    fy_samples,
    normalize_advantages,
    optimizer,
    location_features=True,
):
    losses = []
    diagnostics = []
    for transition in samples:
        target_action, diagnostic = critic_soft_target(
            actor,
            critic,
            transition["state"],
            decoder_name,
            router,
            candidate_count=candidate_count,
            sigma=sigma_target,
            temperature=temperature,
            generator=target_generator,
            normalize_advantages=normalize_advantages,
            location_features=location_features,
        )
        loss, mean_action = fenchel_young_loss(
            actor,
            transition["state"],
            target_action,
            decoder_name,
            router,
            sample_count=fy_samples,
            epsilon=epsilon,
            generator=fy_generator,
            location_features=location_features,
        )
        diagnostic["target_policy_l1"] = float(
            torch.abs(target_action - mean_action).mean()
        )
        diagnostics.append(diagnostic)
        losses.append(loss)
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
    optimizer.step()
    return float(loss.detach()), gradient_norm, diagnostics


def _episode_metrics(environment, episode_return, rollout, objective_scale):
    tracker = environment.simulation.state.tracker
    total_flow_time = tracker.accrued_flow_time(environment.simulation.state.current_time)
    flow_times = [
        tracker.order_completion_times[order_id] - arrival
        for order_id, arrival in tracker.order_arrival_times.items()
    ]
    decisions = len(rollout["fills"])
    return {
        "episode_return": float(episode_return),
        "total_flow_time": float(total_flow_time),
        "objective_per_order": float(total_flow_time) / max(1, environment.order_count),
        "reward_identity_error": abs(
            float(episode_return)
            + float(total_flow_time)
            / (max(1, environment.order_count) * objective_scale)
        ),
        "mean_flow_time": float(np.mean(flow_times)),
        "total_distance": float(sum(tracker.distance_by_picker.values())),
        "completed_tours": len(tracker.completed_tours),
        "decisions": decisions,
        "mean_batch_fill": float(np.mean(rollout["fills"])),
        "mean_batch_orders": float(np.mean(rollout["batch_sizes"])),
        "actor_time_s": rollout["actor_time_s"],
        "decoder_time_s": rollout["decoder_time_s"],
        "mean_actor_latency_s": rollout["actor_time_s"] / max(1, decisions),
        "mean_decoder_latency_s": rollout["decoder_time_s"] / max(1, decisions),
    }


def run_training(
    cfg,
    output_dir,
    *,
    splits,
    dataset_manifest,
    environment_factory,
    tracking_run=None,
):
    """Run the supported generated-data Structured-RL training workflow."""
    output_dir = Path(output_dir)
    learner = cfg["learner"]
    model = cfg["model"]
    decoder_name = str(cfg["decoder"]["name"])
    seed = int(cfg["seed"])
    episodes = int(learner["episodes"])
    updates_per_episode = int(learner["updates_per_episode"])
    batch_size = int(learner["batch_size"])
    replay_capacity = int(learner["replay_capacity"])
    critic_target = str(learner["critic_target"])
    normalize_advantages = bool(learner["normalize_candidate_advantages"])
    location_features = True
    show_progress = bool(cfg.get("progress", True))
    schedule = sorted(set(map(int, learner["checkpoint_episodes"])))
    if not schedule or schedule[0] != 0 or schedule[-1] != episodes:
        raise ValueError("Checkpoint schedule must include episode 0 and the final episode")
    if critic_target not in {"td", "return_to_go"}:
        raise ValueError(f"Unknown critic target: {critic_target!r}")
    torch.manual_seed(seed)

    requested_scale = cfg.get("objective_scale")
    if requested_scale is not None:
        objective_scale = float(requested_scale)
    else:
        objective_scale = reference_objective_scale(
            environment_factory,
            splits["validation"],
            show_progress=show_progress,
        )
    if show_progress:
        print(f"Reward objective_scale={objective_scale:.4f}", flush=True)

    environment = environment_factory()
    environment.objective_scale = objective_scale
    environment.reset(splits["train"][0])
    router = _router(environment, decoder_name)
    probe_obs = _observation(environment, decoder_name)
    feature_count = int(probe_obs.features.shape[1])
    route_cost_scale = probe_obs.route_cost_scale
    hidden = int(model["hidden"])
    critic_kind = str(model["critic"])
    data_spec = dict(cfg["data"])

    actor = OrderScoreActor(feature_count, hidden)
    critic = make_critic(critic_kind, feature_count=feature_count, hidden=hidden)
    target_critic = copy.deepcopy(critic)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(learner["actor_learning_rate"]))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(learner["critic_learning_rate"]))
    collection_generator = torch.Generator().manual_seed(seed)
    target_generator = torch.Generator().manual_seed(seed + 1)
    fy_generator = torch.Generator().manual_seed(seed + 2)
    sample_rng = np.random.default_rng(seed)
    replay = deque(maxlen=replay_capacity)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def make_checkpoint(episode):
        return build_checkpoint(
            actor=actor,
            critic=critic,
            critic_kind=critic_kind,
            feature_count=feature_count,
            actor_hidden=hidden,
            decoder_name=decoder_name,
            data_spec=data_spec,
            selected_episode=episode,
            route_cost_scale=route_cost_scale,
            objective_scale=objective_scale,
        )

    def validate(index):
        return evaluate_policy(
            splits["validation"],
            choose_batch=make_actor_choose_batch(
                actor, decoder_name, location_features=location_features
            ),
            environment_factory=environment_factory,
            objective_scale=objective_scale,
            decoder_name=decoder_name,
            location_features=location_features,
            show_progress=show_progress,
            progress_desc=f"Validation at episode {index}",
        )

    validation_rows = [{"episode": 0, "evaluation": validate(0)}]
    log_validation(tracking_run, 0, validation_rows[0]["evaluation"])
    initial_path = checkpoint_dir / "episode_000.pt"
    save_checkpoint(initial_path, make_checkpoint(0))
    validation_rows[0]["checkpoint"] = str(initial_path)
    best_objective = validation_rows[0]["evaluation"]["mean_objective_per_order"]
    best_episode = 0
    best_actor_state = copy.deepcopy(actor.state_dict())
    best_critic_state = copy.deepcopy(critic.state_dict())
    episode_rows = []
    update_rows = []
    train_ids = splits["train"]
    progress = tqdm(range(episodes), desc="SRL training", unit="episode", disable=not show_progress)
    for episode_index in progress:
        instance_id = train_ids[episode_index % len(train_ids)]
        environment.reset(instance_id)
        observation = _observation(environment, decoder_name)
        transitions, episode_return, rollout = collect_rollout(
            actor,
            environment,
            observation,
            decoder_name=decoder_name,
            router=router,
            sigma_forward=float(learner["sigma_forward"]),
            generator=collection_generator,
            location_features=location_features,
        )
        returns = complete_returns([t["reward"] for t in transitions])
        for transition, return_to_go in zip(transitions, returns):
            transition["return_to_go"] = return_to_go
            replay.append(transition)
        update_started = time.perf_counter()
        episode_updates = []
        for _ in range(updates_per_episode):
            if len(replay) < batch_size:
                break
            samples = [
                replay[index]
                for index in sample_rng.choice(len(replay), size=batch_size, replace=False)
            ]
            critic_loss, critic_gradient = run_critic_update(
                critic,
                target_critic,
                actor,
                samples,
                decoder_name=decoder_name,
                router=router,
                critic_target=critic_target,
                optimizer=critic_optimizer,
                location_features=location_features,
            )
            actor_loss, actor_gradient, diagnostics = run_actor_update(
                actor,
                critic,
                samples,
                decoder_name=decoder_name,
                router=router,
                candidate_count=int(learner["candidate_count"]),
                sigma_target=float(learner["sigma_target"]),
                temperature=float(learner["temperature"]),
                target_generator=target_generator,
                fy_generator=fy_generator,
                epsilon=float(learner["epsilon"]),
                fy_samples=int(learner["fy_samples"]),
                normalize_advantages=normalize_advantages,
                optimizer=actor_optimizer,
                location_features=location_features,
            )
            update = {
                "episode": episode_index + 1,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "actor_gradient_norm": actor_gradient,
                "critic_gradient_norm": critic_gradient,
                "mean_unique_candidates": float(
                    np.mean([row["unique_candidates"] for row in diagnostics])
                ),
                "mean_candidate_q_spread": float(
                    np.mean([row["candidate_q_spread"] for row in diagnostics])
                ),
            }
            update_rows.append(update)
            episode_updates.append(update)
        if critic_target == "td":
            target_critic.load_state_dict(critic.state_dict())
        row = {
            "episode": episode_index + 1,
            "instance_id": instance_id,
            **_episode_metrics(environment, episode_return, rollout, objective_scale),
            "update_time_s": time.perf_counter() - update_started,
            "replay_size": len(replay),
        }
        episode_rows.append(row)
        log_training(tracking_run, row, episode_updates)
        if show_progress:
            progress.set_postfix(flow=f"{row['mean_flow_time']:.1f}s", replay=len(replay))
        if episode_index + 1 in schedule:
            validation = validate(episode_index + 1)
            path = checkpoint_dir / f"episode_{episode_index + 1:03d}.pt"
            save_checkpoint(path, make_checkpoint(episode_index + 1))
            validation_rows.append(
                {"episode": episode_index + 1, "evaluation": validation, "checkpoint": str(path)}
            )
            log_validation(tracking_run, episode_index + 1, validation)
            if validation["mean_objective_per_order"] < best_objective:
                best_objective = validation["mean_objective_per_order"]
                best_episode = episode_index + 1
                best_actor_state = copy.deepcopy(actor.state_dict())
                best_critic_state = copy.deepcopy(critic.state_dict())
    actor.load_state_dict(best_actor_state)
    critic.load_state_dict(best_critic_state)
    selected_path = output_dir / "best.pt"
    save_checkpoint(selected_path, make_checkpoint(best_episode))
    dump_json(output_dir / "dataset_manifest.json", dataset_manifest)
    dump_jsonl(output_dir / "episodes.jsonl", episode_rows)
    dump_jsonl(output_dir / "updates.jsonl", update_rows)
    dump_jsonl(output_dir / "validation.jsonl", validation_rows)
    result = {
        "mode": "train",
        "status": "complete",
        "decoder": decoder_name,
        "selected_checkpoint": str(selected_path),
        "selected_episode": best_episode,
        "selected_validation_objective": best_objective,
        "objective_scale": objective_scale,
        "route_cost_scale": route_cost_scale,
        "feature_count": feature_count,
        "critic_kind": critic_kind,
        "critic_target": critic_target,
        "max_reward_identity_error": max(
            row["reward_identity_error"] for row in episode_rows
        ),
    }
    dump_json(output_dir / "result.json", result)
    if tracking_run is not None:
        tracking_run.summary["selected_episode"] = best_episode
        tracking_run.summary["selected_validation_objective"] = best_objective
        tracking_run.summary["objective_scale"] = objective_scale
    log_artifact(
        tracking_run,
        output_dir,
        [
            output_dir / "resolved_config.json",
            output_dir / "dataset_manifest.json",
            output_dir / "result.json",
            output_dir / "episodes.jsonl",
            output_dir / "updates.jsonl",
            selected_path,
        ],
    )
    return result
