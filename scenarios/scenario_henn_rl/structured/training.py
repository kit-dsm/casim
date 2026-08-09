from __future__ import annotations

import copy
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from scenarios.scenario_henn_rl.structured.environment import StructuredBatchingEpisode
from scenarios.scenario_henn_rl.structured.evaluation import (
    evaluate_structured_actor,
    reference_objective_scale,
)
from scenarios.scenario_henn_rl.structured.models import (
    OrderScoreActor,
    PairwiseStructuredCritic,
    StructuredCritic,
    _copy_state,
    _features,
    _mask,
    critic_soft_target,
    decode_scores,
    discounted_returns,
    fenchel_young_loss,
)
from scenarios.scenario_henn_rl.structured.data import (
    GeneratedHennDataLoader,
    generated_instance_splits,
    generated_manifest,
)
from scenarios.scenario_henn_rl.structured.results import write_json, write_jsonl
from scenarios.scenario_henn_rl.structured.tracking import log_artifact, metric_callback


def pretrain_from_existing_policy(
    actor,
    instance_ids: list[str],
    *,
    epochs: int,
    learning_rate: float,
    epsilon: float,
    fy_samples: int,
    seed: int,
    location_features: bool = True,
) -> dict[str, object]:
    examples = []
    episode = StructuredBatchingEpisode(instance_ids)
    for instance_id in instance_ids:
        state = episode.reset(instance_id=instance_id)
        done = False
        while not done:
            indices = episode.existing_policy_action(state, batching="cw")
            examples.append(
                (_copy_state(state), _mask(indices, len(state["order_ids"])))
            )
            state, _, done, _, _ = episode.step(indices)
    episode.close()
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed + 2)
    final_loss = 0.0
    for _ in range(epochs):
        for state, target in examples:
            loss, _ = fenchel_young_loss(
                actor,
                state,
                target,
                sample_count=fy_samples,
                epsilon=epsilon,
                generator=generator,
                location_features=location_features,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    exact_matches = 0
    item_matches = []
    wait_matches = []
    with torch.no_grad():
        for state, target in examples:
            prediction = decode_scores(
                actor(
                    _features(
                        state, location_features=location_features
                    )
                ),
                state,
            )
            exact_matches += int(torch.equal(prediction, target))
            item_matches.append(float((prediction == target).float().mean()))
            wait_matches.append(
                float((prediction.sum() == 0) == (target.sum() == 0))
            )
    return {
        "examples": len(examples),
        "epochs": epochs,
        "final_loss": final_loss,
        "exact_batch_accuracy": exact_matches / max(1, len(examples)),
        "mean_item_accuracy": float(np.mean(item_matches)),
        "wait_accuracy": float(np.mean(wait_matches)),
        "expert": "existing_cw_s_shape_short_with_validated_release",
    }


def train_structured_rl(
    actor,
    instance_ids: list[str],
    *,
    episodes: int,
    updates_per_episode: int,
    batch_size: int,
    replay_capacity: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
    sigma_forward: float,
    sigma_target: float,
    temperature: float,
    candidate_count: int,
    epsilon: float,
    fy_samples: int,
    gamma: float,
    seed: int,
    validation_ids: list[str],
    checkpoint_every: int | None = None,
    checkpoint_episodes: list[int] | tuple[int, ...] | None = None,
    checkpoint_dir: str | Path | None = None,
    reward_power: float = 1.0,
    sla_threshold_s: float = 0.0,
    objective_scale: float | None = None,
    location_features: bool = True,
    critic_target: str = "td",
    normalize_candidate_advantages: bool = False,
    critic_kind: str = "mean",
    feature_count: int = 7,
    episode_kwargs: dict | None = None,
    metric_callback=None,
    show_progress: bool = False,
) -> tuple[StructuredCritic, dict[str, object]]:
    if critic_target not in {"td", "return_to_go"}:
        raise ValueError(f"Unknown critic target: {critic_target!r}")
    if checkpoint_episodes is None:
        if checkpoint_every is None or checkpoint_every < 1:
            raise ValueError("A positive checkpoint schedule is required")
        checkpoint_episodes = list(range(0, episodes + 1, checkpoint_every))
        if checkpoint_episodes[-1] != episodes:
            checkpoint_episodes.append(episodes)
    checkpoint_episodes = sorted(set(map(int, checkpoint_episodes)))
    if not checkpoint_episodes or checkpoint_episodes[0] != 0:
        raise ValueError("Checkpoint schedule must include episode 0")
    if checkpoint_episodes[-1] != episodes:
        raise ValueError("Checkpoint schedule must include the final episode")
    checkpoint_path = None if checkpoint_dir is None else Path(checkpoint_dir)
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    critic_types = {
        "mean": StructuredCritic,
        "interaction": PairwiseStructuredCritic,
    }
    if critic_kind not in critic_types:
        raise ValueError(f"Unknown critic kind: {critic_kind!r}")
    critic = critic_types[critic_kind](feature_count=feature_count)
    target_critic = copy.deepcopy(critic)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=actor_learning_rate
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=critic_learning_rate
    )
    collection_generator = torch.Generator().manual_seed(seed)
    target_generator = torch.Generator().manual_seed(seed + 1)
    fy_generator = torch.Generator().manual_seed(seed + 2)
    sample_rng = np.random.default_rng(seed)
    replay = deque(maxlen=replay_capacity)
    environment = StructuredBatchingEpisode(
        instance_ids,
        reward_power=reward_power,
        sla_threshold_s=sla_threshold_s,
        objective_scale=objective_scale,
        **(episode_kwargs or {}),
    )
    episode_rows = []
    update_rows = []
    validation_rows = [
        {
            "episode": 0,
            "evaluation": evaluate_structured_actor(
                actor,
                validation_ids,
                location_features=location_features,
                reward_power=reward_power,
                sla_threshold_s=sla_threshold_s,
                objective_scale=objective_scale,
                episode_kwargs=episode_kwargs,
                show_progress=show_progress,
                progress_desc="Initial validation",
            ),
        }
    ]
    if metric_callback is not None:
        metric_callback("validation", 0, validation_rows[0]["evaluation"])
    if checkpoint_path is not None:
        torch.save(actor.state_dict(), checkpoint_path / "episode_000.pt")
        validation_rows[0]["checkpoint"] = str(
            checkpoint_path / "episode_000.pt"
        )
    best_objective = validation_rows[0]["evaluation"][
        "mean_objective_per_order"
    ]
    best_episode = 0
    best_actor_state = copy.deepcopy(actor.state_dict())
    episode_progress = tqdm(
        range(episodes),
        desc="SRL training",
        unit="episode",
        disable=not show_progress,
    )
    for episode_index in episode_progress:
        update_start = len(update_rows)
        instance_id = instance_ids[episode_index % len(instance_ids)]
        state = environment.reset(seed=seed + episode_index, instance_id=instance_id)
        done = False
        episode_return = 0.0
        policy_time = 0.0
        actor_time = 0.0
        oracle_time = 0.0
        decisions = 0
        batch_fills = []
        batch_orders = []
        episode_transitions = []
        while not done:
            actor_started = time.perf_counter()
            with torch.no_grad():
                scores = actor(
                    _features(state, location_features=location_features)
                )
                noise = torch.randn(scores.shape, generator=collection_generator)
            actor_time += time.perf_counter() - actor_started
            oracle_started = time.perf_counter()
            with torch.no_grad():
                action = decode_scores(scores + sigma_forward * noise, state)
            oracle_time += time.perf_counter() - oracle_started
            policy_time = actor_time + oracle_time
            decisions += 1
            indices = (
                torch.nonzero(action, as_tuple=False).flatten().cpu().numpy()
            )
            batch_fills.append(
                float(state["demands"][indices].sum()) / int(state["capacity"])
            )
            batch_orders.append(int(len(indices)))
            next_state, reward, done, _, info = environment.step(indices)
            episode_transitions.append(
                {
                    "state": _copy_state(state),
                    "action": action.clone(),
                    "reward": float(reward),
                    "next_state": _copy_state(next_state),
                    "done": bool(done),
                }
            )
            episode_return += reward
            state = next_state

        returns = discounted_returns(
            [transition["reward"] for transition in episode_transitions],
            gamma,
        )
        for transition, return_to_go in zip(episode_transitions, returns):
            transition["return_to_go"] = return_to_go
            replay.append(transition)

        for _ in range(updates_per_episode):
            if len(replay) < batch_size:
                break
            samples = [
                replay[index]
                for index in sample_rng.choice(
                    len(replay), size=batch_size, replace=False
                )
            ]
            critic_losses = []
            for transition in samples:
                features = _features(
                    transition["state"],
                    location_features=location_features,
                )
                prediction = critic(features, transition["action"])
                with torch.no_grad():
                    target = torch.tensor(
                        transition[
                            "return_to_go" if critic_target == "return_to_go"
                            else "reward"
                        ]
                    )
                    if critic_target == "td" and not transition["done"]:
                        next_features = _features(
                            transition["next_state"],
                            location_features=location_features,
                        )
                        next_action = decode_scores(
                            actor(next_features), transition["next_state"]
                        )
                        target = target + gamma * target_critic(
                            next_features, next_action
                        )
                critic_losses.append(
                    torch.nn.functional.smooth_l1_loss(prediction, target)
                )
            critic_loss = torch.stack(critic_losses).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_gradient_norm = float(
                sum(
                    parameter.grad.detach().pow(2).sum()
                    for parameter in critic.parameters()
                    if parameter.grad is not None
                ).sqrt()
            )
            critic_optimizer.step()

            actor_losses = []
            diagnostics = []
            for transition in samples:
                target_action, diagnostic = critic_soft_target(
                    actor,
                    critic,
                    transition["state"],
                    candidate_count=candidate_count,
                    sigma=sigma_target,
                    temperature=temperature,
                    generator=target_generator,
                    location_features=location_features,
                    normalize_advantages=normalize_candidate_advantages,
                )
                actor_loss, mean_action = fenchel_young_loss(
                    actor,
                    transition["state"],
                    target_action,
                    sample_count=fy_samples,
                    epsilon=epsilon,
                    generator=fy_generator,
                    location_features=location_features,
                )
                actor_losses.append(actor_loss)
                diagnostic["target_policy_l1"] = float(
                    torch.abs(target_action - mean_action).mean()
                )
                diagnostics.append(diagnostic)
            actor_loss = torch.stack(actor_losses).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            )
            actor_optimizer.step()
            update_rows.append(
                {
                    "critic_loss": float(critic_loss.detach()),
                    "actor_loss": float(actor_loss.detach()),
                    "actor_gradient_norm": gradient_norm,
                    "critic_gradient_norm": critic_gradient_norm,
                    "sigma_target_used": sigma_target,
                    "mean_unique_candidates": float(
                        np.mean([row["unique_candidates"] for row in diagnostics])
                    ),
                    "mean_candidate_q_spread": float(
                        np.mean([row["candidate_q_spread"] for row in diagnostics])
                    ),
                    "mean_target_policy_l1": float(
                        np.mean([row["target_policy_l1"] for row in diagnostics])
                    ),
                }
            )
        if critic_target == "td":
            target_critic.load_state_dict(critic.state_dict())
        episode_rows.append(
            {
                "episode": episode_index + 1,
                "instance_id": instance_id,
                "return": episode_return,
                "total_flow_time": info["total_flow_time"],
                "objective_cost": info["objective_cost"],
                "total_distance": info["total_distance"],
                "completed_tours": info["completed_tours"],
                "mean_batch_fill": float(np.mean(batch_fills)),
                "mean_batch_orders": float(np.mean(batch_orders)),
                "orders_per_sim_hour": (
                    len(environment.env.completion_times())
                    * 3600.0
                    / max(
                        1.0,
                        max(environment.env.completion_times().values())
                        - min(environment.env.arrivals.values()),
                    )
                ),
                "reward_identity_error": abs(
                    episode_return
                    + info["objective_cost"] / environment.env.reward_normalizer
                ),
                "wait_actions": info["wait_actions"],
                "dispatch_actions": info["dispatch_actions"],
                "policy_time_s": policy_time,
                "actor_time_s": actor_time,
                "oracle_time_s": oracle_time,
                "mean_policy_latency_s": policy_time / max(1, decisions),
                "mean_oracle_latency_s": oracle_time / max(1, decisions),
                "replay_size": len(replay),
                "mean_flow_time": float(
                    np.mean(list(info["flow_times_by_order"].values()))
                ),
                "p95_flow_time": float(
                    np.quantile(list(info["flow_times_by_order"].values()), 0.95)
                ),
                "mean_tardiness": (
                    float(np.mean(list(info["tardiness_by_order"].values())))
                    if info["tardiness_by_order"]
                    else 0.0
                ),
                "max_tardiness": (
                    float(np.max(list(info["tardiness_by_order"].values())))
                    if info["tardiness_by_order"]
                    else 0.0
                ),
                "violation_fraction": (
                    info["violated_orders"] / len(info["tardiness_by_order"])
                    if info["tardiness_by_order"]
                    else 0.0
                ),
            }
        )
        if metric_callback is not None:
            recent_updates = update_rows[update_start:]
            metrics = dict(episode_rows[-1])
            if recent_updates:
                for key in recent_updates[0]:
                    values = [row[key] for row in recent_updates]
                    if any(value is None for value in values):
                        metrics[key] = None
                    else:
                        metrics[key] = float(np.mean(values))
            metric_callback("train", episode_index + 1, metrics)
        if show_progress:
            episode_progress.set_postfix(
                objective=f"{info['objective_cost']:.1f}",
                flow=f"{episode_rows[-1]['mean_flow_time']:.1f}s",
                replay=len(replay),
            )
        if (episode_index + 1) in checkpoint_episodes:
            validation = evaluate_structured_actor(
                actor,
                validation_ids,
                location_features=location_features,
                reward_power=reward_power,
                sla_threshold_s=sla_threshold_s,
                objective_scale=objective_scale,
                episode_kwargs=episode_kwargs,
                show_progress=show_progress,
                progress_desc=f"Validation at episode {episode_index + 1}",
            )
            row = {"episode": episode_index + 1, "evaluation": validation}
            if checkpoint_path is not None:
                path = checkpoint_path / f"episode_{episode_index + 1:03d}.pt"
                torch.save(actor.state_dict(), path)
                row["checkpoint"] = str(path)
            validation_rows.append(row)
            if metric_callback is not None:
                metric_callback(
                    "validation",
                    episode_index + 1,
                    validation,
                )
            if validation["mean_objective_per_order"] < best_objective:
                best_objective = validation["mean_objective_per_order"]
                best_episode = episode_index + 1
                best_actor_state = copy.deepcopy(actor.state_dict())
    environment.close()
    actor.load_state_dict(best_actor_state)
    return critic, {
        "episodes": episode_rows,
        "updates": update_rows,
        "gamma": gamma,
        "candidate_count": candidate_count,
        "fy_samples": fy_samples,
        "target_copy": (
            "after_each_episode" if critic_target == "td" else "unused"
        ),
        "critic_target": critic_target,
        "normalize_candidate_advantages": normalize_candidate_advantages,
        "critic_kind": critic_kind,
        "feature_count": feature_count,
        "sigma_target": sigma_target,
        "validation_checkpoints": validation_rows,
        "selected_episode": best_episode,
        "selected_validation_objective": best_objective,
        "reward_power": reward_power,
        "sla_threshold_s": sla_threshold_s,
        "objective_scale": objective_scale,
        "checkpoint_episodes": checkpoint_episodes,
        "training_instance_order": list(instance_ids[:episodes]),
        "max_reward_identity_error": max(
            row["reward_identity_error"] for row in episode_rows
        ),
    }


def run_training(cfg: dict, output_dir: Path, tracking_run=None) -> dict[str, object]:
    """Run the supported generated-data SRL training workflow."""
    torch.manual_seed(int(cfg["seed"]))
    data_spec = dict(cfg["data"])
    splits = generated_instance_splits(data_spec)
    if bool(cfg.get("progress", True)):
        print(
            "Preparing SRL run: "
            f"{len(splits['train'])} train / "
            f"{len(splits['validation'])} validation instances, "
            f"{int(cfg['learner']['episodes'])} episodes.",
            flush=True,
        )
        print(
            "Running initial validation before episode 1; "
            "this selects checkpoints without using the test split.",
            flush=True,
        )
    manifest = generated_manifest(data_spec)
    write_json(output_dir / "dataset_manifest.json", manifest)
    loader = GeneratedHennDataLoader(data_spec)
    objective = cfg["objective"]
    model = cfg["model"]
    learner = cfg["learner"]
    feature_count = int(model["feature_count"])
    actor = OrderScoreActor(
        feature_count=feature_count,
        hidden=int(model["hidden"]),
    )
    episode_kwargs = {
        "data_loader": loader,
        "use_order_due_dates": bool(objective["use_order_due_dates"]),
        "include_due_slack": True,
        "decoder": str(model.get("decoder", "knapsack")),
    }
    requested_scale = cfg.get("objective_scale")
    if requested_scale is not None:
        objective_scale = float(requested_scale)
    else:
        objective_scale = reference_objective_scale(
            splits["validation"],
            reward_power=float(objective["power"]),
            episode_kwargs=episode_kwargs,
            show_progress=bool(cfg.get("progress", True)),
            progress_desc="Reference objective scale evaluation",
        )
    critic, training = train_structured_rl(
        actor,
        splits["train"],
        episodes=int(learner["episodes"]),
        updates_per_episode=int(learner["updates_per_episode"]),
        batch_size=int(learner["batch_size"]),
        replay_capacity=int(learner["replay_capacity"]),
        actor_learning_rate=float(learner["actor_learning_rate"]),
        critic_learning_rate=float(learner["critic_learning_rate"]),
        sigma_forward=float(learner["sigma_forward"]),
        sigma_target=float(learner["sigma_target"]),
        temperature=float(learner["temperature"]),
        candidate_count=int(learner["candidate_count"]),
        epsilon=float(learner["epsilon"]),
        fy_samples=int(learner["fy_samples"]),
        critic_target=str(learner.get("critic_target", "td")),
        normalize_candidate_advantages=bool(
            learner.get("normalize_candidate_advantages", False)
        ),
        gamma=float(objective["gamma"]),
        seed=int(cfg["seed"]),
        validation_ids=splits["validation"],
        checkpoint_episodes=list(map(int, learner["checkpoint_episodes"])),
        checkpoint_dir=output_dir / "checkpoints",
        reward_power=float(objective["power"]),
        objective_scale=objective_scale,
        critic_kind=str(model["critic"]),
        feature_count=feature_count,
        episode_kwargs=episode_kwargs,
        metric_callback=metric_callback(tracking_run),
        show_progress=bool(cfg.get("progress", True)),
    )
    checkpoint = {
        "format_version": 1,
        "feature_schema": str(model["feature_schema"]),
        "feature_count": feature_count,
        "actor_hidden": int(model["hidden"]),
        "critic_kind": str(model["critic"]),
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "selected_episode": training["selected_episode"],
        "objective": dict(objective),
        "objective_scale": objective_scale,
        "data": data_spec,
    }
    selected_path = output_dir / "best.pt"
    torch.save(checkpoint, selected_path)
    result = {
        "mode": "train",
        "status": "complete",
        "selected_checkpoint": str(selected_path),
        "training": training,
    }
    result_path = output_dir / "result.json"
    write_json(result_path, result)
    write_jsonl(output_dir / "episodes.jsonl", training["episodes"])
    write_jsonl(output_dir / "updates.jsonl", training["updates"])
    log_artifact(
        tracking_run,
        output_dir,
        [
            selected_path,
            result_path,
            output_dir / "dataset_manifest.json",
            output_dir / "resolved_config.json",
        ],
    )
    if bool(cfg.get("progress", True)):
        print(
            f"Training complete. Selected episode {training['selected_episode']} "
            f"and wrote {selected_path}.",
            flush=True,
        )
    return result
