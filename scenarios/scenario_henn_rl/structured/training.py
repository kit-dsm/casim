from __future__ import annotations

import copy
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from scenarios.scenario_henn_rl.structured.checkpoint import (
    build_checkpoint,
    save_checkpoint,
)
from scenarios.scenario_henn_rl.structured.environment import StructuredBatchingEpisode
from scenarios.scenario_henn_rl.structured.evaluation import (
    evaluate_structured_actor,
    reference_objective_scale,
)
from scenarios.scenario_henn_rl.structured.models import (
    OrderScoreActor,
    make_critic,
    critic_soft_target,
    discounted_returns,
    fenchel_young_loss,
)
from scenarios.scenario_henn_rl.structured.policy import StructuredPolicy
from scenarios.scenario_henn_rl.structured.state import (
    Transition,
    action_mask,
    copy_state,
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
    """Diagnostic imitation pretraining from the C&W existing policy.

    Not on the supported training path; retained for offline experiments.
    """
    episode = StructuredBatchingEpisode(instance_ids)
    policy = StructuredPolicy(actor, episode.decoder, location_features=location_features)
    examples = []
    for instance_id in instance_ids:
        state = episode.reset(instance_id=instance_id)
        done = False
        while not done:
            indices = episode.existing_policy_action(state, batching="cw")
            examples.append((copy_state(state), action_mask(indices, len(state))))
            state, _, done, _, _ = episode.step(indices)
    episode.close()
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed + 2)
    final_loss = 0.0
    for _ in range(epochs):
        for state, target in examples:
            loss, _ = fenchel_young_loss(
                policy,
                state,
                target,
                sample_count=fy_samples,
                epsilon=epsilon,
                generator=generator,
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
            prediction = policy.select_action(state)
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


def collect_rollout(
    policy: StructuredPolicy,
    environment: StructuredBatchingEpisode,
    state,
    *,
    sigma_forward: float,
    generator: torch.Generator,
):
    """Collect one episode of transitions under ``policy``.

    Returns ``(transitions, episode_return, info, metrics)`` where ``metrics``
    separates the rollout actor (network) time from the rollout decoder time.
    """
    done = False
    episode_return = 0.0
    actor_time = 0.0
    decoder_time = 0.0
    decisions = 0
    batch_fills = []
    batch_orders = []
    transitions = []
    while not done:
        actor_started = time.perf_counter()
        with torch.no_grad():
            scores = policy.actor(policy.features(state))
            noise = torch.randn(scores.shape, generator=generator)
        actor_time += time.perf_counter() - actor_started
        decoder_started = time.perf_counter()
        with torch.no_grad():
            action = policy.decode(scores + sigma_forward * noise, state)
        decoder_time += time.perf_counter() - decoder_started
        decisions += 1
        indices = torch.nonzero(action, as_tuple=False).flatten().cpu().numpy()
        batch_fills.append(
            float(state.demands[indices].sum()) / int(state.capacity)
        )
        batch_orders.append(int(len(indices)))
        next_state, reward, done, _, info = environment.step(indices)
        transitions.append(
            Transition(
                state=copy_state(state),
                action=action.clone(),
                reward=float(reward),
                next_state=copy_state(next_state),
                done=bool(done),
            )
        )
        episode_return += reward
        state = next_state
    metrics = {
        "actor_time_s": actor_time,
        "decoder_time_s": decoder_time,
        "policy_time_s": actor_time + decoder_time,
        "decisions": decisions,
        "batch_fills": batch_fills,
        "batch_orders": batch_orders,
    }
    return transitions, episode_return, info, metrics


def run_critic_update(
    critic,
    target_critic,
    policy: StructuredPolicy,
    samples: list[Transition],
    *,
    gamma: float,
    critic_target: str,
    optimizer,
):
    """One critic gradient step; returns loss and gradient norm."""
    critic_losses = []
    for transition in samples:
        features = policy.features(transition.state)
        prediction = critic(features, transition.action)
        with torch.no_grad():
            target = torch.tensor(
                transition.return_to_go
                if critic_target == "return_to_go"
                else transition.reward
            )
            if critic_target == "td" and not transition.done:
                next_features = policy.features(transition.next_state)
                next_action = policy.decode(
                    policy.actor(next_features), transition.next_state
                )
                target = target + gamma * target_critic(next_features, next_action)
        critic_losses.append(
            torch.nn.functional.smooth_l1_loss(prediction, target)
        )
    critic_loss = torch.stack(critic_losses).mean()
    optimizer.zero_grad()
    critic_loss.backward()
    critic_gradient_norm = float(
        sum(
            parameter.grad.detach().pow(2).sum()
            for parameter in critic.parameters()
            if parameter.grad is not None
        ).sqrt()
    )
    optimizer.step()
    return float(critic_loss.detach()), critic_gradient_norm


def run_actor_update(
    policy: StructuredPolicy,
    critic,
    samples: list[Transition],
    *,
    candidate_count: int,
    sigma_target: float,
    temperature: float,
    target_generator: torch.Generator,
    fy_generator: torch.Generator,
    epsilon: float,
    fy_samples: int,
    normalize_advantages: bool,
    optimizer,
):
    """One actor gradient step; returns loss, gradient norm, and diagnostics."""
    actor_losses = []
    diagnostics = []
    for transition in samples:
        target_action, diagnostic = critic_soft_target(
            policy,
            critic,
            transition.state,
            candidate_count=candidate_count,
            sigma=sigma_target,
            temperature=temperature,
            generator=target_generator,
            normalize_advantages=normalize_advantages,
        )
        actor_loss, mean_action = fenchel_young_loss(
            policy,
            transition.state,
            target_action,
            sample_count=fy_samples,
            epsilon=epsilon,
            generator=fy_generator,
        )
        actor_losses.append(actor_loss)
        diagnostic["target_policy_l1"] = float(
            torch.abs(target_action - mean_action).mean()
        )
        diagnostics.append(diagnostic)
    actor_loss = torch.stack(actor_losses).mean()
    optimizer.zero_grad()
    actor_loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.0))
    optimizer.step()
    return float(actor_loss.detach()), gradient_norm, diagnostics


def _resolve_checkpoint_episodes(
    episodes: int,
    checkpoint_every: int | None,
    checkpoint_episodes: list[int] | tuple[int, ...] | None,
) -> list[int]:
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
    return checkpoint_episodes


def _episode_metrics(environment, info, episode_return, rollout_metrics):
    batch_fills = rollout_metrics["batch_fills"]
    batch_orders = rollout_metrics["batch_orders"]
    decisions = rollout_metrics["decisions"]
    actor_time = rollout_metrics["actor_time_s"]
    decoder_time = rollout_metrics["decoder_time_s"]
    return {
        "episode_return": episode_return,
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
        "actor_time_s": actor_time,
        "decoder_time_s": decoder_time,
        "policy_time_s": rollout_metrics["policy_time_s"],
        "mean_actor_latency_s": actor_time / max(1, decisions),
        "mean_decoder_latency_s": decoder_time / max(1, decisions),
        "mean_flow_time": float(np.mean(list(info["flow_times_by_order"].values()))),
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
    feature_schema: str = "deadline_v1",
    actor_hidden: int = 32,
    episode_kwargs: dict | None = None,
    metric_callback=None,
    show_progress: bool = False,
) -> tuple[torch.nn.Module, dict[str, object]]:
    if critic_target not in {"td", "return_to_go"}:
        raise ValueError(f"Unknown critic target: {critic_target!r}")
    checkpoint_episodes = _resolve_checkpoint_episodes(
        episodes, checkpoint_every, checkpoint_episodes
    )
    checkpoint_path = None if checkpoint_dir is None else Path(checkpoint_dir)
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    critic = make_critic(critic_kind, feature_count=feature_count)
    target_critic = copy.deepcopy(critic)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_learning_rate)
    collection_generator = torch.Generator().manual_seed(seed)
    target_generator = torch.Generator().manual_seed(seed + 1)
    fy_generator = torch.Generator().manual_seed(seed + 2)
    sample_rng = np.random.default_rng(seed)
    replay: deque[Transition] = deque(maxlen=replay_capacity)
    environment = StructuredBatchingEpisode(
        instance_ids,
        reward_power=reward_power,
        sla_threshold_s=sla_threshold_s,
        objective_scale=objective_scale,
        **(episode_kwargs or {}),
    )
    policy = StructuredPolicy(
        actor, environment.decoder, location_features=location_features
    )
    decoder_config = environment.decoder.config()
    # The route-cost scale is a layout constant; compute it from one probe reset
    # so every checkpoint (including the untrained episode-0 checkpoint) carries
    # the same scale as the trained policy.
    _probe_state = environment.reset(
        seed=seed, instance_id=instance_ids[0]
    )
    route_cost_scale = float(_probe_state.route_cost_scale)
    episode_rows = []
    update_rows = []

    def validate(episode_index: int, progress_desc: str):
        evaluation = evaluate_structured_actor(
            actor,
            validation_ids,
            location_features=location_features,
            reward_power=reward_power,
            sla_threshold_s=sla_threshold_s,
            objective_scale=objective_scale,
            episode_kwargs=episode_kwargs,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
        return evaluation

    validation_rows = [
        {
            "episode": 0,
            "evaluation": validate(0, "Initial validation"),
        }
    ]
    if metric_callback is not None:
        metric_callback("validation", 0, validation_rows[0]["evaluation"])
    best_objective = validation_rows[0]["evaluation"]["mean_objective_per_order"]
    best_episode = 0
    best_actor_state = copy.deepcopy(actor.state_dict())
    best_critic_state = copy.deepcopy(critic.state_dict())
    if checkpoint_path is not None:
        initial_checkpoint = build_checkpoint(
            actor=actor,
            critic=critic,
            critic_kind=critic_kind,
            feature_schema=feature_schema,
            feature_count=feature_count,
            actor_hidden=actor_hidden,
            decoder_config=decoder_config,
            objective={
                "power": reward_power,
                "sla_threshold_s": sla_threshold_s,
            },
            objective_scale=objective_scale,
            data_spec={},
            selected_episode=0,
            route_cost_scale=route_cost_scale,
        )
        save_checkpoint(checkpoint_path / "episode_000.pt", initial_checkpoint)
        validation_rows[0]["checkpoint"] = str(checkpoint_path / "episode_000.pt")
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
        transitions, episode_return, info, rollout_metrics = collect_rollout(
            policy,
            environment,
            state,
            sigma_forward=sigma_forward,
            generator=collection_generator,
        )
        returns = discounted_returns(
            [transition.reward for transition in transitions], gamma
        )
        for transition, return_to_go in zip(transitions, returns):
            transition.return_to_go = return_to_go
            replay.append(transition)

        update_started = time.perf_counter()
        for _ in range(updates_per_episode):
            if len(replay) < batch_size:
                break
            samples = [
                replay[index]
                for index in sample_rng.choice(
                    len(replay), size=batch_size, replace=False
                )
            ]
            critic_loss, critic_gradient_norm = run_critic_update(
                critic,
                target_critic,
                policy,
                samples,
                gamma=gamma,
                critic_target=critic_target,
                optimizer=critic_optimizer,
            )
            actor_loss, gradient_norm, diagnostics = run_actor_update(
                policy,
                critic,
                samples,
                candidate_count=candidate_count,
                sigma_target=sigma_target,
                temperature=temperature,
                target_generator=target_generator,
                fy_generator=fy_generator,
                epsilon=epsilon,
                fy_samples=fy_samples,
                normalize_advantages=normalize_candidate_advantages,
                optimizer=actor_optimizer,
            )
            update_rows.append(
                {
                    "critic_loss": critic_loss,
                    "actor_loss": actor_loss,
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
        update_time = time.perf_counter() - update_started
        if critic_target == "td":
            target_critic.load_state_dict(critic.state_dict())
        episode_row = {
            "episode": episode_index + 1,
            "instance_id": instance_id,
            **_episode_metrics(environment, info, episode_return, rollout_metrics),
            "update_time_s": update_time,
            "mean_update_latency_s": update_time / max(1, updates_per_episode),
            "replay_size": len(replay),
        }
        episode_rows.append(episode_row)
        if metric_callback is not None:
            recent_updates = update_rows[update_start:]
            metrics = dict(episode_row)
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
                flow=f"{episode_row['mean_flow_time']:.1f}s",
                replay=len(replay),
            )
        if (episode_index + 1) in checkpoint_episodes:
            validation = validate(episode_index + 1, f"Validation at episode {episode_index + 1}")
            row = {"episode": episode_index + 1, "evaluation": validation}
            if checkpoint_path is not None:
                path = checkpoint_path / f"episode_{episode_index + 1:03d}.pt"
                save_checkpoint(
                    path,
                    build_checkpoint(
                        actor=actor,
                        critic=critic,
                        critic_kind=critic_kind,
                        feature_schema=feature_schema,
                        feature_count=feature_count,
                        actor_hidden=actor_hidden,
                        decoder_config=decoder_config,
                        objective={
                            "power": reward_power,
                            "sla_threshold_s": sla_threshold_s,
                        },
                        objective_scale=objective_scale,
                        data_spec={},
                        selected_episode=episode_index + 1,
                        route_cost_scale=route_cost_scale,
                    ),
                )
                row["checkpoint"] = str(path)
            validation_rows.append(row)
            if metric_callback is not None:
                metric_callback("validation", episode_index + 1, validation)
            if validation["mean_objective_per_order"] < best_objective:
                best_objective = validation["mean_objective_per_order"]
                best_episode = episode_index + 1
                best_actor_state = copy.deepcopy(actor.state_dict())
                best_critic_state = copy.deepcopy(critic.state_dict())
    environment.close()
    actor.load_state_dict(best_actor_state)
    critic.load_state_dict(best_critic_state)
    return critic, {
        "episodes": episode_rows,
        "updates": update_rows,
        "gamma": gamma,
        "candidate_count": candidate_count,
        "fy_samples": fy_samples,
        "target_copy": "after_each_episode" if critic_target == "td" else "unused",
        "critic_target": critic_target,
        "normalize_candidate_advantages": normalize_candidate_advantages,
        "critic_kind": critic_kind,
        "feature_count": feature_count,
        "feature_schema": feature_schema,
        "decoder": decoder_config,
        "sigma_target": sigma_target,
        "validation_checkpoints": validation_rows,
        "selected_episode": best_episode,
        "selected_validation_objective": best_objective,
        "reward_power": reward_power,
        "sla_threshold_s": sla_threshold_s,
        "objective_scale": objective_scale,
        "checkpoint_episodes": checkpoint_episodes,
        "training_instance_order": list(instance_ids[:episodes]),
        "route_cost_scale": route_cost_scale,
        "max_reward_identity_error": max(
            row["reward_identity_error"] for row in episode_rows
        ),
        "best_actor_critic_consistent": True,
    }


def _actor_hidden(model: dict) -> int:
    return int(model["hidden"])


def run_training(cfg: dict, output_dir: Path, tracking_run=None) -> dict[str, object]:
    """Run the supported generated-data SRL training workflow."""
    torch.manual_seed(int(cfg["seed"]))
    data_spec = dict(cfg["data"])
    splits = generated_instance_splits(data_spec)
    decoder_name = str(cfg["decoder"]["name"])
    if bool(cfg.get("progress", True)):
        print(
            f"Structured-RL training: decoder={decoder_name}, "
            f"feature_schema={cfg['model']['feature_schema']}, "
            f"objective={cfg['objective']['name']}, "
            f"{len(splits['train'])} train / {len(splits['validation'])} "
            f"validation instances, {int(cfg['learner']['episodes'])} episodes.",
            flush=True,
        )
        print(
            "Running initial validation before episode 1; "
            "this selects checkpoints without using the test split.",
            flush=True,
        )
    write_json(output_dir / "dataset_manifest.json", generated_manifest(data_spec))
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
        "decoder": decoder_name,
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
        feature_schema=str(model["feature_schema"]),
        actor_hidden=_actor_hidden(model),
        episode_kwargs=episode_kwargs,
        metric_callback=metric_callback(tracking_run),
        show_progress=bool(cfg.get("progress", True)),
    )
    checkpoint = build_checkpoint(
        actor=actor,
        critic=critic,
        critic_kind=str(model["critic"]),
        feature_schema=str(model["feature_schema"]),
        feature_count=feature_count,
        actor_hidden=_actor_hidden(model),
        decoder_config=training["decoder"],
        objective=dict(objective),
        objective_scale=objective_scale,
        data_spec=data_spec,
        selected_episode=training["selected_episode"],
        route_cost_scale=float(training.get("route_cost_scale", 0.0)),
    )
    selected_path = output_dir / "best.pt"
    save_checkpoint(selected_path, checkpoint)
    result = {
        "mode": "train",
        "status": "complete",
        "decoder": decoder_name,
        "selected_checkpoint": str(selected_path),
        "selected_episode": training["selected_episode"],
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
            f"({decoder_name} decoder) and wrote {selected_path}.",
            flush=True,
        )
    return result
