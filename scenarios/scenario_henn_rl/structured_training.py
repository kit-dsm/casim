from __future__ import annotations

import copy
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from scenarios.scenario_henn_rl.structured_environment import StructuredBatchingEpisode
from scenarios.scenario_henn_rl.structured_evaluation import evaluate_structured_actor
from scenarios.scenario_henn_rl.structured_policy import (
    StructuredCritic,
    _copy_state,
    _features,
    _mask,
    critic_soft_target,
    decode_scores,
    discounted_returns,
    fenchel_young_loss,
)


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
    location_features: bool = True,
    critic_target: str = "td",
    normalize_candidate_advantages: bool = False,
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
    critic = StructuredCritic()
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
        instance_ids, reward_power=reward_power
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
            ),
        }
    ]
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
    for episode_index in range(episodes):
        instance_id = instance_ids[episode_index % len(instance_ids)]
        state = environment.reset(seed=seed + episode_index, instance_id=instance_id)
        done = False
        episode_return = 0.0
        policy_time = 0.0
        actor_time = 0.0
        oracle_time = 0.0
        decisions = 0
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
                    "mean_unique_candidates": float(
                        np.mean([row["unique_candidates"] for row in diagnostics])
                    ),
                    "mean_candidate_q_spread": float(
                        np.mean([row["candidate_q_spread"] for row in diagnostics])
                    ),
                    "mean_target_entropy": float(
                        np.mean([row["target_entropy"] for row in diagnostics])
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
            }
        )
        if (episode_index + 1) in checkpoint_episodes:
            validation = evaluate_structured_actor(
                actor,
                validation_ids,
                location_features=location_features,
                reward_power=reward_power,
            )
            row = {"episode": episode_index + 1, "evaluation": validation}
            if checkpoint_path is not None:
                path = checkpoint_path / f"episode_{episode_index + 1:03d}.pt"
                torch.save(actor.state_dict(), path)
                row["checkpoint"] = str(path)
            validation_rows.append(row)
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
        "validation_checkpoints": validation_rows,
        "selected_episode": best_episode,
        "selected_validation_objective": best_objective,
        "reward_power": reward_power,
        "checkpoint_episodes": checkpoint_episodes,
        "training_instance_order": list(instance_ids[:episodes]),
        "max_reward_identity_error": max(
            row["reward_identity_error"] for row in episode_rows
        ),
    }
