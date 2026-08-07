from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scenarios.scenario_henn.loader import load_manifest
from scenarios.scenario_henn_rl.environment import (
    HENN_DIR,
    ReleaseTimingEnv,
)


def instance_splits() -> dict[str, list[str]]:
    """Return a deterministic 32/16/16 split stratified by family and size."""
    manifest = load_manifest(HENN_DIR / "reproduction" / "manifest.yaml")
    groups: dict[tuple[str, int], list[str]] = {}
    for instance_id in manifest:
        _, family, size, _ = instance_id.split("_")
        groups.setdefault((family, int(size)), []).append(instance_id)
    result = {"train": [], "validation": [], "test": []}
    for group_index, key in enumerate(sorted(groups)):
        values = sorted(groups[key], key=lambda value: int(value.rsplit("_", 1)[1]))
        if len(values) != 4:
            raise ValueError(f"Expected four arrival variants for {key}")
        test_index = group_index % 4
        validation_index = (group_index + 1) % 4
        for index, value in enumerate(values):
            split = (
                "test"
                if index == test_index
                else "validation"
                if index == validation_index
                else "train"
            )
            result[split].append(value)
    return result


def run_heuristic_episode(
    instance_id: str,
    fill_threshold: float,
    max_age_s: float,
) -> dict[str, object]:
    env = ReleaseTimingEnv([instance_id])
    observation, _ = env.reset(seed=0, options={"instance_id": instance_id})
    del observation
    terminated = False
    episode_return = 0.0
    steps = 0
    info = {}
    while not terminated:
        action = env.configured_policy_action(
            waiting_policy="fill_or_age",
            fill_threshold=fill_threshold,
            max_age_s=max_age_s,
        )
        _, reward, terminated, _, info = env.step(action)
        episode_return += reward
        steps += 1
    env.close()
    return {
        **info,
        "return": episode_return,
        "steps": steps,
        "mean_flow_time": float(info["total_flow_time"]) / len(env.arrivals),
    }


def _heuristic_job(job):
    return run_heuristic_episode(*job)


def evaluate_heuristic(
    instance_ids: list[str],
    *,
    fill_threshold: float,
    max_age_s: float,
    workers: int = 1,
) -> dict[str, object]:
    jobs = [
        (instance_id, fill_threshold, max_age_s)
        for instance_id in instance_ids
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            episodes = list(pool.map(_heuristic_job, jobs))
    else:
        episodes = [_heuristic_job(job) for job in jobs]
    return summarize_episodes(episodes)


def summarize_episodes(episodes: list[dict[str, object]]) -> dict[str, object]:
    flows = np.asarray([row["mean_flow_time"] for row in episodes], dtype=float)
    return {
        "instances": len(episodes),
        "mean_order_flow_time": float(np.mean(flows)),
        "median_order_flow_time": float(np.median(flows)),
        "mean_total_distance": float(
            np.mean([row["total_distance"] for row in episodes])
        ),
        "mean_completed_tours": float(
            np.mean([row["completed_tours"] for row in episodes])
        ),
        "wait_fraction": sum(row["wait_actions"] for row in episodes)
        / max(
            1,
            sum(
                row["wait_actions"] + row["dispatch_actions"]
                for row in episodes
            ),
        ),
        "episodes": episodes,
    }


def tune_heuristic(
    train_ids: list[str],
    validation_ids: list[str],
    *,
    fill_thresholds: list[float],
    max_ages_s: list[float],
    workers: int,
) -> dict[str, object]:
    rows = []
    for fill in fill_thresholds:
        for age in max_ages_s:
            validation = evaluate_heuristic(
                validation_ids,
                fill_threshold=fill,
                max_age_s=age,
                workers=workers,
            )
            rows.append(
                {
                    "fill_threshold": fill,
                    "max_age_s": age,
                    "validation_mean_order_flow_time": validation[
                        "mean_order_flow_time"
                    ],
                }
            )
    best = min(rows, key=lambda row: row["validation_mean_order_flow_time"])
    train = evaluate_heuristic(
        train_ids,
        fill_threshold=best["fill_threshold"],
        max_age_s=best["max_age_s"],
        workers=workers,
    )
    return {"candidates": rows, "selected": best, "train": train}


def _counterfactual_instance(job) -> dict[str, object]:
    instance_id, fill_threshold, max_age_s = job
    started = time.perf_counter()
    reference = ReleaseTimingEnv([instance_id])
    observation, _ = reference.reset(
        seed=0, options={"instance_id": instance_id}
    )
    observations = []
    actions = []
    accrued = []
    normalizers = []
    terminated = False
    info = {}
    while not terminated:
        observations.append(observation.tolist())
        accrued.append(reference.accrued_flow_time())
        normalizers.append(reference.reward_normalizer)
        action = reference.configured_policy_action(
            waiting_policy="fill_or_age",
            fill_threshold=fill_threshold,
            max_age_s=max_age_s,
        )
        actions.append(action)
        observation, _, terminated, _, info = reference.step(action)
    reference_flow = float(info["total_flow_time"])
    reference.close()

    branch = ReleaseTimingEnv([instance_id])
    rows = []
    replay_steps = 0
    for index, reference_action in enumerate(actions):
        branch_observation, _ = branch.reset(
            seed=0, options={"instance_id": instance_id}
        )
        for action in actions[:index]:
            branch_observation, _, terminated, _, _ = branch.step(action)
            replay_steps += 1
            if terminated:
                raise RuntimeError("Reference prefix terminated early")
        if not np.allclose(branch_observation, observations[index], atol=1e-6):
            raise RuntimeError(
                f"Replay mismatch for {instance_id} at decision {index}"
            )
        alternative_action = 1 - reference_action
        _, _, terminated, _, alternative_info = branch.step(alternative_action)
        replay_steps += 1
        while not terminated:
            action = branch.configured_policy_action(
                waiting_policy="fill_or_age",
                fill_threshold=fill_threshold,
                max_age_s=max_age_s,
            )
            _, _, terminated, _, alternative_info = branch.step(action)
            replay_steps += 1
        alternative_flow = float(alternative_info["total_flow_time"])
        flow_by_action = {
            reference_action: reference_flow,
            alternative_action: alternative_flow,
        }
        q_wait = -(flow_by_action[0] - accrued[index]) / normalizers[index]
        q_dispatch = -(flow_by_action[1] - accrued[index]) / normalizers[index]
        advantage = q_dispatch - q_wait
        if abs(advantage) <= 1e-12:
            continue
        rows.append(
            {
                "instance_id": instance_id,
                "decision_index": index,
                "observation": observations[index],
                "reference_action": reference_action,
                "preferred_action": int(advantage > 0.0),
                "q_wait": q_wait,
                "q_dispatch": q_dispatch,
                "advantage_dispatch": advantage,
                "flow_wait": flow_by_action[0],
                "flow_dispatch": flow_by_action[1],
            }
        )
    branch.close()
    return {
        "instance_id": instance_id,
        "reference_flow": reference_flow,
        "reference_decisions": len(actions),
        "replay_steps": replay_steps,
        "wall_time_s": time.perf_counter() - started,
        "labels": rows,
    }


def generate_counterfactuals(
    instance_ids: list[str],
    *,
    fill_threshold: float,
    max_age_s: float,
    workers: int,
) -> dict[str, object]:
    jobs = [
        (instance_id, fill_threshold, max_age_s)
        for instance_id in instance_ids
    ]
    started = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_counterfactual_instance, jobs))
    else:
        results = [_counterfactual_instance(job) for job in jobs]
    labels = []
    summaries = []
    for result in results:
        labels.extend(result["labels"])
        summaries.append(
            {key: value for key, value in result.items() if key != "labels"}
        )
    return {
        "wall_time_s": time.perf_counter() - started,
        "workers": workers,
        "instances": summaries,
        "labels": labels,
    }


def make_vec_env(instance_ids: list[str]):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv([lambda: Monitor(ReleaseTimingEnv(instance_ids))])


def make_ppo(
    instance_ids: list[str],
    seed: int,
    *,
    n_steps: int = 256,
    gae_lambda: float = 0.95,
):
    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy",
        make_vec_env(instance_ids),
        seed=seed,
        n_steps=n_steps,
        batch_size=64,
        gamma=1.0,
        gae_lambda=gae_lambda,
        learning_rate=3e-4,
        device="cpu",
        verbose=0,
    )
    return model


def reference_decisions(
    instance_ids: list[str],
    *,
    fill_threshold: float,
    max_age_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect states and decisions from the fixed release heuristic."""
    observations = []
    actions = []
    env = ReleaseTimingEnv(instance_ids)
    for instance_id in instance_ids:
        observation, _ = env.reset(
            seed=0, options={"instance_id": instance_id}
        )
        terminated = False
        while not terminated:
            action = env.configured_policy_action(
                waiting_policy="fill_or_age",
                fill_threshold=fill_threshold,
                max_age_s=max_age_s,
            )
            observations.append(observation)
            actions.append(action)
            observation, _, terminated, _, _ = env.step(action)
    env.close()
    return np.asarray(observations), np.asarray(actions)


def policy_decision_summary(
    model,
    observations: np.ndarray,
    reference_actions: np.ndarray,
) -> dict[str, float]:
    """Describe PPO's dispatch probabilities on a stable state panel."""
    import torch

    tensor = torch.as_tensor(
        observations, dtype=torch.float32, device=model.device
    )
    model.policy.set_training_mode(False)
    with torch.no_grad():
        probabilities = (
            model.policy.get_distribution(tensor)
            .distribution.probs[:, 1]
            .detach()
            .cpu()
            .numpy()
        )
    waits = reference_actions == 0
    dispatches = reference_actions == 1
    predictions = probabilities >= 0.5
    wait_probabilities = probabilities[waits]
    dispatch_probabilities = probabilities[dispatches]
    return {
        "states": int(len(probabilities)),
        "reference_wait_fraction": float(np.mean(waits)),
        "mean_dispatch_probability": float(np.mean(probabilities)),
        "mean_dispatch_probability_on_reference_wait": float(
            np.mean(probabilities[waits]) if np.any(waits) else np.nan
        ),
        "mean_dispatch_probability_on_reference_dispatch": float(
            np.mean(probabilities[dispatches]) if np.any(dispatches) else np.nan
        ),
        "deterministic_dispatch_fraction": float(np.mean(predictions)),
        "agreement_with_reference": float(
            np.mean(predictions == reference_actions)
        ),
        "dispatch_probability_quantiles_on_reference_wait": [
            float(value)
            for value in np.quantile(wait_probabilities, [0.0, 0.5, 1.0])
        ],
        "dispatch_probability_quantiles_on_reference_dispatch": [
            float(value)
            for value in np.quantile(
                dispatch_probabilities, [0.0, 0.5, 1.0]
            )
        ],
    }


def pretrain_policy(
    model,
    labels: list[dict[str, object]],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    advantage_weighted: bool = False,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    observations = torch.as_tensor(
        np.asarray([row["observation"] for row in labels]),
        dtype=torch.float32,
        device=model.device,
    )
    actions = torch.as_tensor(
        [row["preferred_action"] for row in labels],
        dtype=torch.long,
        device=model.device,
    )
    values = torch.as_tensor(
        [max(row["q_wait"], row["q_dispatch"]) for row in labels],
        dtype=torch.float32,
        device=model.device,
    )
    weights = torch.ones(len(labels), dtype=torch.float32, device=model.device)
    if advantage_weighted:
        weights = torch.as_tensor(
            [abs(row["advantage_dispatch"]) for row in labels],
            dtype=torch.float32,
            device=model.device,
        )
        weights /= weights.mean().clamp_min(1e-12)
    optimizer = model.policy.optimizer
    original_rates = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.policy.set_training_mode(True)
    last_policy_loss = 0.0
    last_value_loss = 0.0
    for _ in range(epochs):
        permutation = torch.randperm(len(labels), generator=generator)
        for start in range(0, len(labels), batch_size):
            indices = permutation[start : start + batch_size].to(model.device)
            distribution = model.policy.get_distribution(observations[indices])
            policy_loss = -(
                distribution.log_prob(actions[indices]) * weights[indices]
            ).mean()
            predicted = model.policy.predict_values(observations[indices]).flatten()
            value_loss = functional.mse_loss(predicted, values[indices])
            loss = policy_loss + 0.5 * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            last_policy_loss = float(policy_loss.detach().cpu())
            last_value_loss = float(value_loss.detach().cpu())
    for group, rate in zip(optimizer.param_groups, original_rates):
        group["lr"] = rate
    model.policy.set_training_mode(False)
    with torch.no_grad():
        predicted = model.policy.get_distribution(observations).get_actions(
            deterministic=True
        )
        accuracy = float((predicted == actions).float().mean().cpu())
    return {
        "labels": len(labels),
        "epochs": epochs,
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "training_accuracy": accuracy,
        "advantage_weighted": advantage_weighted,
    }


def evaluate_model(
    model,
    instance_ids: list[str],
    *,
    deterministic: bool = True,
    dispatch_threshold: float | None = None,
) -> dict[str, object]:
    import torch

    episodes = []
    env = ReleaseTimingEnv(instance_ids)
    for instance_id in instance_ids:
        observation, _ = env.reset(
            seed=0, options={"instance_id": instance_id}
        )
        terminated = False
        episode_return = 0.0
        info = {}
        steps = 0
        while not terminated:
            if dispatch_threshold is None:
                action, _ = model.predict(
                    observation, deterministic=deterministic
                )
            else:
                tensor = torch.as_tensor(
                    observation[None],
                    dtype=torch.float32,
                    device=model.device,
                )
                with torch.no_grad():
                    dispatch_probability = float(
                        model.policy.get_distribution(tensor)
                        .distribution.probs[0, 1]
                        .cpu()
                    )
                action = int(dispatch_probability >= dispatch_threshold)
            observation, reward, terminated, _, info = env.step(
                int(np.asarray(action).item())
            )
            episode_return += reward
            steps += 1
        episodes.append(
            {
                **info,
                "return": episode_return,
                "steps": steps,
                "mean_flow_time": float(info["total_flow_time"])
                / len(env.arrivals),
            }
        )
    env.close()
    return summarize_episodes(episodes)
