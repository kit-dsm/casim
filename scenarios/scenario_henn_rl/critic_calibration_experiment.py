from __future__ import annotations

import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from scenarios.scenario_henn_rl.counterfactuals import (
    audit_structured_critic,
    score_counterfactual_audit,
)
from scenarios.scenario_henn_rl.critic_fitting import (
    calibrate_from_counterfactual_audit,
    overfit_counterfactual_critic,
)
from scenarios.scenario_henn_rl.experiment_support import (
    _critic_calibration_gate,
    _critic_coverage_fit_job,
    _critic_transfer_gate,
    _paired_objective_delta,
    _state_digest,
    _training_order,
    _write_json,
)
from scenarios.scenario_henn_rl.structured_evaluation import evaluate_structured_actor
from scenarios.scenario_henn_rl.structured_policy import (
    OrderScoreActor,
    PairwiseStructuredCritic,
)
from scenarios.scenario_henn_rl.structured_training import train_structured_rl


def run_critic_calibration(
    cfg: DictConfig, splits: dict[str, list[str]], output_dir: Path
) -> dict[str, object]:
    import torch

    settings = cfg.critic_calibration
    seed = int(cfg.experiment.seed)
    power = float(settings.power)
    pilot_episodes = int(settings.pilot_episodes)
    episode_order = _training_order(
        splits["train"], seed=seed, passes=int(cfg.objective_study.passes)
    )
    instance_ids = [str(value) for value in settings.instance_ids]
    unknown = set(instance_ids) - set(splits["validation"])
    if unknown:
        raise ValueError(
            f"Calibration audit instances are not validation data: {sorted(unknown)}"
        )
    counterfactual_ids = [
        str(value) for value in settings.counterfactual_instance_ids
    ]
    unknown = set(counterfactual_ids) - set(splits["train"])
    if unknown:
        raise ValueError(
            "Counterfactual calibration instances are not training data: "
            f"{sorted(unknown)}"
        )
    if set(counterfactual_ids) & set(instance_ids):
        raise ValueError("Counterfactual training and validation instances overlap")

    def train_arm(
        arm_power: float,
        episodes: int,
        checkpoints: list[int],
        directory: Path,
    ):
        torch.manual_seed(seed)
        actor = OrderScoreActor()
        started = time.perf_counter()
        critic, training = train_structured_rl(
            actor,
            episode_order[:episodes],
            episodes=episodes,
            updates_per_episode=int(cfg.srl.updates_per_episode),
            batch_size=int(cfg.srl.batch_size),
            replay_capacity=int(cfg.srl.replay_capacity),
            actor_learning_rate=float(cfg.srl.actor_learning_rate),
            critic_learning_rate=float(cfg.srl.critic_learning_rate),
            sigma_forward=float(cfg.srl.sigma_forward),
            sigma_target=float(cfg.srl.sigma_target),
            temperature=float(settings.temperature),
            candidate_count=int(cfg.srl.candidate_count),
            epsilon=float(cfg.srl.epsilon),
            fy_samples=int(cfg.srl.fy_samples),
            gamma=1.0,
            seed=seed,
            validation_ids=splits["validation"],
            checkpoint_episodes=checkpoints,
            checkpoint_dir=directory / "checkpoints",
            reward_power=arm_power,
            location_features=True,
            critic_target="return_to_go",
            normalize_candidate_advantages=True,
        )
        training["wall_time_s"] = time.perf_counter() - started
        model_path = directory / "selected.pt"
        torch.save(actor.state_dict(), model_path)
        return actor, critic, training, model_path

    result = {
        "mode": "critic_calibration",
        "status": "pilot",
        "seed": seed,
        "splits": splits,
        "training_instance_order": episode_order,
        "configuration": OmegaConf.to_container(cfg, resolve=True),
    }
    _write_json(output_dir / "result.json", result)
    started = time.perf_counter()
    actor, critic, training, model_path = train_arm(
        power,
        pilot_episodes,
        [0, pilot_episodes],
        output_dir / "pilot",
    )
    audit = audit_structured_critic(
        actor,
        critic,
        instance_ids,
        reward_power=power,
        comparison_powers=[1.0, power],
        state_quantiles=[float(value) for value in settings.state_quantiles],
        candidate_count=int(cfg.srl.candidate_count),
        sigma=float(cfg.srl.sigma_target),
        temperature=float(settings.temperature),
        seed=seed + int(round(power * 1000)),
        normalize_advantages=True,
    )
    gate = _critic_calibration_gate(audit["summary"], settings)
    result["pilot"] = {
        "model": str(model_path),
        "training": training,
        "audit": audit,
    }
    result["initial_gate"] = gate
    _write_json(output_dir / "result.json", result)
    if not gate["passed"]:
        fallback_started = time.perf_counter()
        training_audit = audit_structured_critic(
            actor,
            critic,
            counterfactual_ids,
            reward_power=power,
            comparison_powers=[power],
            state_quantiles=[
                float(value) for value in settings.counterfactual_state_quantiles
            ],
            candidate_count=int(cfg.srl.candidate_count),
            sigma=float(cfg.srl.sigma_target),
            temperature=float(settings.temperature),
            seed=seed + int(round(power * 1000)) + 1,
            normalize_advantages=True,
        )
        fit = calibrate_from_counterfactual_audit(
            actor,
            critic,
            training_audit,
            critic_epochs=int(settings.counterfactual_critic_epochs),
            actor_epochs=int(settings.counterfactual_actor_epochs),
            critic_learning_rate=float(settings.counterfactual_critic_learning_rate),
            actor_learning_rate=float(settings.counterfactual_actor_learning_rate),
            epsilon=float(cfg.srl.epsilon),
            fy_samples=int(cfg.srl.fy_samples),
            seed=seed + 3,
        )
        fallback_model = output_dir / "fallback" / "selected.pt"
        fallback_critic = output_dir / "fallback" / "critic.pt"
        fallback_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(actor.state_dict(), fallback_model)
        torch.save(critic.state_dict(), fallback_critic)
        fallback_audit = audit_structured_critic(
            actor,
            critic,
            instance_ids,
            reward_power=power,
            comparison_powers=[1.0, power],
            state_quantiles=[float(value) for value in settings.state_quantiles],
            candidate_count=int(cfg.srl.candidate_count),
            sigma=float(cfg.srl.sigma_target),
            temperature=float(settings.temperature),
            seed=seed + int(round(power * 1000)),
            normalize_advantages=True,
        )
        gate = _critic_calibration_gate(fallback_audit["summary"], settings)
        result["fallback"] = {
            "model": str(fallback_model),
            "critic": str(fallback_critic),
            "training_instances": counterfactual_ids,
            "training_audit": training_audit,
            "fit": fit,
            "validation_audit": fallback_audit,
            "wall_time_s": time.perf_counter() - fallback_started,
        }
        result["gate"] = gate
        _write_json(output_dir / "result.json", result)
        if not gate["passed"]:
            result["status"] = "gate_failed"
            result["wall_time_s"] = time.perf_counter() - started
            _write_json(output_dir / "result.json", result)
            return result
    else:
        result["gate"] = gate

    powers = [float(value) for value in settings.full_powers]
    full = {"arms": {}, "test": {}}
    selected_actors = {}
    checkpoints = [int(value) for value in cfg.objective_study.checkpoints]
    for arm_power in powers:
        key = str(arm_power)
        selected, _, arm_training, selected_path = train_arm(
            arm_power,
            len(episode_order),
            checkpoints,
            output_dir / "full" / f"p{key}",
        )
        selected_actors[key] = selected
        full["arms"][key] = {
            "model": str(selected_path),
            "training": arm_training,
        }
        _write_json(output_dir / "result.json", {**result, "full": full})
    for trained_power, selected in selected_actors.items():
        full["test"][trained_power] = {
            str(evaluation_power): evaluate_structured_actor(
                selected,
                splits["test"],
                reward_power=evaluation_power,
            )
            for evaluation_power in powers
        }
    full["p2_vs_p1"] = _paired_objective_delta(
        full["test"][str(power)][str(power)],
        full["test"]["1.0"][str(power)],
        power,
    )
    result["full"] = full
    result["status"] = "complete"
    result["wall_time_s"] = time.perf_counter() - started
    _write_json(output_dir / "result.json", result)
    return result
