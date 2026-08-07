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


def run_critic_audit(
    cfg: DictConfig, splits: dict[str, list[str]], output_dir: Path
) -> dict[str, object]:
    import torch

    settings = cfg.critic_audit
    seed = int(cfg.experiment.seed)
    selected_episode = int(settings.selected_episode)
    powers = [float(value) for value in settings.powers]
    instance_ids = [str(value) for value in settings.instance_ids]
    unknown = set(instance_ids) - set(splits["validation"])
    if unknown:
        raise ValueError(
            f"Critic audit instances are not in validation: {sorted(unknown)}"
        )
    source_dir = Path(str(settings.source_dir)).resolve()
    source = json.loads((source_dir / "result.json").read_text(encoding="utf-8"))
    episode_order = _training_order(
        splits["train"], seed=seed, passes=int(cfg.objective_study.passes)
    )
    if selected_episode > len(episode_order):
        raise ValueError("Critic reconstruction exceeds the training order")
    result = {
        "mode": "critic_audit",
        "status": "running",
        "seed": seed,
        "source_dir": str(source_dir),
        "selected_episode": selected_episode,
        "instance_ids": instance_ids,
        "state_quantiles": [float(value) for value in settings.state_quantiles],
        "powers": powers,
        "arms": {},
        "configuration": OmegaConf.to_container(cfg, resolve=True),
    }
    _write_json(output_dir / "result.json", result)
    started = time.perf_counter()
    for power in powers:
        key = str(power)
        if source["arms"][key]["training"]["selected_episode"] != selected_episode:
            raise RuntimeError(
                f"Source p={key} did not select episode {selected_episode}"
            )
        torch.manual_seed(seed)
        actor = OrderScoreActor()
        critic, reconstruction = train_structured_rl(
            actor,
            episode_order[:selected_episode],
            episodes=selected_episode,
            updates_per_episode=int(cfg.srl.updates_per_episode),
            batch_size=int(cfg.srl.batch_size),
            replay_capacity=int(cfg.srl.replay_capacity),
            actor_learning_rate=float(cfg.srl.actor_learning_rate),
            critic_learning_rate=float(cfg.srl.critic_learning_rate),
            sigma_forward=float(cfg.srl.sigma_forward),
            sigma_target=float(cfg.srl.sigma_target),
            temperature=float(cfg.srl.temperature),
            candidate_count=int(cfg.srl.candidate_count),
            epsilon=float(cfg.srl.epsilon),
            fy_samples=int(cfg.srl.fy_samples),
            gamma=1.0,
            seed=seed,
            validation_ids=splits["validation"],
            checkpoint_episodes=[0, selected_episode],
            reward_power=power,
            location_features=True,
        )
        frozen = OrderScoreActor()
        frozen.load_state_dict(
            torch.load(
                source["arms"][key]["selected_model"],
                map_location="cpu",
                weights_only=True,
            )
        )
        reproduced_digest = _state_digest(actor)
        frozen_digest = _state_digest(frozen)
        if reproduced_digest != frozen_digest:
            raise RuntimeError(f"Reconstructed p={key} actor does not match source")
        audit = audit_structured_critic(
            actor,
            critic,
            instance_ids,
            reward_power=power,
            comparison_powers=powers,
            state_quantiles=[float(value) for value in settings.state_quantiles],
            candidate_count=int(cfg.srl.candidate_count),
            sigma=float(cfg.srl.sigma_target),
            temperature=float(cfg.srl.temperature),
            seed=seed + int(round(power * 1000)),
        )
        result["arms"][key] = {
            "actor_digest": reproduced_digest,
            "source_actor_digest": frozen_digest,
            "reconstruction": {
                "selected_episode": reconstruction["selected_episode"],
                "max_reward_identity_error": reconstruction[
                    "max_reward_identity_error"
                ],
            },
            "audit": audit,
        }
        _write_json(output_dir / "result.json", result)
    result["status"] = "complete"
    result["wall_time_s"] = time.perf_counter() - started
    _write_json(output_dir / "result.json", result)
    return result
