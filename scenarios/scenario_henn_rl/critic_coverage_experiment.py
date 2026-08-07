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


def run_critic_coverage(
    cfg: DictConfig, splits: dict[str, list[str]], output_dir: Path
) -> dict[str, object]:
    import torch

    settings = cfg.critic_coverage
    source_dir = Path(settings.source_dir).resolve()
    source = json.loads((source_dir / "result.json").read_text(encoding="utf-8"))
    actor_path = source_dir / "pilot" / "selected.pt"
    actor = OrderScoreActor()
    actor.load_state_dict(
        torch.load(actor_path, map_location="cpu", weights_only=True)
    )
    prior_audit = source["fallback"]["training_audit"]
    validation_audit = source["pilot"]["audit"]
    power = float(cfg.critic_calibration.power)
    if float(prior_audit["reward_power"]) != power:
        raise ValueError("Coverage source uses a different reward power")

    base_ids = [str(value) for value in settings.base_instance_ids]
    additional_ids = [str(value) for value in settings.additional_instance_ids]
    used_ids = set(base_ids) | set(additional_ids)
    if used_ids - set(splits["train"]):
        raise ValueError("Coverage labels must use only training instances")
    if set(base_ids) & set(additional_ids):
        raise ValueError("Base and additional coverage instances overlap")
    if {
        row["instance_id"] for row in prior_audit["states"]
    } != set(base_ids):
        raise ValueError("Base coverage instances do not match prior labels")

    result = {
        "mode": "critic_coverage",
        "status": "generating_labels",
        "seed": int(cfg.experiment.seed),
        "source": str(source_dir),
        "test_split_used": False,
        "arms": {},
    }
    started = time.perf_counter()
    dataset_setting = settings.get("dataset_path")
    dataset_path = (
        Path(str(dataset_setting)).resolve()
        if dataset_setting
        else output_dir / "counterfactual_dataset.json"
    )
    if dataset_setting:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    else:
        label_critic = PairwiseStructuredCritic()
        outer = audit_structured_critic(
            actor,
            label_critic,
            base_ids,
            reward_power=power,
            comparison_powers=[power],
            state_quantiles=[float(value) for value in settings.outer_quantiles],
            candidate_count=int(cfg.srl.candidate_count),
            sigma=float(cfg.srl.sigma_target),
            temperature=1.0,
            seed=int(cfg.experiment.seed) + 3001,
            normalize_advantages=True,
        )
        additional = audit_structured_critic(
            actor,
            label_critic,
            additional_ids,
            reward_power=power,
            comparison_powers=[power],
            state_quantiles=[
                float(value) for value in settings.additional_quantiles
            ],
            candidate_count=int(cfg.srl.candidate_count),
            sigma=float(cfg.srl.sigma_target),
            temperature=1.0,
            seed=int(cfg.experiment.seed) + 3002,
            normalize_advantages=True,
        )
        states = [
            *prior_audit["states"],
            *outer["states"],
            *additional["states"],
        ]
        keys = [(row["instance_id"], row["decision_index"]) for row in states]
        if len(keys) != len(set(keys)):
            raise RuntimeError("Coverage state selection contains duplicates")
        dataset = {
            "reward_power": power,
            "states": states,
            "nested_state_counts": [int(value) for value in settings.state_counts],
            "base_instance_ids": base_ids,
            "additional_instance_ids": additional_ids,
            "candidate_labels": sum(
                len(row["candidate_rows"]) for row in states
            ),
        }
        _write_json(dataset_path, dataset)
    state_counts = [int(value) for value in settings.state_counts]
    if len(dataset["states"]) != state_counts[-1]:
        raise RuntimeError(
            f"Expected {state_counts[-1]} coverage states, got {len(dataset['states'])}"
        )
    result["dataset"] = {
        "path": str(dataset_path),
        "states": len(dataset["states"]),
        "candidate_labels": dataset["candidate_labels"],
        "instance_ids": sorted(used_ids),
    }
    result["status"] = "fitting"
    _write_json(output_dir / "result.json", result)

    worker_settings = {
        "epochs": int(settings.epochs),
        "learning_rate": float(settings.learning_rate),
    }
    jobs = [
        (
            str(dataset_path),
            str(actor_path),
            size,
            worker_settings,
            str(output_dir),
            int(cfg.experiment.seed),
        )
        for size in state_counts
    ]
    with ProcessPoolExecutor(max_workers=int(settings.workers)) as executor:
        fitted = list(executor.map(_critic_coverage_fit_job, jobs))

    result["status"] = "validating"
    for arm in fitted:
        critic = PairwiseStructuredCritic()
        critic.load_state_dict(
            torch.load(arm["model"], map_location="cpu", weights_only=True)
        )
        validation = score_counterfactual_audit(
            actor,
            critic,
            validation_audit,
            temperature=float(cfg.critic_calibration.temperature),
            normalize_advantages=True,
        )
        arm["validation"] = validation
        arm["gate"] = _critic_transfer_gate(
            validation["summary"], cfg.critic_calibration
        )
        result["arms"][str(arm["state_count"])] = arm
        _write_json(output_dir / "result.json", result)
    result["status"] = "complete"
    result["any_gate_passed"] = any(
        arm["gate"]["passed"] for arm in result["arms"].values()
    )
    result["wall_time_s"] = time.perf_counter() - started
    _write_json(output_dir / "result.json", result)
    return result
