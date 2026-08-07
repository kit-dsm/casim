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


def run_critic_overfit(
    cfg: DictConfig, splits: dict[str, list[str]], output_dir: Path
) -> dict[str, object]:
    import torch

    settings = cfg.critic_overfit
    source_dir = Path(settings.source_dir).resolve()
    source = json.loads((source_dir / "result.json").read_text(encoding="utf-8"))
    training_audit = source["fallback"]["training_audit"]
    training_ids = {
        row["instance_id"] for row in training_audit["states"]
    }
    if not training_ids <= set(splits["train"]):
        raise ValueError("Critic memorization data must contain only training instances")
    if training_ids & (set(splits["validation"]) | set(splits["test"])):
        raise ValueError("Critic memorization data overlap validation or test data")

    actor = OrderScoreActor()
    actor.load_state_dict(
        torch.load(
            source_dir / "pilot" / "selected.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    result = {
        "mode": "critic_overfit",
        "status": "running",
        "seed": int(cfg.experiment.seed),
        "source": str(source_dir),
        "training_instances": sorted(training_ids),
        "arms": {},
    }
    started = time.perf_counter()
    interaction_aware = bool(settings.interaction_aware)
    reused_model = settings.get("model_path")
    if reused_model:
        if not interaction_aware:
            raise ValueError("Model reuse is only supported for the interaction critic")
        model_path = Path(str(reused_model)).resolve()
        memorization = json.loads(
            (model_path.parent / "result.json").read_text(encoding="utf-8")
        )
        arm = memorization["arms"]["pairwise_interaction"]
        if not arm.get("memorized"):
            raise ValueError("Reused interaction critic did not pass memorization")
        critic = PairwiseStructuredCritic()
        critic.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        result["arms"]["pairwise_interaction"] = {
            **arm,
            "model": str(model_path),
            "reused": True,
        }
    else:
        arms = (
            [("pairwise_interaction", str(settings.interaction_loss), True)]
            if interaction_aware
            else [
                ("regression", "regression", False),
                ("pairwise_ranking", "pairwise_ranking", False),
            ]
        )
        for offset, (name, loss_kind, use_interactions) in enumerate(arms):
            critic, arm = overfit_counterfactual_critic(
                actor,
                training_audit,
                epochs=int(settings.epochs),
                learning_rate=float(settings.learning_rate),
                loss_kind=loss_kind,
                seed=int(cfg.experiment.seed) + offset,
                interaction_aware=use_interactions,
            )
            model_path = output_dir / f"{name}.pt"
            torch.save(critic.state_dict(), model_path)
            arm["model"] = str(model_path)
            result["arms"][name] = arm
            _write_json(output_dir / "result.json", result)

    thresholds = {
        "mean_spearman": float(settings.min_spearman),
        "top_accuracy": float(settings.min_top_accuracy),
    }
    for arm in result["arms"].values():
        arm["memorized"] = all(
            arm["final"][metric] >= threshold
            for metric, threshold in thresholds.items()
        )
    if interaction_aware:
        passed = result["arms"]["pairwise_interaction"]["memorized"]
        conclusion = (
            "interaction_aware_critic_passed_memorization_gate"
            if passed
            else "interaction_aware_critic_failed_memorization_gate"
        )
    else:
        regression_passed = result["arms"]["regression"]["memorized"]
        ranking_passed = result["arms"]["pairwise_ranking"]["memorized"]
        if regression_passed:
            conclusion = "representation_can_memorize_with_existing_regression"
        elif ranking_passed:
            conclusion = "representation_can_memorize_but_regression_loss_is_limiting"
        else:
            conclusion = "memorization_failed_with_current_representation_and_losses"

    if interaction_aware and passed and bool(settings.validate):
        validation_ids = [str(value) for value in settings.validation_instance_ids]
        if set(validation_ids) - set(splits["validation"]):
            raise ValueError("Critic transfer audit must use validation instances")
        validation_audit = audit_structured_critic(
            actor,
            critic,
            validation_ids,
            reward_power=float(cfg.critic_calibration.power),
            comparison_powers=[1.0, float(cfg.critic_calibration.power)],
            state_quantiles=[
                float(value) for value in settings.validation_state_quantiles
            ],
            candidate_count=int(cfg.srl.candidate_count),
            sigma=float(cfg.srl.sigma_target),
            temperature=float(cfg.critic_calibration.temperature),
            seed=int(cfg.experiment.seed)
            + int(round(float(cfg.critic_calibration.power) * 1000)),
            normalize_advantages=True,
        )
        transfer_gate = _critic_transfer_gate(
            validation_audit["summary"], cfg.critic_calibration
        )
        result["validation"] = {
            "audit": validation_audit,
            "gate": transfer_gate,
        }
        conclusion = (
            "interaction_aware_critic_passed_validation_transfer_gate"
            if transfer_gate["passed"]
            else "interaction_aware_critic_failed_validation_transfer_gate"
        )
    result.update(
        {
            "thresholds": thresholds,
            "conclusion": conclusion,
            "status": "complete",
            "wall_time_s": time.perf_counter() - started,
        }
    )
    _write_json(output_dir / "result.json", result)
    return result
