from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from scenarios.scenario_henn_rl.critic_audit_experiment import run_critic_audit
from scenarios.scenario_henn_rl.critic_calibration_experiment import (
    run_critic_calibration,
)
from scenarios.scenario_henn_rl.critic_coverage_experiment import (
    run_critic_coverage,
)
from scenarios.scenario_henn_rl.critic_memorization_experiment import (
    run_critic_overfit,
)
from scenarios.scenario_henn_rl.candidate_feature_experiment import (
    run_candidate_feature_study,
)
from scenarios.scenario_henn_rl.experiment_support import _ids_for_mode, _write_json
from scenarios.scenario_henn_rl.objective_experiments import (
    run_comparison,
    run_objective_study,
)
from scenarios.scenario_henn_rl.policy_experiments import run_policy_experiment


def run(cfg: DictConfig) -> dict[str, object]:
    torch.manual_seed(int(cfg.experiment.seed))
    output_dir = Path(cfg.experiment.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = _ids_for_mode(cfg)
    mode = str(cfg.experiment.mode)
    if mode == "critic_calibration":
        return run_critic_calibration(cfg, splits, output_dir)
    if mode == "critic_overfit":
        return run_critic_overfit(cfg, splits, output_dir)
    if mode == "critic_coverage":
        return run_critic_coverage(cfg, splits, output_dir)
    if mode == "candidate_features":
        return run_candidate_feature_study(cfg, splits, output_dir)
    if mode == "critic_audit":
        return run_critic_audit(cfg, splits, output_dir)
    if mode == "objective_study":
        return run_objective_study(cfg, splits, output_dir)
    if mode == "comparison":
        result = run_comparison(cfg, splits)
        _write_json(output_dir / "result.json", result)
        return result
    return run_policy_experiment(cfg, splits, output_dir)


@hydra.main(
    version_base="1.3",
    config_path="config",
    config_name="structured_batching_config",
)
def main(cfg: DictConfig) -> None:
    result = run(cfg)
    print(
        json.dumps(
            {
                "status": result.get("status", "complete"),
                "mode": result["mode"],
            }
        )
    )


if __name__ == "__main__":
    main()
