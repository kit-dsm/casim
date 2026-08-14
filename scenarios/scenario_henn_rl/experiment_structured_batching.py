from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from scenarios.scenario_henn_rl.structured.audit import run_audit
from scenarios.scenario_henn_rl.structured.data import run_generation
from scenarios.scenario_henn_rl.structured.evaluation import run_evaluation
from scenarios.scenario_henn_rl.structured.results import write_json
from scenarios.scenario_henn_rl.structured.tracking import (
    log_artifact,
    start_tracking,
)
from scenarios.scenario_henn_rl.structured.training import run_training


def run(cfg: DictConfig) -> dict[str, object]:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    output_dir = Path(resolved["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.json"
    write_json(config_path, resolved)
    action = str(resolved["experiment"]["action"])
    decoder_name = str(resolved.get("decoder", {}).get("name", "knapsack"))
    objective_name = str(resolved.get("objective", {}).get("name", "unknown"))
    print(
        f"Structured-batching run: action={action}, decoder={decoder_name}, "
        f"objective={objective_name}, output_dir={output_dir}",
        flush=True,
    )
    if resolved["tracking"].get("enabled") and bool(
        resolved.get("progress", True)
    ):
        print("Initializing W&B tracking...", flush=True)
    tracking_run = start_tracking(resolved["tracking"], resolved)
    if tracking_run is not None and bool(resolved.get("progress", True)):
        url = getattr(tracking_run, "url", None)
        print(
            f"W&B tracking enabled{f': {url}' if url else '.'}",
            flush=True,
        )
    try:
        if action == "generate":
            result = run_generation(resolved["data"], output_dir)
            result = {"mode": "generate", "status": "complete", **result}
            result_path = output_dir / "result.json"
            write_json(result_path, result)
            log_artifact(
                tracking_run,
                output_dir,
                [config_path, result_path, output_dir / "dataset_manifest.json"],
            )
            return result
        if action == "train":
            return run_training(resolved, output_dir, tracking_run)
        if action == "evaluate":
            return run_evaluation(resolved, output_dir, tracking_run)
        if action == "audit":
            return run_audit(resolved, output_dir, tracking_run)
        raise ValueError(f"Unknown structured experiment action: {action}")
    finally:
        if tracking_run is not None:
            tracking_run.finish()


@hydra.main(version_base="1.3", config_path="config", config_name="structured")
def main(cfg: DictConfig) -> None:
    result = run(cfg)
    print(json.dumps({"status": result["status"], "mode": result["mode"]}))


if __name__ == "__main__":
    main()
