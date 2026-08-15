from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from casim.io_helpers import dump_json
from scenarios.scenario_henn_rl.data import (
    GeneratedHennDataLoader,
    generated_instance_splits,
    generated_manifest,
    run_generation,
)
from learning.structured_batching.evaluate import run_evaluation
from learning.structured_batching.tracking import log_artifact, start_tracking
from learning.structured_batching.train import run_training
from scenarios.scenario_henn_rl.runtime import build_environment


def run(cfg: DictConfig) -> dict[str, object]:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    output_dir = Path(resolved["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.json"
    dump_json(config_path, resolved)
    action = str(resolved["experiment"]["action"])
    decoder_name = str(resolved.get("decoder", {}).get("name", "knapsack"))
    print(
        f"Structured batching: action={action}, decoder={decoder_name}, "
        f"output_dir={output_dir}",
        flush=True,
    )
    tracking_run = start_tracking(resolved["tracking"], resolved, output_dir)
    try:
        if action == "generate":
            result = run_generation(resolved["data"], output_dir)
            result = {"mode": "generate", "status": "complete", **result}
            result_path = output_dir / "result.json"
            dump_json(result_path, result)
            log_artifact(tracking_run, output_dir, [config_path, result_path])
            return result
        data_spec = dict(resolved["data"])
        splits = generated_instance_splits(data_spec)
        manifest = generated_manifest(data_spec)
        loader = GeneratedHennDataLoader(data_spec)

        def environment_factory():
            return build_environment(data_loader=loader)

        if action == "train":
            return run_training(
                resolved,
                output_dir,
                splits=splits,
                dataset_manifest=manifest,
                environment_factory=environment_factory,
                tracking_run=tracking_run,
            )
        if action == "evaluate":
            return run_evaluation(
                resolved,
                output_dir,
                splits=splits,
                dataset_manifest=manifest,
                environment_factory=environment_factory,
                tracking_run=tracking_run,
            )
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
