"""Optional Weights & Biases instrumentation for Structured-RL experiments."""

from __future__ import annotations

from numbers import Real
from pathlib import Path

import numpy as np


def start_tracking(settings: dict, resolved_config: dict, output_dir: Path):
    """Start a W&B run when enabled; disabled experiments do not import W&B."""
    if not bool(settings.get("enabled", False)):
        return None

    import wandb

    run = wandb.init(
        project=str(settings["project"]),
        entity=settings.get("entity"),
        group=settings.get("group"),
        name=settings.get("name") or output_dir.name,
        mode=str(settings.get("mode", "online")),
        tags=list(settings.get("tags", [])),
        config=resolved_config,
        dir=str(output_dir),
        job_type=str(resolved_config["experiment"]["action"]),
    )
    run.define_metric("train/episode")
    run.define_metric("train/*", step_metric="train/episode")
    run.define_metric("update/*", step_metric="train/episode")
    run.define_metric("validation/episode")
    run.define_metric("validation/*", step_metric="validation/episode")
    return run


def log_training(run, episode: dict, updates: list[dict]) -> None:
    if run is None:
        return
    values = {
        f"train/{key}": value
        for key, value in _numeric_values(episode).items()
        if key != "episode"
    }
    values["train/episode"] = int(episode["episode"])
    values["train/updates"] = len(updates)
    if updates:
        for key in _numeric_values(updates[0]):
            if key == "episode":
                continue
            samples = [row[key] for row in updates if key in row]
            values[f"update/{key}"] = float(np.mean(samples))
    run.log(values)


def log_validation(run, episode: int, evaluation: dict) -> None:
    if run is None:
        return
    values = {
        f"validation/{key}": value
        for key, value in _numeric_values(evaluation).items()
    }
    values["validation/episode"] = int(episode)
    run.log(values)


def log_evaluation(run, name: str, evaluation: dict) -> None:
    if run is None:
        return
    run.log(
        {
            f"evaluation/{name}/{key}": value
            for key, value in _numeric_values(evaluation).items()
        }
    )


def log_artifact(run, output_dir: Path, files: list[Path]) -> None:
    if run is None:
        return

    import wandb

    artifact = wandb.Artifact(output_dir.name, type="structured-rl-run")
    for path in files:
        if path.is_file():
            artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)


def _numeric_values(values: dict) -> dict[str, float | int]:
    return {
        str(key): int(value) if isinstance(value, bool) else value
        for key, value in values.items()
        if isinstance(value, Real)
    }
