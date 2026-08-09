from __future__ import annotations

from pathlib import Path


def start_tracking(settings: dict, resolved_config: dict):
    if not bool(settings.get("enabled", False)):
        return None
    import wandb

    run = wandb.init(
        project=str(settings["project"]),
        entity=settings.get("entity"),
        group=settings.get("group"),
        name=settings.get("name"),
        mode=str(settings.get("mode", "online")),
        config=resolved_config,
    )
    if hasattr(run, "define_metric"):
        run.define_metric("train/episode")
        run.define_metric("train/*", step_metric="train/episode")
        run.define_metric("validation/episode")
        run.define_metric("validation/*", step_metric="validation/episode")
    return run


def metric_callback(run):
    if run is None:
        return None

    def log(namespace: str, step: int, metrics: dict) -> None:
        values = {
            f"{namespace}/{key}": value
            for key, value in _numeric_values(metrics).items()
        }
        values[f"{namespace}/episode"] = int(step)
        run.log(values)

    return log


def log_evaluation(run, name: str, evaluation: dict) -> None:
    if run is None:
        return
    import wandb

    summary = {
        f"{name}/{key}": value
        for key, value in _numeric_values(evaluation).items()
    }
    run.log(summary)
    episodes = evaluation.get("episodes", [])
    if episodes:
        columns = [
            "instance_id",
            "mean_flow_time",
            "p95_flow_time",
            "mean_tardiness",
            "max_tardiness",
            "violation_fraction",
            "objective_per_order",
            "mean_batch_fill",
            "mean_batch_orders",
            "orders_per_sim_hour",
        ]
        run.log(
            {
                f"{name}/instances": wandb.Table(
                    columns=columns,
                    data=[
                        [row.get(column) for column in columns]
                        for row in episodes
                    ],
                )
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


def _numeric_values(value: dict, prefix: str = "") -> dict[str, float | int]:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, bool):
            result[name] = int(item)
        elif isinstance(item, (int, float)):
            result[name] = item
        elif isinstance(item, dict):
            result.update(_numeric_values(item, name))
    return result
