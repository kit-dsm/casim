from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from scenarios.scenario_henn_rl.learning import instance_splits
from scenarios.scenario_henn_rl.critic_fitting import overfit_counterfactual_critic
from scenarios.scenario_henn_rl.structured_evaluation import evaluate_existing_policy
from scenarios.scenario_henn_rl.structured_policy import OrderScoreActor


def _critic_calibration_gate(summary, settings) -> dict[str, object]:
    checks = {
        "spearman": {
            "value": summary["mean_spearman"],
            "threshold": float(settings.min_spearman),
            "passed": summary["mean_spearman"] >= float(settings.min_spearman),
        },
        "critic_regret_fraction": {
            "value": summary["mean_critic_regret_fraction"],
            "threshold": float(settings.max_critic_regret_fraction),
            "passed": summary["mean_critic_regret_fraction"]
            <= float(settings.max_critic_regret_fraction),
        },
        "actor_regret_fraction": {
            "value": summary["mean_actor_regret_fraction"],
            "threshold": float(settings.max_actor_regret_fraction),
            "passed": summary["mean_actor_regret_fraction"]
            <= float(settings.max_actor_regret_fraction),
        },
        "soft_target_capture_fraction": {
            "value": summary["soft_target_capture_fraction"],
            "threshold": float(settings.min_soft_target_capture_fraction),
            "passed": summary["soft_target_capture_fraction"]
            >= float(settings.min_soft_target_capture_fraction),
        },
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def _critic_transfer_gate(summary, settings) -> dict[str, object]:
    calibration = _critic_calibration_gate(summary, settings)
    checks = {
        key: value
        for key, value in calibration["checks"].items()
        if key != "actor_regret_fraction"
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "actor_regret_excluded": "actor was frozen during critic validation",
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _ids_for_mode(cfg: DictConfig):
    splits = instance_splits()
    if cfg.experiment.mode in {
        "study", "evaluate", "comparison", "objective_study", "critic_audit",
        "critic_calibration", "critic_overfit", "critic_coverage",
        "candidate_features"
    }:
        return splits
    if cfg.experiment.mode not in {"pilot", "temperature"}:
        raise ValueError(f"Unknown structured mode: {cfg.experiment.mode}")
    result = {}
    for split, instance_ids in splits.items():
        by_size = {}
        for instance_id in instance_ids:
            size = int(instance_id.split("_")[2])
            by_size.setdefault(size, instance_id)
        result[split] = [by_size[size] for size in sorted(by_size)]
    return result


def _baseline_job(job):
    waiting, batching, selector, instance_ids, time_limit_s, *rest = job
    reward_power = float(rest[0]) if rest else 1.0
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    started = time.perf_counter()
    try:
        evaluation = evaluate_existing_policy(
            instance_ids,
            batching=batching,
            waiting_policy=waiting,
            selector=selector,
            time_limit_s=time_limit_s,
            reward_power=reward_power,
        )
        return {
            "status": "complete",
            "waiting": waiting,
            "batching": batching,
            "selector": selector,
            "reward_power": reward_power,
            "wall_time_s": time.perf_counter() - started,
            "evaluation": evaluation,
        }
    except Exception as exc:
        return {
            "status": "error",
            "waiting": waiting,
            "batching": batching,
            "selector": selector,
            "reward_power": reward_power,
            "wall_time_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _critic_coverage_fit_job(job):
    dataset_path, actor_path, size, settings, output_dir, seed = job
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch

    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    audit = {
        "reward_power": dataset["reward_power"],
        "states": dataset["states"][:size],
    }
    actor = OrderScoreActor()
    actor.load_state_dict(
        torch.load(actor_path, map_location="cpu", weights_only=True)
    )
    started = time.perf_counter()
    critic, training = overfit_counterfactual_critic(
        actor,
        audit,
        epochs=int(settings["epochs"]),
        learning_rate=float(settings["learning_rate"]),
        loss_kind="pairwise_top",
        seed=seed,
        interaction_aware=True,
    )
    model_path = Path(output_dir) / f"states_{size}.pt"
    torch.save(critic.state_dict(), model_path)
    return {
        "state_count": size,
        "model": str(model_path),
        "training": training,
        "wall_time_s": time.perf_counter() - started,
    }


def _paired_delta(left, right) -> dict[str, object]:
    left_rows = {row["instance_id"]: row for row in left["episodes"]}
    right_rows = {row["instance_id"]: row for row in right["episodes"]}
    rows = []
    for instance_id in sorted(left_rows):
        left_flow = float(left_rows[instance_id]["mean_flow_time"])
        right_flow = float(right_rows[instance_id]["mean_flow_time"])
        delta = left_flow - right_flow
        rows.append(
            {
                "instance_id": instance_id,
                "family": instance_id.split("_")[1],
                "orders": int(instance_id.split("_")[2]),
                "left_flow_time": left_flow,
                "right_flow_time": right_flow,
                "delta_s": delta,
                "delta_percent": 100.0 * delta / right_flow,
            }
        )
    values = [row["delta_s"] for row in rows]
    def grouped(field: str) -> dict[str, dict[str, float | int]]:
        result = {}
        for value in sorted({row[field] for row in rows}):
            members = [row for row in rows if row[field] == value]
            result[str(value)] = {
                "instances": len(members),
                "mean_delta_s": float(
                    np.mean([row["delta_s"] for row in members])
                ),
                "mean_delta_percent": float(
                    np.mean([row["delta_percent"] for row in members])
                ),
                "left_wins": sum(row["delta_s"] < 0.0 for row in members),
            }
        return result

    return {
        "mean_delta_s": float(np.mean(values)),
        "median_delta_s": float(np.median(values)),
        "left_wins": sum(value < 0.0 for value in values),
        "by_family": grouped("family"),
        "by_order_count": grouped("orders"),
        "instances": rows,
    }


def _training_order(
    train_ids: list[str], *, seed: int, passes: int
) -> list[str]:
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(passes):
        shuffled = list(train_ids)
        rng.shuffle(shuffled)
        result.extend(shuffled)
    return result


def _objective_key(power: float) -> str:
    return "mean_flow_power_" + str(power).replace(".", "_")


def _evaluation_objective(evaluation: dict, power: float) -> float:
    values = []
    for row in evaluation["episodes"]:
        flows = np.asarray(row["flow_times"], dtype=float)
        values.append(float(np.mean(flows**power)))
    return float(np.mean(values))


def _select_objective_baselines(
    validation_rows: list[dict], powers: list[float]
) -> dict[str, dict]:
    """Select solely from validation evaluations supplied by the caller."""
    return {
        str(power): min(
            validation_rows,
            key=lambda row: _evaluation_objective(
                row["evaluation"], power
            ),
        )
        for power in powers
    }


def _paired_objective_delta(left: dict, right: dict, power: float) -> dict:
    left_rows = {row["instance_id"]: row for row in left["episodes"]}
    right_rows = {row["instance_id"]: row for row in right["episodes"]}
    rows = []
    for instance_id in sorted(left_rows):
        left_value = float(
            np.mean(np.asarray(left_rows[instance_id]["flow_times"]) ** power)
        )
        right_value = float(
            np.mean(np.asarray(right_rows[instance_id]["flow_times"]) ** power)
        )
        delta = left_value - right_value
        rows.append(
            {
                "instance_id": instance_id,
                "family": instance_id.split("_")[1],
                "orders": int(instance_id.split("_")[2]),
                "left_objective": left_value,
                "right_objective": right_value,
                "delta": delta,
                "delta_percent": 100.0 * delta / right_value,
            }
        )

    def grouped(field):
        return {
            str(value): {
                "instances": len(members := [r for r in rows if r[field] == value]),
                "mean_delta": float(np.mean([r["delta"] for r in members])),
                "mean_delta_percent": float(
                    np.mean([r["delta_percent"] for r in members])
                ),
                "left_wins": sum(r["delta"] < 0.0 for r in members),
            }
            for value in sorted({row[field] for row in rows})
        }

    return {
        "power": power,
        "mean_delta": float(np.mean([row["delta"] for row in rows])),
        "median_delta": float(np.median([row["delta"] for row in rows])),
        "left_wins": sum(row["delta"] < 0.0 for row in rows),
        "by_family": grouped("family"),
        "by_order_count": grouped("orders"),
        "instances": rows,
    }


def _state_digest(actor) -> str:
    digest = hashlib.sha256()
    for value in actor.state_dict().values():
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
