from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from scenarios.scenario_henn_rl.learning import (
    evaluate_heuristic,
    evaluate_model,
    generate_counterfactuals,
    instance_splits,
    make_ppo,
    make_vec_env,
    policy_decision_summary,
    pretrain_policy,
    reference_decisions,
    tune_heuristic,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _train_arm(
    *,
    seed: int,
    train_ids: list[str],
    validation_ids: list[str],
    test_ids: list[str],
    labels: list[dict[str, object]],
    cfg: DictConfig,
    output_dir: Path,
    pretrained: bool,
) -> dict[str, object]:
    model = make_ppo(train_ids, seed, n_steps=int(cfg.ppo.n_steps))
    supervised = None
    before = None
    if pretrained:
        started = time.perf_counter()
        supervised = pretrain_policy(
            model,
            labels,
            epochs=int(cfg.supervised.epochs),
            batch_size=int(cfg.supervised.batch_size),
            learning_rate=float(cfg.supervised.learning_rate),
            seed=seed,
        )
        supervised["wall_time_s"] = time.perf_counter() - started
        before = {
            "validation": evaluate_model(model, validation_ids),
            "test": evaluate_model(model, test_ids),
        }
    started = time.perf_counter()
    model.learn(total_timesteps=int(cfg.ppo.timesteps))
    training_time = time.perf_counter() - started
    after = {
        "validation": evaluate_model(model, validation_ids),
        "test": evaluate_model(model, test_ids),
    }
    name = "supervised_ppo" if pretrained else "ppo_scratch"
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "models" / f"{name}_{seed}")
    actual_timesteps = int(model.num_timesteps)
    model.get_env().close()
    return {
        "arm": name,
        "seed": seed,
        "supervised": supervised,
        "evaluation_before_rl": before,
        "training_wall_time_s": training_time,
        "timesteps": actual_timesteps,
        "timesteps_per_s": actual_timesteps / max(training_time, 1e-12),
        "evaluation_after_rl": after,
    }


def run_pilot(cfg: DictConfig, output_dir: Path) -> dict[str, object]:
    splits = instance_splits()
    by_size = {}
    for instance_id in splits["train"]:
        size = int(instance_id.split("_")[2])
        by_size.setdefault(size, instance_id)
    pilot_ids = [by_size[size] for size in sorted(by_size)]
    generated = generate_counterfactuals(
        pilot_ids,
        fill_threshold=1.0,
        max_age_s=900.0,
        workers=min(int(cfg.experiment.workers), len(pilot_ids)),
    )
    _write_jsonl(output_dir / "pilot_labels.jsonl", generated.pop("labels"))
    return {"mode": "pilot", "instances": pilot_ids, "generation": generated}


def run_study(cfg: DictConfig, output_dir: Path) -> dict[str, object]:
    splits = instance_splits()
    workers = max(1, int(cfg.experiment.workers))
    tuning = tune_heuristic(
        splits["train"],
        splits["validation"],
        fill_thresholds=[float(value) for value in cfg.heuristic.fill_thresholds],
        max_ages_s=[float(value) for value in cfg.heuristic.max_ages_s],
        workers=workers,
    )
    selected = tuning["selected"]
    fill = float(selected["fill_threshold"])
    age = float(selected["max_age_s"])
    baselines = {
        "always_dispatch": evaluate_heuristic(
            splits["test"],
            fill_threshold=0.0,
            max_age_s=0.0,
            workers=workers,
        ),
        "tuned_heuristic": evaluate_heuristic(
            splits["test"],
            fill_threshold=fill,
            max_age_s=age,
            workers=workers,
        ),
    }
    labels_path = cfg.experiment.get("labels_path")
    if labels_path:
        source = Path(str(labels_path)).resolve()
        labels = _read_jsonl(source)
        generated = {
            "reused": True,
            "source": str(source),
            "workers": 0,
        }
    else:
        generated = generate_counterfactuals(
            splits["train"],
            fill_threshold=fill,
            max_age_s=age,
            workers=workers,
        )
        labels = generated.pop("labels")
        _write_jsonl(output_dir / "counterfactual_labels.jsonl", labels)
    arms = []
    for seed in [int(value) for value in cfg.ppo.seeds]:
        arms.append(
            _train_arm(
                seed=seed,
                train_ids=splits["train"],
                validation_ids=splits["validation"],
                test_ids=splits["test"],
                labels=labels,
                cfg=cfg,
                output_dir=output_dir,
                pretrained=False,
            )
        )
        arms.append(
            _train_arm(
                seed=seed,
                train_ids=splits["train"],
                validation_ids=splits["validation"],
                test_ids=splits["test"],
                labels=labels,
                cfg=cfg,
                output_dir=output_dir,
                pretrained=True,
            )
        )
    return {
        "mode": "study",
        "splits": splits,
        "heuristic_tuning": tuning,
        "baselines": baselines,
        "counterfactual_generation": generated,
        "label_summary": {
            "count": len(labels),
            "dispatch_preferred_fraction": sum(
                row["preferred_action"] for row in labels
            )
            / max(1, len(labels)),
            "mean_absolute_advantage": sum(
                abs(row["advantage_dispatch"]) for row in labels
            )
            / max(1, len(labels)),
        },
        "training_arms": arms,
        "configuration": OmegaConf.to_container(cfg, resolve=True),
    }


def _read_progress(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for source in csv.DictReader(stream):
            row = {}
            for key, value in source.items():
                if value in (None, ""):
                    row[key] = None
                    continue
                number = float(value)
                row[key] = number if math.isfinite(number) else None
            rows.append(row)
    return rows


def _checkpoint_evaluation(
    model,
    validation_ids: list[str],
    observations,
    reference_actions,
    thresholds: list[float] | None = None,
) -> dict[str, object]:
    calibrated = []
    for threshold in thresholds or []:
        evaluation = evaluate_model(
            model,
            validation_ids,
            dispatch_threshold=threshold,
        )
        calibrated.append(
            {
                "threshold": threshold,
                "mean_order_flow_time": evaluation["mean_order_flow_time"],
                "wait_fraction": evaluation["wait_fraction"],
            }
        )
    return {
        "timesteps": int(model.num_timesteps),
        "policy": policy_decision_summary(
            model, observations, reference_actions
        ),
        "validation_default": evaluate_model(model, validation_ids),
        "validation_thresholds": calibrated,
    }


def _train_stabilization_arm(
    *,
    model,
    name: str,
    added_timesteps: int,
    checkpoint_freq: int,
    output_dir: Path,
    validation_ids: list[str],
    test_ids: list[str],
    observations,
    reference_actions,
    thresholds: list[float],
    calibration_checkpoints: int,
) -> dict[str, object]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure

    arm_dir = output_dir / name
    checkpoint_dir = arm_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(arm_dir / "sb3"), ["csv"]))
    initial_path = arm_dir / "initial.zip"
    model.save(initial_path)
    callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="ppo",
    )
    started = time.perf_counter()
    model.learn(
        total_timesteps=added_timesteps,
        callback=callback,
        reset_num_timesteps=False,
    )
    training_time = time.perf_counter() - started
    final_path = arm_dir / "final.zip"
    model.save(final_path)
    model.get_env().close()

    paths_by_step = {int(PPO.load(initial_path).num_timesteps): initial_path}
    for path in checkpoint_dir.glob("ppo_*_steps.zip"):
        paths_by_step[int(path.stem.split("_")[1])] = path
    paths_by_step[int(PPO.load(final_path).num_timesteps)] = final_path
    evaluations = []
    ordered_paths = sorted(paths_by_step.items())
    calibrated_indices = {0}
    calibrated_indices.update(
        range(
            max(0, len(ordered_paths) - calibration_checkpoints),
            len(ordered_paths),
        )
    )
    for index, (timesteps, path) in enumerate(ordered_paths):
        checkpoint = PPO.load(path, device="cpu")
        evaluation = _checkpoint_evaluation(
            checkpoint,
            validation_ids,
            observations,
            reference_actions,
            thresholds if index in calibrated_indices else None,
        )
        evaluation["timesteps"] = timesteps
        evaluation["model_path"] = str(path)
        evaluations.append(evaluation)

    candidates = [
        {
            "timesteps": row["timesteps"],
            "model_path": row["model_path"],
            **threshold,
        }
        for row in evaluations
        for threshold in row["validation_thresholds"]
    ]
    selected = min(candidates, key=lambda row: row["mean_order_flow_time"])
    selected_default = min(
        evaluations,
        key=lambda row: row["validation_default"]["mean_order_flow_time"],
    )
    selected_model = PPO.load(selected["model_path"], device="cpu")
    test_default = evaluate_model(selected_model, test_ids)
    test_calibrated = evaluate_model(
        selected_model,
        test_ids,
        dispatch_threshold=float(selected["threshold"]),
    )
    selected_default_model = PPO.load(
        selected_default["model_path"], device="cpu"
    )
    test_selected_default = evaluate_model(
        selected_default_model, test_ids
    )
    progress = _read_progress(arm_dir / "sb3" / "progress.csv")
    return {
        "arm": name,
        "gae_lambda": float(selected_model.gae_lambda),
        "added_timesteps": added_timesteps,
        "training_wall_time_s": training_time,
        "timesteps_per_s": added_timesteps / max(training_time, 1e-12),
        "checkpoints": evaluations,
        "selected_on_validation": selected,
        "selected_default_on_validation": {
            "timesteps": selected_default["timesteps"],
            "model_path": selected_default["model_path"],
            "evaluation": selected_default["validation_default"],
        },
        "test_default": test_default,
        "test_calibrated": test_calibrated,
        "test_selected_default": test_selected_default,
        "training_metrics": progress,
    }


def run_stabilization(cfg: DictConfig, output_dir: Path) -> dict[str, object]:
    """Continue seed 11 and isolate the effect of GAE lambda."""
    from stable_baselines3 import PPO

    splits = instance_splits()
    settings = cfg.stabilization
    seed = int(settings.seed)
    threshold_min = float(settings.threshold_min)
    threshold_step = float(settings.threshold_step)
    threshold_count = round(
        (float(settings.threshold_max) - threshold_min) / threshold_step
    )
    thresholds = [
        round(threshold_min + index * threshold_step, 10)
        for index in range(threshold_count + 1)
    ]
    observations, actions = reference_decisions(
        splits["validation"],
        fill_threshold=float(settings.fill_threshold),
        max_age_s=float(settings.max_age_s),
    )
    base_path = Path(str(settings.base_model)).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base PPO model not found: {base_path}")
    arms = []
    for gae_lambda in [float(value) for value in settings.gae_lambdas]:
        model = PPO.load(
            base_path,
            env=make_vec_env(splits["train"]),
            device="cpu",
            custom_objects={"gae_lambda": gae_lambda},
        )
        model.set_random_seed(seed)
        arms.append(
            _train_stabilization_arm(
                model=model,
                name=f"continue_gae_{gae_lambda:g}",
                added_timesteps=int(settings.added_timesteps),
                checkpoint_freq=int(settings.checkpoint_freq),
                output_dir=output_dir,
                validation_ids=splits["validation"],
                test_ids=splits["test"],
                observations=observations,
                reference_actions=actions,
                thresholds=thresholds,
                calibration_checkpoints=int(
                    settings.calibration_checkpoints
                ),
            )
        )

    always_dispatch = evaluate_heuristic(
        splits["validation"],
        fill_threshold=0.0,
        max_age_s=0.0,
        workers=1,
    )
    tolerance = float(settings.calibration_tolerance)
    calibration_poor = all(
        min(
            row["validation_default"]["mean_order_flow_time"]
            for row in arm["checkpoints"]
        )
        >= always_dispatch["mean_order_flow_time"]
        and (
            always_dispatch["mean_order_flow_time"]
            - arm["selected_on_validation"]["mean_order_flow_time"]
        )
        / always_dispatch["mean_order_flow_time"]
        >= tolerance
        for arm in arms
    )
    weighted_arm = None
    if calibration_poor:
        labels = _read_jsonl(Path(str(settings.labels_path)).resolve())
        best_lambda = min(
            arms,
            key=lambda arm: arm["selected_on_validation"][
                "mean_order_flow_time"
            ],
        )["gae_lambda"]
        model = make_ppo(
            splits["train"],
            seed,
            n_steps=int(cfg.ppo.n_steps),
            gae_lambda=float(best_lambda),
        )
        supervised = pretrain_policy(
            model,
            labels,
            epochs=int(cfg.supervised.epochs),
            batch_size=int(cfg.supervised.batch_size),
            learning_rate=float(cfg.supervised.learning_rate),
            seed=seed,
            advantage_weighted=True,
        )
        weighted_arm = _train_stabilization_arm(
            model=model,
            name="advantage_weighted_ppo",
            added_timesteps=int(settings.total_weighted_timesteps),
            checkpoint_freq=int(settings.checkpoint_freq),
            output_dir=output_dir,
            validation_ids=splits["validation"],
            test_ids=splits["test"],
            observations=observations,
            reference_actions=actions,
            thresholds=thresholds,
            calibration_checkpoints=int(settings.calibration_checkpoints),
        )
        weighted_arm["supervised"] = supervised
    return {
        "mode": "stabilization",
        "seed": seed,
        "base_model": str(base_path),
        "validation_always_dispatch": always_dispatch,
        "continuation_arms": arms,
        "calibration_poor": calibration_poor,
        "advantage_weighted_arm": weighted_arm,
        "configuration": OmegaConf.to_container(cfg, resolve=True),
    }


def run_diagnosis(cfg: DictConfig, output_dir: Path) -> dict[str, object]:
    """Trace one ordinary SB3 PPO run without changing the algorithm."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure

    splits = instance_splits()
    seed = int(cfg.diagnosis.seed)
    fill = float(cfg.diagnosis.fill_threshold)
    age = float(cfg.diagnosis.max_age_s)
    observations, actions = reference_decisions(
        splits["validation"], fill_threshold=fill, max_age_s=age
    )
    model = make_ppo(splits["train"], seed, n_steps=int(cfg.ppo.n_steps))
    log_dir = output_dir / "sb3"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(log_dir), ["csv"]))
    checkpoints = [
        {
            "timesteps": 0,
            "policy": policy_decision_summary(model, observations, actions),
            "validation": evaluate_model(model, splits["validation"]),
        }
    ]
    callback = CheckpointCallback(
        save_freq=int(cfg.diagnosis.checkpoint_freq),
        save_path=str(checkpoint_dir),
        name_prefix="ppo",
    )
    started = time.perf_counter()
    model.learn(
        total_timesteps=int(cfg.diagnosis.timesteps), callback=callback
    )
    training_time = time.perf_counter() - started
    model.save(output_dir / "ppo_final")
    model.get_env().close()

    for path in sorted(
        checkpoint_dir.glob("ppo_*_steps.zip"),
        key=lambda item: int(item.stem.split("_")[1]),
    ):
        timesteps = int(path.stem.split("_")[1])
        checkpoint = PPO.load(path, device="cpu")
        checkpoints.append(
            {
                "timesteps": timesteps,
                "policy": policy_decision_summary(
                    checkpoint, observations, actions
                ),
                "validation": evaluate_model(
                    checkpoint, splits["validation"]
                ),
            }
        )
    final_model = PPO.load(output_dir / "ppo_final", device="cpu")
    test = evaluate_model(final_model, splits["test"])
    progress_path = log_dir / "progress.csv"
    return {
        "mode": "diagnosis",
        "seed": seed,
        "training_wall_time_s": training_time,
        "timesteps": int(cfg.diagnosis.timesteps),
        "timesteps_per_s": int(cfg.diagnosis.timesteps)
        / max(training_time, 1e-12),
        "ppo_parameters": {
            "gamma": final_model.gamma,
            "gae_lambda": final_model.gae_lambda,
            "ent_coef": final_model.ent_coef,
            "n_steps": final_model.n_steps,
            "batch_size": final_model.batch_size,
        },
        "reference_panel": {
            "states": int(len(actions)),
            "wait_fraction": float((actions == 0).mean()),
        },
        "training_metrics": _read_progress(progress_path),
        "checkpoints": checkpoints,
        "test": test,
        "configuration": OmegaConf.to_container(cfg, resolve=True),
    }


def run(cfg: DictConfig) -> dict[str, object]:
    output_dir = Path(cfg.experiment.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "splits.json", instance_splits())
    if cfg.experiment.mode == "pilot":
        result = run_pilot(cfg, output_dir)
    elif cfg.experiment.mode == "study":
        result = run_study(cfg, output_dir)
    elif cfg.experiment.mode == "diagnosis":
        result = run_diagnosis(cfg, output_dir)
    elif cfg.experiment.mode == "stabilization":
        result = run_stabilization(cfg, output_dir)
    else:
        raise ValueError(f"Unknown experiment mode: {cfg.experiment.mode}")
    _write_json(output_dir / "result.json", result)
    return result


@hydra.main(
    version_base="1.3",
    config_path="config",
    config_name="henn_rl_config",
)
def main(cfg: DictConfig) -> None:
    result = run(cfg)
    print(json.dumps({"status": "complete", "mode": result["mode"]}))


if __name__ == "__main__":
    main()
