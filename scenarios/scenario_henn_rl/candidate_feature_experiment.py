from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from scipy.stats import spearmanr

from scenarios.scenario_henn_rl.counterfactuals import _reference_trajectory
from scenarios.scenario_henn_rl.experiment_support import (
    _critic_transfer_gate,
    _write_json,
)
from scenarios.scenario_henn_rl.structured_environment import StructuredBatchingEpisode
from scenarios.scenario_henn_rl.structured_policy import OrderScoreActor


ROUTE_FEATURES = ["route_distance"]
ROUTE_AGE_FEATURES = [
    "route_distance",
    "selected_age_sum_s",
    "selected_age_max_s",
    "remaining_age_sum_s",
    "remaining_age_max_s",
]
ALL_FEATURES = [
    "route_distance",
    "route_service_time_s",
    "route_saving",
    "selected_orders",
    "selected_demand",
    "capacity_fill",
    "selected_pick_positions",
    "selected_age_sum_s",
    "selected_age_mean_s",
    "selected_age_max_s",
    "remaining_orders",
    "remaining_demand",
    "remaining_age_sum_s",
    "remaining_age_mean_s",
    "remaining_age_max_s",
    "aisles_visited",
    "aisle_span",
    "position_span",
    "visible_orders",
    "visible_demand",
    "episode_progress",
]


def _extract_rows(actor, audit: dict[str, object]) -> list[dict[str, object]]:
    power = str(audit["reward_power"])
    by_instance: dict[str, list[dict[str, object]]] = {}
    for state_row in audit["states"]:
        by_instance.setdefault(str(state_row["instance_id"]), []).append(state_row)

    rows = []
    for instance_id, state_rows in by_instance.items():
        trajectory = _reference_trajectory(actor, instance_id)
        episode = StructuredBatchingEpisode([instance_id])
        state = episode.reset(instance_id=instance_id)
        wanted = {int(row["decision_index"]): row for row in state_rows}
        for decision_index, reference in enumerate(trajectory):
            if decision_index in wanted:
                source = wanted[decision_index]
                if not np.array_equal(state["order_ids"], reference["state"]["order_ids"]):
                    raise RuntimeError("Feature replay reached a different order buffer")
                values = np.asarray(
                    [float(candidate["true_q"][power]) for candidate in source["candidate_rows"]]
                )
                scale = max(float(values.std()), 1e-12)
                standardized = (values - values.mean()) / scale
                for candidate_index, (candidate, target) in enumerate(
                    zip(source["candidate_rows"], standardized)
                ):
                    decision_fraction = decision_index / max(
                        1, len(trajectory) - 1
                    )
                    rows.append(
                        {
                            "instance_id": instance_id,
                            "decision_index": decision_index,
                            "candidate_index": candidate_index,
                            "true_q": float(candidate["true_q"][power]),
                            "target": float(target),
                            "stage": (
                                "early"
                                if decision_fraction < 1 / 3
                                else "middle"
                                if decision_fraction < 2 / 3
                                else "late"
                            ),
                            "features": episode.candidate_operational_features(
                                candidate["order_ids"]
                            ),
                        }
                    )
            selected = reference["actor_order_ids"]
            indices = np.asarray(
                [
                    index
                    for index, order_id in enumerate(state["order_ids"])
                    if int(order_id) in set(selected)
                ],
                dtype=int,
            )
            state, _, done, _, _ = episode.step(indices)
            if done:
                break
        episode.close()
    return rows


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def _ridge_predict(model, x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    return np.column_stack([np.ones(len(z)), z]) @ model["coefficients"]


def _state_metrics(rows, predicted: np.ndarray) -> dict[str, float]:
    grouped: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault((row["instance_id"], row["decision_index"]), []).append(index)
    state_results = []
    actor_regret_total = 0.0
    soft_advantage_total = 0.0
    for indices in grouped.values():
        true = np.asarray([rows[index]["true_q"] for index in indices])
        scores = predicted[indices]
        correlation = (
            np.nan
            if float(np.ptp(scores)) < 1e-12
            else spearmanr(scores, true).statistic
        )
        best = int(np.argmax(true))
        chosen = int(np.argmax(scores))
        spread = max(float(np.ptp(true)), 1e-12)
        normalized = (scores - scores.mean()) / max(float(scores.std()), 1e-6)
        weights = np.exp(normalized - normalized.max())
        weights /= weights.sum()
        actor_regret = float(true[best] - true[0])
        actor_regret_total += actor_regret
        soft_advantage_total += float(weights @ true - true[0])
        state_results.append(
            (
                None if np.isnan(correlation) else float(correlation),
                float(true[best] - true[chosen]) / spread,
                int(best == chosen),
                actor_regret / spread,
            )
        )
    correlations = [row[0] for row in state_results if row[0] is not None]
    return {
        "states": len(state_results),
        "mean_spearman": (
            float(np.mean(correlations)) if correlations else None
        ),
        "mean_critic_regret_fraction": float(np.mean([row[1] for row in state_results])),
        "critic_top_accuracy": float(np.mean([row[2] for row in state_results])),
        "mean_actor_regret_fraction": float(
            np.mean([row[3] for row in state_results])
        ),
        "soft_target_capture_fraction": soft_advantage_total / max(actor_regret_total, 1e-12),
    }


def _breakdowns(rows, predicted: np.ndarray) -> dict[str, object]:
    dimensions = {
        "family": [row["instance_id"].split("_")[1] for row in rows],
        "order_count": [row["instance_id"].split("_")[2] for row in rows],
        "episode_stage": [row["stage"] for row in rows],
    }
    result = {}
    for name, values in dimensions.items():
        result[name] = {}
        for value in sorted(set(values)):
            keep = np.asarray([member == value for member in values])
            result[name][value] = _state_metrics(
                [row for row, include in zip(rows, keep) if include],
                predicted[keep],
            )
    return result


def _matrix(rows, feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"][name]) for name in feature_names] for row in rows],
        dtype=float,
    )


def _grouped_cv(rows, feature_names: list[str], alpha: float) -> dict[str, object]:
    instances = sorted({row["instance_id"] for row in rows})
    predictions = np.empty(len(rows), dtype=float)
    folds = []
    x = _matrix(rows, feature_names)
    y = np.asarray([row["target"] for row in rows])
    for instance_id in instances:
        test = np.asarray([row["instance_id"] == instance_id for row in rows])
        model = _ridge_fit(x[~test], y[~test], alpha)
        predictions[test] = _ridge_predict(model, x[test])
        folds.append({"instance_id": instance_id, **_state_metrics(
            [row for row, keep in zip(rows, test) if keep], predictions[test]
        )})
    return {
        "summary": _state_metrics(rows, predictions),
        "breakdowns": _breakdowns(rows, predictions),
        "folds": folds,
    }


class _FeatureMLP(torch.nn.Module):
    def __init__(self, feature_count: int, hidden: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(feature_count, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, features):
        return self.network(features).flatten()


def _mlp_fit(
    x: np.ndarray,
    y: np.ndarray,
    rows,
    *,
    hidden: int,
    epochs: int,
    learning_rate: float,
    seed: int,
):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    features = torch.as_tensor((x - mean) / scale, dtype=torch.float32)
    target = torch.as_tensor(y, dtype=torch.float32)
    groups: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((row["instance_id"], row["decision_index"]), []).append(index)
    torch.manual_seed(seed)
    model = _FeatureMLP(x.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        predicted = model(features)
        losses = []
        for indices in groups.values():
            indices = torch.as_tensor(indices, dtype=torch.long)
            state_prediction = predicted[indices]
            state_target = target[indices]
            pairs = torch.triu_indices(len(indices), len(indices), offset=1)
            differences = state_target[pairs[0]] - state_target[pairs[1]]
            keep = differences.abs() > 1e-8
            pair_logits = (
                state_prediction[pairs[0]] - state_prediction[pairs[1]]
            )[keep]
            labels = (differences[keep] > 0).to(pair_logits.dtype)
            ranking = torch.nn.functional.binary_cross_entropy_with_logits(
                pair_logits, labels
            )
            top = torch.nn.functional.cross_entropy(
                state_prediction.unsqueeze(0), state_target.argmax().reshape(1)
            )
            losses.append(ranking + top)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {"mean": mean, "scale": scale, "network": model}


def _mlp_predict(model, x: np.ndarray) -> np.ndarray:
    features = torch.as_tensor(
        (x - model["mean"]) / model["scale"], dtype=torch.float32
    )
    with torch.no_grad():
        return model["network"](features).numpy()


def _grouped_mlp_cv(
    rows,
    feature_names: list[str],
    *,
    hidden: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    instances = sorted({row["instance_id"] for row in rows})
    predictions = np.empty(len(rows), dtype=float)
    folds = []
    x = _matrix(rows, feature_names)
    y = np.asarray([row["target"] for row in rows])
    for fold, instance_id in enumerate(instances):
        test = np.asarray([row["instance_id"] == instance_id for row in rows])
        training_rows = [row for row, keep in zip(rows, ~test) if keep]
        model = _mlp_fit(
            x[~test], y[~test], training_rows,
            hidden=hidden, epochs=epochs, learning_rate=learning_rate,
            seed=seed + fold,
        )
        predictions[test] = _mlp_predict(model, x[test])
        folds.append({"instance_id": instance_id, **_state_metrics(
            [row for row, keep in zip(rows, test) if keep], predictions[test]
        )})
    return {
        "summary": _state_metrics(rows, predictions),
        "breakdowns": _breakdowns(rows, predictions),
        "folds": folds,
    }


def _serialize_model(model, feature_names: list[str]) -> dict[str, object]:
    return {
        "features": feature_names,
        "mean": model["mean"].tolist(),
        "scale": model["scale"].tolist(),
        "coefficients": model["coefficients"].tolist(),
    }


def run_candidate_feature_study(
    cfg: DictConfig, splits: dict[str, list[str]], output_dir: Path
) -> dict[str, object]:
    settings = cfg.candidate_features
    started = time.perf_counter()
    source_dir = Path(settings.source_dir).resolve()
    calibration_dir = Path(settings.validation_source_dir).resolve()
    dataset = json.loads(Path(settings.dataset_path).resolve().read_text(encoding="utf-8"))
    calibration = json.loads((calibration_dir / "result.json").read_text(encoding="utf-8"))
    actor = OrderScoreActor()
    actor.load_state_dict(
        torch.load(source_dir / "pilot" / "selected.pt", map_location="cpu", weights_only=True)
    )
    training_ids = {str(value) for value in dataset["base_instance_ids"] + dataset["additional_instance_ids"]}
    validation_audit = calibration["pilot"]["audit"]
    validation_ids = {str(row["instance_id"]) for row in validation_audit["states"]}
    if training_ids - set(splits["train"]):
        raise ValueError("Candidate-feature training labels include non-training instances")
    if validation_ids - set(splits["validation"]):
        raise ValueError("Candidate-feature audit includes non-validation instances")
    if training_ids & validation_ids:
        raise ValueError("Candidate-feature train and validation instances overlap")

    training_rows = _extract_rows(actor, dataset)
    feature_sets = {
        "route_only": ROUTE_FEATURES,
        "route_age": ROUTE_AGE_FEATURES,
        "all_operational": ALL_FEATURES,
    }
    alpha = float(settings.ridge_alpha)
    cv = {
        name: _grouped_cv(training_rows, features, alpha)
        for name, features in feature_sets.items()
    }
    linear_model = _ridge_fit(
        _matrix(training_rows, ALL_FEATURES),
        np.asarray([row["target"] for row in training_rows]),
        alpha,
    )
    linear_training = _state_metrics(
        training_rows,
        _ridge_predict(linear_model, _matrix(training_rows, ALL_FEATURES)),
    )
    if (
        linear_training["mean_spearman"] < 0.8
        or linear_training["mean_critic_regret_fraction"] > 0.12
    ):
        cv["all_operational_mlp"] = _grouped_mlp_cv(
            training_rows,
            ALL_FEATURES,
            hidden=int(settings.mlp_hidden),
            epochs=int(settings.mlp_epochs),
            learning_rate=float(settings.mlp_learning_rate),
            seed=int(cfg.experiment.seed),
        )
        feature_sets["all_operational_mlp"] = ALL_FEATURES
    selected_name = min(
        cv,
        key=lambda name: (
            cv[name]["summary"]["mean_critic_regret_fraction"],
            -cv[name]["summary"]["mean_spearman"],
        ),
    )
    selected_features = feature_sets[selected_name]
    if selected_name == "all_operational_mlp":
        model = _mlp_fit(
            _matrix(training_rows, selected_features),
            np.asarray([row["target"] for row in training_rows]),
            training_rows,
            hidden=int(settings.mlp_hidden), epochs=int(settings.mlp_epochs),
            learning_rate=float(settings.mlp_learning_rate),
            seed=int(cfg.experiment.seed),
        )
        predict = _mlp_predict
        serialized_model = {
            "features": selected_features,
            "kind": "one_hidden_layer_pairwise_top_mlp",
            "hidden": int(settings.mlp_hidden),
        }
    else:
        model = _ridge_fit(
            _matrix(training_rows, selected_features),
            np.asarray([row["target"] for row in training_rows]),
            alpha,
        )
        predict = _ridge_predict
        serialized_model = {"kind": "ridge", **_serialize_model(model, selected_features)}
    validation_rows = _extract_rows(actor, validation_audit)
    validation_predictions = predict(
        model, _matrix(validation_rows, selected_features)
    )
    validation = _state_metrics(validation_rows, validation_predictions)
    gate = _critic_transfer_gate(validation, cfg.critic_calibration)
    result = {
        "mode": "candidate_features",
        "status": "complete",
        "seed": int(cfg.experiment.seed),
        "reward_power": float(dataset["reward_power"]),
        "test_split_used": False,
        "training": {
            "instances": sorted(training_ids),
            "states": len(dataset["states"]),
            "candidate_labels": len(training_rows),
            "selection": "leave-one-training-instance-out CV; minimum regret, Spearman tie-break",
            "ridge_alpha": alpha,
            "all_operational_linear_in_sample": linear_training,
            "arms": cv,
        },
        "selected": selected_name,
        "model": serialized_model,
        "validation": {
            "instances": sorted(validation_ids),
            "metrics": validation,
            "breakdowns": _breakdowns(validation_rows, validation_predictions),
            "gate": gate,
        },
        "wall_time_s": time.perf_counter() - started,
    }
    _write_json(output_dir / "result.json", result)
    return result
