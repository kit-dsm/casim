from __future__ import annotations

import copy
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from scenarios.scenario_henn_rl.experiment_support import _write_json
from scenarios.scenario_henn_rl.learning import instance_splits
from scenarios.scenario_henn_rl.objective_experiments import train_from_config
from scenarios.scenario_henn_rl.structured_evaluation import (
    evaluate_existing_policy,
    evaluate_structured_actor,
)
from scenarios.scenario_henn_rl.structured_policy import OrderScoreActor
from scenarios.scenario_henn_rl.structured_training import pretrain_from_existing_policy


def run_policy_experiment(
    cfg: DictConfig, splits: dict[str, list[str]], output_dir: Path
) -> dict[str, object]:
    seed = int(cfg.experiment.seed)
    train_ids = splits["train"]
    validation_ids = splits["validation"]
    test_ids = splits["test"]
    if cfg.experiment.mode == "temperature":
        candidates = []
        states = []
        for temperature in [float(value) for value in cfg.srl.temperatures]:
            arm_cfg = OmegaConf.merge(
                cfg, {"srl": {"temperature": temperature}}
            )
            torch.manual_seed(seed)
            actor = OrderScoreActor()
            training = train_from_config(
                actor,
                arm_cfg,
                train_ids,
                validation_ids,
                location_features=True,
            )
            selected = min(
                training["validation_checkpoints"],
                key=lambda row: row["evaluation"][
                    "mean_order_flow_time"
                ],
            )
            candidates.append(
                {
                    "temperature": temperature,
                    "training": training,
                    "selected_validation": selected,
                }
            )
            states.append(copy.deepcopy(actor.state_dict()))
        best_index = min(
            range(len(candidates)),
            key=lambda index: candidates[index]["selected_validation"][
                "evaluation"
            ]["mean_order_flow_time"],
        )
        selected_actor = OrderScoreActor()
        selected_actor.load_state_dict(states[best_index])
        torch.save(
            selected_actor.state_dict(), output_dir / "temperature_selected.pt"
        )
        selected_cfg = OmegaConf.merge(
            cfg,
            {
                "srl": {
                    "temperature": candidates[best_index]["temperature"]
                }
            },
        )
        torch.manual_seed(seed)
        no_location_actor = OrderScoreActor()
        no_location_training = train_from_config(
            no_location_actor,
            selected_cfg,
            train_ids,
            validation_ids,
            location_features=False,
        )
        torch.save(
            no_location_actor.state_dict(),
            output_dir / "temperature_selected_no_location.pt",
        )
        result = {
            "mode": "temperature",
            "splits": splits,
            "candidates": candidates,
            "selected_temperature": candidates[best_index]["temperature"],
            "selected_test": evaluate_structured_actor(
                selected_actor, test_ids
            ),
            "selected_full_test": evaluate_structured_actor(
                selected_actor, instance_splits()["test"]
            ),
            "no_location_at_selected_temperature": {
                "training": no_location_training,
                "full_test": evaluate_structured_actor(
                    no_location_actor,
                    instance_splits()["test"],
                    location_features=False,
                ),
            },
            "configuration": OmegaConf.to_container(cfg, resolve=True),
        }
        _write_json(output_dir / "result.json", result)
        return result
    baselines = {}
    if bool(cfg.experiment.evaluate_baselines):
        baselines = {
            name: {
                "validation": evaluate_existing_policy(
                    validation_ids,
                    batching=name,
                    time_limit_s=float(cfg.baselines.ls_time_limit_s),
                ),
                "test": evaluate_existing_policy(
                    test_ids,
                    batching=name,
                    time_limit_s=float(cfg.baselines.ls_time_limit_s),
                ),
            }
            for name in ("fifo", "cw", "ls")
        }
    if cfg.experiment.mode == "evaluate":
        model_dir = Path(str(cfg.experiment.model_dir)).resolve()
        policies = {}
        for name, filename, location_features in (
            ("srl_scratch", "srl_scratch.pt", True),
            ("imitation_srl", "imitation_srl.pt", True),
            ("srl_no_location", "srl_no_location.pt", False),
        ):
            actor = OrderScoreActor()
            actor.load_state_dict(
                torch.load(
                    model_dir / filename,
                    map_location="cpu",
                    weights_only=True,
                )
            )
            policies[name] = {
                "validation": evaluate_structured_actor(
                    actor,
                    validation_ids,
                    location_features=location_features,
                ),
                "test": evaluate_structured_actor(
                    actor,
                    test_ids,
                    location_features=location_features,
                ),
            }
        result = {
            "mode": "evaluate",
            "splits": splits,
            "model_dir": str(model_dir),
            "baselines": baselines,
            "policies": policies,
            "configuration": OmegaConf.to_container(cfg, resolve=True),
        }
        _write_json(output_dir / "result.json", result)
        return result

    torch.manual_seed(seed)
    scratch = OrderScoreActor()
    scratch_training = train_from_config(
        scratch,
        cfg,
        train_ids,
        validation_ids,
        location_features=True,
    )
    scratch_test = evaluate_structured_actor(scratch, test_ids)
    torch.save(scratch.state_dict(), output_dir / "srl_scratch.pt")

    torch.manual_seed(seed)
    imitation = OrderScoreActor()
    imitation_summary = pretrain_from_existing_policy(
        imitation,
        train_ids,
        epochs=int(cfg.imitation.epochs),
        learning_rate=float(cfg.imitation.learning_rate),
        epsilon=float(cfg.srl.epsilon),
        fy_samples=int(cfg.srl.fy_samples),
        seed=int(cfg.experiment.seed),
    )
    imitation_only_test = evaluate_structured_actor(imitation, test_ids)
    fine_tuned = copy.deepcopy(imitation)
    fine_tuning = train_from_config(
        fine_tuned,
        cfg,
        train_ids,
        validation_ids,
        location_features=True,
    )
    fine_tuned_test = evaluate_structured_actor(fine_tuned, test_ids)
    torch.save(fine_tuned.state_dict(), output_dir / "imitation_srl.pt")

    no_location = None
    if bool(cfg.experiment.location_ablation):
        torch.manual_seed(seed)
        ablated = OrderScoreActor()
        ablated_training = train_from_config(
            ablated,
            cfg,
            train_ids,
            validation_ids,
            location_features=False,
        )
        no_location = {
            "training": ablated_training,
            "test": evaluate_structured_actor(
                ablated, test_ids, location_features=False
            ),
        }
        torch.save(ablated.state_dict(), output_dir / "srl_no_location.pt")
    result = {
        "mode": cfg.experiment.mode,
        "splits": splits,
        "baselines": baselines,
        "arms": {
            "srl_scratch": {
                "training": scratch_training,
                "test": scratch_test,
            },
            "structured_imitation": {
                "training": imitation_summary,
                "test": imitation_only_test,
            },
            "imitation_srl": {
                "training": fine_tuning,
                "test": fine_tuned_test,
            },
            "srl_no_location": no_location,
        },
        "configuration": OmegaConf.to_container(cfg, resolve=True),
    }
    _write_json(output_dir / "result.json", result)
    return result
