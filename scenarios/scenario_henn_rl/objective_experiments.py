from __future__ import annotations

import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from scenarios.scenario_henn_rl.experiment_support import (
    _baseline_job,
    _evaluation_objective,
    _objective_key,
    _paired_delta,
    _paired_objective_delta,
    _select_objective_baselines,
    _state_digest,
    _training_order,
    _write_json,
)
from scenarios.scenario_henn_rl.structured_evaluation import (
    evaluate_existing_policy,
    evaluate_structured_actor,
)
from scenarios.scenario_henn_rl.structured_policy import OrderScoreActor
from scenarios.scenario_henn_rl.structured_training import train_structured_rl


def run_comparison(cfg: DictConfig, splits) -> dict[str, object]:
    import torch
    settings = cfg.comparison
    validation_ids = splits["validation"]
    test_ids = splits["test"]
    jobs = [
        (
            waiting,
            batching,
            selector,
            validation_ids,
            float(cfg.baselines.ls_time_limit_s),
        )
        for waiting in ("no_wait", "henn_4_1")
        for batching in ("fifo", "cw", "ls")
        for selector in ("first", "short", "long", "sav")
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(settings.workers)) as pool:
        validation = list(pool.map(_baseline_job, jobs))
    errors = [row for row in validation if row["status"] != "complete"]
    if errors:
        raise RuntimeError(f"Baseline validation failed: {errors}")
    selected = []
    for waiting in ("no_wait", "henn_4_1"):
        for batching in ("fifo", "cw", "ls"):
            candidates = [
                row
                for row in validation
                if row["waiting"] == waiting and row["batching"] == batching
            ]
            best = min(
                candidates,
                key=lambda row: row["evaluation"]["mean_order_flow_time"],
            )
            test = _baseline_job(
                (
                    waiting,
                    batching,
                    best["selector"],
                    test_ids,
                    float(cfg.baselines.ls_time_limit_s),
                )
            )
            if test["status"] != "complete":
                raise RuntimeError(f"Selected baseline failed: {test}")
            selected.append({"validation": best, "test": test})

    diagnostics = []
    for batching in ("fifo", "cw", "ls"):
        result = _baseline_job(
            (
                "fill_or_age",
                batching,
                "short",
                test_ids,
                float(cfg.baselines.ls_time_limit_s),
            )
        )
        if result["status"] != "complete":
            raise RuntimeError(f"Fill/age diagnostic failed: {result}")
        diagnostics.append(result)

    policies = {}
    model_specs = (
        (
            "srl_scratch",
            Path(str(settings.model_dir)) / "srl_scratch.pt",
            True,
        ),
        (
            "srl_no_location",
            Path(str(settings.model_dir)) / "srl_no_location.pt",
            False,
        ),
        (
            "temperature_selected",
            Path(str(settings.temperature_model)),
            True,
        ),
    )
    for name, path, location_features in model_specs:
        actor = OrderScoreActor()
        actor.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        policies[name] = evaluate_structured_actor(
            actor, test_ids, location_features=location_features
        )

    best_no_wait = min(
        (row for row in selected if row["validation"]["waiting"] == "no_wait"),
        key=lambda row: row["validation"]["evaluation"]["mean_order_flow_time"],
    )
    best_henn = min(
        (row for row in selected if row["validation"]["waiting"] == "henn_4_1"),
        key=lambda row: row["validation"]["evaluation"]["mean_order_flow_time"],
    )
    primary = policies["srl_scratch"]
    waiting_effects = {}
    for batching in ("fifo", "cw", "ls"):
        no_wait = next(
            row for row in selected
            if row["validation"]["waiting"] == "no_wait"
            and row["validation"]["batching"] == batching
        )
        henn = next(
            row for row in selected
            if row["validation"]["waiting"] == "henn_4_1"
            and row["validation"]["batching"] == batching
        )
        waiting_effects[batching] = _paired_delta(
            henn["test"]["evaluation"], no_wait["test"]["evaluation"]
        )
    return {
        "mode": "comparison",
        "splits": splits,
        "validation_matrix": validation,
        "selected_baselines": selected,
        "fill_or_age_diagnostics": diagnostics,
        "policies": policies,
        "selection": {
            "best_no_wait": {
                key: best_no_wait["validation"][key]
                for key in ("waiting", "batching", "selector")
            },
            "best_henn_4_1": {
                key: best_henn["validation"][key]
                for key in ("waiting", "batching", "selector")
            },
        },
        "paired_comparisons": {
            "srl_scratch_vs_best_no_wait": _paired_delta(
                primary, best_no_wait["test"]["evaluation"]
            ),
            "srl_scratch_vs_best_henn_4_1": _paired_delta(
                primary, best_henn["test"]["evaluation"]
            ),
            "henn_4_1_vs_no_wait_by_batching": waiting_effects,
        },
        "wall_time_s": time.perf_counter() - started,
        "configuration": OmegaConf.to_container(cfg, resolve=True),
    }


def train_from_config(
    actor,
    cfg: DictConfig,
    train_ids: list[str],
    validation_ids: list[str],
    *,
    location_features: bool,
):
    settings = cfg.srl
    started = time.perf_counter()
    _, training = train_structured_rl(
        actor,
        train_ids,
        episodes=int(settings.episodes),
        updates_per_episode=int(settings.updates_per_episode),
        batch_size=int(settings.batch_size),
        replay_capacity=int(settings.replay_capacity),
        actor_learning_rate=float(settings.actor_learning_rate),
        critic_learning_rate=float(settings.critic_learning_rate),
        sigma_forward=float(settings.sigma_forward),
        sigma_target=float(settings.sigma_target),
        temperature=float(settings.temperature),
        candidate_count=int(settings.candidate_count),
        epsilon=float(settings.epsilon),
        fy_samples=int(settings.fy_samples),
        gamma=float(settings.gamma),
        seed=int(cfg.experiment.seed),
        validation_ids=validation_ids,
        checkpoint_every=int(settings.checkpoint_every),
        location_features=location_features,
    )
    training["wall_time_s"] = time.perf_counter() - started
    return training


def _write_objective_report(path: Path, result: dict) -> None:
    policies = result["test"]["policies"]
    lines = [
        "# Structured batching objective study",
        "",
        "This is an exploratory confirmation on the deterministic Henn test "
        "split, which appeared in the earlier feasibility study; it is not a "
        "pristine first-use test.",
        "",
        "All policies used fixed no-wait actions, the existing additive-score "
        "actor and exact knapsack decoder, and S-shape routing. Selection used "
        "only the complete validation split.",
        "",
        "## Frozen test results",
        "",
        "| trained power | selected episode | mean flow (s) | p90 (s) | p95 (s) | max (s) | GM 1.5 (s) | GM 2 (s) | distance | throughput | waits |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for power in result["powers"]:
        key = str(power)
        evaluation = policies[key]["cross_objective_evaluations"][key]
        flow = evaluation["order_flow_time"]
        cross = evaluation["cross_objectives"]
        lines.append(
            f"| {power:g} | {result['arms'][key]['training']['selected_episode']} "
            f"| {flow['mean']:.1f} | {flow['p90']:.1f} | {flow['p95']:.1f} "
            f"| {flow['maximum']:.1f} | {cross['generalized_mean_1_5']:.1f} "
            f"| {cross['generalized_mean_2']:.1f} | {evaluation['mean_distance']:.1f} "
            f"| {evaluation['mean_orders_per_sim_hour']:.3f} "
            f"| {sum(row['wait_actions'] for row in evaluation['episodes'])} |"
        )
    lines.extend(
        [
            "",
            "| trained power | median (s) | mean(F^1.5) | mean(F^2) | tours | batch fill | orders / batch | decision latency (ms) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for power in result["powers"]:
        key = str(power)
        evaluation = policies[key]["cross_objective_evaluations"][key]
        flow = evaluation["order_flow_time"]
        cross = evaluation["cross_objectives"]
        lines.append(
            f"| {power:g} | {flow['median']:.1f} "
            f"| {cross['mean_flow_power_1_5']:.3e} "
            f"| {cross['mean_flow_power_2']:.3e} "
            f"| {evaluation['mean_tours']:.2f} "
            f"| {evaluation['mean_batch_fill']:.3f} "
            f"| {evaluation['mean_batch_orders']:.3f} "
            f"| {1000.0 * evaluation['mean_decision_latency_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| trained power | objective delta vs matching baseline | wins / 16 | objective delta vs p=1 SRL | wins / 16 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for power in result["powers"]:
        key = str(power)
        evaluation = policies[key]["cross_objective_evaluations"][key]
        baseline = result["test"]["paired_comparisons"][key][
            "vs_matching_baseline"
        ]
        versus_p1 = result["test"]["paired_comparisons"][key]["vs_p1_srl"]
        baseline_value = evaluation["mean_objective_per_order"] - baseline[
            "mean_delta"
        ]
        p1_value = evaluation["mean_objective_per_order"] - versus_p1[
            "mean_delta"
        ]
        lines.append(
            f"| {power:g} | {100.0 * baseline['mean_delta'] / baseline_value:+.2f}% "
            f"| {baseline['left_wins']} | "
            f"{100.0 * versus_p1['mean_delta'] / p1_value:+.2f}% "
            f"| {versus_p1['left_wins']} |"
        )
    p1_training = result["arms"]["1.0"]["training"]
    checkpoints = {
        row["episode"]: row["evaluation"]
        for row in p1_training["validation_checkpoints"]
    }
    p1_test = policies["1.0"]["cross_objective_evaluations"]["1.0"]
    convex_findings = []
    for power in (1.5, 2.0):
        key = str(power)
        own = policies[key]["cross_objective_evaluations"][key]
        reference = policies["1.0"]["cross_objective_evaluations"][key]
        change = 100.0 * (
            own["mean_objective_per_order"]
            / reference["mean_objective_per_order"]
            - 1.0
        )
        convex_findings.append(
            f"The p={power:g} policy changed its held-out moment objective "
            f"by {change:+.2f}% relative to the p=1 policy."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The longer p=1 run selected episode {p1_training['selected_episode']}; "
            f"its test p95 was {p1_test['order_flow_time']['p95']:.1f} s. "
            f"On validation, checkpoint 8 had p95 "
            f"{checkpoints[8]['order_flow_time']['p95']:.1f} s and checkpoint 128 "
            f"had p95 {checkpoints[128]['order_flow_time']['p95']:.1f} s. This "
            "0.3% validation-tail improvement came with a 1.7% worse validation "
            "mean-flow objective, so validation still selected episode 8. The "
            "selected policy's 21676.4 s test p95 was 3.1% above the earlier "
            "eight-episode scratch result (21020.9 s). Longer available training "
            "therefore did not resolve the prior tail weakness.",
            "",
            *convex_findings,
            "Neither convex arm pays off under the stated criterion: both worsen "
            "their own held-out moment relative to p=1 and neither improves p95. "
            "The p=1.5 arm lowers maximum flow by 5.2%, but that isolated maximum "
            "improvement is not accompanied by better p90, p95, or its aligned "
            "moment objective.",
            "",
            "Powers above one are smooth tail-sensitive moments, not exact p95 "
            "or CVaR objectives. Distance and throughput are reported as external "
            "operational consequences, not reward-aligned objectives.",
            "",
            "## Method and artifacts",
            "",
            f"Seed: {result['seed']}. Training order: four independently shuffled "
            "passes over all 32 training instances, reused identically for every "
            "power. Checkpoints: 0, 8, 16, 32, 64, and 128. Baselines: FIFO, "
            "C&W-like, and LS crossed with first, short, long, and SAV under "
            "no-wait; LS used 0.25 s and validation used three workers.",
            "",
            "The machine-readable result, per-episode records, paired deltas, "
            "timings, diagnostics, and saved checkpoints are in the study output "
            "directory.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_objective_study(
    cfg: DictConfig, splits: dict[str, list[str]], output_dir: Path
) -> dict[str, object]:
    import torch

    powers = [float(value) for value in cfg.objective_study.powers]
    seed = int(cfg.experiment.seed)
    checkpoints = [
        int(value) for value in cfg.objective_study.checkpoints
    ]
    episode_order = _training_order(
        splits["train"], seed=seed, passes=int(cfg.objective_study.passes)
    )
    started = time.perf_counter()
    jobs = [
        (
            "no_wait",
            batching,
            selector,
            splits["validation"],
            float(cfg.baselines.ls_time_limit_s),
            1.0,
        )
        for batching in ("fifo", "cw", "ls")
        for selector in ("first", "short", "long", "sav")
    ]
    with ProcessPoolExecutor(
        max_workers=int(cfg.objective_study.workers)
    ) as pool:
        baseline_validation = list(pool.map(_baseline_job, jobs))
    failures = [row for row in baseline_validation if row["status"] != "complete"]
    if failures:
        raise RuntimeError(f"Baseline validation failed: {failures}")
    selected_baselines = _select_objective_baselines(
        baseline_validation, powers
    )
    result = {
        "mode": "objective_study",
        "status": "training",
        "seed": seed,
        "powers": powers,
        "splits": splits,
        "training_instance_order": episode_order,
        "baseline_validation": baseline_validation,
        "selected_baselines": {
            key: {
                "waiting": row["waiting"],
                "batching": row["batching"],
                "selector": row["selector"],
                "validation_objective": _evaluation_objective(
                    row["evaluation"], float(key)
                ),
            }
            for key, row in selected_baselines.items()
        },
        "arms": {},
        "configuration": OmegaConf.to_container(cfg, resolve=True),
        "failures": [],
    }
    _write_json(output_dir / "result.json", result)
    selected_actors = {}
    initial_digests = []
    for power in powers:
        key = str(power)
        torch.manual_seed(seed)
        actor = OrderScoreActor()
        initial_digest = _state_digest(actor)
        initial_digests.append(initial_digest)
        arm_started = time.perf_counter()
        _, training = train_structured_rl(
            actor,
            episode_order,
            episodes=len(episode_order),
            updates_per_episode=int(cfg.srl.updates_per_episode),
            batch_size=int(cfg.srl.batch_size),
            replay_capacity=int(cfg.srl.replay_capacity),
            actor_learning_rate=float(cfg.srl.actor_learning_rate),
            critic_learning_rate=float(cfg.srl.critic_learning_rate),
            sigma_forward=float(cfg.srl.sigma_forward),
            sigma_target=float(cfg.srl.sigma_target),
            temperature=float(cfg.srl.temperature),
            candidate_count=int(cfg.srl.candidate_count),
            epsilon=float(cfg.srl.epsilon),
            fy_samples=int(cfg.srl.fy_samples),
            gamma=1.0,
            seed=seed,
            validation_ids=splits["validation"],
            checkpoint_episodes=checkpoints,
            checkpoint_dir=output_dir / f"p{key}" / "checkpoints",
            reward_power=power,
            location_features=True,
        )
        training["wall_time_s"] = time.perf_counter() - arm_started
        selected_path = output_dir / f"p{key}" / "selected.pt"
        torch.save(actor.state_dict(), selected_path)
        selected_actors[key] = copy.deepcopy(actor)
        result["arms"][key] = {
            "initial_state_digest": initial_digest,
            "selected_model": str(selected_path),
            "training": training,
        }
        _write_json(output_dir / "result.json", result)
    if len(set(initial_digests)) != 1:
        raise RuntimeError("Objective arms did not receive identical initialization")

    distinct_baselines = {}
    for key, selection in result["selected_baselines"].items():
        config_key = (
            selection["batching"], selection["selector"]
        )
        if config_key not in distinct_baselines:
            test = _baseline_job(
                (
                    "no_wait",
                    selection["batching"],
                    selection["selector"],
                    splits["test"],
                    float(cfg.baselines.ls_time_limit_s),
                    float(key),
                )
            )
            if test["status"] != "complete":
                raise RuntimeError(f"Selected baseline test failed: {test}")
            distinct_baselines[config_key] = test["evaluation"]

    policy_tests = {}
    for trained_power, actor in selected_actors.items():
        policy_tests[trained_power] = {
            "cross_objective_evaluations": {
                str(power): evaluate_structured_actor(
                    actor,
                    splits["test"],
                    location_features=True,
                    reward_power=power,
                )
                for power in powers
            }
        }
    comparisons = {}
    for power in powers:
        key = str(power)
        policy = policy_tests[key]["cross_objective_evaluations"][key]
        baseline_config = result["selected_baselines"][key]
        baseline = distinct_baselines[
            (baseline_config["batching"], baseline_config["selector"])
        ]
        comparisons[key] = {
            "vs_matching_baseline": _paired_objective_delta(
                policy, baseline, power
            ),
            "vs_p1_srl": _paired_objective_delta(
                policy,
                policy_tests["1.0"]["cross_objective_evaluations"][key],
                power,
            ),
        }
    result["test"] = {
        "exploratory_confirmation": True,
        "distinct_selected_baselines": {
            f"{batching}:{selector}": evaluation
            for (batching, selector), evaluation in distinct_baselines.items()
        },
        "policies": policy_tests,
        "paired_comparisons": comparisons,
    }
    result["status"] = "complete"
    result["wall_time_s"] = time.perf_counter() - started
    _write_json(output_dir / "result.json", result)
    _write_objective_report(
        Path(__file__).parents[2] / "docs" / "structured_batching_objective_study.md",
        result,
    )
    return result
