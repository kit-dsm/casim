from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from ware_ops_algos.algorithms import CombinedRoutingSolution

from casim.events.operational_events import OrderArrival
from casim.io_helpers import dump_json, dump_jsonl
from casim.setup import build_runtime
from ware_ops_algos.domain_models import load_and_flatten_data_card
from scenarios.scenario_henn.algorithm import (
    HennWakeUp,
    decide_henn,
    single_order_service_times,
)
from scenarios.scenario_henn.loader import load_manifest
from scenarios.scenario_henn.reference import (
    compare_benchmarks,
    load_references,
    normalize_actual,
)
from scenarios.scenario_henn.scenario_specific_hooks import build_sim_hooks


def _instance_config(
    cfg: DictConfig,
    instance_id: str,
    output_dir: Path,
) -> DictConfig:
    instance_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=False)
    )
    OmegaConf.update(
        instance_cfg,
        "input.load.instance_id",
        instance_id,
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        instance_cfg,
        "experiment.output_dir",
        str(output_dir),
        merge=False,
    )
    OmegaConf.update(
        instance_cfg,
        "experiment.instance_name",
        (
            f"{instance_id}/"
            f"{instance_cfg.batching.name}-{instance_cfg.selection.name}"
        ),
        merge=False,
    )
    return instance_cfg


def _next_arrival(simulation) -> float | None:
    arrivals = [
        event.time
        for event in simulation.events
        if isinstance(event, OrderArrival)
    ]
    return min(arrivals) if arrivals else None


def _trace_row(
    decision_index: int,
    snapshot,
    decision,
    solver_name: str,
    objective_value: float | None,
) -> dict[str, object]:
    row = {
        "decision": decision_index,
        "current_time_s": float(snapshot.dynamic_warehouse_info.time),
        "available_order_ids": sorted(
            int(order.order_id) for order in snapshot.orders.orders
        ),
        "action": decision.action,
        "reason": decision.reason,
        "wait_until_s": decision.wait_until,
        "solver": solver_name,
        "candidate_objective": objective_value,
    }
    row.update(decision.details)
    if decision.solution is not None:
        row["dispatch_time_s"] = float(snapshot.dynamic_warehouse_info.time)
        row["planned_jobs"] = [
            {
                "order_ids": sorted(job.job.route.batch.order_numbers),
                "route_start_time_s": float(job.start_time),
                "completion_time_s": float(job.end_time),
            }
            for job in decision.solution.jobs
        ]
    return row


def run_henn_experiment(
    cfg: DictConfig,
    simulation,
    decision_engine,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    try:
        initial_domain = simulation.reset(hooks=build_sim_hooks(cfg))
    except Exception as exc:
        raise RuntimeError(f"INPUT_DATA: {exc}") from exc
    arrivals = {
        int(order.order_id): float(order.order_date)
        for order in initial_domain.orders.orders
    }
    solver = decision_engine.solver_for(initial_domain.problem_class)
    single_service_cache: dict[int, float] = {}
    trace: list[dict[str, object]] = []
    decision_index = 0

    while True:
        try:
            done, snapshot = simulation.run()
        except Exception as exc:
            raise RuntimeError(f"SIMULATION: {exc}") from exc
        if done:
            break
        if snapshot is None:
            raise RuntimeError("Simulation paused without a state snapshot")

        decision_index += 1
        if decision_index > int(cfg.experiment.max_decisions):
            raise RuntimeError(
                f"Exceeded {cfg.experiment.max_decisions} decisions"
            )

        try:
            result = decision_engine.solve(snapshot, action=None)
        except Exception as exc:
            raise RuntimeError(f"CANDIDATE_GENERATION: {exc}") from exc
        if result is None:
            raise RuntimeError(
                "CANDIDATE_GENERATION: CoSy produced no candidate solution"
            )
        candidate, solver_name, objective_value = result
        try:
            if not isinstance(candidate, CombinedRoutingSolution):
                raise TypeError(
                    "The configured Henn CoSy endpoint must return "
                    "CombinedRoutingSolution"
                )
            single_order_service_times(
                snapshot,
                solver,
                single_service_cache,
            )
        except Exception as exc:
            raise RuntimeError(f"CANDIDATE_GENERATION: {exc}") from exc

        next_arrival = _next_arrival(simulation)
        try:
            decision = decide_henn(
                candidate=candidate,
                snapshot=snapshot,
                current_time=float(simulation.state.current_time),
                next_arrival=next_arrival,
                stream_exhausted=next_arrival is None,
                selector=str(cfg.selection.name),
                single_services=single_service_cache,
                waiting_policy=str(cfg.waiting.name),
                fill_threshold=float(cfg.waiting.fill_threshold or 0.75),
                max_age_s=float(cfg.waiting.max_age_s or 300.0),
            )
        except Exception as exc:
            raise RuntimeError(f"DECISION_RULE: {exc}") from exc
        trace.append(
            _trace_row(
                decision_index,
                snapshot,
                decision,
                solver_name,
                objective_value,
            )
        )

        if decision.action == "wait":
            if decision.wait_until is not None:
                simulation.add_event(HennWakeUp(decision.wait_until))
            continue

        try:
            events, committed = decision_engine.commit(
                snapshot, decision.solution
            )
            simulation.step(
                events,
                snapshot.problem_class,
                committed,
            )
        except Exception as exc:
            raise RuntimeError(f"SIMULATION: {exc}") from exc

    algorithm_id = (
        f"{cfg.waiting.name}+{cfg.batching.name}+"
        f"s_shape+{cfg.selection.name}"
    )
    actual = normalize_actual(
        instance_id=str(cfg.input.load.instance_id),
        algorithm_id=algorithm_id,
        completed_tours=simulation.state.tracker.completed_tours,
        arrivals=arrivals,
    )
    return actual, trace


def _write_report(
    path: Path,
    actual: dict[str, object],
    comparisons: list[dict[str, object]],
) -> None:
    lines = [
        f"# Henn Algorithm 4.1: {actual['instance_id']}",
        "",
        f"Algorithm: `{actual['algorithm_id']}`",
        "",
        f"- Completion time: {actual['completion_time_s']:.3f} s",
        f"- Maximum turnover: {actual['max_turnover_time_s']:.3f} s",
        f"- Executed batches: {len(actual['batches'])}",
        "",
        "Gil and Best Known values are cross-algorithm benchmarks, not "
        "Algorithm 4.1 reproduction targets.",
        "",
        "| Benchmark | Completion | Maximum turnover |",
        "|---|---:|---:|",
    ]
    for report in comparisons:
        completion = report["completion_time"]
        turnover = report["max_turnover_time"]
        lines.append(
            f"| {report['reference_algorithm_id']} | "
            f"{completion['status']} ({completion['signed_gap']:+.3f} s) | "
            f"{turnover['status']} ({turnover['signed_gap']:+.3f} s) |"
        )
    lines.extend(
        [
            "",
            "Reference batch membership, dispatch times, and route sequences "
            "are `NOT_AVAILABLE`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_instance(
    cfg: DictConfig,
    instance_id: str,
    references: dict[str, dict[str, dict[str, object]]],
    output_root: Path,
) -> dict[str, object]:
    instance_dir = output_root / "instances" / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    instance_cfg = _instance_config(cfg, instance_id, instance_dir)

    resolved = OmegaConf.to_yaml(instance_cfg, resolve=True)
    (instance_dir / "resolved_config.yaml").write_text(
        resolved,
        encoding="utf-8",
    )
    config_digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()

    try:
        data_card = load_and_flatten_data_card(instance_cfg.data_card)
        simulation, decision_engine = build_runtime(
            instance_cfg, data_card
        )
    except Exception as exc:
        raise RuntimeError(f"CONFIGURATION: {exc}") from exc
    actual, trace = run_henn_experiment(
        instance_cfg,
        simulation,
        decision_engine,
    )
    actual["config_digest"] = config_digest
    try:
        comparisons = compare_benchmarks(
            actual,
            references[instance_id],
            absolute_tolerance_s=float(
                instance_cfg.comparison.absolute_tolerance_s
            ),
            relative_tolerance=float(
                instance_cfg.comparison.relative_tolerance
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"OBJECTIVE_COMPARISON: {exc}") from exc

    dump_json(instance_dir / "actual.json", actual)
    dump_json(instance_dir / "comparison.json", comparisons)
    dump_jsonl(instance_dir / "decision_trace.jsonl", trace)
    _write_report(instance_dir / "report.md", actual, comparisons)

    comparison_by_id = {
        report["reference_algorithm_id"]: report
        for report in comparisons
    }
    gil = comparison_by_id["gil_2020_grasp_vnd"]
    best = comparison_by_id["best_known"]
    return {
        "instance_id": instance_id,
        "status": "COMPLETED",
        "algorithm_id": actual["algorithm_id"],
        "batching": str(cfg.batching.name),
        "selection": str(cfg.selection.name),
        "completion_time_s": actual["completion_time_s"],
        "max_turnover_time_s": actual["max_turnover_time_s"],
        "gil_completion": gil["completion_time"]["status"],
        "gil_turnover": gil["max_turnover_time"]["status"],
        "best_completion": best["completion_time"]["status"],
        "best_turnover": best["max_turnover_time"]["status"],
        "first_observable_difference": (
            gil["first_observable_difference"]
            or best["first_observable_difference"]
        ),
    }


def _write_summary(output_root: Path, rows: list[dict[str, object]]) -> None:
    dump_json(output_root / "summary.json", rows)
    fieldnames = sorted({key for row in rows for key in row})
    with (output_root / "summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if row["status"] != "COMPLETED"]
    dump_json(output_root / "failures.json", failures)


def _failure(
    instance_id: str,
    phase: str,
    exc: Exception,
) -> dict[str, object]:
    message = str(exc)
    prefix, separator, remainder = message.partition(":")
    valid_phases = {
        "INPUT_DATA",
        "REFERENCE_DATA",
        "CONFIGURATION",
        "CANDIDATE_GENERATION",
        "DECISION_RULE",
        "SIMULATION",
        "OBJECTIVE_COMPARISON",
    }
    if separator and prefix in valid_phases:
        phase = prefix
        message = remainder.strip()
    return {
        "instance_id": instance_id,
        "status": "ERROR",
        "failure_category": phase,
        "message": message,
    }


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="henn_config",
)
def main(cfg: DictConfig) -> None:
    output_root = Path(cfg.experiment.output_dir).resolve()
    manifest_path = Path(cfg.input.data_loader.instances_dir) / str(
        cfg.input.load.manifest_path
    )
    try:
        entries = load_manifest(manifest_path)
    except Exception as exc:
        raise RuntimeError(f"INPUT_DATA: {exc}") from exc
    try:
        references = load_references(cfg.reference.workbook_path)
    except Exception as exc:
        raise RuntimeError(f"REFERENCE_DATA: {exc}") from exc
    if set(entries) != set(references):
        raise ValueError(
            "Manifest and workbook instance IDs do not map one-to-one"
        )

    if cfg.experiment.mode == "single":
        instance_ids = [str(cfg.experiment.instance_id)]
    elif cfg.experiment.mode == "reference_set":
        instance_ids = list(entries)
    else:
        raise ValueError(f"Unknown experiment mode: {cfg.experiment.mode}")

    rows = []
    for instance_id in instance_ids:
        phase = "CONFIGURATION"
        try:
            phase = "INPUT_DATA"
            if instance_id not in entries:
                raise ValueError(f"Unknown instance ID: {instance_id}")
            phase = "SIMULATION"
            rows.append(
                run_instance(
                    cfg,
                    instance_id,
                    references,
                    output_root,
                )
            )
        except Exception as exc:
            failure = _failure(instance_id, phase, exc)
            dump_json(
                output_root
                / "instances"
                / instance_id
                / "failure.json",
                failure,
            )
            rows.append(failure)
            if cfg.experiment.mode == "single":
                _write_summary(output_root, rows)
                raise
    _write_summary(output_root, rows)


if __name__ == "__main__":
    main()
