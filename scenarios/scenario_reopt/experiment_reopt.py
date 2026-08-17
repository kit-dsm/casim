from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from casim.io_helpers import dump_json, dump_jsonl
from casim.setup import build_runtime
from scenarios.scenario_reopt.scenario_specific_hooks import build_sim_hooks


def _committed_order_ids(solution) -> list[int]:
    return sorted(
        int(order_id)
        for job in solution.jobs
        for order_id in job.order_numbers
    )


def _base_result(
    cfg: DictConfig,
    makespan: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        "instance": str(cfg.experiment.instance_name),
        "variant": str(cfg.variant.name),
        "objective": "makespan",
        "objective_value": float(makespan),
        "makespan": float(makespan),
        "solver": str(cfg.variant.solver),
        "status": "success",
        "exactness": OmegaConf.to_container(
            cfg.variant.exactness,
            resolve=True,
        ),
    }
    return result


def run_experiment(cfg: DictConfig) -> dict[str, object]:
    variant = str(cfg.variant.name)
    if variant not in {"cios", "reopt", "no_wait"}:
        raise ValueError(f"Unsupported reoptimization variant: {variant}")

    output_dir = Path(cfg.experiment.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    simulation, decision_engine = build_runtime(cfg)
    initial_domain = simulation.reset(hooks=build_sim_hooks())

    if str(cfg.variant.execution) == "complete_information":
        solver = decision_engine.solver_for("OBRSP")
        solution, solver_name, makespan = solver.solve(
            initial_domain
        )
        result = _base_result(cfg, makespan)
        result["algorithm"] = solver_name
        result["runtime_s"] = float(solution.execution_time)
        result["explored_states"] = int(solution.explored_states)
        result["solver_status"] = solution.solver_status
        result["objective_value"] = solution.objective_value
        result["is_optimal"] = solution.is_optimal
        result["batches"] = [
            {
                "order_ids": sorted(int(value) for value in job.order_numbers),
                "item_sequence": [
                    list(node) for node in job.job.route.item_sequence
                ],
                "start_time": float(job.start_time),
                "end_time": float(job.end_time),
            }
            for job in solution.jobs
        ]
        dump_json(output_dir / "result.json", result)
        return result

    trace: list[dict[str, object]] = []
    while True:
        done, snapshot = simulation.run()
        if done:
            break
        if snapshot is None:
            raise RuntimeError("Simulation paused without a decision snapshot")

        decision_time = float(snapshot.dynamic_warehouse_info.time)
        available = sorted(
            int(order.order_id) for order in snapshot.orders.orders
        )
        events, solution = decision_engine.on_trigger(snapshot)
        decision_record = decision_engine.decision_tracker.decisions[-1]
        row: dict[str, object] = {
            "decision_time": decision_time,
            "available_order_ids": available,
            "committed_order_ids": _committed_order_ids(solution),
            "solver": "LorenzDPSolver" if variant == "reopt" else "CoSySolver",
            "pipeline_identifier": decision_record["pipeline"],
            "algorithm_runtime_s": float(solution.execution_time),
            "decision_elapsed_s": float(decision_record["decision_elapsed_s"]),
            "solver_status": solution.solver_status,
            "objective_value": solution.objective_value,
            "objective_bound": solution.objective_bound,
            "optimality_gap": solution.optimality_gap,
            "is_optimal": solution.is_optimal,
        }
        row["planned_completion_time"] = float(
            max(job.end_time for job in solution.jobs)
        )
        simulation.step(events)
        trace.append(row)

    completed = simulation.state.tracker.completed_tours
    if not completed:
        raise RuntimeError("Online run completed without a picker tour")
    makespan = max(float(tour[2]) for tour in completed)
    result = _base_result(cfg, makespan)
    result["completion_reason"] = simulation.state.completion_reason
    result["status"] = (
        "success"
        if simulation.state.completion_reason == "drained"
        else simulation.state.completion_reason
    )
    result["unfinished_work"] = simulation.state.unfinished_work()
    result["pipeline_identifiers"] = sorted(
        decision_engine.decision_tracker.pipeline_counts
    )
    if variant == "no_wait":
        pipelines = sorted(
            decision_engine.decision_tracker.pipeline_counts.keys()
        )
        if len(pipelines) != 1:
            raise RuntimeError(
                "The no_wait configuration must execute exactly one pipeline; "
                f"observed {pipelines}"
            )
        result["pipeline_identifier"] = pipelines[0]
    result["num_completed_orders"] = len(
        simulation.state.order_manager.completed_orders
    )
    result["num_interventions"] = len(
        simulation.state.tracker.interventions
    )
    result["total_distance"] = float(
        sum(simulation.state.tracker.distance_by_picker.values())
    )
    result["distance_by_picker"] = {
        str(key): float(value)
        for key, value in simulation.state.tracker.distance_by_picker.items()
    }
    result["completed_tours"] = [
        {
            "tour_id": int(tour[0]),
            "start_time": float(tour[1]),
            "end_time": float(tour[2]),
            "order_ids": sorted(int(value) for value in tour[3]),
            "picker_id": int(tour[4]),
            "n_lines": int(tour[7]),
        }
        for tour in completed
    ]
    if simulation.state.tracker.interventions:
        result["interventions"] = simulation.state.tracker.interventions
    dump_json(output_dir / "result.json", result)
    dump_jsonl(output_dir / "decision_trace.jsonl", trace)
    return result


@hydra.main(
    version_base="1.3",
    config_path="config",
    config_name="reopt_config",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.experiment.output_dir).resolve()
    try:
        result = run_experiment(cfg)
    except Exception as error:
        dump_json(
            output_dir / "result.json",
            {
                "instance": str(cfg.experiment.instance_name),
                "variant": str(cfg.variant.name),
                "solver": str(cfg.variant.solver),
                "status": "failed",
                "error": {
                    "category": type(error).__name__,
                    "message": str(error),
                },
            },
        )
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
