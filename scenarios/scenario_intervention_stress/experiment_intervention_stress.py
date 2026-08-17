from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import DictConfig

from casim.io_helpers import dump_json, dump_jsonl
from casim.viz.app import launch
from casim.setup import build_runtime
from scenarios.scenario_intervention_stress.scenario_specific_hooks import (
    add_orders_hook,
)


def _decision_row(snapshot, solution, decision) -> dict:
    dynamic = snapshot.dynamic_warehouse_info
    row = {
        "time": float(dynamic.time),
        "problem": str(snapshot.problem_class),
        "pipeline": str(decision["pipeline"]),
        "algorithm_runtime_s": float(solution.execution_time),
        "decision_elapsed_s": float(decision["decision_elapsed_s"]),
        "solver_status": solution.solver_status,
        "objective_value": solution.objective_value,
        "objective_bound": solution.objective_bound,
        "optimality_gap": solution.optimality_gap,
        "is_optimal": solution.is_optimal,
        "available_order_ids": sorted(
            int(order.order_id) for order in snapshot.orders.orders
        ),
    }
    if dynamic.active_tour_id is not None:
        old_batch = set(dynamic.buffered_batches[0].order_numbers)
        new_batch = set(solution.routes[0].batch.order_numbers)
        row.update(
            {
                "picker_id": int(dynamic.current_picker.id),
                "tour_id": int(dynamic.active_tour_id),
                "route_version": int(dynamic.route_version),
                "origin_type": dynamic.origin_type,
                "inserted_order_ids": sorted(new_batch - old_batch),
                "old_order_ids": sorted(old_batch),
                "new_order_ids": sorted(new_batch),
                "bin_owners_before": [
                    list(owners) for owners in dynamic.cart_bin_order_ids
                ],
                "bin_owners_after": [
                    list(owners)
                    for _, owners in sorted(
                        solution.routes[0].batch.bin_assignments.items()
                    )
                ],
                "new_residual_distance": float(
                    solution.routes[0].distance
                ),
            }
        )
    else:
        row["committed_order_ids"] = sorted(
            int(order_id)
            for job in solution.jobs
            for order_id in job.order_numbers
        )
        row["picker_ids"] = sorted(
            {int(job.picker_id) for job in solution.jobs}
        )
    return row


def _validate_configuration(simulation, initial_domain) -> list[str]:
    warnings = []
    conditions = simulation.conditions_map.get(("OBRSP", "none"))
    if not conditions or not simulation.state.intervention_enabled:
        return warnings
    order_threshold = None
    for cond in conditions:
        if cond.key == "orders":
            order_threshold = cond.value
            break
    if order_threshold is None:
        return warnings
    arrivals = sorted(
        float(order.order_date or 0.0)
        for order in initial_domain.orders.orders
    )
    threshold = int(order_threshold)
    first_dispatch = (
        arrivals[threshold - 1]
        if 0 < threshold <= len(arrivals)
        else None
    )
    later_arrivals = (
        sum(value > first_dispatch for value in arrivals)
        if first_dispatch is not None
        else 0
    )
    if later_arrivals == 0:
        if first_dispatch is None:
            warnings.append(
                "The order threshold exceeds this finite order stream; "
                "dispatch starts only during final draining, so "
                "arrival-driven intervention cannot occur."
            )
        else:
            warnings.append(
                "The order threshold is first reachable at or after the "
                "final arrival; arrival-driven intervention cannot occur "
                "before final draining."
            )
    return warnings


def run(cfg: DictConfig) -> dict:
    simulation, decision_engine = build_runtime(cfg)
    initial_domain = simulation.reset(hooks=[add_orders_hook])
    configuration_warnings = _validate_configuration(simulation, initial_domain)

    decisions = []
    while True:
        done, snapshot = simulation.run()
        if done:
            break
        events, solution = decision_engine.on_trigger(snapshot)
        decision = decision_engine.decision_tracker.decisions[-1]
        decisions.append(_decision_row(snapshot, solution, decision))
        simulation.step(events)

    tracker = simulation.state.tracker
    completed = tracker.completed_tours
    result = {
        "scenario": "intervention_stress",
        "policy": str(cfg.intervention_repo.name),
        "status": (
            "success"
            if simulation.state.completion_reason == "drained"
            else simulation.state.completion_reason
        ),
        "completion_reason": simulation.state.completion_reason,
        "configuration_warnings": configuration_warnings,
        "unfinished_work": simulation.state.unfinished_work(),
        "makespan": max(
            (float(tour[2]) for tour in completed),
            default=0.0,
        ),
        "completed_orders": len(
            simulation.state.order_manager.completed_orders
        ),
        "completed_tours": len(completed),
        "total_distance": float(sum(tracker.distance_by_picker.values())),
        "distance_by_picker": {
            str(key): float(value)
            for key, value in tracker.distance_by_picker.items()
        },
        "interventions": len(tracker.interventions),
        "inserted_orders": sum(
            len(intervention["inserted_order_ids"])
            for intervention in tracker.interventions
        ),
        "decisions": len(decisions),
        "decision_problem_counts": dict(
            Counter(row["problem"] for row in decisions)
        ),
        "pipeline_counts": dict(
            decision_engine.decision_tracker.pipeline_counts
        ),
        "algorithm_runtime_s": sum(
            row["algorithm_runtime_s"] for row in decisions
        ),
        "decision_elapsed_s": sum(
            row["decision_elapsed_s"] for row in decisions
        ),
    }
    output_dir = Path(cfg.experiment.output_dir)
    dump_json(output_dir / "result.json", result)
    dump_jsonl(output_dir / "decisions.jsonl", decisions)
    return result


@hydra.main(
    version_base="1.3",
    config_path="config",
    config_name="intervention_stress_config",
)
def main(cfg: DictConfig) -> None:
    result = run(cfg)
    print(json.dumps(result, sort_keys=True))
    if cfg.viz.launch:
        launch(
            Path(cfg.experiment.output_dir) / "viz",
            port=int(cfg.viz.port),
        )


if __name__ == "__main__":
    main()
