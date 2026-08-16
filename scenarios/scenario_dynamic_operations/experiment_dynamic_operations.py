from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import DictConfig

from casim.io_helpers import dump_json, dump_jsonl
from casim.setup import build_runtime
from casim.viz.app import launch
from scenarios.scenario_dynamic_operations.scenario_specific_hooks import (
    build_hooks,
)


def _projected_count(snapshot) -> int:
    if snapshot.orders.orders:
        return len(snapshot.orders.orders)
    return len(snapshot.dynamic_warehouse_info.buffered_batches or [])


def run(cfg: DictConfig) -> dict:
    simulation, decision_engine = build_runtime(cfg)
    initial_domain = simulation.reset(hooks=build_hooks(cfg))

    decisions = []
    while True:
        done, snapshot = simulation.run()
        if done:
            break
        state = simulation.state
        raw_before = len(state.order_manager.get_order_buffer())
        batches_before = len(state.order_manager.get_pick_list_buffer())
        selected = decision_engine.on_trigger(snapshot)
        if selected is None:
            raise RuntimeError(
                f"No decision for {snapshot.problem_class} at "
                f"t={snapshot.dynamic_warehouse_info.time}"
            )
        events, solution = selected
        decision = decision_engine.decision_tracker.decisions[-1]
        commitment = decision_engine.decision_tracker.commitments[-1]
        row = {
            "time": float(snapshot.dynamic_warehouse_info.time),
            "problem": str(snapshot.problem_class),
            "replanning": str(snapshot.dynamic_warehouse_info.replanning),
            "pipeline": str(decision[3]),
            "raw_buffer_before": raw_before,
            "batch_buffer_before": batches_before,
            "projected_candidates": _projected_count(snapshot),
            "solution_count": int(commitment["returned"]),
            "committed_count": int(commitment["committed"]),
            "deferred_count": int(commitment["deferred"]),
            "algorithm_runtime_s": float(solution.execution_time),
            "decision_elapsed_s": float(decision[7]),
            "replanned_tours": len(
                snapshot.dynamic_warehouse_info.replannable_tours or []
            ) if snapshot.dynamic_warehouse_info.replanning == "unstarted" else 0,
        }
        if hasattr(solution, "batches"):
            row["committed_order_count"] = sum(
                len(batch.orders) for batch in solution.batches
            )
        elif hasattr(solution, "jobs"):
            row["committed_order_count"] = sum(
                len(job.order_numbers) for job in solution.jobs
            )
        else:
            row["committed_order_count"] = 0
        simulation.step(events, snapshot.problem_class, solution)
        row["raw_buffer_after"] = len(
            state.order_manager.get_order_buffer()
        )
        row["batch_buffer_after"] = len(
            state.order_manager.get_pick_list_buffer()
        )
        decisions.append(row)

    state = simulation.state
    tracker = state.tracker
    end_time = float(state.current_time)
    available = tracker.available_time_by_picker(end_time)
    completed_orders = len(state.order_manager.completed_orders)
    result = {
        "scenario": "dynamic_operations",
        "policy": str(cfg.engines.name),
        "profile": str(cfg.simulation.name),
        "status": state.completion_reason,
        "completion_reason": state.completion_reason,
        "simulation_time": end_time,
        "received_orders": len(initial_domain.orders.orders),
        "completed_orders": completed_orders,
        "completed_tours": len(tracker.completed_tours),
        "on_time_orders": len(tracker.all_on_time),
        "delayed_orders": len(tracker.all_delayed),
        "on_time_ratio": tracker.on_time_ratio,
        "total_distance": float(sum(tracker.distance_by_picker.values())),
        "utilization": tracker.current_utilization(end_time),
        "available_time_by_picker": {
            str(key): float(value) for key, value in available.items()
        },
        "buffer_flow": {
            "batched_orders": sum(
                row["committed_order_count"]
                for row in decisions
                if row["problem"] == "OBP"
            ),
            "scheduled_orders": sum(
                row["committed_order_count"]
                for row in decisions
                if row["problem"] == "ORSP"
            ),
            "replanned_tours": sum(
                row["replanned_tours"] for row in decisions
            ),
            "raw_orders": len(state.order_manager.get_order_buffer()),
            "buffered_batches": len(
                state.order_manager.get_pick_list_buffer()
            ),
            "nonterminal_tours": len(
                state.unfinished_work()["nonterminal_tours"]
            ),
            "unfinished_orders": (
                len(initial_domain.orders.orders)
                - completed_orders
            ),
        },
        "unfinished_work": state.unfinished_work(),
        "decisions": len(decisions),
        "decision_problem_counts": dict(
            Counter(row["problem"] for row in decisions)
        ),
        "interventions": len(tracker.interventions),
        "route_replacements": sum(
            bool(value["replaced"]) for value in tracker.interventions
        ),
        "algorithm_runtime_s": sum(
            row["algorithm_runtime_s"] for row in decisions
        ),
        "decision_elapsed_s": sum(
            row["decision_elapsed_s"] for row in decisions
        ),
        "simulation_decision_latency": 0.0,
    }
    output = Path(cfg.experiment.output_dir)
    dump_json(output / "result.json", result)
    dump_jsonl(output / "decisions.jsonl", decisions)
    return result


@hydra.main(
    version_base="1.3",
    config_path="config",
    config_name="dynamic_operations_config",
)
def main(cfg: DictConfig) -> None:
    result = run(cfg)
    print(json.dumps(result, sort_keys=True))
    if cfg.viz.launch:
        launch(Path(cfg.experiment.output_dir) / "viz", port=int(cfg.viz.port))


if __name__ == "__main__":
    main()
