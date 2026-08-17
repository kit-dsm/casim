from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from casim.io_helpers import dump_json, dump_jsonl
from casim.setup import build_runtime
from scenarios.scenario_ijpe.scenario_specific_hooks import build_sim_hooks


def run(cfg: DictConfig) -> dict:
    simulation, decision_engine = build_runtime(cfg)
    initial_domain = simulation.reset(hooks=build_sim_hooks(cfg))

    while True:
        done, snapshot = simulation.run()
        if done:
            break
        events, solution = decision_engine.on_trigger(snapshot)
        simulation.step(events)

    tracker = simulation.state.tracker
    decisions = [
        {
            "problem": row["problem_class"],
            "replanning": row["replanning"],
            "solution_order_count": row["solution_order_count"],
            "pipeline": row["pipeline"],
            "objective": row["objective"],
            "objective_value": row["objective_value"],
            "algorithm_runtime_s": row["algorithm_runtime_s"],
            "decision_elapsed_s": row["decision_elapsed_s"],
        }
        for row in decision_engine.decision_tracker.decisions
    ]
    result = {
        "scenario": "scenario_ijpe",
        "input": str(cfg.input.load.orders_path),
        "simulation": str(cfg.simulation.name),
        "status": "success",
        "completion_reason": simulation.state.completion_reason,
        "simulation_time": float(simulation.state.current_time),
        "initial_orders": len(initial_domain.orders.orders),
        "completed_orders": len(simulation.state.order_manager.completed_orders),
        "completed_tours": len(tracker.completed_tours),
        "on_time_orders": len(tracker.all_on_time),
        "delayed_orders": len(tracker.all_delayed),
        "total_distance": float(sum(tracker.distance_by_picker.values())),
        "shifted_orders": len(tracker.shifted_orders),
        "unfinished_work": simulation.state.unfinished_work(),
        "decisions": len(decisions),
    }
    output = Path(cfg.experiment.output_dir)
    dump_json(output / "result.json", result)
    dump_jsonl(output / "decisions.jsonl", decisions)
    return result


@hydra.main(
    config_path="config",
    config_name="ijpe_config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
