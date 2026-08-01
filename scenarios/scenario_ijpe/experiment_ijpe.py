from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from scenarios.experiment_commons import (
    load_and_flatten_data_card,
    setup_decision_engine,
    setup_scenario,
)
from scenarios.scenario_ijpe.scenario_specific_hooks import build_sim_hooks


def run(cfg: DictConfig) -> dict:
    card = load_and_flatten_data_card(cfg.data_card)
    simulation = setup_scenario(cfg)
    decision_engine = setup_decision_engine(
        cfg,
        card,
        simulation.state_adapters,
    )
    decision_engine.event_map = {
        name: instantiate(event)
        for name, event in cfg.engines.decision_engine.event_map.items()
    }
    initial_domain = simulation.reset(hooks=build_sim_hooks(cfg))

    while True:
        done, snapshot = simulation.run()
        if done:
            break
        selected = decision_engine.on_trigger(snapshot)
        if selected is None:
            raise RuntimeError(
                f"No decision for {snapshot.problem_class} at "
                f"t={snapshot.dynamic_warehouse_info.time}"
            )
        events, solution = selected
        simulation.step(events, snapshot.problem_class, solution)

    tracker = simulation.state.tracker
    decisions = [
        {
            "problem": row[0],
            "input_count": row[1],
            "pipeline": row[2],
            "objective_value": row[3],
            "objective": row[4],
            "algorithm_runtime_s": row[5],
            "decision_elapsed_s": row[6],
        }
        for row in decision_engine.decision_tracker.decisions
    ]
    result = {
        "scenario": "scenario_ijpe",
        "input": str(cfg.input.load.orders_path),
        "simulation": str(cfg.simulation.name),
        "status": simulation.state.completion_reason,
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
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "decisions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
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
