"""Configured Walking vs. Waiting demonstration: build, reset, run, decide, step, report."""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from scenarios.experiment_commons import (
    load_and_flatten_data_card, setup_decision_engine, setup_scenario,
)
from scenarios.scenario_walk_or_wait.hooks import seed_orders, seed_pickers


def report(sim, output_dir: Path, expected_orders: int) -> dict:
    state = sim.state
    completed = state.tracker.completed_tours
    completion = {}
    for _, _, end, order_ids, _, _, _, _ in completed:
        for order_id in order_ids:
            if order_id in completion:
                raise ValueError(f"Order {order_id} completed twice")
            completion[order_id] = end
    if len(completion) != expected_orders:
        raise ValueError(f"Expected {expected_orders} completed orders, got {len(completion)}")
    known = {
        order_id: state.order_manager.get_order_from_history(order_id)
        for order_id in completion
    }
    total_lines = sum(lines for _, _, _, _, _, _, _, lines in completed)
    if total_lines == 0:
        raise ValueError("No picked items for paper metrics")
    metrics = {
        "mean_order_completion_time": sum(
            end - known[order_id].order_date for order_id, end in completion.items()
        ) / len(completion),
        "mean_length_per_item": sum(state.tracker.distance_by_picker.values()) / total_lines,
        "mean_tardiness": sum(
            max(0.0, end - known[order_id].due_date)
            for order_id, end in completion.items()
        ) / len(completion),
        "orders": len(completion),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


@hydra.main(config_path="config", config_name="walk_or_wait_config", version_base="1.3")
def main(cfg: DictConfig):
    data_card = load_and_flatten_data_card(cfg.data_card)
    sim = setup_scenario(cfg)
    sim.reset(hooks=[seed_orders, seed_pickers])
    decision_engine = setup_decision_engine(cfg, data_card)

    done = False
    while not done:
        done, snapshot = sim.run()
        if not done:
            events, solution = decision_engine.on_trigger(snapshot)
            sim.step(events, snapshot.problem_class, solution, snapshot)

    print(report(sim, Path(cfg.experiment.output_dir), len(cfg.simulation.orders)))


if __name__ == "__main__":
    main()
