"""Configured stochastic waiting study using CASIM's usual experiment loop."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from scenarios.experiment_commons import (
    load_and_flatten_data_card, setup_decision_engine, setup_scenario,
)
from scenarios.scenario_stochastic_waiting.scenario_specific_hooks import (
    add_orders_hook, picker_arrival_hook,
)


@hydra.main(config_path="config", config_name="stochastic_waiting_config", version_base="1.3")
def main(cfg: DictConfig):
    data_card = load_and_flatten_data_card(cfg.data_card)
    sim = setup_scenario(cfg)
    decision_engine = setup_decision_engine(cfg, data_card)

    sim.reset(hooks=[add_orders_hook, picker_arrival_hook])
    done = False
    while not done:
        done, snapshot = sim.run()
        if not done:
            events, solution = decision_engine.on_trigger(snapshot)
            sim.step(events, snapshot.problem_class, solution, snapshot)

    print(f"Decisions: {decision_engine.decision_tracker.num_decisions}")
    if cfg.viz.launch:
        from casim.viz.app import launch
        launch(Path(cfg.experiment.output_dir) / "viz", port=cfg.viz.port)


if __name__ == "__main__":
    main()
