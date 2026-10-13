"""Run the four-order analytical waiting example in CASIM."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from casim.runner import apply_decision
from scenarios.experiment_commons import (
    load_and_flatten_data_card, setup_experiment,
)
from scenarios.scenario_stochastic_waiting.scenario_specific_hooks import (
    add_orders_hook, picker_arrival_hook,
)


@hydra.main(config_path="config", config_name="stochastic_waiting_config", version_base="1.3")
def main(cfg: DictConfig):
    data_card = load_and_flatten_data_card(cfg.data_card)
    sim, decision_engine = setup_experiment(cfg, data_card)

    sim.reset(hooks=[add_orders_hook, picker_arrival_hook])
    while True:
        done, snapshot = sim.run()
        if done:
            break
        result = decision_engine.on_trigger(snapshot)
        apply_decision(sim, decision_engine, result, snapshot)

    print(f"Decisions: {decision_engine.decision_tracker.num_decisions}")
    if cfg.viz.launch:
        from casim.viz.app import launch
        launch(Path(cfg.experiment.output_dir) / "viz", port=cfg.viz.port)


if __name__ == "__main__":
    main()
