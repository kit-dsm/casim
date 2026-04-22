import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from scenarios.experiment_commons import load_and_flatten_data_card, setup_scenario, setup_decision_engine
from casim.io_helpers import dump_pickle
from casim.viz.app import launch
from casim.viz.gantt_chart import gantt_chart
from scenarios.scenario_grocery_retailer.scenario_specific_hooks import add_orders_hook, picker_arrival_hook, \
    shift_start_hook, make_truck_schedule_hook, wms_run_hook, make_dock_manager_hook

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


@hydra.main(config_path="config", config_name="grocery_retailer_config")
def main(cfg: DictConfig):
    datacard = load_and_flatten_data_card(cfg.data_card)

    sim = setup_scenario(cfg)
    decision_engine = setup_decision_engine(cfg, datacard)

    sim.reset(hooks=[add_orders_hook,
                     picker_arrival_hook,
                     shift_start_hook,
                     make_truck_schedule_hook(bin_minutes=30,
                                              sweep_time_sec=18*3600),
                     wms_run_hook,
                     make_dock_manager_hook(K_dock=98)
                     ])

    done = False
    while not done:
        done, state_snapshot = sim.run()
        if done:
            dt = decision_engine.decision_tracker
            dump_pickle(str(Path(cfg.experiment.output_dir) / "decisions.pkl"), {
                "decisions": dt.decisions,
                "pipeline_counts": dict(dt.pipeline_counts),
            })
            print(f"'Total decisions':{dt.num_decisions}")
            for pipeline, count in sorted(dt.pipeline_counts.items(),
                                          key=lambda x: -x[1]):
                pct = 100 * count / dt.num_decisions
                print(f"{pipeline} {count:} {pct}%")
            break
        events_to_add, solution = decision_engine.on_trigger(state_snapshot)
        sim.step(events_to_add, state_snapshot.problem_class, solution)

    if cfg.viz.launch:
        viz_dir = Path(cfg.experiment.output_dir) / "viz"
        launch(viz_dir, port=cfg.viz.port, debug=False)

    fig = gantt_chart(sim.state.tracker)
    fig.write_html(str(Path(cfg.project_root) / "gantt.html"))

if __name__ == "__main__":
    main()
