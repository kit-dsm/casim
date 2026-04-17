import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.operational_events import PickerArrival
from casim.simulation_engine import SimulationEngine
from casim.viz.app import launch
from scenarios.experiment_commons import setup_scenario, setup_decision_engine, load_and_flatten_data_card
from scenarios.io_helpers import dump_pickle

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def picker_arrival_hook(sim: SimulationEngine,
                        domain: SimWarehouseDomain):
    min_order_date = np.inf
    for o in domain.orders.orders:
        if o.order_date < min_order_date:
            min_order_date = o.order_date
    for resource in domain.resources.resources:
        sim.add_event(PickerArrival(time=min_order_date,
                                    picker_id=resource.id))


def add_orders_hook(sim: SimulationEngine,
                    domain: SimWarehouseDomain):
    orders = domain.orders.orders
    for order in orders:
        sim.add_order(order)


@hydra.main(config_path="config", config_name="henn_online_config")
def main(cfg: DictConfig):
    datacard = load_and_flatten_data_card(cfg.data_card)

    sim = setup_scenario(cfg)
    decision_engine = setup_decision_engine(cfg, datacard)

    sim.reset(hooks=[add_orders_hook,
                     picker_arrival_hook,
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


if __name__ == "__main__":
    main()
