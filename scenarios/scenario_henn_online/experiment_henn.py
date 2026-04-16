import logging
import os

import hydra
import numpy as np
from omegaconf import DictConfig

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.operational_events import PickerArrival
from casim.simulation_engine import SimulationEngine
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


def save_static_info(sim: SimulationEngine,
                     domain: SimWarehouseDomain):
    target = sim.cache_path / "vis_input"
    if not os.path.exists(target):
        os.mkdir(target)
    dump_pickle(str(target / "inital_domain.pkl"), domain)


@hydra.main(config_path="config", config_name="henn_online_config")
def main(cfg: DictConfig):
    datacard = load_and_flatten_data_card(cfg.data_card)

    sim = setup_scenario(cfg)
    decision_engine = setup_decision_engine(cfg, datacard)

    sim.reset(hooks=[add_orders_hook,
                     picker_arrival_hook,
                     save_static_info])

    done = False
    while not done:
        done, state_snapshot = sim.run()
        if done:
            break
        events_to_add = decision_engine.on_trigger(state_snapshot)
        sim.step(events_to_add)


if __name__ == "__main__":
    main()
