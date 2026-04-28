import os
import shutil
import unittest
from pathlib import Path

import numpy as np
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.operational_events import PickerArrival
from casim.simulation_engine.simulation_engine import SimulationEngine
from scenarios.experiment_commons import (
    load_and_flatten_data_card,
    setup_scenario,
    setup_decision_engine,
)

TEST_DIR = Path(__file__).parent
os.environ["PROJECT_ROOT"] = TEST_DIR.as_posix()

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


class TestHennOffline(unittest.TestCase):
    def setUp(self):
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        self.tmp_dir = TEST_DIR / "tmp_output"
        (self.tmp_dir / "event_logs").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _load_cfg(self, config_name="test_henn_online_config", overrides=None):
        with initialize(version_base=None, config_path="./config"):
            return compose(config_name=config_name, overrides=overrides or [])

    def test_henn_offline(self):
        cfg = self._load_cfg()
        datacard = load_and_flatten_data_card(cfg.data_card)
        sim = setup_scenario(cfg)
        decision_engine = setup_decision_engine(cfg, datacard)

        sim.reset(hooks=[add_orders_hook,
                         picker_arrival_hook])

        done = False
        while not done:
            done, state_snapshot = sim.run()
            if done:
                break
            events_to_add, solution = decision_engine.on_trigger(state_snapshot)
            sim.step(events_to_add, state_snapshot.problem_class, solution)


if __name__ == "__main__":
    unittest.main()
