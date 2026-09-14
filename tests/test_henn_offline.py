import os
import shutil
import unittest
from pathlib import Path

import numpy as np
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.operational_events import (
    FlushRemainingOrders,
    NodeArrival,
    PickerArrival,
)
from casim.simulation_engine.simulation_engine import SimulationEngine
from scenarios.experiment_commons import (
    load_and_flatten_data_card,
    setup_scenario,
    setup_decision_engine,
)
from scenarios.scenario_henn_online.algorithm import (
    HennWakeUp,
    decide_henn,
    insert_into_active_tour,
    single_order_service_times,
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
    sim.add_event(FlushRemainingOrders(max(o.order_date for o in orders)))


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
        cfg = self._load_cfg(overrides=[
            "scenario.simulation_engine.problems.OBRP.conditions.1.threshold=1",
        ])
        datacard = load_and_flatten_data_card(cfg.data_card)
        sim = setup_scenario(cfg)
        decision_engine = setup_decision_engine(cfg, datacard)
        sim.triggers_map[HennWakeUp] = "OBRP"

        sim.reset(hooks=[add_orders_hook,
                         picker_arrival_hook])

        single_services = {}
        actions = []
        insertions = 0
        while True:
            done, state_snapshot = sim.run()
            if done:
                break
            active_tour_id = (
                state_snapshot.dynamic_warehouse_info.active_tour_id
            )
            if active_tour_id is not None:
                result = decision_engine.solve(state_snapshot)
                self.assertIsNotNone(result)
                candidate, _, _ = result
                tour = sim.state.tour_manager.get_tour(active_tour_id)
                picker = sim.state.resource_manager.get_resource(
                    tour.assigned_resource
                )
                replacement = insert_into_active_tour(
                    candidate,
                    tour,
                    picker.current_location,
                    state_snapshot.layout,
                )
                if replacement is None:
                    version = sim.state.resume_active_tour(active_tour_id)
                else:
                    version = sim.state.commit_active_plan(
                        active_tour_id,
                        replacement,
                        picker_id=tour.assigned_resource,
                        expected_version=(
                            state_snapshot.dynamic_warehouse_info.route_version
                        ),
                    )
                    insertions += 1
                sim.add_event(NodeArrival(
                    sim.state.current_time,
                    active_tour_id,
                    version,
                ))
                continue
            result = decision_engine.solve(state_snapshot)
            self.assertIsNotNone(result)
            candidate, _, _ = result
            single_order_service_times(
                state_snapshot,
                decision_engine.get_solver("OBRP"),
                single_services,
            )
            decision = decide_henn(
                candidate,
                state_snapshot,
                sim.state.current_time,
                sim.state.input_closed,
                cfg.henn.selector,
                single_services,
                cfg.henn.waiting_policy,
                cfg.henn.fill_threshold,
                cfg.henn.max_age_s,
            )
            actions.append(decision.action)
            if decision.action == "wait":
                sim.add_event(HennWakeUp(decision.wait_until))
                continue
            events_to_add, solution = decision_engine.commit(
                state_snapshot,
                decision.solution,
            )
            sim.step(events_to_add, state_snapshot.problem_class, solution)

        self.assertIn("dispatch", actions)
        self.assertGreater(insertions, 0)
        self.assertTrue(sim.state.input_closed)
        self.assertEqual([], sim.state.order_manager.get_order_buffer())


if __name__ == "__main__":
    unittest.main()
