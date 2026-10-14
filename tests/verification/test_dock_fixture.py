"""Capture the synthetic two-picker dock run for Task 4.9 parity."""

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from casim.events.operational_events import TruckDeparture
from casim.setup import resolve
from casim.state.dock_manager import DockManager
from scenarios.experiment_commons import load_and_flatten_data_card, setup_experiment
from ware_ops_algos.domain_models import Dock, Resources
from casim.domain_objects.sim_domain import SimWarehouseDomain
from scenarios.scenario_walk_or_wait.hooks import seed_orders, seed_pickers


CONFIG = Path(__file__).resolve().parents[2] / "scenarios" / "scenario_walk_or_wait" / "config"


class DockTrace:
    def __init__(self):
        self.events = []
        self.decisions = []

    def on_reset(self, sim, domain):
        pass

    def on_event(self, event, sim):
        self.events.append([type(event).__name__, event.time,
                            sim.state.dock_manager.n_staged_pallets])

    def on_decision(self, decision, result, snapshot, sim):
        self.decisions.append([decision, sim.state.current_time,
                               [type(command).__name__ for command in result.commands]])

    def on_done(self, sim):
        pass


def run_fixture(tmp_path, seed, capacity):
    random.seed(seed)
    np.random.seed(seed)
    with initialize_config_dir(config_dir=str(CONFIG), version_base="1.3"):
        cfg = compose(config_name="walk_or_wait_config", overrides=[
            "policy=wait_0", "simulation=explicit_wait0", "cosy_repo=paper_patrol",
            f"instances_base={CONFIG.parents[1].as_posix()}",
            f"cache_base='{(tmp_path / 'cache').as_posix()}'",
            f"experiment.output_dir='{tmp_path.as_posix()}'",
        ])
    sim, engine = setup_experiment(cfg, load_and_flatten_data_card(cfg.data_card))
    original_load = sim.data_loader.load

    def load_two_pickers(**kwargs):
        domain = original_load(**kwargs)
        second = copy.deepcopy(domain.resources.resources[0])
        second.id = 1
        return SimWarehouseDomain(
            problem_class=domain.problem_class,
            objective=domain.objective,
            layout=domain.layout,
            articles=domain.articles,
            orders=domain.orders,
            resources=Resources(domain.resources.tpe,
                                [domain.resources.resources[0], second]),
            storage=domain.storage,
            warehouse_info=domain.warehouse_info,
            information=domain.information,
            dock=Dock(capacity=capacity),
            dynamic_warehouse_info=domain.dynamic_warehouse_info,
        )

    sim.data_loader.load = load_two_pickers
    trace = DockTrace()
    sim.event_loggers.append(trace)

    def trucks(sim, domain):
        sim.add_event(TruckDeparture(time=20, capacity=capacity))
        sim.add_event(TruckDeparture(time=100, capacity=capacity))

    sim.reset(hooks=[seed_pickers, seed_orders, trucks])
    while True:
        done, snapshot = sim.run()
        if done:
            break
        sim.step(engine.on_trigger(snapshot), snapshot)
    tracker = sim.state.tracker
    return {
        "seed": seed, "capacity": capacity, "events": trace.events,
        "decisions": trace.decisions,
        "completed_tours": tracker.completed_tours,
        "distance_by_picker": dict(tracker.distance_by_picker),
        "last_time": sim.state.current_time,
    }


@pytest.mark.parametrize("seed", (17, 29))
@pytest.mark.parametrize("capacity", (2, None))
def test_dock_fixture_matches_prechange_trace(tmp_path, seed, capacity):
    """Keep events, dock fill, decisions and KPIs equal to the prechange run."""
    expected_runs = json.loads(Path(__file__).with_name("dock_baseline.json").read_text())
    expected = next(result for result in expected_runs
                    if result["seed"] == seed and result["capacity"] == capacity)
    actual = json.loads(json.dumps(run_fixture(tmp_path, seed, capacity), default=str))
    assert actual == expected


def test_dock_manager_counts_without_dock_and_uses_pallet_yield():
    unbounded = DockManager(None)
    unbounded.stage_pallets(unbounded.pallets_per_tour)
    assert unbounded.capacity is None
    assert unbounded.n_staged_pallets == 1

    dock = DockManager(Dock(capacity=4, pallets_per_tour=2))
    dock.stage_pallets(dock.pallets_per_tour)
    assert dock.n_staged_pallets == 2
    dock.release_pallets(1)
    assert dock.n_staged_pallets == 1


def test_check_rejects_dock_capacity_condition_without_dock(tmp_path):
    with initialize_config_dir(config_dir=str(CONFIG), version_base="1.3"):
        cfg = compose(config_name="walk_or_wait_config", overrides=[
            "policy=wait_0", "simulation=explicit_wait0", "cosy_repo=paper_patrol",
            f"instances_base={CONFIG.parents[1].as_posix()}",
            f"cache_base='{(tmp_path / 'cache').as_posix()}'",
            f"experiment.output_dir='{tmp_path.as_posix()}'",
        ])
    condition = {"_target_": "casim.simulation_engine.conditions.DockCapacityCondition"}
    cfg.decisions.OBRSPW.conditions.append(OmegaConf.create(condition))
    with pytest.raises(ValueError, match="DockCapacityCondition requires a warehouse Dock"):
        resolve(cfg, load_and_flatten_data_card(cfg.data_card))
