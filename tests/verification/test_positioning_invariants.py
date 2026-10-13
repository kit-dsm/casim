"""The three-order center example keeps every order and every executed route prefix."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from ware_ops_algos.planning.commands import ReplaceSuffix
from ware_ops_algos.planning.lifecycle import TourStates

from casim.runner import apply_decision
from scenarios.experiment_commons import load_and_flatten_data_card, setup_experiment
from scenarios.scenario_walk_or_wait.hooks import seed_orders, seed_pickers


CONFIG = Path(__file__).resolve().parents[2] / "scenarios" / "scenario_walk_or_wait" / "config"


def test_center_example_preserves_plan_invariants(tmp_path, invariant_logger):
    with initialize_config_dir(config_dir=str(CONFIG), version_base="1.3"):
        cfg = compose(config_name="walk_or_wait_config", overrides=[
            "policy=center", "simulation=explicit_wait0", "cosy_repo=paper_center",
            "insertion_repo=center", f"instances_base={CONFIG.parents[1].as_posix()}",
            f"cache_base={(tmp_path / 'cache').as_posix()}",
            f"experiment.output_dir={tmp_path.as_posix()}",
        ])
    sim, engine = setup_experiment(cfg, load_and_flatten_data_card(cfg.data_card))
    sim.event_loggers.append(invariant_logger)
    sim.reset(hooks=[seed_pickers, seed_orders])
    while True:
        done, snapshot = sim.run()
        if done:
            break
        apply_decision(sim, engine, engine.on_trigger(snapshot), snapshot)
    tracker = sim.state.tracker
    assert len(tracker.completed_tours) == 2
    assert sum(len(record[3]) for record in tracker.completed_tours) == 3
    assert tracker.distance_by_picker[0] == 88
    assert tracker.completed_tours[-1][2] == 91


def test_two_admissions_keep_cumulative_prefix_and_distance(tmp_path, invariant_logger):
    with initialize_config_dir(config_dir=str(CONFIG), version_base="1.3"):
        cfg = compose(config_name="walk_or_wait_config", overrides=[
            "policy=wait_0", "simulation=explicit_wait0", "cosy_repo=paper_patrol",
            "insertion_repo=paper_remaining_route",
            f"instances_base={CONFIG.parents[1].as_posix()}",
            f"cache_base={(tmp_path / 'cache').as_posix()}",
            f"experiment.output_dir={tmp_path.as_posix()}",
        ])
    cfg.simulation.orders = [
        {"arrival_s": 8.5, "due_s": 400.0, "picks": [[2, 3]]},
        {"arrival_s": 10.5, "due_s": 400.0, "picks": [[2, 8]]},
        {"arrival_s": 30.0, "due_s": 400.0, "picks": [[6, 9]]},
    ]
    sim, engine = setup_experiment(cfg, load_and_flatten_data_card(cfg.data_card))
    sim.event_loggers.append(invariant_logger)
    sim.reset(hooks=[seed_pickers, seed_orders])
    admissions = []
    distance_at_second = remaining_after_second = first_tour_distance = None
    while True:
        done, snapshot = sim.run()
        first_tour = sim.state.tour_manager.all_tours.get(1)
        if (first_tour is not None and first_tour.status is TourStates.DONE
                and first_tour_distance is None):
            first_tour_distance = sim.state.tracker.distance_by_picker[0]
        if done:
            break
        result = engine.on_trigger(snapshot)
        replacements = [command for command in result.commands
                        if isinstance(command, ReplaceSuffix) and command.tour_id == 1]
        if replacements:
            view = next(view for view in snapshot.dynamic_warehouse_info.plan.tours
                        if view.tour_id == 1)
            assert replacements[0].route.routing_origin.edge_destination is not None
            admissions.append((view.pinned_prefix, replacements[0].route.distance))
        apply_decision(sim, engine, result, snapshot)
        if len(admissions) == 2 and distance_at_second is None:
            distance_at_second = sim.state.tracker.distance_by_picker[0]
            remaining_after_second = admissions[-1][1]
            joined = sim.state.tour_manager.get_tour(1).executed_prefix
            junction = len(admissions[0][0])
            assert joined[:junction] == list(admissions[0][0])
            assert joined[junction - 1] != joined[junction]
    assert len(admissions) == 2
    assert admissions[1][0][:len(admissions[0][0])] == admissions[0][0]
    assert first_tour_distance == distance_at_second + remaining_after_second


@pytest.mark.parametrize(("policy", "makespan", "distance"), [
    ("wait_2", 152, 144),
    ("wait_3", 137, 128),
])
def test_partly_picked_tour_can_admit_order(tmp_path, invariant_logger,
                                            policy, makespan, distance):
    with initialize_config_dir(config_dir=str(CONFIG), version_base="1.3"):
        cfg = compose(config_name="walk_or_wait_config", overrides=[
            f"policy={policy}", "simulation=explicit_example",
            f"instances_base={CONFIG.parents[1].as_posix()}",
            f"cache_base={(tmp_path / 'cache').as_posix()}",
            f"experiment.output_dir={tmp_path.as_posix()}",
        ])
    sim, engine = setup_experiment(cfg, load_and_flatten_data_card(cfg.data_card))
    sim.event_loggers.append(invariant_logger)
    sim.reset(hooks=[seed_pickers, seed_orders])
    admitted_after_pick = False
    while True:
        done, snapshot = sim.run()
        if done:
            break
        result = engine.on_trigger(snapshot)
        plan = snapshot.dynamic_warehouse_info.plan
        if (snapshot.dynamic_warehouse_info.context.decision == "ATIP"
                and any(isinstance(command, ReplaceSuffix) and command.added_order_ids
                        for command in result.commands)):
            old = next(tour for tour in plan.tours
                       if tour.status is TourStates.STARTED)
            admitted_after_pick |= bool(old.batch.order_numbers - old.active_order_ids)
        apply_decision(sim, engine, result, snapshot)
    assert admitted_after_pick
    tracker = sim.state.tracker
    assert sum(len(record[3]) for record in tracker.completed_tours) == 6
    assert tracker.completed_tours[-1][2] == makespan
    assert tracker.distance_by_picker[0] == distance
