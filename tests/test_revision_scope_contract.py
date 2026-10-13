"""Integration checks for the new decision transaction and draft lifecycle."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ware_ops_algos.algorithms import SchedulingSolution, WaitingSolution
from ware_ops_algos.domain_models import TourStates

from casim.decision_engine.commands import (
    CommandViolation, CreateTour, ReleaseTour, Wait,
)
from casim.decision_engine.conversion import to_commands
from casim.decision_engine.decision_engine import DecisionResult
from casim.events.operational_events import PickerTourQuery, Reconsider
from scenarios.experiment_commons import load_and_flatten_data_card, setup_experiment
from casim.config.schema import DecisionDefinitions
from casim.setup import resolve
from scenarios.scenario_stochastic_waiting.scenario_specific_hooks import (
    add_orders_hook, picker_arrival_hook,
)


CONFIG_DIR = (Path(__file__).resolve().parents[1] / "scenarios"
              / "scenario_stochastic_waiting" / "config")


def test_committed_schema_matches_config_model():
    root = Path(__file__).resolve().parents[1]
    expected = DecisionDefinitions.model_json_schema()
    actual = json.loads((root / "schemas" / "decision.schema.json").read_text())
    assert actual == expected


def test_problem_class_is_derived_from_named_decision(tmp_path):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="stochastic_waiting_config", overrides=[
            f"instances_base={tmp_path.as_posix()}",
            f"cache_base={(tmp_path / 'cache').as_posix()}",
            f"experiment.output_dir={tmp_path.as_posix()}",
        ])
    OmegaConf.set_struct(cfg.decisions, False)
    config = cfg.decisions.pop("OBRSPW")
    cfg.decisions["planning"] = config
    setups = resolve(cfg, load_and_flatten_data_card(cfg.data_card))
    assert setups["planning"].problem_class == "OBRSP"
    assert setups["planning"].name == "planning"


def test_overlapping_decisions_require_distinct_precedence(tmp_path):
    walk_config = (Path(__file__).resolve().parents[1] / "scenarios"
                   / "scenario_walk_or_wait" / "config")
    with initialize_config_dir(config_dir=str(walk_config), version_base="1.3"):
        cfg = compose(config_name="walk_or_wait_config", overrides=[
            f"instances_base={tmp_path.as_posix()}",
            f"cache_base={(tmp_path / 'cache').as_posix()}",
            f"experiment.output_dir={tmp_path.as_posix()}",
        ])
    cfg.decisions.OBRSPW.precedence = cfg.decisions.ATIP.precedence
    with pytest.raises(ValueError, match="both write.*orders\\[buffered\\]"):
        resolve(cfg, load_and_flatten_data_card(cfg.data_card))


def test_periodic_reconsider_uses_current_decision_version(tmp_path):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="stochastic_waiting_config", overrides=[
            f"instances_base={tmp_path.as_posix()}",
            f"cache_base={(tmp_path / 'cache').as_posix()}",
            f"experiment.output_dir={tmp_path.as_posix()}",
        ])
    cfg.decisions.OBRSPW.triggers.append({"every_s": 5})
    sim, engine = setup_experiment(cfg, load_and_flatten_data_card(cfg.data_card))
    sim.reset(hooks=[add_orders_hook, picker_arrival_hook])
    original = next(event for event in sim.events
                    if isinstance(event, Reconsider) and event.reason == "periodic")
    done, snapshot = sim.run()
    assert not done
    sim.step(engine.on_trigger(snapshot), snapshot)
    assert original.is_stale(sim.state)
    assert any(isinstance(event, Reconsider)
               and event.version == sim.state.decision_versions["OBRSPW"]
               and event.time == snapshot.context.time + 5
               for event in sim.events)


def _first_wait(tmp_path):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="stochastic_waiting_config", overrides=[
            f"instances_base={tmp_path.as_posix()}",
            f"cache_base={(tmp_path / 'cache').as_posix()}",
            f"experiment.output_dir={tmp_path.as_posix()}",
        ])
    sim, decision_engine = setup_experiment(
        cfg, load_and_flatten_data_card(cfg.data_card))
    sim.reset(hooks=[add_orders_hook, picker_arrival_hook])
    while True:
        done, snapshot = sim.run()
        assert not done
        result = decision_engine.on_trigger(snapshot)
        if any(isinstance(command, Wait) and command.reconsider_at is not None
               for command in result.commands):
            return sim, decision_engine, snapshot, result
        sim.step(result, snapshot)


def test_rejected_decision_leaves_state_and_events_untouched(tmp_path):
    sim, _, snapshot, result = _first_wait(tmp_path)
    before = deepcopy(sim.state)
    events_before = tuple(sim.events)
    bad = ReleaseTour(expected_decision_version=snapshot.context.version,
                      tour_id=999)
    with pytest.raises(CommandViolation):
        sim.step(DecisionResult((*result.commands, bad)), snapshot)
    assert sim.state.decision_versions == before.decision_versions
    assert sim.state.tour_manager.all_tours == before.tour_manager.all_tours
    assert tuple(sim.events) == events_before


def test_drop_policy_skips_the_whole_invalid_decision(tmp_path):
    sim, _, snapshot, result = _first_wait(tmp_path)
    sim.on_violation = "drop"
    version = sim.state.decision_versions.get("OBRSPW", 0)
    events = tuple(sim.events)
    accepted = sim.decision_engine.decision_tracker.num_decisions
    bad = ReleaseTour(expected_decision_version=version, tour_id=999)
    sim.step(DecisionResult((*result.commands, bad)), snapshot)
    assert sim.state.decision_versions.get("OBRSPW", 0) == version
    assert tuple(sim.events) == events
    assert sim.decision_engine.decision_tracker.num_decisions == accepted


def test_repair_policy_reprojects_a_stale_decision_once(tmp_path):
    sim, _, snapshot, result = _first_wait(tmp_path)
    sim.on_violation = "repair"
    sim.state.decision_versions["OBRSPW"] = snapshot.context.version + 1
    sim.step(result, snapshot)
    assert sim.state.decision_versions["OBRSPW"] == snapshot.context.version + 2


def test_wait_keeps_draft_and_old_reconsider_becomes_stale(tmp_path):
    sim, decision_engine, snapshot, result = _first_wait(tmp_path)
    sim.step(result, snapshot)
    draft = next(iter(sim.state.tour_manager.all_tours.values()))
    assert draft.status == TourStates.SCHEDULED
    assert not any(isinstance(event, PickerTourQuery) for event in sim.events)

    plan_snapshot = sim.state_adapters["OBRSPW"].transform_state(
        sim.state, "OBRSPW")
    job = decision_engine._draft_solution(plan_snapshot).jobs[0]
    unchanged = to_commands(
        WaitingSolution(action="wait"), plan_snapshot,
        candidate_plan=SchedulingSolution(jobs=[job]),
    )
    assert [type(command) for command in unchanged] == [Wait]

    old = next(event for event in sim.events if isinstance(event, Reconsider))
    adapter = sim.state_adapters["OBRSPW"]
    sim.state.apply_all("OBRSPW", (
        Wait(expected_decision_version=sim.state.decision_versions["OBRSPW"],
             reconsider_at=old.time + 1),
    ), adapter.selection, adapter.scope)
    assert old.is_stale(sim.state)


def test_waiting_release_starts_no_earlier_than_reconsideration(tmp_path):
    sim, decision_engine, snapshot, result = _first_wait(tmp_path)
    sim.step(result, snapshot)
    deadline = next(event.time for event in sim.events
                    if isinstance(event, Reconsider))
    done, release_snapshot = sim.run()
    assert not done
    assert release_snapshot.context.time == deadline
    release = decision_engine.on_trigger(release_snapshot)
    assert any(isinstance(command, ReleaseTour) for command in release.commands)
    sim.step(release, release_snapshot)
    assert all(tour.status != TourStates.STARTED
               for tour in sim.state.tour_manager.all_tours.values())
    while True:
        done, next_snapshot = sim.run()
        if done:
            break
        sim.step(decision_engine.on_trigger(next_snapshot), next_snapshot)
    assert all(tour.start_time >= deadline
               for tour in sim.state.tour_manager.all_tours.values()
               if tour.status == TourStates.DONE)
