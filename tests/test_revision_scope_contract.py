"""Integration checks for named decisions and strict live application."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ware_ops_algos.planning.commands import CommitTour
from ware_ops_algos.planning.lifecycle import TourStates

from casim.config.schema import DecisionDefinitions
from casim.decision_engine.decision_engine import DecisionResult
from casim.domain_objects.sim_domain import ReconsiderReason
from casim.events.operational_events import PickerTourQuery, Reconsider
from casim.setup import resolve
from casim.state.apply import CommandViolation
from casim.config.data_card import load_and_flatten_data_card
from casim.run import prepare
from tests.scenario_configs import analytic_config, walk_config


def _config(tmp_path):
    return analytic_config(tmp_path)


def _first_wait(tmp_path):
    cfg = _config(tmp_path)
    prepared = prepare(cfg)
    sim, engine = prepared.sim, prepared.engine
    prepared.reset()
    for _ in range(100):
        done, snapshot = sim.run()
        assert not done
        result = engine.on_trigger(snapshot)
        if result.waiting and result.reconsider_at is not None:
            return sim, engine, snapshot, result
        sim.step(result, snapshot)
    raise AssertionError("Waiting decision was never reached")


def test_committed_schema_matches_config_model():
    root = Path(__file__).resolve().parents[1]
    expected = DecisionDefinitions.model_json_schema()
    actual = json.loads((root / "schemas" / "decision.schema.json").read_text())
    assert actual == expected


def test_problem_class_is_declared_independently_of_decision_name(tmp_path):
    cfg = _config(tmp_path)
    OmegaConf.set_struct(cfg.decisions, False)
    config = cfg.decisions.pop("OBRSP")
    cfg.decisions["planning"] = config
    setups = resolve(cfg, load_and_flatten_data_card(cfg.data_card))
    assert setups["planning"].problem_class == "OBRSP"
    assert setups["planning"].name == "planning"


def test_overlapping_decisions_require_distinct_precedence(tmp_path):
    cfg = walk_config(tmp_path)
    cfg.decisions.OBRSP.precedence = cfg.decisions.active_tour_admission.precedence
    with pytest.raises(ValueError, match="both write.*orders"):
        resolve(cfg, load_and_flatten_data_card(cfg.data_card))


def test_periodic_reconsider_uses_current_decision_version(tmp_path):
    cfg = _config(tmp_path)
    cfg.decisions.OBRSP.triggers.append({"every_s": 5})
    prepared = prepare(cfg)
    sim, engine = prepared.sim, prepared.engine
    prepared.reset()
    original = next(event for event in sim.events
                    if isinstance(event, Reconsider)
                    and event.reason is ReconsiderReason.PERIODIC)
    done, snapshot = sim.run()
    assert not done
    sim.step(engine.on_trigger(snapshot), snapshot)
    assert original.is_stale(sim.state)
    assert any(isinstance(event, Reconsider)
               and event.version == sim.state.decision_versions["OBRSP"]
               and event.time == snapshot.dynamic_warehouse_info.plan.time + 5
               for event in sim.events)


def test_invalid_live_reference_leaves_state_and_events_untouched(tmp_path):
    sim, _, snapshot, result = _first_wait(tmp_path)
    before = deepcopy(sim.state.tour_manager.all_tours)
    events_before = tuple(sim.events)
    version = sim.state.decision_versions.get("OBRSP", 0)
    with pytest.raises(CommandViolation):
        sim.step(DecisionResult((*result.commands, CommitTour(tour_id=999))), snapshot)
    assert sim.state.decision_versions.get("OBRSP", 0) == version
    assert sim.state.tour_manager.all_tours == before
    assert tuple(sim.events) == events_before


def test_wait_keeps_draft_and_old_reconsider_becomes_stale(tmp_path):
    sim, _, snapshot, result = _first_wait(tmp_path)
    sim.step(result, snapshot)
    draft = next(iter(sim.state.tour_manager.all_tours.values()))
    assert draft.status is TourStates.PLANNED
    assert sim.state.decision_waiting["OBRSP"]
    assert not any(isinstance(event, PickerTourQuery) for event in sim.events)
    old = next(event for event in sim.events
               if isinstance(event, Reconsider)
               and event.reason is ReconsiderReason.WAIT)
    sim.state.apply_all("OBRSP", ())
    assert old.is_stale(sim.state)


def test_waiting_release_starts_no_earlier_than_reconsideration(tmp_path):
    sim, engine, snapshot, result = _first_wait(tmp_path)
    sim.step(result, snapshot)
    deadline = next(event.time for event in sim.events
                    if isinstance(event, Reconsider)
                    and event.reason is ReconsiderReason.WAIT)
    for _ in range(100):
        done, next_snapshot = sim.run()
        assert not done
        if next_snapshot.dynamic_warehouse_info.plan.time == deadline:
            assert ReconsiderReason.WAIT in next_snapshot.dynamic_warehouse_info.context.reconsider
            release = engine.on_trigger(next_snapshot)
            assert any(isinstance(command, CommitTour) for command in release.commands), (
                release.waiting, release.reconsider_at,
                [type(command).__name__ for command in release.commands],
                next_snapshot.dynamic_warehouse_info.context.solve)
            sim.step(release, next_snapshot)
            break
        sim.step(engine.on_trigger(next_snapshot), next_snapshot)
    else:
        raise AssertionError("Wait deadline was never reached")
    while True:
        done, next_snapshot = sim.run()
        if done:
            break
        sim.step(engine.on_trigger(next_snapshot), next_snapshot)
    assert all(tour.start_time >= deadline
               for tour in sim.state.tour_manager.all_tours.values()
               if tour.status is TourStates.DONE)
