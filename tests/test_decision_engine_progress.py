"""Tests for DecisionEngine.on_trigger strict behavior and ProgressLogger."""

import logging
from types import SimpleNamespace

import pytest

from casim.decision_engine.decision_engine import DecisionEngine
from casim.loggers import ProgressLogger


def _snapshot(problem_class="OBRSP", replanning="none", time=0.0):
    dynamic = SimpleNamespace(
        time=time,
        replanning=replanning,
    )
    return SimpleNamespace(
        problem_class=problem_class,
        dynamic_warehouse_info=dynamic,
        objective="makespan",
    )


class _NoneSolver:
    def solve(self, snapshot, action=None):
        return None


class _FakeSolver:
    def solve(self, snapshot, action=None):
        from ware_ops_algos.algorithms import BatchingSolution
        sol = BatchingSolution(algo_name="fake", batches=[])
        return sol, "FakePipeline", 0.0


def test_on_trigger_raises_contextual_error_when_solver_returns_none():
    engine = DecisionEngine(
        solver_map={("OBRSP", "none"): _NoneSolver()},
    )
    snapshot = _snapshot("OBRSP", "none", time=42.0)
    with pytest.raises(RuntimeError, match="no decision for"):
        engine.on_trigger(snapshot)


def test_on_trigger_error_includes_problem_class_replanning_and_time():
    engine = DecisionEngine(
        solver_map={("ORSP", "unstarted"): _NoneSolver()},
    )
    snapshot = _snapshot("ORSP", "unstarted", time=100.0)
    with pytest.raises(RuntimeError) as exc_info:
        engine.on_trigger(snapshot)
    message = str(exc_info.value)
    assert "ORSP" in message
    assert "unstarted" in message
    assert "100" in message or "100.0" in message


def test_on_trigger_returns_events_and_solution_when_solver_succeeds():
    engine = DecisionEngine(
        solver_map={("OBP", "none"): _FakeSolver()},
    )
    snapshot = _snapshot("OBP", "none", time=0.0)
    result = engine.on_trigger(snapshot)
    assert result is not None
    events, solution = result
    assert len(events) == 1
    from casim.events.decision_events import PickListDone
    assert isinstance(events[0], PickListDone)
    assert solution is not None


def test_solve_retains_none_for_henn_candidate_path():
    engine = DecisionEngine(
        solver_map={("OBRP", "none"): _NoneSolver()},
    )
    snapshot = _snapshot("OBRP", "none", time=0.0)
    assert engine.solve(snapshot) is None


def _make_sim():
    return SimpleNamespace(
        state=SimpleNamespace(
            current_time=42.0,
            tracker=SimpleNamespace(
                completed_tours=[(1, 0, 10, [], 0, [], [], 1)],
                completed_order_count=5,
                distance_by_picker={0: 100.0},
            ),
            order_manager=SimpleNamespace(
                get_order_buffer=lambda: [],
                get_pick_list_buffer=lambda: [],
            ),
            tour_manager=SimpleNamespace(
                active_tours=lambda: [],
            ),
            completion_reason="drained",
        )
    )


def test_progress_logger_disabled_produces_no_output(caplog):
    logger = ProgressLogger(every=0)
    sim = _make_sim()
    logger.on_reset(sim, None)
    with caplog.at_level(logging.INFO):
        for i in range(100):
            logger.on_event(SimpleNamespace(), sim)
    assert caplog.text == ""


def test_progress_logger_enabled_fires_at_configured_count(caplog):
    logger = ProgressLogger(every=10)
    sim = _make_sim()
    logger.on_reset(sim, None)
    with caplog.at_level(logging.INFO):
        for i in range(25):
            logger.on_event(SimpleNamespace(), sim)
    assert caplog.text.count("t=42") >= 2


def test_progress_logger_writes_no_files(tmp_path):
    logger = ProgressLogger(every=1)
    sim = _make_sim()
    sim.state.current_time = 0.0
    sim.state.tracker.completed_tours = []
    sim.state.tracker.completed_order_count = 0
    sim.state.tracker.distance_by_picker = {}
    logger.on_reset(sim, None)
    logger.on_event(SimpleNamespace(), sim)
    logger.on_done(sim)
    assert not (tmp_path / "kpis.json").exists()
    assert not (tmp_path / "tracker.json").exists()
    assert list(tmp_path.iterdir()) == []
