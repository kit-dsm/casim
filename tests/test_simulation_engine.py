"""Focused tests for SimulationEngine tuple bindings, step, and drain detection."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from ware_ops_algos.algorithms import BatchObject, GreedyItemAssignment
from ware_ops_algos.domain_models import Order

from casim.events.operational_events import (
    FlushRemainingOrders,
    OrderArrival,
    ProcessEvent,
)
from casim.simulation_engine.simulation_engine import SimulationEngine
from scenarios.scenario_reopt.loader import ReoptDataLoader


ROOT = Path(__file__).parents[1]
SCENARIO = ROOT / "scenarios" / "scenario_reopt"


def _domain():
    return ReoptDataLoader(SCENARIO).load(
        SCENARIO / "data" / "paper_example.json"
    )


def _engine(tmp_path, *, state_adapters=None, triggers_map=None,
            conditions_map=None, hook=None, completion_mode="drain",
            horizon_time=None):
    domain = _domain()
    engine = SimulationEngine(
        state_adapters=state_adapters or {},
        triggers_map=triggers_map or {},
        conditions_map=conditions_map or {},
        loader_kwargs={},
        data_loader=SimpleNamespace(load=lambda **_: domain),
        event_loggers=[],
        completion_mode=completion_mode,
        horizon_time=horizon_time,
    )
    engine.reset(hooks=[hook] if hook else [])
    return engine


def test_triggers_map_accepts_tuple_bindings():
    from casim.simulation_engine.state_adapter import StateAdapter

    domain = _domain()
    adapter = StateAdapter(
        problem_class="OBRSP",
        replanning="none",
        orders={"source": "buffered"},
        resources={"source": "dispatchable", "scope": "trigger_if_present"},
    )
    binding = ("OBRSP", "none")
    engine = SimulationEngine(
        state_adapters={binding: adapter},
        triggers_map={OrderArrival: binding},
        conditions_map={binding: {"orders": 1, "pickers": 1}},
        loader_kwargs={},
        data_loader=SimpleNamespace(load=lambda **_: domain),
        event_loggers=[],
        completion_mode="drain",
    )
    engine.reset()
    assert engine.triggers_map[OrderArrival] == ("OBRSP", "none")
    assert engine.conditions_map[("OBRSP", "none")] == {"orders": 1, "pickers": 1}


def test_step_accepts_only_events_argument():
    domain = _domain()
    engine = SimulationEngine(
        state_adapters={},
        triggers_map={},
        conditions_map={},
        loader_kwargs={},
        data_loader=SimpleNamespace(load=lambda **_: domain),
        event_loggers=[],
        completion_mode="drain",
    )
    engine.reset()
    engine.state.current_time = 0.0

    class NoOpProcessEvent(ProcessEvent):
        def __init__(self, t):
            super().__init__(t)

        def handle(self, state):
            return []

    engine.step([])
    engine.step([NoOpProcessEvent(0.0)])


def test_step_rejects_non_process_events():
    domain = _domain()
    engine = SimulationEngine(
        state_adapters={},
        triggers_map={},
        conditions_map={},
        loader_kwargs={},
        data_loader=SimpleNamespace(load=lambda **_: domain),
        event_loggers=[],
        completion_mode="drain",
    )
    engine.reset()

    class NotAProcessEvent:
        time = 0.0

    with pytest.raises(TypeError, match="ProcessEvent"):
        engine.step([NotAProcessEvent()])


def test_step_rejects_events_at_wrong_time():
    domain = _domain()
    engine = SimulationEngine(
        state_adapters={},
        triggers_map={},
        conditions_map={},
        loader_kwargs={},
        data_loader=SimpleNamespace(load=lambda **_: domain),
        event_loggers=[],
        completion_mode="drain",
    )
    engine.reset()

    class BadProcessEvent(ProcessEvent):
        def __init__(self, t):
            super().__init__(t)

        def handle(self, state):
            return []

    engine.state.current_time = 10.0
    with pytest.raises(ValueError, match="current simulation time"):
        engine.step([BadProcessEvent(5.0)])


def test_drain_no_progress_detection_raises():
    domain = _domain()

    class NoOpProcessEvent(ProcessEvent):
        def __init__(self, t):
            super().__init__(t)

        def handle(self, state):
            return []

    binding = ("OBRSP", "none")
    from casim.simulation_engine.state_adapter import StateAdapter
    adapter = StateAdapter(
        problem_class="OBRSP",
        replanning="none",
        orders={"source": "buffered"},
        resources={"source": "dispatchable", "scope": "trigger_if_present"},
    )

    def add_drain_order(engine, domain):
        engine.add_order(domain.orders.orders[0])
        engine.add_event(FlushRemainingOrders(0.0))

    engine = SimulationEngine(
        state_adapters={binding: adapter},
        triggers_map={FlushRemainingOrders: binding},
        conditions_map={binding: {}},
        loader_kwargs={},
        data_loader=SimpleNamespace(load=lambda **_: domain),
        event_loggers=[],
        completion_mode="drain",
    )
    engine.reset(hooks=[add_drain_order])

    done, snapshot = engine.run()
    assert not done
    assert snapshot is not None
    assert engine._drain_signature is not None

    with pytest.raises(RuntimeError, match="no operational progress"):
        engine.step([NoOpProcessEvent(0.0)])


def test_immediate_trigger_after_step_produces_pending_snapshot():
    domain = _domain()
    binding = ("OBRSP", "none")

    from casim.simulation_engine.state_adapter import StateAdapter
    adapter = StateAdapter(
        problem_class="OBRSP",
        replanning="none",
        orders={"source": "buffered"},
        resources={"source": "dispatchable", "scope": "trigger_if_present"},
    )

    def add_order(engine, domain):
        engine.add_order(domain.orders.orders[0])

    engine = SimulationEngine(
        state_adapters={binding: adapter},
        triggers_map={OrderArrival: binding},
        conditions_map={binding: {}},
        loader_kwargs={},
        data_loader=SimpleNamespace(load=lambda **_: domain),
        event_loggers=[],
        completion_mode="drain",
    )
    engine.reset(hooks=[add_order])

    done, snapshot = engine.run()
    assert not done
    assert snapshot is not None
    assert snapshot.problem_class == "OBRSP"
