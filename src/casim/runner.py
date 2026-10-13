"""Apply a decision and record its accepted choice at the caller boundary."""

from time import perf_counter

from casim.decision_engine.decision_engine import DecisionEngine, DecisionResult
from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.simulation_engine.simulation_engine import SimulationEngine


def apply_decision(sim: SimulationEngine, engine: DecisionEngine,
                   result: DecisionResult, snapshot: SimWarehouseDomain) -> None:
    started = perf_counter()
    sim.step(result, snapshot)
    engine.on_accepted(result, snapshot)
    engine.decision_tracker.on_commands(
        sim.state.current_time, snapshot.dynamic_warehouse_info.context.decision,
        result.commands, result.conversion_seconds, perf_counter() - started,
    )
