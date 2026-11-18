"""The observer interface the simulation engine calls during a run."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from casim.domain_objects.sim_domain import SimWarehouseDomain
    from casim.events.base_events import Event
    from casim.simulation_engine import SimulationEngine


class EventLogger:
    def on_reset(self, sim: SimulationEngine, domain: SimWarehouseDomain) -> None: ...
    def on_event(self, event: Event, sim: SimulationEngine) -> None: ...
    def on_decision(self, problem_class, solution, state_snapshot, sim: SimulationEngine) -> None: ...
    def on_done(self, sim: SimulationEngine) -> None: ...
    def on_time_end(self, sim: SimulationEngine) -> None:
        """After the last event of a timestamp, before time advances."""
