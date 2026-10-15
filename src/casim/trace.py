"""The normalized event trace of a run and its digest."""

from __future__ import annotations

from hashlib import sha256
import json

from casim.loggers import EventLogger


class TraceRecorder(EventLogger):
    """Records (event type, time, tour id, picker id) in processing order."""

    def __init__(self):
        self.trace: list[tuple] = []

    def on_event(self, event, sim) -> None:
        tour_id = getattr(event, "tour_id", None)
        picker_id = getattr(event, "picker_id", None)
        if picker_id is None and tour_id is not None:
            picker_id = sim.state.tour_manager.get_tour(tour_id).assigned_resource
        self.trace.append((type(event).__name__, round(event.time, 9), tour_id, picker_id))

    def digest(self) -> str:
        return sha256(json.dumps(self.trace, separators=(",", ":")).encode()).hexdigest()
