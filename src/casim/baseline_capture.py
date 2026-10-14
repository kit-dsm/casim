"""Optional, read-only event and invariant capture for Phase B baselines.

Attaches to SimulationEngine only when the Hydra flag ``experiment.audit`` is
set.  It observes events and decisions, runs the invariant checks after every
step and records the data needed for independent KPI recomputation.  It never
modifies state or the event queue.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from ware_ops_algos.planning.invariants import check_plan, check_progress
from ware_ops_algos.planning.lifecycle import TourStates

from casim.events.operational_events import (
    NodeArrival, OrderArrival, TravelEvent,
)
from casim.loggers import EventLogger


class BaselineCapture(EventLogger):
    """Observe a run without changing its state or event queue."""

    def __init__(self, path: Path):
        self.path = path
        self.trace: list[tuple] = []
        self.commits: list[dict] = []
        self.decisions: Counter = Counter()
        self.legs: list[dict] = []
        self._pending_legs: dict[int, dict] = {}
        self._committed_ids: set[int] = set()
        self._before = None
        self.order_arrivals: dict[str, float] = {}
        self.violations: dict[str, Counter] = {
            "check_plan": Counter(), "check_progress": Counter(),
        }
        self.first_violations: dict[str, dict[str, dict]] = {
            "check_plan": {}, "check_progress": {},
        }

    def on_reset(self, sim, domain) -> None:
        self._before = self._full_plan(sim)

    def _full_plan(self, sim):
        adapter = next(iter(sim.state_adapters.values()))
        return adapter.full_plan_view(sim.state)

    def _check(self, sim, source: str) -> None:
        after = self._full_plan(sim)
        for kind, findings in (
            ("check_plan", check_plan(after)),
            ("check_progress", check_progress(self._before, after)),
        ):
            for finding in findings:
                code = str(finding.code)
                self.violations[kind][code] += 1
                self.first_violations[kind].setdefault(code, {
                    "at": source, "time": after.time,
                    "tour_id": finding.tour_id, "message": finding.message,
                })
        self._before = after

    def on_event(self, event, sim) -> None:
        state = sim.state
        tour_id = getattr(event, "tour_id", None)
        picker_id = getattr(event, "picker_id", None)
        if picker_id is None and tour_id is not None:
            tour = state.tour_manager.get_tour(tour_id)
            picker_id = tour.assigned_resource
        self.trace.append((type(event).__name__, round(event.time, 9),
                           tour_id, picker_id))
        if isinstance(event, OrderArrival):
            self.order_arrivals[str(event.order.order_id)] = event.order.order_date
        elif isinstance(event, TravelEvent):
            tour = state.tour_manager.get_tour(event.tour_id)
            origin = tour.current_node()
            dest = tour.next_node()
            distance, reason = self._leg_distance(state, tour, origin, dest)
            self._pending_legs[event.tour_id] = {
                "tour_id": event.tour_id, "picker_id": picker_id,
                "start": event.time, "end": tour.edge_end_time,
                "distance": distance, "recompute_reason": reason,
                "origin": origin.position, "destination": dest.position,
            }
        elif isinstance(event, NodeArrival):
            leg = self._pending_legs.pop(event.tour_id, None)
            if leg is not None:
                leg["completed_fraction"] = 1.0
                self.legs.append(leg)
        self._check(sim, type(event).__name__)

    def on_decision(self, decision, result, snapshot, sim) -> None:
        state = sim.state
        self.decisions[decision] += 1
        for tour_id, leg in list(self._pending_legs.items()):
            tour = state.tour_manager.get_tour(tour_id)
            if tour.edge_origin is None:
                elapsed = state.current_time - leg["start"]
                duration = leg["end"] - leg["start"]
                leg["completed_fraction"] = (
                    1.0 if duration == 0 else max(0.0, min(1.0, elapsed / duration)))
                self.legs.append(leg)
                del self._pending_legs[tour_id]
        for tour in state.tour_manager.get_all_tours():
            if tour.tour_id in self._committed_ids:
                continue
            if tour.status in (TourStates.SCHEDULED, TourStates.PENDING,
                               TourStates.STARTED, TourStates.DONE):
                self._committed_ids.add(tour.tour_id)
                self.commits.append({
                    "tour_id": tour.tour_id,
                    "picker_id": tour.assigned_resource,
                    "order_ids": list(tour.order_numbers),
                    "start": tour.start_time, "end": tour.end_time_planned,
                })
        self._check(sim, f"decision:{decision}")

    def on_done(self, sim) -> None:
        state = sim.state
        tours = []
        for tour in state.tour_manager.get_all_tours():
            if tour.status is not TourStates.DONE:
                continue
            picker = state.resource_manager.get_resource(tour.assigned_resource)
            tours.append({
                "tour_id": tour.tour_id, "picker_id": tour.assigned_resource,
                "order_ids": list(tour.order_numbers),
                "start": tour.start_time, "end": tour.end_time,
                "picks": len(tour.completed_picks),
                "speed": picker.speed, "time_per_pick": picker.time_per_pick,
                "tour_setup_time": picker.tour_setup_time or 0.0,
            })
        payload = {
            "trace_digest": sha256(json.dumps(self.trace,
                                               separators=(",", ":")).encode()).hexdigest(),
            "event_count": len(self.trace),
            "committed_tours": self.commits,
            "decisions": dict(self.decisions),
            "completed_tours": tours,
            "route_legs": self.legs,
            "order_arrivals": self.order_arrivals,
            "violations": {kind: dict(counts)
                           for kind, counts in self.violations.items()},
            "first_violations": self.first_violations,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _leg_distance(state, tour, origin, dest) -> tuple:
        """Independently compute one leg's distance, or mark not recomputable."""
        lm = state.layout_manager
        if (tour.cursor == 0 and tour.routing_origin is not None
                and tour.routing_origin.edge_destination is not None):
            pa = tour.routing_origin.position
            pb = tour.routing_origin.edge_destination
        else:
            pa = origin.position
            pb = dest.position
        return BaselineCapture._position_distance(lm, pa, pb)

    @staticmethod
    def _position_distance(lm, pa, pb) -> tuple:
        """Distance between two positions via graph or aisle geometry."""
        layout = lm.get_layout()
        dm = layout.layout_network.distance_matrix
        graph_nodes = set(dm.index)
        if pa in graph_nodes and pb in graph_nodes:
            return (float(dm.at[pa, pb]), None)
        na = BaselineCapture._nearest_node(pa, graph_nodes)
        if na is None:
            return (None, f"position {pa} not in any aisle")
        nb = BaselineCapture._nearest_node(pb, graph_nodes)
        if nb is None:
            return (None, f"position {pb} not in any aisle")
        graph_dist = float(dm.at[na, nb])
        total = abs(pa[1] - na[1]) + graph_dist + abs(pb[1] - nb[1])
        return (total, None)

    @staticmethod
    def _nearest_node(pos, graph_nodes) -> tuple | None:
        """Nearest graph node in the same aisle (same x coordinate)."""
        same_aisle = [n for n in graph_nodes if n[0] == pos[0]]
        if not same_aisle:
            return None
        return min(same_aisle, key=lambda n: abs(n[1] - pos[1]))
