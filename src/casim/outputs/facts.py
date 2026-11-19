"""The realised facts of a run (docs/measures_and_ranking_design.md, "Facts").

``FactsRecorder`` observes the executed events and writes, in ``facts/``,

- ``tours.csv``: one row per committed tour, also one that never started:
  ``tour_id``, ``picker_id``, ``status`` at the end, ``commit_time``,
  ``planned_start`` (of the route, as last planned), ``start`` (of the
  setup, at dispatch), ``setup_end`` (start of the route), ``end``, ``setup`` (s, as
  executed; the resource's setup time for a tour that never started),
  ``route_distance`` (planned), ``distance`` (walked), ``lines`` (picked),
  ``orders_at_start``, ``order_ids``. Drafts that were never committed are
  left out;
- ``orders.csv``: one row per order: ``order_id``, source-marked ``affected``, ``release_time``,
  ``due_date``, ``lines``, ``tour_id``, ``tour_start``, ``completion`` (end
  of its tour, empty if not completed), and for comparisons with recorded
  data ``picker_id``, ``first_pick_start``, ``last_pick_end``;
- ``picks.csv``: one row per picked line (R1.1 of docs/ijpe/plan.md):
  ``tour_id``, ``picker_id``, ``order_id``, ``article_id``, ``node_x``,
  ``node_y``, ``walk_start``, ``walk_distance``, ``pick_start``, ``pick_end``.
  The walk starts at the end of the tour's previous pick, or at the route
  start for the first pick; its distance is the route length walked since.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ware_ops_algos.planning.lifecycle import TourStates

from casim.events.operational_events import PickComplete, TourEnd, TourStart, TravelEvent
from casim.simulation_engine.event_logger import EventLogger

FACTS_DIR = "facts"
TOUR_COLUMNS = ("tour_id", "picker_id", "status", "commit_time", "planned_start", "start",
                "setup_end", "end", "setup", "route_distance", "distance", "lines",
                "orders_at_start", "order_ids")
ORDER_COLUMNS = ("order_id", "affected", "release_time", "due_date", "lines", "tour_id", "tour_start",
                 "completion", "picker_id", "first_pick_start", "last_pick_end")
PICK_COLUMNS = ("tour_id", "picker_id", "order_id", "article_id", "node_x", "node_y",
                "walk_start", "walk_distance", "pick_start", "pick_end")


class FactsRecorder(EventLogger):
    def __init__(self, source=None):
        self.source = source
        self.picks: list[dict] = []
        self.tours: dict[int, dict] = {}
        self.commit_time: dict[int, float] = {}
        self._walk: dict[int, list[float]] = {}  # tour -> [walk start, distance so far]

    def on_decision(self, problem_class, result, state_snapshot, sim) -> None:
        # A tour is committed by the first decision after which it is neither
        # planned nor cancelled (a draft dropped before it was committed).
        for tour in sim.state.tour_manager.get_all_tours():
            if (tour.tour_id not in self.commit_time
                    and tour.status not in (TourStates.PLANNED, TourStates.CANCELLED)):
                self.commit_time[tour.tour_id] = sim.state.current_time

    def on_event(self, event, sim) -> None:
        if isinstance(event, TourStart):
            tour = sim.state.tour_manager.get_tour(event.tour_id)
            # The setup began at dispatch; it lasts the resource's setup time
            # or the realized one (C8).
            self.tours[event.tour_id] = {
                "start": tour.dispatched_at, "setup_end": event.time,
                "end": None, "distance": 0.0, "lines": 0}
            self._walk[event.tour_id] = [event.time, 0.0]
        elif isinstance(event, TravelEvent):
            tour = sim.state.tour_manager.get_tour(event.tour_id)
            if tour.edge_distance is not None and event.tour_id in self._walk:
                self._walk[event.tour_id][1] += tour.edge_distance
                self.tours[event.tour_id]["distance"] += tour.edge_distance
        elif isinstance(event, PickComplete):
            tour = sim.state.tour_manager.get_tour(event.tour_id)
            pick = tour.completed_picks[-1]
            walk_start, distance = self._walk[event.tour_id]
            self.picks.append({
                "tour_id": event.tour_id, "picker_id": tour.assigned_resource,
                "order_id": pick.order_number, "article_id": pick.article_id,
                "node_x": pick.pick_node[0], "node_y": pick.pick_node[1],
                "walk_start": walk_start, "walk_distance": distance,
                "pick_start": event.pick_start, "pick_end": event.time})
            self.tours[event.tour_id]["lines"] += 1
            self._walk[event.tour_id] = [event.time, 0.0]
        elif isinstance(event, TourEnd) and event.tour_id in self.tours:
            self.tours[event.tour_id]["end"] = event.time

    def orders(self, sim) -> list[dict]:
        affected = set(getattr(self.source, "affected_order_ids", ()))
        picks: dict = {}
        for pick in self.picks:
            row = picks.setdefault(pick["order_id"], {
                "tour": pick["tour_id"], "picker": pick["picker_id"],
                "first": pick["pick_start"], "last": pick["pick_end"]})
            row["first"] = min(row["first"], pick["pick_start"])
            row["last"] = max(row["last"], pick["pick_end"])
        table = []
        for order in sorted(sim.state.order_manager.known_orders(), key=lambda o: o.order_id):
            row = picks.get(order.order_id)
            tour = self.tours.get(row["tour"]) if row else None
            table.append({
                "order_id": order.order_id, "affected": order.order_id in affected,
                "release_time": order.order_date,
                "due_date": order.due_date, "lines": len(order.order_positions),
                "tour_id": row["tour"] if row else None,
                "tour_start": tour["start"] if tour else None,
                "completion": tour["end"] if tour else None,
                "picker_id": row["picker"] if row else None,
                "first_pick_start": row["first"] if row else None,
                "last_pick_end": row["last"] if row else None})
        return table

    def tour_rows(self, sim) -> list[dict]:
        """Every committed tour with its executed facts (empty if never started)."""
        resources = sim.state.resource_manager
        rows = []
        for tour in sorted(sim.state.tour_manager.get_all_tours(), key=lambda t: t.tour_id):
            if tour.tour_id not in self.commit_time:
                continue
            executed = self.tours.get(tour.tour_id, {})
            rows.append({
                "tour_id": tour.tour_id, "picker_id": tour.assigned_resource,
                "status": tour.status.value, "commit_time": self.commit_time[tour.tour_id],
                "planned_start": tour.planned_start,
                "start": executed.get("start"), "setup_end": executed.get("setup_end"),
                "end": executed.get("end"),
                "setup": (executed["setup_end"] - executed["start"] if executed
                          else resources.get_resource(tour.assigned_resource).tour_setup_time
                          or 0.0),
                "route_distance": tour.route.distance,
                "distance": executed.get("distance", 0.0),
                "lines": executed.get("lines", 0),
                "orders_at_start": len(tour.original_route.batch.orders),
                "order_ids": " ".join(map(str, sorted(tour.order_numbers)))})
        return rows

    def write(self, sim, output_dir: Path) -> None:
        directory = Path(output_dir) / FACTS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        for name, columns, rows in (
                ("tours.csv", TOUR_COLUMNS, self.tour_rows(sim)),
                ("orders.csv", ORDER_COLUMNS, self.orders(sim)),
                ("picks.csv", PICK_COLUMNS, self.picks)):
            with open(directory / name, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
