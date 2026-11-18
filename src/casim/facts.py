"""The realised facts of a run (docs/measures_and_ranking_design.md, "Facts").

``FactsRecorder`` observes the executed events and writes, in ``facts/``,

- ``tours.csv``: one row per started tour: ``tour_id``, ``picker_id``,
  ``commit_time``, ``start`` (of the setup), ``setup_end`` (start of the
  route), ``end``, ``distance``, ``lines``, ``order_ids``;
- ``orders.csv``: one row per order: ``order_id``, ``release_time``,
  ``due_date``, ``lines``, ``tour_id``, ``tour_start``, ``completion`` (end
  of its tour, empty if not completed), and for comparisons with recorded
  data ``picker_id``, ``first_pick_start``, ``last_pick_end``;
- ``picks.csv``: one row per picked line (R1.1 of docs/ijpe/plan.md):
  ``tour_id``, ``picker_id``, ``order_id``, ``article_id``, ``node_x``,
  ``node_y``, ``walk_start``, ``walk_distance``, ``pick_start``, ``pick_end``.
  The walk starts at the end of the tour's previous pick, or at the route
  start for the first pick; its distance is the route length walked since.

The older ``tours.csv`` and ``orders.csv`` next to ``facts/`` are to be derived
from these tables, or removed, after the revision.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ware_ops_algos.planning.lifecycle import TourStates

from casim.events.operational_events import PickComplete, TourEnd, TourStart, TravelEvent
from casim.loggers import EventLogger

FACTS_DIR = "facts"
TOUR_COLUMNS = ("tour_id", "picker_id", "commit_time", "start", "setup_end", "end",
                "distance", "lines", "order_ids")
ORDER_COLUMNS = ("order_id", "release_time", "due_date", "lines", "tour_id", "tour_start",
                 "completion", "picker_id", "first_pick_start", "last_pick_end")
PICK_COLUMNS = ("tour_id", "picker_id", "order_id", "article_id", "node_x", "node_y",
                "walk_start", "walk_distance", "pick_start", "pick_end")


class FactsRecorder(EventLogger):
    def __init__(self):
        self.picks: list[dict] = []
        self.tours: dict[int, dict] = {}
        self.commit_time: dict[int, float] = {}
        self._walk: dict[int, list[float]] = {}  # tour -> [walk start, distance so far]

    def on_decision(self, problem_class, result, state_snapshot, sim) -> None:
        # A tour is committed by the first decision after which it is no longer planned.
        for tour in sim.state.tour_manager.get_all_tours():
            if tour.tour_id not in self.commit_time and tour.status is not TourStates.PLANNED:
                self.commit_time[tour.tour_id] = sim.state.current_time

    def on_event(self, event, sim) -> None:
        if isinstance(event, TourStart):
            tour = sim.state.tour_manager.get_tour(event.tour_id)
            resource = sim.state.resource_manager.get_resource(tour.assigned_resource)
            self.tours[event.tour_id] = {
                "tour_id": event.tour_id, "picker_id": tour.assigned_resource,
                "commit_time": self.commit_time.get(event.tour_id),
                "start": event.time - (resource.tour_setup_time or 0.0),
                "setup_end": event.time, "end": None, "distance": 0.0, "lines": 0,
                "order_ids": " ".join(map(str, sorted(tour.order_numbers)))}
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
                "order_id": order.order_id, "release_time": order.order_date,
                "due_date": order.due_date, "lines": len(order.order_positions),
                "tour_id": row["tour"] if row else None,
                "tour_start": tour["start"] if tour else None,
                "completion": tour["end"] if tour else None,
                "picker_id": row["picker"] if row else None,
                "first_pick_start": row["first"] if row else None,
                "last_pick_end": row["last"] if row else None})
        return table

    def write(self, sim, output_dir: Path) -> None:
        directory = Path(output_dir) / FACTS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        for name, columns, rows in (
                ("tours.csv", TOUR_COLUMNS, [self.tours[t] for t in sorted(self.tours)]),
                ("orders.csv", ORDER_COLUMNS, self.orders(sim)),
                ("picks.csv", PICK_COLUMNS, self.picks)):
            with open(directory / name, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
