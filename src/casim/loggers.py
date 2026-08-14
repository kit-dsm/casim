from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import TYPE_CHECKING

from ware_ops_algos.domain_models import Resource

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.domain_objects.tour_model import TourStates
from casim.events.operational_events import Event
from casim.events.operational_events import OrderArrival
from casim.io_helpers import dump_pickle
from casim.state import State
from casim.trackers import ExperimentTracker

if TYPE_CHECKING:
    from casim.simulation_engine import SimulationEngine


class EventLogger:
    def on_reset(self, sim: SimulationEngine, domain: SimWarehouseDomain) -> None: ...
    def on_event(self, event: Event, sim: SimulationEngine) -> None: ...
    def on_done(self, sim: SimulationEngine) -> None: ...


class DashLogger(EventLogger):
    """Write compact replayable operational-state deltas for the Dash app."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = None
        self._previous_state = None

    def on_reset(self, sim, domain):
        dump_pickle(
            str(self.out_dir / "static.pkl"),
            {
                "layout": domain.layout,
                "storage_locations": domain.storage.locations,
                "orders": domain.orders.orders,
            },
        )
        self.events_file = gzip.open(
            self.out_dir / "events.jsonl.gz",
            "wt",
            encoding="utf-8",
        )
        self._previous_state = None
        self._write_record(
            event_id=-1,
            event_type="Reset",
            event_time=sim.state.current_time,
            event_context={},
            sim=sim,
        )

    def on_event(self, event, sim):
        context = {
            name: getattr(event, name)
            for name in (
                "picker_id",
                "tour_id",
                "route_version",
                "expected_route_version",
            )
            if getattr(event, name, None) is not None
        }
        order = getattr(event, "order", None)
        if order is not None:
            context["order_id"] = int(order.order_id)
        self._write_record(
            event_id=event.id,
            event_type=event.__class__.__name__,
            event_time=sim.state.current_time,
            event_context=context,
            sim=sim,
        )

    def on_done(self, sim):
        if self.events_file is not None:
            self.events_file.close()
            self.events_file = None

    def _write_record(
        self,
        *,
        event_id: int,
        event_type: str,
        event_time: float,
        event_context: dict,
        sim,
    ) -> None:
        full_snapshot = self._previous_state is None or event_type in {
            "PickListDone",
            "SequencingDone",
        }
        picker_ids = None if full_snapshot else set()
        tour_ids = None if full_snapshot else set()
        if not full_snapshot:
            picker_id = event_context.get("picker_id")
            tour_id = event_context.get("tour_id")
            if tour_id is not None:
                tour_ids.add(int(tour_id))
                tour = sim.state.tour_manager.all_tours.get(int(tour_id))
                if tour is not None and tour.assigned_resource is not None:
                    picker_ids.add(int(tour.assigned_resource))
            if picker_id is not None:
                picker_ids.add(int(picker_id))
        include_buffers = full_snapshot or event_type in {
            "OrderArrival",
            "PickListDone",
            "SequencingDone",
            "ActiveRouteReplacement",
            "TourEnd",
        }
        current = self._operational_state(
            sim,
            picker_ids=picker_ids,
            tour_ids=tour_ids,
            include_buffers=include_buffers,
        )
        delta = (
            current
            if self._previous_state is None
            else _state_delta(self._previous_state, current)
        )
        record = {
            "event_id": int(event_id),
            "event_type": event_type,
            "time": float(event_time),
            "event": event_context,
            "state": delta,
        }
        self.events_file.write(
            json.dumps(record, separators=(",", ":"), sort_keys=True)
            + "\n"
        )
        if full_snapshot:
            self._previous_state = current
        else:
            _merge_state(self._previous_state, current)

    def _operational_state(
        self,
        sim,
        *,
        picker_ids: set[int] | None = None,
        tour_ids: set[int] | None = None,
        include_buffers: bool = True,
    ) -> dict:
        state = sim.state
        order_manager = state.order_manager
        tour_manager = state.tour_manager
        pickers = {}
        active_picker_tour = {}
        for picker in state.resources.resources:
            if picker_ids is not None and picker.id not in picker_ids:
                continue
            tour = tour_manager.get_active_tour_for_picker(picker.id)
            active_picker_tour[picker.id] = (
                tour.tour_id if tour is not None else None
            )
            position, edge_progress = self._execution_position(
                picker,
                tour,
                state.current_time,
            )
            if tour is None:
                phase = "idle"
                route_suffix = []
                route_version = None
                remaining_picks = 0
                completed_picks = 0
                remaining_pick_ids = []
                completed_pick_ids = []
                bin_owners = []
                locked_bins = []
                replan_requested = False
                intervention_event_pending = False
                waiting_for = None
            else:
                phase = (
                    "waiting"
                    if tour.waiting_for
                    else
                    "picking"
                    if tour.is_picking
                    else "travelling"
                    if tour.is_travelling
                    else "at_node"
                )
                route_suffix = [
                    list(position),
                    *[
                        list(node.position)
                        for node in tour.annotated_route[tour.cursor + 1 :]
                    ],
                ]
                route_version = tour.route_version
                remaining_picks = len(tour.remaining_picks)
                completed_picks = len(tour.completed_picks)
                remaining_pick_ids = [
                    [
                        pick.order_number,
                        pick.article_id,
                        list(pick.pick_node),
                    ]
                    for pick in tour.remaining_picks
                ]
                completed_pick_ids = [
                    [
                        pick.order_number,
                        pick.article_id,
                        list(pick.pick_node),
                    ]
                    for pick in tour.completed_picks
                ]
                bin_owners = [
                    sorted(cart_bin.order_ids)
                    for cart_bin in tour.cart_bins
                ]
                locked_bins = [
                    cart_bin.bin_id
                    for cart_bin in tour.cart_bins
                    if cart_bin.locked
                ]
                replan_requested = bool(tour.replan_requested)
                intervention_event_pending = bool(
                    tour.intervention_event_pending
                )
                waiting_for = tour.waiting_for
            pickers[int(picker.id)] = {
                "id": int(picker.id),
                "position": list(position),
                "occupied": bool(picker.occupied),
                "active_tour_id": active_picker_tour[picker.id],
                "phase": phase,
                "route_version": route_version,
                "edge_progress": edge_progress,
                "remaining_picks": remaining_picks,
                "completed_picks": completed_picks,
                "remaining_pick_ids": remaining_pick_ids,
                "completed_pick_ids": completed_pick_ids,
                "route_suffix": route_suffix,
                "bin_owners": bin_owners,
                "locked_bins": locked_bins,
                "replan_requested": replan_requested,
                "intervention_event_pending": intervention_event_pending,
                "waiting_for": waiting_for,
            }
        tours = {
            int(tour_id): {
                "status": (
                    tour.status.value
                    if isinstance(tour.status, TourStates)
                    else str(tour.status)
                ),
                "picker_id": tour.assigned_resource,
                "order_ids": list(tour.order_numbers),
                "start_time": tour.start_time,
                "end_time": tour.end_time,
                "route_version": tour.route_version,
                "waiting_for": tour.waiting_for,
                "remaining_picks": len(tour.remaining_picks),
                "completed_picks": len(tour.completed_picks),
                "bin_owners": [
                    sorted(cart_bin.order_ids)
                    for cart_bin in tour.cart_bins
                ],
                "locked_bins": [
                    cart_bin.bin_id
                    for cart_bin in tour.cart_bins
                    if cart_bin.locked
                ],
            }
            for tour_id, tour in tour_manager.all_tours.items()
            if tour_ids is None or tour_id in tour_ids
        }
        result = {
            "pickers": pickers,
            "tours": tours,
            "active_picker_tour": active_picker_tour,
            "pending_event_count": len(sim.events),
            "intervention_count": len(state.tracker.interventions),
            "last_intervention": (
                state.tracker.interventions[-1]
                if state.tracker.interventions
                else None
            ),
        }
        if include_buffers:
            result.update(
                {
                    "buffered_order_ids": sorted(
                        order.order_id
                        for order in order_manager.get_order_buffer()
                    ),
                    "completed_order_ids": sorted(
                        order.order_id
                        for order in order_manager.completed_orders
                    ),
                    "pick_list_buffer": [
                        [order.order_id for order in batch.orders]
                        for batch in order_manager.get_pick_list_buffer()
                    ],
                }
            )
        return result

    @staticmethod
    def _execution_position(
        picker: Resource,
        tour,
        time: float,
    ) -> tuple[tuple[float, float], float | None]:
        if tour is None or not tour.is_travelling:
            return DashLogger._pos(picker), None
        duration = tour.edge_arrives_at - tour.edge_started_at
        progress = (
            1.0
            if duration <= 0
            else (time - tour.edge_started_at) / duration
        )
        progress = min(1.0, max(0.0, progress))
        origin_x, origin_y = tour.edge_origin.position
        destination_x, destination_y = tour.edge_destination.position
        return (
            (
                origin_x + (destination_x - origin_x) * progress,
                origin_y + (destination_y - origin_y) * progress,
            ),
            progress,
        )

    @staticmethod
    def _pos(p: Resource):
        loc = p.current_location
        if hasattr(loc, "position"):
            loc = loc.position
        return (loc[0], loc[1])


class KPILogger(EventLogger):
    def __init__(self, out_dir: Path, print_every: int | None = 5000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.print_every = print_every
        self._event_count = 0

    def on_reset(self, sim, domain):
        self._event_count = 0

    def on_event(self, event, sim) -> None:
        state = sim.state
        self._event_count += 1
        if self.print_every and self._event_count % self.print_every == 0:
            self._print_progress(state)

    def on_done(self, sim: SimulationEngine) -> None:
        state = sim.state
        summary = self._summary(state)
        summary["pending_order_arrival_ids"] = sorted(
            event.order_id
            for event in sim.events
            if isinstance(event, OrderArrival)
        )
        self._print_summary(summary)
        with open(self.out_dir / "kpis.json", "w") as f:
            json.dump(_jsonable(summary), f, indent=2)

        tracker_dict = self._tracker_to_dict(state.tracker)

        with open(self.out_dir / "tracker.json", "w") as f:
            json.dump(_jsonable(tracker_dict), f, indent=2)

    @staticmethod
    def _summary(state: State) -> dict:
        t = state.tracker
        horizon = state.current_time
        horizon_h = horizon / 3600 if horizon else 0

        return {
            "completion_reason": state.completion_reason,
            "unfinished_work": state.unfinished_work(),
            "makespan": t.completion_makespan,
            "num_tours": len(t.completed_tours),
            "num_orders_completed": t.completed_order_count,
            "avg_tour_makespan": t.average_tour_makespan,
            "avg_batch_size": t.average_batch_size,
            "orders_per_hour": (
                t.completed_order_count / horizon_h if horizon_h else 0.0
            ),
            "tours_per_hour": len(t.completed_tours) / horizon_h if horizon_h else 0.0,
            "distance_by_picker": dict(t.distance_by_picker),
            "idle_time_by_picker": dict(t.idle_time_by_picker),
            "total_distance": sum(t.distance_by_picker.values()),
            "total_delayed": t.total_delayed,
            "dock_utilization": dict(t.dock_utilization),
            "total_tour_times": t.total_processing_time,
            "picks_per_hour": (
                t.completed_line_count / horizon_h if horizon_h else 0.0
            ),
            "total_processing_time": t.total_processing_time
            # "total_on_time": t.total_on_time
        }

    def _print_progress(self, state: State) -> None:
        t = state.tracker
        print(
            f"[t={state.current_time:>10.0f}] "
            f"events={self._event_count:>6d}  "
            f"tours={len(t.completed_tours):>4d}  "
            f"avg_batch={t.average_batch_size:.2f}  "
            f"dist={sum(t.distance_by_picker.values()):.0f},"
            f"on_time_ratio={100 * t.on_time_ratio:.1f}% "
            f"delayed_ratio={100 * t.delayed_ratio:.1f} "
            + (
                f"dock_fill={state.n_staged_pallets}"
                if state.dock_capacity is not None
                else ""
            )
        )

    @staticmethod
    def _tracker_to_dict(tracker: ExperimentTracker) -> dict:
        return {
            "distance_by_picker": dict(tracker.distance_by_picker),
            "idle_time_by_picker": dict(tracker.idle_time_by_picker),
            "idle_intervals": tracker.idle_intervals,
            "completed_tours": tracker.completed_tours,
            "truck_departures": tracker.truck_departures,
            "dock_utilization": tracker.dock_utilization,
            "batch_buffer": tracker.batch_buffer,
            "avg_makespan": tracker.avg_makespan,
            "all_delayed": tracker.all_delayed,
            "all_on_time": tracker.all_on_time,
            "avg_utilization": tracker.picker_utilization,
            "delayed_tours_exp_finish": tracker.delayed_expected_finish,
            "total_processing_time": tracker.total_processing_time
        }

    @staticmethod
    def _print_summary(s: dict) -> None:
        print("\n" + "=" * 50)
        print("KPI Summary")
        print("=" * 50)
        print(f"  makespan:            {s['makespan']:.0f}")
        print(f"  tours completed:     {s['num_tours']}")
        print(f"  orders completed:    {s['num_orders_completed']}")
        print(f"  avg tour makespan:   {s['avg_tour_makespan']:.1f}")
        print(f"  avg batch size:      {s['avg_batch_size']:.2f}")
        print(f"  orders/hour:         {s['orders_per_hour']:.1f}")
        print(f"  total distance:      {s['total_distance']:.0f}")
        print(f"  total delayed:      {s['total_delayed']:.0f}")
        print(f"  total tour times:    {s['total_tour_times']:.0f}")
        print(f"  total processing time:    {s['total_processing_time']:.0f}")
        for pid, d in s['distance_by_picker'].items():
            idle = s['idle_time_by_picker'].get(pid, 0)
            print(f"    picker {pid}: distance={d:.0f}  idle={idle:.0f}")
        print("=" * 50 + "\n")


def _jsonable(obj):
    """Coerce dict-of-numbers-with-int-keys to json-friendly form."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return obj


def _state_delta(previous, current):
    """Return changed dictionary branches; lists are replaced as values."""
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return current if previous != current else {}
    changed = {}
    for key, value in current.items():
        if key not in previous:
            changed[key] = value
            continue
        old_value = previous[key]
        if isinstance(old_value, dict) and isinstance(value, dict):
            nested = _state_delta(old_value, value)
            if nested:
                changed[key] = nested
        elif old_value != value:
            changed[key] = value
    return changed


def _merge_state(target, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_state(target[key], value)
        else:
            target[key] = value
