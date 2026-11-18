"""KPIs of a run: progress while it runs, and the summary of ``kpis.json``."""

from __future__ import annotations

import logging

from casim.outputs.tracker import ExperimentTracker
from casim.simulation_engine.event_logger import EventLogger
from casim.state import State

logger = logging.getLogger(__name__)


class KPILogger(EventLogger):
    """Logs progress every ``print_every`` events and the summary at the end."""

    def __init__(self, print_every: int | None = 5000):
        self.print_every = print_every
        self._event_count = 0

    def on_reset(self, sim, domain):
        self._event_count = 0

    def on_event(self, event, sim) -> None:
        self._event_count += 1
        if self.print_every and self._event_count % self.print_every == 0:
            t = sim.state.tracker
            logger.info(
                f"[t={sim.state.current_time:>10.0f}] events={self._event_count:>6d}  "
                f"tours={len(t.completed_tours):>4d}  avg_batch={t.average_batch_size:.2f}  "
                f"dist={sum(t.distance_by_picker.values()):.0f}, "
                f"on_time_ratio={100 * t.on_time_ratio:.1f}% "
                f"delayed_ratio={100 * t.delayed_ratio:.1f} "
                f"dock_fill={sim.state.dock_manager.n_staged_pallets}")

    def on_done(self, sim) -> None:
        logger.info("%s", format_summary(summary(sim.state)))


def summary(state: State) -> dict:
    """The KPIs of ``kpis.json``."""
    t = state.tracker
    horizon_h = state.current_time / 3600 if state.current_time else 0
    total_orders = sum(len(tour.order_ids) for tour in t.completed_tours)
    total_lines = sum(tour.lines for tour in t.completed_tours)
    completion_delays = [
        tour.end - state.order_manager.get_order_from_history(order_id).order_date
        for tour in t.completed_tours for order_id in tour.order_ids]
    return {
        "makespan": max((tour.end for tour in t.completed_tours), default=0.0),
        "mean_order_completion_time": (sum(completion_delays) / len(completion_delays)
                                       if completion_delays else 0.0),
        "num_tours": len(t.completed_tours),
        "num_orders_completed": total_orders,
        "avg_tour_makespan": t.average_tour_makespan,
        "avg_batch_size": t.average_batch_size,
        "orders_per_hour": total_orders / horizon_h if horizon_h else 0.0,
        "tours_per_hour": len(t.completed_tours) / horizon_h if horizon_h else 0.0,
        "distance_by_picker": dict(t.distance_by_picker),
        "idle_time_by_picker": dict(t.idle_time_by_picker),
        "busy_time_by_picker": dict(t.busy_time_by_picker),
        "absent_time_by_picker": dict(t.absent_time_by_picker),
        "total_distance": sum(t.distance_by_picker.values()),
        "total_delayed": t.total_delayed,
        "dock_utilization": dict(t.dock_utilization),
        "total_tour_times": sum(t.process_times),
        "picks_per_hour": total_lines / horizon_h if horizon_h else 0.0,
        "total_processing_time": t.total_processing_time,
    }


def timeline(tracker: ExperimentTracker) -> dict:
    """The time series of ``tracker.json`` (plots of the dock and utilisation)."""
    return {
        "distance_by_picker": dict(tracker.distance_by_picker),
        "idle_time_by_picker": dict(tracker.idle_time_by_picker),
        "idle_intervals": tracker.idle_intervals,
        "completed_tours": [tour.to_dict() for tour in tracker.completed_tours],
        "truck_departures": tracker.truck_departures,
        "dock_utilization": tracker.dock_utilization,
        "batch_buffer": tracker.batch_buffer,
        "avg_makespan": tracker.avg_makespan,
        "all_delayed": tracker.all_delayed,
        "all_on_time": tracker.all_on_time,
        "avg_utilization": tracker.picker_utilization,
        "delayed_tours_exp_finish": tracker.delayed_expected_finish,
        "total_processing_time": tracker.total_processing_time,
    }


def format_summary(s: dict) -> str:
    """The KPI summary of one run as text."""
    lines = ["=" * 50, "KPI Summary", "=" * 50,
             f"  makespan:            {s['makespan']:.0f}",
             f"  mean order completion time: {s['mean_order_completion_time']:.3f}",
             f"  tours completed:     {s['num_tours']}",
             f"  orders completed:    {s['num_orders_completed']}",
             f"  avg tour makespan:   {s['avg_tour_makespan']:.1f}",
             f"  avg batch size:      {s['avg_batch_size']:.2f}",
             f"  orders/hour:         {s['orders_per_hour']:.1f}",
             f"  total distance:      {s['total_distance']:.0f}",
             f"  total delayed:       {s['total_delayed']:.0f}",
             f"  total tour times:    {s['total_tour_times']:.0f}",
             f"  total processing time: {s['total_processing_time']:.0f}"]
    for pid, d in s["distance_by_picker"].items():
        idle = s["idle_time_by_picker"].get(pid, 0)
        lines.append(f"    picker {pid}: distance={d:.0f}  idle={idle:.0f}")
    lines.append("=" * 50)
    return "\n".join(lines)


def jsonable(obj):
    """Dicts with int keys and tuples as JSON-friendly values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    return obj
