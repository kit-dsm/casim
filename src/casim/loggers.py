from __future__ import annotations
from typing import TYPE_CHECKING

import copy
import json
from pathlib import Path

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.state import State
from scenarios.io_helpers import dump_pickle
if TYPE_CHECKING:
    from casim.simulation_engine import SimulationEngine


class EventLogger:
    def on_reset(self, sim: SimulationEngine, domain: SimWarehouseDomain) -> None: ...
    def on_event(self, event: Event, sim: SimulationEngine) -> None: ...
    def on_done(self, sim: SimulationEngine) -> None: ...


class DashLogger(EventLogger):
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots = []

    def on_reset(self, sim, domain):
        self.snapshots = []
        dump_pickle(str(self.out_dir / "static.pkl"), {
            "layout": domain.layout,
            "storage_locations": domain.storage.locations,
            "orders": domain.orders.orders
        })

    def on_event(self, event, sim):
        state = sim.state
        om = state.order_manager
        tm = state.tour_manager

        self.snapshots.append({
            "event_id": event.id,
            "event_type": event.__class__.__name__,
            "time": state.current_time,

            "pickers": copy.deepcopy(state.resource_manager.get_resources()),

            "buffered_order_ids": list(om._order_buffer.keys()),
            "pick_list_buffer": [
                [o.order_id for o in pl.orders] for pl in om._pick_list_buffer
            ],

            "tours": {
                t_id: {
                    "status": t.status,
                    "picker_id": t.assigned_resource,
                    "order_ids": list(t.order_numbers),
                    "start_time": t.start_time,
                }
                for t_id, t in tm.all_tours.items()
            },
            "active_picker_tour": dict(tm._active_picker_tour),
        })

    def on_done(self, state):
        dump_pickle(str(self.out_dir / "events.pkl"), self.snapshots)


class KPILogger(EventLogger):
    def __init__(self, out_dir: Path, print_every: int | None = 500):
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
        self._print_summary(summary)
        dump_pickle(str(self.out_dir / "kpis.pkl"), summary)
        with open(self.out_dir / "kpis.json", "w") as f:
            json.dump(_jsonable(summary), f, indent=2)

    @staticmethod
    def _summary(state: State) -> dict:
        t = state.tracker
        horizon = state.current_time
        horizon_h = horizon / 3600 if horizon else 0

        total_orders = sum(len(oids) for _, _, _, oids in t.completed_tours)

        return {
            "makespan": horizon,
            "num_tours": len(t.completed_tours),
            "num_orders_completed": total_orders,
            "avg_tour_makespan": t.average_tour_makespan,
            "avg_batch_size": t.average_batch_size,
            "orders_per_hour": total_orders / horizon_h if horizon_h else 0.0,
            "tours_per_hour": len(t.completed_tours) / horizon_h if horizon_h else 0.0,
            "distance_by_picker": dict(t.distance_by_picker),
            "idle_time_by_picker": dict(t.idle_time_by_picker),
            "total_distance": sum(t.distance_by_picker.values()),
        }

    def _print_progress(self, state: State) -> None:
        t = state.tracker
        print(
            f"[t={state.current_time:>10.0f}] "
            f"events={self._event_count:>6d}  "
            f"tours={len(t.completed_tours):>4d}  "
            f"avg_batch={t.average_batch_size:.2f}  "
            f"dist={sum(t.distance_by_picker.values()):.0f}"
        )

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