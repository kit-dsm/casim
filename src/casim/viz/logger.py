"""Record a run's states for the Dash viewer (casim.viz.app)."""

from __future__ import annotations

import pickle
from pathlib import Path

from ware_ops_algos.domain_models import Resource

from casim.simulation_engine.event_logger import EventLogger


def dump_pickle(path: str, data) -> None:
    with open(path, "wb") as handle:
        pickle.dump(data, handle)


class DashLogger(EventLogger):
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = None

    def on_reset(self, sim, domain):
        dump_pickle(
            str(self.out_dir / "static.pkl"),
            {
                "layout": domain.layout,
                "storage_locations": domain.storage.locations,
                "orders": domain.orders.orders,
            },
        )

        self.events_file = open(self.out_dir / "events.pkl", "wb")

    def on_event(self, event, sim):
        snapshot = self._snapshot(sim)
        snapshot.update(event_id=event.id, event_type=event.__class__.__name__)
        pickle.dump(snapshot, self.events_file, protocol=pickle.HIGHEST_PROTOCOL)

    def on_decision(self, problem_class, result, state_snapshot, sim):
        command_names = [type(command).__name__ for command in result.commands]
        label = f"{problem_class}: {', '.join(command_names) or 'no plan changes'}"
        snapshot = self._snapshot(sim)
        snapshot.update(event_id=None, event_type="Decision",
                        decision_summary=label, commands=command_names)
        pickle.dump(snapshot, self.events_file, protocol=pickle.HIGHEST_PROTOCOL)

    def _snapshot(self, sim):
        state = sim.state
        om = state.order_manager
        tm = state.tour_manager

        return {
            "time": state.current_time,
            "pickers": [
                {
                    "id": p.id,
                    "position": self._pos(p),
                    "status": state.resource_manager.resource_status(p.id, tm).value,
                    "current_tour": getattr(p, "current_tour_id", None),
                }
                for p in state.resource_manager.get_resources().resources
            ],
            "buffered_order_ids": list(om._order_buffer.keys()),
            "pick_list_buffer": [
                [o.order_id for o in pl.orders] for pl in om._pick_list_buffer
            ],
            "tours": {
                t_id: {
                    "status": t.status,
                    "picker_id": t.assigned_resource,
                    "order_ids": list(t.order_numbers),
                    "start_time": t.planned_start,
                }
                for t_id, t in tm.all_tours.items()
            },
            "active_picker_tour": dict(tm._active_picker_tour),
        }

    def on_done(self, state):
        if self.events_file is not None:
            self.events_file.close()

    def _pos(self, p: Resource):
        loc = p.current_location
        if hasattr(loc, "position"):
            loc = loc.position
        return (loc[0], loc[1])
