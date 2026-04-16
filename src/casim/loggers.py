import copy
from pathlib import Path

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.domain_objects.tour_model import TourStates
from casim.events.base_events import Event
from casim.state import State
from scenarios.io_helpers import dump_pickle


class EventLogger:
    def on_reset(self, domain: SimWarehouseDomain): ...
    def on_event(self, event: Event, state: State) -> None: ...
    def on_done(self, state: State) -> None: ...


class DashLogger(EventLogger):
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots = []

    def on_reset(self, domain):
        self.snapshots = []
        dump_pickle(str(self.out_dir / "static.pkl"), {
            "layout": domain.layout,
            "storage_locations": domain.storage.locations,
            "orders": domain.orders.orders
        })

    def on_event(self, event, state):
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
    def on_done(self, state: State) -> None:
        tours = []
        for t_id in state.tour_manager.all_tours.keys():
            t = state.tour_manager.all_tours[t_id]
            assert t.status == TourStates.DONE
            tours.append(t)

        processed_orders = []
        for o_id in state.order_manager._order_history.keys():
            o = state.order_manager._order_history[o_id]
            processed_orders.append(o)
        print(len(processed_orders))
        processed_orders = state.tracker.processed_orders
        # sol = self._evaluate_due_dates(tours, processed_orders)
        # print("Max makespan", sol["completion_time"].max() / 1000)
        # print("Final average makespan", self.state.tracker.average_tour_makespan)
        # print("Final # on time", sol["on_time"].sum())
        # sol.to_csv("./solution.csv")

        print("makespan", state.current_time)
        # self.state.tracker.final_makespan = sol["completion_time"].max()
        state.tracker.final_makespan = state.current_time
        # self.state.tracker.save_logs()
        # print(sol)