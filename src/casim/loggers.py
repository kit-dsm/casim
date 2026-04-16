import copy
from pathlib import Path

from casim.domain_objects.tour_model import TourStates
from casim.events.base_events import Event
from casim.state import State
from scenarios.io_helpers import dump_pickle


class EventLogger:
    def on_event(self, event: Event, state: State) -> None: ...
    def on_done(self, state: State) -> None: ...


class DashLogger(EventLogger):
    def __init__(self, out_path: Path):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots = []

    def on_event(self, event, state):
        self.snapshots.append({
            "event": f"{str(event.id).zfill(5)} - {event.__class__.__name__}",
            "time": state.current_time,
            "pickers": copy.deepcopy(state.resource_manager.get_resources()),
        })

    def on_done(self, state):
        dump_pickle(str(self.out_path), self.snapshots)


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