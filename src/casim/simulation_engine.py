import copy
import heapq
import logging
from pathlib import Path
from typing import Callable, Type

from ware_ops_algos.algorithms import TourStates
from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import Order

from casim.decision_engine.decision_engine import DecisionEngine
from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.events.operational_events import OrderArrival, FlushRemainingOrders
from casim.loggers import EventLogger
from casim.state import State
from casim.state.conditions import Condition
from casim.state.state_snapshot import StateSnapshot

logger = logging.getLogger(__name__)

class SimulationEngine:
    def __init__(self,
                 state_transformers: dict[str, StateSnapshot],
                 # decision_engine: DecisionEngine,
                 triggers: dict[Type[Event], str],
                 conditions_map: dict[str, Condition],
                 cache_path: str | Path,
                 domain_cache_path: str,
                 loader_kwargs: dict,
                 data_loader: DataLoader = None,
                 reset_hook: Callable[['SimulationEngine'], None] = None,
                 event_loggers: list[EventLogger] | None = None):

        self.cache_path = cache_path
        self.state_transformers = state_transformers
        self.state: State | None = None
        self.triggers = triggers
        self.conditions_map = conditions_map
        self.domain_cache_path = domain_cache_path
        # self.decision_engine = decision_engine
        self.events = []
        # self.event_manager = EventManager()
        self.reset_hook = reset_hook
        self.loader_kwargs = loader_kwargs
        self.even_loggers = event_loggers or []
        if data_loader:
            self.data_loader = data_loader

    def load_data(self) -> SimWarehouseDomain:
        domain = self.data_loader.load(**self.loader_kwargs)
        return domain

    def reset(self, hooks: list[Callable[['SimulationEngine', SimWarehouseDomain], None]] = None):
        domain = self.load_data()
        orders = domain.orders.orders
        self.state = State(
            layout=domain.layout,
            articles=domain.articles,
            storage=domain.storage,
            resources=domain.resources
        )

        if hooks:
            for hook in hooks:
                hook(self, domain)

    def add_order(self, order: Order):
        self.add_event(OrderArrival(order.order_date, order))

    def add_event(self, event: Event):
        heapq.heappush(self.events, event)

    def run(self) -> [bool, SimWarehouseDomain]:
        while self.events:
            event = heapq.heappop(self.events)
            logger.info(f"Event {event} popped at state time: {self.state.current_time}, events start: {event.time}")
            self.state.current_time = event.time
            events_to_add = event.handle(self.state)

            for e in events_to_add:
                self.add_event(e)

            for el in self.even_loggers:
                el.on_event(event, self.state)

            if event.__class__ in self.triggers.keys():
                problem = self.triggers[event.__class__]
                state_transformer = self.state_transformers[problem]
                state_snapshot = state_transformer.transform_state(self.state, problem)
                condition = self.conditions_map[problem]
                if condition.get_decision(state_snapshot):  # if decision necessary -> "pause"
                    return False, state_snapshot

            if not self.events and len(self.state.order_manager.get_order_buffer()) > 0:
                self.state.done_flag = True
                self.add_event(FlushRemainingOrders(self.state.current_time))

            # if self.track_states:
            #     self.state.statistics.append(copy.deepcopy({
            #         "event": f"{str(event.id).zfill(5)} - {event.__class__.__name__}",
            #         "time": self.state.current_time,
            #         "pickers": self.state.resource_manager.get_resources()
            #     }))


        logger.info("Simulation complete")
        for el in self.even_loggers:
            el.on_done(self.state)
        return True, None

    def step(self, events_to_add):
        if events_to_add:
            for e in events_to_add:
                self.add_event(e)

    def _is_done(self):
        if self.events:
            return False
        elif not self.events and len(self.state.order_manager.get_order_buffer()) > 0:
            return False
        elif not self.events:
            return True

    # def _write_solutions(self):
    #     tours = []
    #     for t_id in self.state.tour_manager.all_tours.keys():
    #         t = self.state.tour_manager.all_tours[t_id]
    #         assert t.status == TourStates.DONE
    #         tours.append(t)
    #
    #     processed_orders = []
    #     for o_id in self.state.order_manager._order_history.keys():
    #         o = self.state.order_manager._order_history[o_id]
    #         processed_orders.append(o)
    #     print(len(processed_orders))
    #     processed_orders = self.state.tracker.processed_orders
    #     # sol = self._evaluate_due_dates(tours, processed_orders)
    #     # print("Max makespan", sol["completion_time"].max() / 1000)
    #     # print("Final average makespan", self.state.tracker.average_tour_makespan)
    #     # print("Final # on time", sol["on_time"].sum())
    #     # sol.to_csv("./solution.csv")
    #
    #     print("makespan", self.state.current_time)
    #     # self.state.tracker.final_makespan = sol["completion_time"].max()
    #     self.state.tracker.final_makespan = self.state.current_time
    #     # self.state.tracker.save_logs()
    #     # print(sol)
