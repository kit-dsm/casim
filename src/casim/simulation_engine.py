import heapq
import logging
from pathlib import Path
from typing import Callable, Type

from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import Order

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
        self.events = []
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
        self.state = State(
            layout=domain.layout,
            articles=domain.articles,
            storage=domain.storage,
            resources=domain.resources
        )

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

        logger.info("Simulation complete")
        for el in self.even_loggers:
            el.on_done(self.state)
        return True, None

    def step(self, events_to_add):
        if events_to_add:
            for e in events_to_add:
                self.add_event(e)
