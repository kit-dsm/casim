import heapq
import logging
from typing import Callable, Type

from tqdm import tqdm

from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import Order

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.events.operational_events import OrderArrival, FlushRemainingOrders
from casim.loggers import EventLogger
from casim.state import State
from casim.simulation_engine.conditions import Condition
from casim.simulation_engine.state_adapter import StateAdapter

logger = logging.getLogger(__name__)


class SimulationEngine:
    def __init__(self,
                 state_adapters: dict[str, StateAdapter],
                 triggers_map: dict[Type[Event], str],
                 conditions_map: dict[str, Condition],
                 domain_cache_path: str,
                 loader_kwargs: dict,
                 data_loader: DataLoader = None,
                 reset_hook: Callable[['SimulationEngine'], None] = None,
                 event_loggers: list[EventLogger] | None = None):

        self.state_adapters = state_adapters
        self.state: State | None = None
        self.triggers_map = triggers_map
        self.conditions_map = conditions_map
        self.domain_cache_path = domain_cache_path
        self.events = []
        self.reset_hook = reset_hook
        self.loader_kwargs = loader_kwargs
        self.event_loggers = event_loggers or []
        self._initial_domain = None
        if data_loader:
            self.data_loader = data_loader

    def load_data(self) -> SimWarehouseDomain:
        domain = self.data_loader.load(**self.loader_kwargs)
        return domain

    def reset(self, hooks: list[Callable[['SimulationEngine', SimWarehouseDomain], None]] = ()):
        domain = self.load_data()
        self._initial_domain = domain
        self.state = State(
            layout=domain.layout,
            articles=domain.articles,
            storage=domain.storage,
            resources=domain.resources,
            active_objective = domain.objective
        )

        for hook in (hooks or []):
            hook(self, domain)

        for el in self.event_loggers:
            el.on_reset(self, domain)

    def add_order(self, order: Order):
        self.add_event(OrderArrival(order.order_date, order))

    def add_event(self, event: Event):
        heapq.heappush(self.events, event)

    def run(self) -> [bool, SimWarehouseDomain | None]:
        if not hasattr(self, "_pbar"):
            self._pbar = tqdm(
                desc="sim events", unit=" ev", unit_scale=True,
                bar_format="{desc}: {n_fmt} [{elapsed}, {rate_fmt}] sim_t={postfix}",
            )
        while self.events:
            event = heapq.heappop(self.events)
            # logger.info(f"Event {event} popped at state time: {self.state.current_time}, events start: {event.time}")
            self.state.current_time = event.time
            events_to_add = event.handle(self.state)

            self._pbar.update(1)
            if self._pbar.n % 1000 == 0:
                self._pbar.set_postfix_str(f"{self.state.current_time:.0f}s")

            for e in events_to_add:
                self.add_event(e)

            for el in self.event_loggers:
                el.on_event(event, self)

            if event.__class__ in self.triggers_map:
                problem = self.triggers_map[event.__class__]
                state_transformer = self.state_adapters[problem]
                state_snapshot = state_transformer.transform_state(self.state, problem)
                conditions = self.conditions_map.get(problem) or []
                if (all(c.get_decision(state_snapshot) for c in conditions if c is not None) or
                        isinstance(event, FlushRemainingOrders)):
                    return False, state_snapshot

            if not self.events and self.state.order_manager.get_order_buffer():
                self.state.done_flag = True
                # self.add_event(FlushRemainingOrders(self.state.current_time))

        logger.info("Simulation complete")
        if hasattr(self, "_pbar"):
            self._pbar.close()
            del self._pbar
        for el in self.event_loggers:
            el.on_done(self)
        return True, None

    def step(self, events_to_add, problem_class, solution):
        state_adapter = self.state_adapters[problem_class]
        state_adapter.cleanup_state(self.state, solution)
        if events_to_add:
            for e in events_to_add:
                self.add_event(e)
