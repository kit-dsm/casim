"""Order events and project each due warehouse decision once per timestamp."""

from __future__ import annotations

import heapq
import logging
from dataclasses import replace
from typing import Callable

from tqdm import tqdm

from ware_ops_algos.domain_models import Order

from casim.data_loader import DataLoader
from casim.decision_engine.decision_engine import DecisionResult
from casim.domain_objects.sim_domain import ReconsiderReason, SimWarehouseDomain
from casim.events.base_events import Event
from casim.events.operational_events import OrderArrival, ActiveTourOpportunity, Reconsider
from casim.loggers import EventLogger
from casim.simulation_engine.conditions import Condition
from casim.simulation_engine.state_adapter import StateAdapter
from casim.state import State
from casim.state.views import OperationsView


logger = logging.getLogger(__name__)


class SimulationEngine:
    def __init__(self,
                 state_adapters: dict[str, StateAdapter],
                 triggers_map: dict[type[Event], tuple[str, ...]],
                 conditions_map: dict[str, tuple[Condition, ...]],
                 domain_cache_path: str,
                 loader_kwargs: dict,
                 problem_classes: dict[str, str],
                 data_loader: DataLoader,
                 precedence: dict[str, int | None] | None = None,
                 periods: dict[str, float] | None = None,
                 reset_hook: Callable[['SimulationEngine'], None] | None = None,
                 event_loggers: list[EventLogger] | None = None):
        self.state_adapters = state_adapters
        self.triggers_map = triggers_map
        self.conditions_map = conditions_map
        self.problem_classes = problem_classes
        self.domain_cache_path = domain_cache_path
        self.loader_kwargs = loader_kwargs
        self.data_loader = data_loader
        self.precedence = precedence or {}
        self.periods = periods or {}
        self.reset_hook = reset_hook
        self.event_loggers = event_loggers or []
        self.state: State | None = None
        self.events: list[Event] = []
        self._pending_decisions: dict[str, tuple[set[str], set[ReconsiderReason]]] = {}
        self._decided_at_time: set[str] = set()
        self._decision_time: float | None = None
        self._period_anchor = 0.0
        self._initial_domain: SimWarehouseDomain | None = None

    def load_data(self) -> SimWarehouseDomain:
        return self.data_loader.load(**self.loader_kwargs)

    def reset(self, hooks: list[Callable[['SimulationEngine', SimWarehouseDomain], None]] = ()):
        domain = self.load_data()
        self._initial_domain = domain
        self.events.clear()
        self._pending_decisions.clear()
        self._decided_at_time.clear()
        self._decision_time = None
        self.state = State(
            layout=domain.layout, articles=domain.articles,
            storage=domain.storage, resources=domain.resources,
            active_objective=domain.objective,
            warehouse_info=domain.warehouse_info,
            information=domain.information,
        )
        self._period_anchor = self.state.current_time
        for hook in hooks or ():
            hook(self, domain)
        for decision, every_s in self.periods.items():
            self.add_event(Reconsider(self.state.current_time + every_s,
                                      decision, 0, ReconsiderReason.PERIODIC))
        for event_logger in self.event_loggers:
            event_logger.on_reset(self, domain)

    def add_order(self, order: Order):
        self.add_event(OrderArrival(order.order_date, order))

    def add_event(self, event: Event):
        heapq.heappush(self.events, event)

    def _operations(self) -> OperationsView:
        return OperationsView(
            is_break=self.state.is_break,
            n_staged_pallets=(self.state.dock_manager.n_staged_pallets
                              if hasattr(self.state, "dock_manager") else 0),
        )

    def _due(self, decision: str, cause: str | None,
             reason: ReconsiderReason | None = None) -> None:
        causes, reconsider = self._pending_decisions.setdefault(
            decision, (set(), set()))
        if cause is not None:
            causes.add(cause)
        if reason is not None:
            reconsider.add(reason)

    def run(self) -> tuple[bool, SimWarehouseDomain | None]:
        if not hasattr(self, "_pbar"):
            self._pbar = tqdm(
                desc="sim events", unit=" ev", unit_scale=True,
                bar_format="{desc}: {n_fmt} [{elapsed}, {rate_fmt}] sim_t={postfix}",
            )
        while self.events or self._pending_decisions:
            if self._pending_decisions and (
                    not self.events or self.events[0].time > self.state.current_time):
                decision = min(self._pending_decisions,
                               key=lambda name: (self.precedence.get(name) is None,
                                                 self.precedence.get(name) or 0, name))
                causes, reasons = self._pending_decisions.pop(decision)
                if decision in self._decided_at_time:
                    continue
                snapshot = self.state_adapters[decision].transform_state(
                    self.state, decision, self.problem_classes[decision])
                dynamic = snapshot.dynamic_warehouse_info
                context = replace(dynamic.context,
                                  cause=tuple(sorted(causes)),
                                  reconsider=tuple(sorted(reasons, key=lambda item: item.value)))
                snapshot.dynamic_warehouse_info = replace(dynamic, context=context)
                conditions = self.conditions_map.get(decision, ())
                solve = (all(condition.get_decision(snapshot, self._operations())
                             for condition in conditions)
                         or "FlushRemainingOrders" in causes
                         or ReconsiderReason.WAIT in reasons)
                snapshot.dynamic_warehouse_info = replace(
                    snapshot.dynamic_warehouse_info,
                    context=replace(context, solve=solve))
                self._decided_at_time.add(decision)
                return False, snapshot

            event = heapq.heappop(self.events)
            if event.is_stale(self.state):
                continue
            if self._decision_time != event.time:
                self._decision_time = event.time
                self._decided_at_time.clear()
            self.state.current_time = event.time
            events_to_add = event.handle(self.state)
            self._pbar.update(1)
            if self._pbar.n % 1000 == 0:
                self._pbar.set_postfix_str(f"{self.state.current_time:.0f}s")
            for emitted in events_to_add:
                if (isinstance(emitted, ActiveTourOpportunity)
                        and type(emitted) not in self.triggers_map):
                    continue
                self.add_event(emitted)
            for event_logger in self.event_loggers:
                event_logger.on_event(event, self)
            if isinstance(event, Reconsider):
                self._due(event.decision, None, event.reason)
            else:
                for decision in self.triggers_map.get(type(event), ()):
                    self._due(decision, type(event).__name__)

        logger.info("Simulation complete")
        if hasattr(self, "_pbar"):
            self._pbar.close()
            del self._pbar
        for event_logger in self.event_loggers:
            event_logger.on_done(self)
        return True, None

    def step(self, result: DecisionResult, state_snapshot: SimWarehouseDomain):
        dynamic = state_snapshot.dynamic_warehouse_info
        decision = dynamic.context.decision
        events = self.state.apply_all(decision, result.commands)
        self.state.decision_waiting[decision] = result.waiting
        self.state.decision_reconsider_at[decision] = result.reconsider_at
        self.state.decision_considered_active_orders[decision] = set(
            result.considered_active_orders)
        if dynamic.context.solve:
            self.state.last_solved_information_version[decision] = (
                self.state.information_version)
        if (result.reconsider_at is not None
                and result.reconsider_at > self.state.current_time):
            reason = (ReconsiderReason.WAIT if result.waiting
                      else ReconsiderReason.COMMITMENT_WINDOW)
            self.add_event(Reconsider(result.reconsider_at, decision,
                                      self.state.decision_versions[decision], reason))
        self._schedule_periodic(decision)
        for event in events:
            self.add_event(event)
        for event_logger in self.event_loggers:
            event_logger.on_decision(decision, result, state_snapshot, self)

    def _schedule_periodic(self, decision: str) -> None:
        if decision in self.periods and not self.state.done_flag:
            period = self.periods[decision]
            elapsed = max(0.0, self.state.current_time - self._period_anchor)
            tick = int(elapsed // period) + 1
            self.add_event(Reconsider(
                self._period_anchor + tick * period, decision,
                self.state.decision_versions.get(decision, 0),
                ReconsiderReason.PERIODIC))
