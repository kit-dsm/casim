import heapq
import copy
import logging
import time
from dataclasses import replace
from typing import Callable, Type

from tqdm import tqdm

from casim.data_loader import DataLoader
from ware_ops_algos.domain_models import Order, SimWarehouseDomain
from casim.events.base_events import Event
from casim.events.operational_events import (
    OrderArrival, ActiveTourOpportunity, Reconsider,
)
from casim.decision_engine.commands import CommandViolation, Wait
from casim.loggers import EventLogger
from casim.state import State
from casim.simulation_engine.conditions import Condition
from casim.decision_engine.decision_engine import DecisionResult
from casim.interfaces import StateProjector

logger = logging.getLogger(__name__)


class SimulationEngine:
    def __init__(self,
                 state_adapters: dict[str, StateProjector],
                 triggers_map: dict[Type[Event], str],
                 conditions_map: dict[str, tuple[Condition, ...]],
                 domain_cache_path: str,
                 loader_kwargs: dict,
                 problem_classes: dict[str, str] | None = None,
                 data_loader: DataLoader = None,
                 reset_hook: Callable[['SimulationEngine'], None] = None,
                 event_loggers: list[EventLogger] | None = None):

        self.state_adapters = state_adapters
        self.problem_classes = problem_classes or {}
        self.state: State | None = None
        self.triggers_map = triggers_map
        self.conditions_map = conditions_map
        self.domain_cache_path = domain_cache_path
        self.events = []
        self._pending_decisions: dict[str, set[str]] = {}
        self._decided_at_time: set[str] = set()
        self._decision_time: float | None = None
        self.precedence = {name: index for index, name in enumerate(state_adapters)}
        self.periods: dict[str, float] = {}
        self._period_anchor: float = 0.0
        self.on_violation = "reject"
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
        self.events = []
        self._pending_decisions.clear()
        self._decided_at_time.clear()
        self._decision_time = None
        self.state = State(
            layout=domain.layout,
            articles=domain.articles,
            storage=domain.storage,
            resources=domain.resources,
            active_objective = domain.objective,
            warehouse_info=domain.warehouse_info,
            information=domain.information,
        )
        self._period_anchor = self.state.current_time

        for hook in (hooks or []):
            hook(self, domain)

        for decision, every_s in self.periods.items():
            self.add_event(Reconsider(self.state.current_time + every_s,
                                      decision, 0, reason="periodic"))

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
        while self.events or self._pending_decisions:
            if self._pending_decisions and (
                not self.events or self.events[0].time > self.state.current_time
            ):
                problem = min(self._pending_decisions,
                              key=lambda name: (self.precedence.get(name, 0), name))
                causes = self._pending_decisions.pop(problem)
                if problem in self._decided_at_time:
                    continue
                projected = self.state_adapters[problem].transform_state(
                    self.state, problem, self.problem_classes.get(problem, problem))
                if projected is None:
                    raise ValueError(f"{type(self.state_adapters[problem]).__name__} returned None")
                projected.context = replace(projected.context, cause=tuple(sorted(causes)))
                projected.warehouse_info = copy.deepcopy(self.state.warehouse_info)
                conditions = self.conditions_map.get(problem) or ()
                self._decided_at_time.add(problem)
                should_solve = (all(condition.get_decision(projected)
                                    for condition in conditions
                                    if condition is not None)
                                or "FlushRemainingOrders" in causes)
                if (should_solve and "Reconsider:release_window" in causes
                        and projected.memory.current_information_version ==
                        projected.memory.last_information_version):
                    projected.context = replace(projected.context, solve=False)
                    return False, projected
                if should_solve:
                    return False, projected
                if projected.plan.resources and any(tour.status.value == "scheduled"
                       for tour in projected.plan.tours):
                    projected.context = replace(projected.context, solve=False)
                    return False, projected
                self._schedule_periodic(problem)
                continue
            event = heapq.heappop(self.events)
            if event.is_stale(self.state):
                continue
            if self._decision_time != event.time:
                self._decision_time = event.time
                self._decided_at_time.clear()
            # logger.info(f"Event {event} popped at state time: {self.state.current_time}, events start: {event.time}")
            self.state.current_time = event.time
            events_to_add = event.handle(self.state)

            self._pbar.update(1)
            if self._pbar.n % 1000 == 0:
                self._pbar.set_postfix_str(f"{self.state.current_time:.0f}s")

            for e in events_to_add:
                if isinstance(e, ActiveTourOpportunity) and type(e) not in self.triggers_map:
                    continue
                self.add_event(e)

            for el in self.event_loggers:
                el.on_event(event, self)

            problem = (event.decision if isinstance(event, Reconsider)
                       else self.triggers_map.get(type(event)))
            if problem is not None:
                cause = (f"Reconsider:{event.reason}" if isinstance(event, Reconsider)
                         else type(event).__name__)
                self._pending_decisions.setdefault(problem, set()).add(cause)

        logger.info("Simulation complete")
        if hasattr(self, "_pbar"):
            self._pbar.close()
            del self._pbar
        for el in self.event_loggers:
            el.on_done(self)
        return True, None

    def step(self, result: DecisionResult, state_snapshot: SimWarehouseDomain):
        decision = state_snapshot.context.decision
        adapter = self.state_adapters[decision]
        apply_start = time.perf_counter()
        try:
            events = self.state.apply_all(decision, result.commands,
                                          adapter.selection, adapter.scope)
        except CommandViolation as violation:
            if self.on_violation == "drop":
                logger.warning("Dropped %s decision: %s: %s",
                               decision, violation.code.value, violation)
                return
            if self.on_violation != "repair":
                raise
            if getattr(self, "decision_engine", None) is None:
                raise
            fresh = adapter.transform_state(
                self.state, decision, self.problem_classes.get(decision, decision))
            fresh.context = replace(fresh.context, cause=("repair",))
            result = self.decision_engine.on_trigger(fresh)
            events = self.state.apply_all(decision, result.commands,
                                          adapter.selection, adapter.scope)
            state_snapshot = fresh
        decision_engine = getattr(self, "decision_engine", None)
        if decision_engine is not None:
            decision_engine.on_accepted(result, state_snapshot)
            decision_engine.decision_tracker.on_commands(
                self.state.current_time, decision, result.commands,
                result.conversion_seconds, time.perf_counter() - apply_start,
            )
        if state_snapshot.context.solve:
            self.state.last_solved_information_version[decision] = (
                self.state.information_version)
        for command in result.commands:
            if isinstance(command, Wait) and command.reconsider_at is not None:
                if command.reconsider_at > self.state.current_time:
                    self.add_event(Reconsider(
                        command.reconsider_at, decision,
                        self.state.decision_versions[decision], command.reason))
        self._schedule_periodic(decision)
        for event in (*events, *result.events):
            self.add_event(event)
        for event_logger in self.event_loggers:
            event_logger.on_decision(decision, result,
                                     state_snapshot, self)

    def _schedule_periodic(self, decision: str) -> None:
        if decision in self.periods and not self.state.done_flag:
            period = self.periods[decision]
            elapsed = max(0.0, self.state.current_time - self._period_anchor)
            tick = int(elapsed // period) + 1
            self.add_event(Reconsider(
                self._period_anchor + tick * period, decision,
                self.state.decision_versions.get(decision, 0), reason="periodic"))
