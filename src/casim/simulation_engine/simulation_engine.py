import heapq
import logging
from typing import Callable, Type

from tqdm import tqdm

from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import Order

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.operational_events import (
    Event,
    FlushRemainingOrders,
    InterventionRequest,
    OrderArrival,
    ProcessEvent,
)
from casim.loggers import EventLogger
from casim.state import State
from casim.simulation_engine.state_adapter import StateAdapter

logger = logging.getLogger(__name__)


class _NullProgress:
    """No-op progress bar used when ``show_progress`` is disabled."""

    n = 0

    def update(self, amount=1):
        self.n += amount

    def set_postfix_str(self, value):
        return None

    def close(self):
        return None


class SimulationEngine:
    def __init__(self,
                 state_adapters: dict[str, StateAdapter],
                 triggers_map: dict[Type[Event], str],
                 conditions_map: dict[str, dict],
                 loader_kwargs: dict,
                 data_loader: DataLoader = None,
                 event_loggers: list[EventLogger] | None = None,
                 completion_mode: str = "drain",
                 horizon_time: float | None = None,
                 intervention_enabled: bool = False,
                 active_batch_insertion_enabled: bool = False,
                 show_progress: bool = True):

        self.state_adapters = state_adapters
        self.state: State | None = None
        self.triggers_map = triggers_map
        self.conditions_map = conditions_map
        self.events = []
        self.loader_kwargs = loader_kwargs
        self.event_loggers = event_loggers or []
        if completion_mode not in {"drain", "horizon"}:
            raise ValueError(
                "completion_mode must be 'drain' or 'horizon'"
            )
        if completion_mode == "horizon" and horizon_time is None:
            raise ValueError(
                "horizon_time is required in horizon completion mode"
            )
        self.completion_mode = completion_mode
        self.horizon_time = (
            float(horizon_time) if horizon_time is not None else None
        )
        self._drain_problem = self.triggers_map.get(FlushRemainingOrders)
        self._drain_signature = None
        self._finished = False
        self._pending_snapshot = None
        self.intervention_enabled = intervention_enabled
        self.active_batch_insertion_enabled = active_batch_insertion_enabled
        self.show_progress = bool(show_progress)
        if data_loader:
            self.data_loader = data_loader

    def reset(self, hooks: list[Callable[['SimulationEngine', SimWarehouseDomain], None]] = ()):
        domain = self.data_loader.load(**self.loader_kwargs)
        self.state = State(
            layout=domain.layout,
            articles=domain.articles,
            storage=domain.storage,
            resources=domain.resources,
            active_objective = domain.objective
        )
        self.state.intervention_enabled = self.intervention_enabled
        self.state.active_batch_insertion_enabled = (
            self.active_batch_insertion_enabled
        )
        self.events = []
        self._drain_signature = None
        self._finished = False
        self._pending_snapshot = None

        for hook in (hooks or []):
            hook(self, domain)

        for el in self.event_loggers:
            el.on_reset(self, domain)
        return domain

    def add_order(self, order: Order):
        version = self.state.register_order(order)
        self.add_event(OrderArrival(
            order.order_date,
            order_id=order.order_id,
            release_version=version,
        ))

    def add_event(self, event: Event):
        heapq.heappush(self.events, event)

    @staticmethod
    def _work_signature(work: dict[str, object]) -> tuple:
        return (
            tuple(work["buffered_order_ids"]),
            tuple(work["buffered_batch_order_ids"]),
            tuple(
                (
                    tour_id,
                    value["status"],
                    tuple(value["order_ids"]),
                )
                for tour_id, value in sorted(
                    work["nonterminal_tours"].items()
                )
            ),
            tuple(work["reservation_tour_ids"]),
            tuple(work["occupied_picker_ids"]),
            tuple(work["orphaned_order_ids"]),
        )

    def _finish(self, reason: str):
        self.state.done_flag = True
        self.state.completion_reason = reason
        self._finished = True
        logger.info("Simulation complete: %s", reason)
        if hasattr(self, "_pbar"):
            self._pbar.close()
            del self._pbar
        for event_logger in self.event_loggers:
            event_logger.on_done(self)
        return True, None

    def _conditions_hold(
        self,
        problem: str,
        snapshot: SimWarehouseDomain,
    ) -> bool:
        requires = self.conditions_map.get(problem) or {}
        if not requires:
            return True
        dynamic = snapshot.dynamic_warehouse_info
        drain = (
            self.state.input_closed
            and problem == self._drain_problem
        )
        if drain:
            if "orders" in requires and not snapshot.orders.orders:
                return False
            if "batches" in requires and not dynamic.buffered_batches:
                return False
            remaining = {
                k: v for k, v in requires.items()
                if k not in ("orders", "batches")
            }
        else:
            remaining = requires
        if "orders" in remaining:
            if len(snapshot.orders.orders) < int(remaining["orders"]):
                return False
        if "batches" in remaining:
            if len(dynamic.buffered_batches) < int(remaining["batches"]):
                return False
        if "pickers" in remaining:
            if len(snapshot.resources.resources) < int(remaining["pickers"]):
                return False
        if remaining.get("not_on_break") and dynamic.is_break:
            return False
        if "dock_capacity" in remaining:
            threshold = int(remaining["dock_capacity"])
            if dynamic.n_staged_pallets > threshold:
                return False
        return True

    def run(self) -> tuple[bool, SimWarehouseDomain | None]:
        if self._finished:
            return True, None
        if self._pending_snapshot is not None:
            snapshot = self._pending_snapshot
            self._pending_snapshot = None
            return False, snapshot
        if not hasattr(self, "_pbar"):
            if self.show_progress:
                self._pbar = tqdm(
                    desc="sim events", unit=" ev", unit_scale=True,
                    bar_format="{desc}: {n_fmt} [{elapsed}, {rate_fmt}] sim_t={postfix}",
                )
            else:
                self._pbar = _NullProgress()
        while self.events:
            if (
                self.completion_mode == "horizon"
                and self.events[0].time > self.horizon_time
            ):
                self.state.current_time = self.horizon_time
                return self._finish("horizon_complete")
            event = heapq.heappop(self.events)
            self.state.current_time = event.time
            events_to_add = event.handle(self.state)

            self._pbar.update(1)
            if self._pbar.n % 1000 == 0:
                self._pbar.set_postfix_str(f"{self.state.current_time:.0f}s")

            for e in events_to_add:
                self.add_event(e)

            for el in self.event_loggers:
                el.on_event(event, self)

            state_snapshot = self._snapshot_for_trigger(event)
            if state_snapshot is not None:
                return False, state_snapshot

        unfinished = self.state.unfinished_work()
        if self.completion_mode == "horizon":
            self.state.current_time = self.horizon_time
            return self._finish("horizon_complete")
        if self.state.has_unfinished_work():
            raise RuntimeError(
                "Simulation stalled with unfinished work: "
                f"{unfinished}"
            )
        return self._finish("drained")

    def _snapshot_for_trigger(
        self,
        event: Event,
    ) -> SimWarehouseDomain | None:
        if (
            event.__class__ not in self.triggers_map
            or getattr(event, "cancelled", False)
        ):
            return None
        problem = self.triggers_map[event.__class__]
        state_snapshot = self.state_adapters[problem].transform_state(
            self.state,
            problem,
            trigger=event,
        )
        if not self._conditions_hold(problem, state_snapshot):
            return None
        if (
            self.completion_mode == "drain"
            and self.state.input_closed
            and problem == self._drain_problem
        ):
            self._drain_signature = self._work_signature(
                self.state.unfinished_work()
            )
        return state_snapshot

    def step(self, events_to_add, problem_class, solution=None):
        """Synchronously commit a zero-latency decision result.

        Process events are the operational hand-off of a decision.  They run
        before the simulator can pop another event at the same timestamp.
        Events returned by the commitment are queued normally.
        """
        previous_drain_signature = self._drain_signature
        self._drain_signature = None
        for event in events_to_add or []:
            if not isinstance(event, ProcessEvent):
                raise TypeError(
                    "DecisionEngine may only hand ProcessEvent instances to "
                    f"SimulationEngine.step, got {type(event).__name__}"
                )
            if float(event.time) != float(self.state.current_time):
                raise ValueError(
                    "Zero-latency process events must use the current "
                    "simulation time"
                )
            follow_up = event.handle(self.state)
            for next_event in follow_up:
                self.add_event(next_event)
            for event_logger in self.event_loggers:
                event_logger.on_event(event, self)
            state_snapshot = self._snapshot_for_trigger(event)
            if state_snapshot is not None:
                if self._pending_snapshot is not None:
                    raise RuntimeError(
                        "One decision commitment produced multiple immediate "
                        "decision triggers"
                    )
                self._pending_snapshot = state_snapshot
        if previous_drain_signature is not None:
            current = self._work_signature(self.state.unfinished_work())
            if current == previous_drain_signature:
                raise RuntimeError(
                    "Drain decision made no operational progress for "
                    f"problem {problem_class}"
                )
        remaining = self.state.unfinished_work()
        if (
            self.completion_mode == "drain"
            and self.state.input_closed
            and self._drain_problem is not None
            and (
                remaining["buffered_order_ids"]
                or remaining["buffered_batch_order_ids"]
            )
            and not any(
                isinstance(event, FlushRemainingOrders)
                for event in self.events
            )
        ):
            self.add_event(
                FlushRemainingOrders(self.state.current_time)
            )
