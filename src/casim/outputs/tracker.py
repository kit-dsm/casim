from collections import defaultdict
from dataclasses import asdict, dataclass

from ware_ops_algos.planning.commands import CancelTour, CreateTour, ReplaceSuffix, Reschedule


@dataclass(frozen=True)
class CompletedTour:
    """A finished tour: when it ran, its orders and their due-date outcome."""

    tour_id: int
    start: float
    end: float
    order_ids: list[int]
    picker_id: int
    on_time: list[int]
    delayed: list[int]
    lines: int

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentTracker:
    def __init__(self, n_pickers):
        self.first_tour_start = None
        self.committed_times: dict[int, tuple[float, float]] = {}
        self.distance_by_picker: dict[int, float] = defaultdict(float)
        # picker_id -> [(time, derived resource state)], one entry per change
        self._state_log: dict[int, list[tuple[float, str]]] = defaultdict(list)
        self._horizon_start: float | None = None
        self._horizon_end: float | None = None
        self.pick_durations_by_picker: dict[int, list[float]] = defaultdict(list)
        self.completed_tours: list[CompletedTour] = []
        self.truck_departures: list[tuple[float, int]] = []
        self.dock_utilization: list[tuple[float, int]] = []
        self.batch_buffer: list[tuple[float, int]] = []
        self.avg_makespan: list[tuple[float, float]] = []
        self.picker_utilization: list[tuple[float, float]] = []
        self.all_delayed = []
        self.all_on_time = []
        self.n_pickers: int = n_pickers
        self.process_times: list[float] = []
        # truck departure: (time, delayed_tour_ids, expected_finished_time)
        self.delayed_expected_finish: list[tuple[float, list[int], list[float]]] = []
        # shifted orders: [(order_id, from, to)]
        self.shifted_orders: list[tuple[str, float, float]] = []
        # ingested orders [(order_id, due_at)]
        self.ingested_orders: list[tuple[int, float]] = []
        # (tour_id, start_time, end_time, list[order_ids], picker_id)

    def on_commit(self, tour_id, planned_start, planned_end):
        """Keep the planned times at commit, to report deviations of execution."""
        self.committed_times[tour_id] = (planned_start, planned_end)

    def on_travel(self, picker_id, distance):
        self.distance_by_picker[picker_id] += distance

    def on_resource_state(self, picker_id, time, state: str) -> None:
        """Log a resource's derived state; only changes are kept."""
        if self._horizon_start is None:
            self._horizon_start = time
        log = self._state_log[picker_id]
        if not log or log[-1][1] != state:
            log.append((time, state))
        self._horizon_end = time

    def close(self, time) -> None:
        """End the horizon at the last event."""
        self._horizon_end = time

    @property
    def horizon(self) -> float:
        if self._horizon_start is None:
            return 0.0
        return self._horizon_end - self._horizon_start

    def _intervals(self, states: frozenset[str]):
        for picker_id, log in self._state_log.items():
            for (start, state), (end, _) in zip(log, [*log[1:], (self._horizon_end, None)]):
                if state in states and end > start:
                    yield picker_id, start, end

    def _time_in(self, states: frozenset[str]) -> dict[int, float]:
        totals = {picker_id: 0.0 for picker_id in self._state_log}
        for picker_id, start, end in self._intervals(states):
            totals[picker_id] += end - start
        return totals

    @property
    def idle_time_by_picker(self) -> dict[int, float]:
        """Present and not busy: idle, or reserved and waiting for a later start."""
        return self._time_in(frozenset(("idle", "reserved")))

    @property
    def busy_time_by_picker(self) -> dict[int, float]:
        """With a pending (setup) or started tour."""
        return self._time_in(frozenset(("busy",)))

    @property
    def absent_time_by_picker(self) -> dict[int, float]:
        return self._time_in(frozenset(("off_shift",)))

    @property
    def idle_intervals(self) -> list[tuple[int, float, float]]:
        """Maximal intervals in which a resource is present and not busy."""
        merged = []
        for picker_id, start, end in self._intervals(frozenset(("idle", "reserved"))):
            if merged and merged[-1][0] == picker_id and merged[-1][2] == start:
                merged[-1] = (picker_id, merged[-1][1], end)
            else:
                merged.append((picker_id, start, end))
        return merged

    def on_pick_end(self, tour_id, picker_id, order_id, item_id, start_time, end_time):
        self.pick_durations_by_picker[picker_id].append(end_time - start_time)

    def on_tour_start(self, time):
        if self.first_tour_start == None:
            self.first_tour_start = time
        # self.batch_buffer.append((time, batch_buffer))

    def on_tour_end(self,
                    tour_id,
                    start_time,
                    end_time,
                    order_ids,
                    picker_id,
                    on_time,
                    delayed,
                    n_pallets_dock,
                    n_lines):
        self.completed_tours.append(CompletedTour(
            tour_id=tour_id, start=start_time, end=end_time, order_ids=list(order_ids),
            picker_id=picker_id, on_time=list(on_time), delayed=list(delayed), lines=n_lines))
        for delayed_order_id in delayed:
            self.all_delayed.append(delayed_order_id)
        for on_time_order_id in on_time:
            self.all_on_time.append(on_time_order_id)
        self.dock_utilization.append((end_time, n_pallets_dock))
        self.avg_makespan.append((end_time, self.average_tour_makespan))
        util = self.current_utilization(end_time)
        self.picker_utilization.append((end_time, util))
        self.process_times.append(end_time - start_time)

    def on_truck_departure(self, time, capacity):
        self.truck_departures.append((time, capacity))

    def on_batch_arrival(self, time, batch_buffer):
        self.batch_buffer.append((time, batch_buffer))

    def on_truck_departure_delays(self, time, open_tours, expected_finish_times):
        self.delayed_expected_finish.append((time, open_tours, expected_finish_times))

    def on_volume_shift(self, order_nr, from_time, to_time):
        self.shifted_orders.append((order_nr, from_time, to_time))

    def on_order_ingestion(self, order_nr, due_date):
        self.ingested_orders.append((order_nr, due_date))

    def current_utilization(self, current_time: float) -> float:
        elapsed = current_time - self.first_tour_start
        if elapsed <= 0:
            return 0.0

        total_tour_time = sum(tour.duration for tour in self.completed_tours)
        return total_tour_time / (elapsed * self.n_pickers)

    @property
    def total_processing_time(self):
        return sum(self.process_times)

    @property
    def total_delayed(self) -> int:
        return len(self.all_delayed)

    @property
    def all_orders_fulfilled(self) -> int:
        return len(self.all_delayed) + len(self.all_on_time)

    @property
    def on_time_ratio(self) -> float:
        total = self.all_orders_fulfilled
        return len(self.all_on_time) / total if total > 0 else 0.0

    @property
    def delayed_ratio(self) -> float:
        total = self.all_orders_fulfilled
        return len(self.all_delayed) / total if total > 0 else 0.0

    @property
    def tour_durations(self) -> list[float]:
        return [tour.duration for tour in self.completed_tours]

    @property
    def average_tour_makespan(self) -> float:
        d = self.tour_durations
        return sum(d) / len(d) if d else 0.0

    @property
    def average_batch_size(self) -> float:
        if not self.completed_tours:
            return 0.0
        total_orders = sum(len(tour.order_ids) for tour in self.completed_tours)
        return total_orders / len(self.completed_tours)


class DecisionTracker:
    def __init__(self):
        self.decisions: list[tuple] = []
        self.pipeline_counts: dict[str, int] = defaultdict(int)
        self.command_decisions: list[dict] = []

    def on_commands(self, time, decision, result, snapshot, apply_seconds):
        commands = result.commands
        context = snapshot.dynamic_warehouse_info.context
        visible = {tour.tour_id for tour in snapshot.dynamic_warehouse_info.plan.tours}
        cancelled = {command.tour_id for command in commands
                     if isinstance(command, CancelTour)}
        changed = {command.tour_id for command in commands
                   if isinstance(command, (ReplaceSuffix, Reschedule))} - cancelled
        self.command_decisions.append({
            "time": time,
            "decision": decision,
            "trigger": list(context.cause),
            "reconsider": [reason.value for reason in context.reconsider],
            "candidates": [{"pipeline": name, "score": score}
                           for name, score in result.candidate_scores],
            "chosen_pipeline": result.solver_name,
            "chosen_score": result.objective_value,
            "solve_seconds": result.solve_seconds,
            "commands": [type(command).__name__ for command in commands],
            "kept_tours": len(visible - changed - cancelled),
            "changed_tours": len(changed),
            "cancelled_tours": len(cancelled),
            "created_tours": sum(isinstance(command, CreateTour) for command in commands),
            "conversion_seconds": result.conversion_seconds,
            "apply_seconds": apply_seconds,
        })

    def on_decision(self, problem_class, input_ids, selected_pipeline,
                    kpi_value, kpi, runtime, elapsed):
        self.decisions.append((
            problem_class,
            input_ids,
            selected_pipeline,
            kpi_value,
            kpi,
            runtime,
            elapsed
        ))
        self.pipeline_counts[selected_pipeline] += 1

    @property
    def num_decisions(self) -> int:
        return len(self.decisions)
