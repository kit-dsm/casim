from collections import defaultdict


class ExperimentTracker:
    def __init__(self, n_pickers):
        self.first_tour_start = None
        self.distance_by_picker: dict[int, float] = defaultdict(float)
        self.idle_time_by_picker: dict[int, float] = defaultdict(float)
        self._idle_start: dict[int, float] = {}
        self.idle_intervals: list[tuple[int, float, float]] = []
        self.pick_durations_by_picker: dict[int, list[float]] = defaultdict(list)
        self.completed_tours: list[tuple[int, float, float, list[int], int, list[int], list[int], int]] = []
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
        self.interventions: list[dict] = []
        self.availability_intervals: list[dict] = []
        self._availability_start: dict[int, float] = {}
        self._available_time_closed_by_picker: dict[int, float] = (
            defaultdict(float)
        )
        self._completed_tour_time = 0.0
        self._completed_order_count = 0
        self._completed_line_count = 0
        self._completion_makespan = 0.0
        self._accrued_flow_time = 0.0
        self._flow_time_updated_at = 0.0
        self._active_order_ids: set[object] = set()
        self.order_arrival_times: dict[object, float] = {}
        self.order_completion_times: dict[object, float] = {}
        # (tour_id, start_time, end_time, list[order_ids], picker_id)

    def _advance_flow_time(self, time: float) -> None:
        time = float(time)
        if time < self._flow_time_updated_at:
            raise ValueError("Flow-time observations must be chronological")
        self._accrued_flow_time += (
            time - self._flow_time_updated_at
        ) * len(self._active_order_ids)
        self._flow_time_updated_at = time

    def on_order_arrival(self, order_id, time: float) -> None:
        """Start accumulating the operational flow time of one order."""
        self._advance_flow_time(time)
        if order_id in self.order_arrival_times:
            raise ValueError(f"Order {order_id} arrived more than once")
        self.order_arrival_times[order_id] = float(time)
        self._active_order_ids.add(order_id)

    def on_orders_completed(self, order_ids, time: float) -> None:
        """Stop accumulating flow time for newly completed original orders."""
        self._advance_flow_time(time)
        for order_id in order_ids:
            if order_id not in self._active_order_ids:
                continue
            self._active_order_ids.remove(order_id)
            self.order_completion_times[order_id] = float(time)

    def accrued_flow_time(self, current_time: float) -> float:
        """Return total flow time accrued by completed and waiting orders."""
        current_time = float(current_time)
        if current_time < self._flow_time_updated_at:
            raise ValueError("Flow time cannot be queried in the past")
        return self._accrued_flow_time + (
            current_time - self._flow_time_updated_at
        ) * len(self._active_order_ids)

    def on_travel(self, picker_id, distance):
        self.distance_by_picker[picker_id] += distance

    def on_idle_start(self, picker_id, time):
        self._idle_start[picker_id] = time

    def on_idle_end(self, picker_id, time):
        start = self._idle_start.pop(picker_id, None)
        if start is not None:
            self.idle_time_by_picker[picker_id] += time - start
            self.idle_intervals.append((picker_id, start, time))

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
                    n_lines,
                    completed_order_ids=()):
        duration = end_time - start_time
        self.completed_tours.append((tour_id, start_time, end_time, list(order_ids), picker_id, on_time, delayed, n_lines))
        self._completed_tour_time += duration
        self._completed_order_count += len(order_ids)
        self._completed_line_count += n_lines
        self._completion_makespan = max(self._completion_makespan, end_time)
        for delayed_order_id in delayed:
            self.all_delayed.append(delayed_order_id)
        for on_time_order_id in on_time:
            self.all_on_time.append(on_time_order_id)
        self.dock_utilization.append((end_time, n_pallets_dock))
        self.avg_makespan.append((end_time, self.average_tour_makespan))
        util = self.current_utilization(end_time)
        self.picker_utilization.append((end_time, util))
        self.process_times.append(duration)
        self.on_orders_completed(completed_order_ids, end_time)
        # print(f"tour_id:  {tour_id} with order_ids: {order_ids} completed at time {end_time:.0f} with makespan {end_time - start_time:.1f} and picker utilization {util:.2%} and dock utilization {n_pallets_dock} pallets")

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

    def on_intervention(self, row: dict) -> None:
        self.interventions.append(row)

    def on_availability_change(
        self,
        picker_id: int,
        available: bool,
        time: float,
    ) -> None:
        if available:
            self._availability_start.setdefault(picker_id, float(time))
            return
        start = self._availability_start.pop(picker_id, None)
        if start is not None:
            self._available_time_closed_by_picker[picker_id] += (
                float(time) - start
            )
            self.availability_intervals.append(
                {
                    "picker_id": int(picker_id),
                    "start": float(start),
                    "end": float(time),
                }
            )

    def available_time_by_picker(self, end_time: float) -> dict[int, float]:
        totals = dict(self._available_time_closed_by_picker)
        for picker_id, start in self._availability_start.items():
            totals[picker_id] = totals.get(picker_id, 0.0) + max(
                0.0, float(end_time) - start
            )
        return totals

    def current_utilization(self, current_time: float) -> float:
        available_time = sum(
            self.available_time_by_picker(current_time).values()
        )
        return (
            self._completed_tour_time / available_time
            if available_time
            else 0.0
        )

    @property
    def total_processing_time(self):
        return self._completed_tour_time

    @property
    def completed_order_count(self) -> int:
        return self._completed_order_count

    @property
    def completed_line_count(self) -> int:
        return self._completed_line_count

    @property
    def completion_makespan(self) -> float:
        return self._completion_makespan

    @property
    def total_flow_time(self) -> float:
        return self.accrued_flow_time(self._flow_time_updated_at)

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
        return [end - start for _, start, end, _, _, _, _, _ in self.completed_tours]

    @property
    def average_tour_makespan(self) -> float:
        return (
            self._completed_tour_time / len(self.completed_tours)
            if self.completed_tours
            else 0.0
        )

    @property
    def average_batch_size(self) -> float:
        if not self.completed_tours:
            return 0.0
        return self._completed_order_count / len(self.completed_tours)


class DecisionTracker:
    def __init__(self):
        self.decisions: list[tuple] = []
        self.commitments: list[dict] = []
        self.pipeline_counts: dict[str, int] = defaultdict(int)

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

    def on_commitment(
        self,
        *,
        returned: int,
        committed: int,
        policy: str,
    ) -> None:
        self.commitments.append(
            {
                "returned": int(returned),
                "committed": int(committed),
                "deferred": int(returned - committed),
                "policy": policy,
            }
        )

    @property
    def num_decisions(self) -> int:
        return len(self.decisions)
