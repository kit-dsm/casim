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
        # (tour_id, start_time, end_time, list[order_ids], picker_id)

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
                    n_lines):
        self.completed_tours.append((tour_id, start_time, end_time, list(order_ids), picker_id, on_time, delayed, n_lines))
        for delayed_order_id in delayed:
            self.all_delayed.append(delayed_order_id)
        for on_time_order_id in on_time:
            self.all_on_time.append(on_time_order_id)
        self.dock_utilization.append((end_time, n_pallets_dock))
        self.avg_makespan.append((end_time, self.average_tour_makespan))
        util = self.current_utilization(end_time)
        self.picker_utilization.append((end_time, util))
        self.process_times.append(end_time - start_time)
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

    def current_utilization(self, current_time: float) -> float:
        elapsed = current_time - self.first_tour_start
        if elapsed <= 0:
            return 0.0

        total_tour_time = sum(
            end - start
            for _, start, end, _, _, _, _, _ in self.completed_tours
        )
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
        return [end - start for _, start, end, _, _, _, _, _ in self.completed_tours]

    @property
    def average_tour_makespan(self) -> float:
        d = self.tour_durations
        return sum(d) / len(d) if d else 0.0

    @property
    def average_batch_size(self) -> float:
        if not self.completed_tours:
            return 0.0
        total_orders = sum(len(order_ids) for _, _, _, order_ids, _, _, _, _ in self.completed_tours)
        return total_orders / len(self.completed_tours)


class DecisionTracker:
    def __init__(self):
        self.decisions: list[tuple] = []
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

    @property
    def num_decisions(self) -> int:
        return len(self.decisions)
