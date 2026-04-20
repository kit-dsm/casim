from collections import defaultdict


class ExperimentTracker:
    def __init__(self):
        self.distance_by_picker: dict[int, float] = defaultdict(float)
        self.idle_time_by_picker: dict[int, float] = defaultdict(float)
        self._idle_start: dict[int, float] = {}
        self.idle_intervals: list[tuple[int, float, float]] = []
        self.pick_durations_by_picker: dict[int, list[float]] = defaultdict(list)
        self.completed_tours: list[tuple[int, float, float, list[int], int]] = []
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

    def on_tour_end(self, tour_id, start_time, end_time, order_ids, picker_id):
        self.completed_tours.append((tour_id, start_time, end_time, list(order_ids), picker_id))

    @property
    def tour_durations(self) -> list[float]:
        return [end - start for _, start, end, _, _ in self.completed_tours]

    @property
    def average_tour_makespan(self) -> float:
        d = self.tour_durations
        return sum(d) / len(d) if d else 0.0

    @property
    def average_batch_size(self) -> float:
        if not self.completed_tours:
            return 0.0
        total_orders = sum(len(order_ids) for _, _, _, order_ids, _ in self.completed_tours)
        return total_orders / len(self.completed_tours)


class DecisionTracker:
    def __init__(self):
        self.decisions: list[tuple] = []
        # (problem_class [input_ids, selected_pipeline, kpi_value, alternatives])
        self.pipeline_counts: dict[str, int] = defaultdict(int)

    def on_decision(self, problem_class, input_ids, selected_pipeline,
                    kpi_value, kpi, runtime):
        self.decisions.append((
            problem_class,
            tuple(input_ids),
            selected_pipeline,
            kpi_value,
            kpi,
            runtime
        ))
        self.pipeline_counts[selected_pipeline] += 1

    @property
    def num_decisions(self) -> int:
        return len(self.decisions)
