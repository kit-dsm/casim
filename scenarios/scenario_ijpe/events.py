from ware_ops_algos.algorithms import NodeType, Job

from casim.domain_objects.tour_model import TourStates
from casim.events.base_events import BaseTourEvent, Event, ProcessEvent
from casim.state import State


class SequencingDone(ProcessEvent):
    priority_score = 0
    def __init__(self, time: float, sequencing: Job):
        super().__init__(time)
        self.sequencing = sequencing

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        # state.order_manager.clear_order_buffer(self.sequencing.route.pick_list.orders)
        # state.add_sequencing_to_planning_state(self.sequencing)
        events_to_return = []
        picker_id = self.sequencing.picker_id
        picker = state.resource_manager.get_resource(picker_id)
        if not picker.occupied:
            events_to_return.append(PickerTourQuery(self.time,
                                                    picker_id))
            state.resource_manager.mark_picker_occupied(picker_id)
        return events_to_return


class BreakStart(Event):
    priority_score = 0
    def __init__(self, time: float, break_duration: float):
        super().__init__(time)
        self.break_duration = break_duration

    def handle(self, state: State) -> list[Event]:
        state.is_break = True
        print(f"break start at {self.time}")
        print(f"break duration {self.break_duration}")
        print(f"break end at {self.time}")
        return [BreakEnd(self.time + self.break_duration)]


class BreakEnd(Event):
    priority_score = 0
    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: State) -> list[Event]:
        state.is_break = False
        print(f"break end at {self.time}")
        return []



class PickerTourQuery(Event):
    def __init__(self, time: float, picker_id: int):
        super().__init__(time)
        self.picker_id = picker_id

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        picker = state.resource_manager.get_resource(
            self.picker_id)
        next_tour_id = state.tour_manager.get_next_tour_for_picker(
            self.picker_id)
        if next_tour_id is not None:
            next_tour = state.tour_manager.get_tour(next_tour_id)
            start_time = state.current_time
            if next_tour.start_time is not None:
                start_time = max(next_tour.start_time, self.time)
            next_tour.status = TourStates.PENDING
            if picker.tour_setup_time:
                start_time += picker.tour_setup_time
            return [TourStart(start_time, next_tour_id)]
        else:
            picker.occupied = False
            return [PickerIdle(state.current_time, picker.id)]

class PickerIdle(Event):
    priority_score = 1

    def __init__(self, time: float, picker_id: int):
        super().__init__(time)
        self.picker_id = picker_id

    def handle(self, state: 'State') -> list['Event']:
        super().handle(state)
        state.tracker.on_idle_start(self.picker_id, self.time)
        return []


class TourStart(BaseTourEvent):
    def handle(self, state):
        tour = self.get_tour(state)
        res = state.resource_manager.get_resource(tour.assigned_resource)
        state.tracker.on_idle_end(tour.assigned_resource, self.time)
        state.tour_manager.start_tour(tour.tour_id, self.time)
        state.resource_manager.mark_picker_occupied(tour.assigned_resource)
        state.resource_manager.update_resource_location(tour.assigned_resource,
                                                        tour.annotated_route[0])
        state.tracker.on_tour_start(self.time)
        duration = self._compute_duration(tour, res, state)
        return [TourEnd(self.time + duration, tour.tour_id)]

    def _compute_duration(self, tour, picker, state):
        total = 0.0
        nodes = tour.annotated_route
        for a, b in zip(nodes, nodes[1:]):
            total += state.layout_manager.get_distance(a, b) / picker.speed
        for n in nodes:
            if n.node_type == NodeType.PICK:
                total += picker.time_per_pick
        return total

class TourEnd(BaseTourEvent):
    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        tour = self.get_tour(state)
        res = state.resource_manager.get_resource(tour.assigned_resource)
        state.resource_manager.update_resource_location(
            tour.assigned_resource, tour.annotated_route[-1]  # depot
        )
        state.tour_manager.finish_tour(tour.tour_id, self.time)
        om = state.order_manager
        on_time = []
        delayed = []
        for o_id in tour.order_numbers:
            o = om.get_order_from_history(o_id)
            if o.due_date:
                if o.due_date < self.time:
                    delayed.append(o_id)
            else:
                on_time.append(o_id)

        # state.tracker.update_on_tour_end(tour_start=tour.start_time, tour_finish=self.time, order_manager=om)
        assert self.tour_id != state.tour_manager.get_next_tour_for_picker(res.id), (f"{self.tour_id}, " 
                                                                                     f"{state.tour_manager._picker_tour_queues}")
        n_pallets_dock = 0
        if hasattr(state, "dock_manager"):
            state.dock_manager.stage_pallets(1)
            n_pallets_dock = state.dock_manager.n_staged_pallets
        state.tracker.on_tour_end(tour.tour_id,
                                  tour.start_time,
                                  self.time,
                                  tour.order_numbers,
                                  tour.assigned_resource,
                                  on_time,
                                  delayed,
                                  n_pallets_dock)
        return [PickerTourQuery(self.time, res.id)]