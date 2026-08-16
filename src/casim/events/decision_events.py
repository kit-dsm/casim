import logging

from ware_ops_algos.algorithms import (
    BatchingSolution,
    Route,
    SchedulingSolution,
)

from casim.events.operational_events import (
    Event,
    NodeArrival,
    PickerTourQuery,
    ProcessEvent,
    TravelEvent,
)
from casim.state import State

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PickListDone(ProcessEvent):
    def __init__(self, time: float, solution: BatchingSolution):
        super().__init__(time)
        self.solution = solution

    def handle(self, state: State) -> list[Event]:
        state.commit_batching_solution(self.solution)
        return []


class SequencingDone(ProcessEvent):
    priority_score = 0
    def __init__(
        self,
        time: float,
        solution: SchedulingSolution,
        replace_tour_ids: tuple[int, ...] = (),
    ):
        super().__init__(time)
        self.solution = solution
        self.replace_tour_ids = tuple(replace_tour_ids)

    def handle(self, state: State) -> list[Event]:
        picker_ids = state.commit_scheduling_decision(
            self.solution,
            replace_tour_ids=self.replace_tour_ids,
        )
        return [
            PickerTourQuery(self.time, picker_id)
            for picker_id in picker_ids
        ]


class ActiveRouteReplacement(ProcessEvent):
    """Atomically replace the unexecuted suffix of one active tour."""

    def __init__(
        self,
        time: float,
        route: Route,
        picker_id: int | None = None,
        tour_id: int | None = None,
        expected_route_version: int | None = None,
        resumes_execution: bool = False,
    ):
        super().__init__(time)
        self.route = route
        self.picker_id = picker_id
        self.tour_id = tour_id
        self.expected_route_version = expected_route_version
        self.resumes_execution = resumes_execution

    def handle(self, state: State) -> list[Event]:
        if (
            self.tour_id is None
            or self.expected_route_version is None
            or self.picker_id is None
        ):
            raise ValueError(
                "Active route replacement requires picker, tour, and version"
            )
        action, version = state.commit_active_plan(
            int(self.tour_id),
            self.route,
            picker_id=int(self.picker_id),
            expected_version=int(self.expected_route_version),
            time=self.time,
            resumes_execution=self.resumes_execution,
        )
        if action == "node":
            return [NodeArrival(self.time, self.tour_id, version)]
        if action == "travel":
            return [TravelEvent(self.time, self.tour_id, version)]
        return []
