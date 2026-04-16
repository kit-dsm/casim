from ware_ops_algos.algorithms import TourPlanningState

from casim.state import State


class Event:
    event_counter = 0
    priority_score = 1  # Default priority for operational events

    def __init__(self, time: float):
        self.time = time
        self.id = Event.event_counter
        Event.event_counter += 1

    def __lt__(self, other: 'Event'):
        """Sorts by time first, then by priority_score (lower is higher priority),
           then by ID."""
        if self.time != other.time:
            return self.time < other.time

        # Tie-breaker logic
        if self.priority_score != other.priority_score:
            return self.priority_score < other.priority_score

        return self.id < other.id

    def __le__(self, other: 'Event'):
        return self.__lt__(other) or self.__eq__(other)

    def __eq__(self, other: 'Event'):
        return (self.time == other.time and
                self.priority_score == other.priority_score and
                self.id == other.id)

    def handle(self, state: 'State') -> list['Event']:
        # if self.time > state.current_time:
        # assert self.time >= state.current_time, f"{self.time}, {state.current_time}, {self}"
        #     state.current_time = self.time
        return []


class ProcessEvent(Event):
    """
    A ProcessEvent represents events created as a result of a decision.
    It contains the respective solution for the decision problem which
    is added to the state.
    May return status update events to initiate operational EventChain.
    """
    priority_score = 0
    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, state: State) -> list[Event]:
        super().handle(state)
        return []


class BaseTourEvent(Event):
    priority_score = 1
    def __init__(self, time: float, tour_id: int):
        super().__init__(time)
        self.tour_id = tour_id

    def get_tour(self, state: State) -> TourPlanningState:
        return state.tour_manager.get_tour(self.tour_id)
