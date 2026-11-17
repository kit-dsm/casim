"""Dispatch, presence and planned times with two pickers (Task 4.14).

Each test runs a two-picker state through its events without decisions: the
commands of a decision are applied with ``apply`` at a chosen time. Pickers
have speed 1 and no setup, and every tour is the all-aisles patrol without
picks, so a tour's actual duration is the route distance ``d``. A tour's
planned end can be set apart from ``d`` to model a tour that ends earlier or
later than planned.
"""

import copy
import heapq

import pytest
from ware_ops_algos.algorithms import Job, ScheduledJob
from ware_ops_algos.algorithms.routing.patrol import all_aisles_patrol_route
from ware_ops_algos.domain_models import Resources
from ware_ops_algos.planning.commands import CommitTour, CreateTour, Reschedule
from ware_ops_algos.planning.lifecycle import ResourceStates, TourStates

from casim.events.operational_events import PickerArrival, PickerDeparture
from casim.run import prepare
from casim.state import State
from casim.state.apply import apply
from casim.state.resource_checks import check_resources
from tests.scenario_configs import walk_config


class Sim:
    """A two-picker state and its event heap."""

    def __init__(self, tmp_path):
        domain = prepare(walk_config(tmp_path, "cosy_repo=paper_patrol",
                                     "variants=explicit_wait0")).sim.load_data()
        first = domain.resources.resources[0]
        first.speed, first.tour_setup_time = 1.0, 0
        second = copy.deepcopy(first)
        second.id = 1
        self.state = State(
            layout=domain.layout, articles=domain.articles, storage=domain.storage,
            resources=Resources(domain.resources.tpe, [first, second]),
            active_objective=domain.objective)
        self.route = all_aisles_patrol_route(self.state.layout_manager.get_layout())
        self.d = self.route.distance
        self.heap = []
        self.findings = []

    def push(self, *events):
        for event in events:
            heapq.heappush(self.heap, event)

    def run_until(self, time):
        """Process every event up to ``time``; check the resources after each timestamp."""
        while self.heap and self.heap[0].time <= time:
            event = heapq.heappop(self.heap)
            if event.is_stale(self.state):
                continue
            if event.time > self.state.current_time:
                self.findings += check_resources(self.state)
            self.state.current_time = event.time
            self.push(*event.handle(self.state))
        self.findings += check_resources(self.state)
        self.state.current_time = max(self.state.current_time, time)

    def decide(self, time, *commands):
        """Apply a decision's commands at ``time``, as SimulationEngine.step does."""
        self.run_until(time)
        self.push(*apply(self.state, commands))

    def commit(self, time, picker_id, start, end=None, earliest="start", processing=None):
        """Create and commit one tour; its earliest start defaults to its start.

        ``processing`` is the tour's estimated processing time (default ``d``);
        the tour still takes ``d`` to run.
        """
        processing = self.d if processing is None else processing
        job = Job(-1, processing, 0, float("inf"), 0, route=self.route, batch=self.route.batch)
        end = start + self.d if end is None else end
        earliest = start if earliest == "start" else earliest
        before = set(self.state.tour_manager.all_tours)
        self.decide(time, CreateTour(ScheduledJob(job=job, picker_id=picker_id,
                                                  start_time=start, end_time=end),
                                     earliest_start=earliest),
                    CommitTour(creation_index=0))
        (tour_id,) = set(self.state.tour_manager.all_tours) - before
        return self.state.tour_manager.get_tour(tour_id)

    def status(self, picker_id):
        return self.state.resource_manager.resource_status(picker_id, self.state.tour_manager)


@pytest.fixture
def sim(tmp_path):
    sim = Sim(tmp_path)
    sim.push(PickerArrival(0, 0), PickerArrival(0, 1))
    sim.run_until(0)
    return sim


def test_both_pickers_arrive_idle(sim):
    assert sim.status(0) is ResourceStates.IDLE and sim.status(1) is ResourceStates.IDLE


def test_tour_rescheduled_earlier_starts_at_the_new_time(sim):
    tour = sim.commit(0, 0, start=100)
    sim.run_until(10)
    assert tour.status is TourStates.SCHEDULED and sim.status(0) is ResourceStates.RESERVED
    sim.decide(10, Reschedule(tour.tour_id, 0, 50, 50 + sim.d, earliest_start=50))
    sim.run_until(50)
    assert tour.status is TourStates.STARTED and tour.actual_start == 50
    sim.run_until(200 + sim.d)
    assert tour.status is TourStates.DONE and tour.actual_end == 50 + sim.d
    assert sim.findings == []


def test_tour_moved_to_another_idle_picker_runs_there(sim):
    tour = sim.commit(0, 0, start=100)
    sim.decide(10, Reschedule(tour.tour_id, 1, 20, 20 + sim.d, earliest_start=20))
    sim.run_until(20)
    assert tour.assigned_resource == 1 and tour.status is TourStates.STARTED
    assert tour.actual_start == 20
    assert sim.status(0) is ResourceStates.IDLE and sim.status(1) is ResourceStates.BUSY
    sim.run_until(200 + sim.d)
    assert sim.findings == []


def test_earlier_commit_while_the_picker_waits_for_a_later_tour(sim):
    later = sim.commit(0, 0, start=100 + sim.d)
    earlier = sim.commit(10, 0, start=20)
    sim.run_until(20)
    assert earlier.status is TourStates.STARTED and earlier.actual_start == 20
    assert later.status is TourStates.SCHEDULED
    sim.run_until(200 + 2 * sim.d)
    assert later.actual_start == 100 + sim.d
    assert sim.findings == []


def test_break_with_one_picker_busy_and_one_idle(sim):
    busy = sim.commit(0, 0, start=0)
    after = sim.commit(0, 0, start=sim.d)
    sim.push(PickerDeparture(1, 0, duration=30), PickerDeparture(1, 1, duration=30))
    sim.run_until(1)
    # The idle picker leaves at once, the busy one finishes its tour first.
    assert sim.status(1) is ResourceStates.OFF_SHIFT
    assert sim.status(0) is ResourceStates.BUSY
    sim.run_until(31)
    assert sim.status(1) is ResourceStates.IDLE
    sim.run_until(sim.d)
    assert busy.status is TourStates.DONE and sim.status(0) is ResourceStates.OFF_SHIFT
    # The next tour waits for the return; its planned start is projected there.
    assert after.status is TourStates.SCHEDULED and after.planned_start == sim.d + 30
    sim.run_until(sim.d + 30)
    assert after.status is TourStates.STARTED and after.actual_start == sim.d + 30
    sim.run_until(3 * sim.d + 100)
    assert sim.findings == []


def test_shift_end_keeps_queued_tours_and_their_planned_times(sim):
    tour = sim.commit(0, 0, start=100)
    sim.push(PickerDeparture(10, 0))
    sim.run_until(500)
    assert sim.status(0) is ResourceStates.OFF_SHIFT
    assert tour.status is TourStates.SCHEDULED
    assert (tour.planned_start, tour.planned_end) == (100, 100 + sim.d)
    sim.push(PickerArrival(500, 0))
    sim.run_until(500)
    assert tour.status is TourStates.STARTED and tour.actual_start == 500
    assert sim.findings == []


def test_departure_and_arrival_at_one_timestamp(sim):
    # The departure comes first, the arrival cancels it: the picker stays present.
    sim.push(PickerDeparture(10, 0), PickerArrival(10, 0))
    sim.run_until(10)
    assert sim.status(0) is ResourceStates.IDLE
    # The other way round, the picker is gone.
    sim.push(PickerArrival(20, 1), PickerDeparture(20, 1))
    sim.run_until(20)
    assert sim.status(1) is ResourceStates.OFF_SHIFT
    assert sim.findings == []


@pytest.mark.parametrize("binding", [True, False])
def test_tour_that_ends_early_under_binding_and_forecast(sim, binding):
    planned = 100 + sim.d
    first = sim.commit(0, 0, start=0, end=planned)
    second = sim.commit(0, 0, start=planned, earliest="start" if binding else None)
    sim.run_until(sim.d)
    assert first.status is TourStates.DONE
    if binding:
        # A timetable: the next tour keeps its start.
        assert second.status is TourStates.SCHEDULED and second.planned_start == planned
        sim.run_until(planned)
        assert second.actual_start == planned
    else:
        # A sequence: the next tour starts as soon as the picker is free.
        assert second.status is TourStates.STARTED and second.actual_start == sim.d
    sim.run_until(3 * planned)
    assert sim.findings == []


def test_tour_that_ends_late_delays_the_next_and_its_projection(sim):
    planned = sim.d - 10
    first = sim.commit(0, 0, start=0, end=planned)
    second = sim.commit(0, 0, start=planned)
    sim.run_until(planned)
    # The running tour's forecast end is its actual end; the next tour is projected after it.
    assert first.status is TourStates.STARTED and first.planned_end == sim.d
    assert second.status is TourStates.SCHEDULED and second.planned_start == sim.d
    sim.run_until(sim.d)
    assert second.status is TourStates.STARTED and second.actual_start == sim.d
    sim.run_until(3 * sim.d)
    assert sim.findings == []


def test_tour_that_overruns_its_forecast_is_projected_when_it_ends(sim):
    # The estimate is 10 short of the actual duration d: the running tour passes
    # its forecast end, and the next tour's projection is renewed at TourEnd.
    first = sim.commit(0, 0, start=0, end=sim.d - 10, processing=sim.d - 10)
    second = sim.commit(0, 0, start=sim.d - 10, earliest=None)
    sim.run_until(sim.d - 5)
    assert first.status is TourStates.STARTED and first.planned_end == sim.d - 10
    sim.run_until(sim.d)
    assert second.status is TourStates.STARTED and second.actual_start == sim.d
    sim.run_until(3 * sim.d)
    assert sim.findings == []
