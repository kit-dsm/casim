"""Realized execution times (C8): execution asks the source, planning keeps its times."""

from types import SimpleNamespace

from ware_ops_algos.algorithms import FIFOScheduling, Job
from ware_ops_algos.algorithms.routing.patrol import all_aisles_patrol_route
from ware_ops_algos.planning.lifecycle import TourStates

from casim.events.operational_events import Dispatch, PickComplete, TourEnd, TourStart
from casim.outputs.facts import FactsRecorder
from casim.run import prepare
from tests.scenario_configs import walk_config


class Realized:
    """Setup 7 s per tour, 2 s plus the amount per line; records what it was asked."""

    def __init__(self):
        self.setups, self.picks = [], []

    def setup_s(self, picker_id, order_numbers):
        self.setups.append((picker_id, tuple(order_numbers)))
        return 7.0

    def pick_s(self, picker_id, pick):
        self.picks.append((picker_id, pick.order_number, pick.article_id))
        return 2.0 + pick.amount


def _run_one_tour(tmp_path, realized):
    sim = prepare(walk_config(tmp_path, "cosy_repo=paper_patrol", "variants=explicit_wait0")).sim
    sim.reset()
    state = sim.state
    state.execution_times = realized
    state.resource_manager.arrive(0)
    picker = state.resource_manager.get_resource(0)
    picker.tour_setup_time = 3
    route = all_aisles_patrol_route(state.layout_manager.get_layout())
    job = Job(-1, route.distance, 0, float("inf"), 0, route=route, batch=route.batch)
    scheduled = FIFOScheduling(state.resource_manager.get_resources(),
                               available_from={0: 0}).solve([job]).jobs[0]
    tour_id = state.add_sequencing_to_planning_state(scheduled, scheduled.start_time)
    state.tour_manager.advance(tour_id, TourStates.SCHEDULED)
    planned_end = state.tour_manager.get_tour(tour_id).planned_end
    events = [Dispatch(0, 0, state.resource_manager.bump_version(0))]
    seen = []
    facts = FactsRecorder()
    while events:
        event = events.pop(0)
        state.current_time = event.time
        seen.append(event)
        events.extend(event.handle(state))
        facts.on_event(event, SimpleNamespace(state=state))
        events.sort(key=lambda e: e.time)
        if isinstance(event, TourEnd):
            break
    return seen, planned_end, route, picker, facts


def test_without_realized_times_execution_uses_the_resource_times(tmp_path):
    seen, planned_end, route, picker, facts = _run_one_tour(tmp_path, None)
    start = next(e for e in seen if isinstance(e, TourStart))
    assert start.time == 3
    assert seen[-1].time == planned_end


def test_realized_setup_and_pick_times_drive_execution_only(tmp_path):
    realized = Realized()
    seen, planned_end, route, picker, facts = _run_one_tour(tmp_path, realized)
    start = next(e for e in seen if isinstance(e, TourStart))
    assert start.time == 7
    # The facts report the executed setup: from dispatch (0) to the route start.
    executed = facts.tours[start.tour_id]
    assert (executed["start"], executed["setup_end"]) == (0, 7)
    assert [picker_id for picker_id, _ in realized.setups] == [0]
    picks = list(route.batch.pick_positions)
    assert len(realized.picks) == len(picks)
    assert all(picker_id == 0 for picker_id, _, _ in realized.picks)
    durations = [e.time - e.pick_start for e in seen if isinstance(e, PickComplete)]
    assert sorted(durations) == sorted(2.0 + p.amount for p in picks)
    # The plan was made with the resource's times, before any realization.
    assert planned_end == 3 + route.distance / picker.speed + sum(
        picker.time_per_pick if picker.pick_time_model is None else picker.pick_time_model(p.amount)
        for p in picks)
    expected_end = 7 + route.distance / picker.speed + sum(2.0 + p.amount for p in picks)
    assert abs(seen[-1].time - expected_end) < 1e-6
