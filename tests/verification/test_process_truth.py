"""Realized process times (casim/domain_objects/process_truth.py).

The truth fixes how long each line and setup take; execution uses it,
planning keeps the resource's time model, and no decision snapshot carries it.
"""

from types import SimpleNamespace

from ware_ops_algos.algorithms import FIFOScheduling, Job
from ware_ops_algos.algorithms.routing.patrol import all_aisles_patrol_route
from ware_ops_algos.domain_models import pick_time
from ware_ops_algos.planning.lifecycle import TourStates

from casim.domain_objects.process_truth import ProcessTruth
from casim.events.operational_events import Dispatch, PickComplete, TourEnd, TourStart
from casim.outputs.facts import FactsRecorder
from casim.run import prepare
from tests.scenario_configs import walk_config


def _line(order, article, amount):
    return SimpleNamespace(order_number=order, article_id=article, amount=amount)


def _run_one_tour(tmp_path, truth):
    sim = prepare(walk_config(tmp_path, "cosy_repo=paper_patrol", "variants=explicit_wait0")).sim
    sim.reset()
    state = sim.state
    state.truth = truth
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
    events, seen, facts = [Dispatch(0, 0, state.resource_manager.bump_version(0))], [], FactsRecorder()
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


def test_without_truth_execution_uses_the_resource_time_model(tmp_path):
    seen, planned_end, _, _, facts = _run_one_tour(tmp_path, None)
    start = next(e for e in seen if isinstance(e, TourStart))
    assert start.time == 3
    assert seen[-1].time == planned_end
    assert (facts.tours[start.tour_id]["start"], facts.tours[start.tour_id]["setup_end"]) == (0, 3)


def test_truth_drives_execution_and_planning_keeps_the_resource_times(tmp_path):
    probe = prepare(walk_config(tmp_path / "probe", "cosy_repo=paper_patrol",
                                "variants=explicit_wait0")).sim
    probe.reset()
    picks = all_aisles_patrol_route(probe.state.layout_manager.get_layout()).batch.pick_positions
    line_s = {}
    for pick in picks:
        key = (pick.order_number, pick.article_id, pick.amount)
        line_s[key] = (*line_s.get(key, ()), 4.0 + len(line_s))
    truth = ProcessTruth(line_s=line_s, setup_s={0: (7.0,)})
    seen, planned_end, route, picker, facts = _run_one_tour(tmp_path, truth)
    start = next(e for e in seen if isinstance(e, TourStart))
    assert start.time == 7
    assert (facts.tours[start.tour_id]["start"], facts.tours[start.tour_id]["setup_end"]) == (0, 7)
    durations = sorted(e.time - e.pick_start for e in seen if isinstance(e, PickComplete))
    assert durations == sorted(t for times in line_s.values() for t in times)
    # The plan was made with the resource's times.
    assert planned_end == 3 + route.distance / picker.speed + sum(
        pick_time(picker, p.amount) for p in route.batch.pick_positions)
    assert abs(seen[-1].time - (7 + route.distance / picker.speed + sum(durations))) < 1e-6


def test_lines_with_one_key_take_their_times_in_turn_and_setups_run_out():
    truth = ProcessTruth(line_s={(1, 2, 3): (10.0, 20.0)}, setup_s={0: (5.0, 6.0)})
    assert [truth.pick(_line(1, 2, 3)) for _ in range(3)] == [10.0, 20.0, 10.0]
    assert truth.pick(_line(1, 9, 3)) is None
    assert [truth.setup(0), truth.setup(0), truth.setup(0), truth.setup(1)] == [5.0, 6.0, None, None]


def test_no_decision_snapshot_carries_the_truth(tmp_path):
    sim = prepare(walk_config(tmp_path, "cosy_repo=paper_patrol", "variants=explicit_wait0")).sim
    sim.reset()
    sim.state.truth = ProcessTruth(setup_s={0: (1.0,)})
    for name, adapter in sim.state_adapters.items():
        snapshot = adapter.transform_state(sim.state, name, "OBRSP")
        assert snapshot.truth is None
