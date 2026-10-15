"""A scheduled tour enters setup at its start and travels after one setup."""

from ware_ops_algos.algorithms import FIFOScheduling, Job
from ware_ops_algos.algorithms.routing.patrol import all_aisles_patrol_route
from ware_ops_algos.planning.lifecycle import ResourceStates, TourStates

from casim.events.operational_events import PickerTourQuery, TourEnd, TourStart
from casim.run import prepare
from tests.scenario_configs import walk_config


def test_setup_begins_at_scheduled_start_and_occurs_once(tmp_path):
    cfg = walk_config(tmp_path, "cosy_repo=paper_patrol", "variants=explicit_wait0")
    sim = prepare(cfg).sim
    sim.reset()
    state = sim.state
    state.resource_manager.set_picker_available(0)
    picker = state.resource_manager.get_resource(0)
    picker.tour_setup_time = 3
    route = all_aisles_patrol_route(state.layout_manager.get_layout())
    job = Job(-1, route.distance, 0, float("inf"), 0,
              route=route, batch=route.batch)
    scheduled = FIFOScheduling(state.resource_manager.get_resources(),
                               available_from={0: 10}).solve([job]).jobs[0]
    assert scheduled.start_time == 10
    assert scheduled.end_time == 10 + 3 + route.distance / picker.speed

    tour_id = state.add_sequencing_to_planning_state(scheduled)
    state.tour_manager.advance(tour_id, TourStates.SCHEDULED)
    state.current_time = 0
    follow_up = PickerTourQuery(0, 0).handle(state)
    assert len(follow_up) == 1
    assert isinstance(follow_up[0], PickerTourQuery)
    assert follow_up[0].time == 10
    assert state.tour_manager.get_tour(tour_id).status is TourStates.SCHEDULED
    assert state.resource_manager.resource_status(0, state.tour_manager) is ResourceStates.RESERVED

    state.current_time = 10
    follow_up = follow_up[0].handle(state)
    assert len(follow_up) == 1
    assert isinstance(follow_up[0], TourStart)
    assert follow_up[0].time == 13
    assert state.tour_manager.get_tour(tour_id).status is TourStates.PENDING

    events = follow_up
    while events:
        event = events.pop(0)
        state.current_time = event.time
        events.extend(event.handle(state))
        if isinstance(event, TourEnd):
            assert event.time == scheduled.end_time
            break
    else:
        raise AssertionError("Tour never ended")
