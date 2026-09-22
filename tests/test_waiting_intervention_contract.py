"""Operational boundary checks; the stochastic numerical model is tested elsewhere."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from ware_ops_algos.algorithms import CombinedRoutingSolution

from casim.events.operational_events import NodeArrival, TravelEvent
from casim.domain_objects.tour_model import TourStates
from casim.simulation_engine.state_adapter import ReORSPAdapter
from scenarios.experiment_commons import (
    load_and_flatten_data_card, setup_decision_engine, setup_scenario,
)
from scenarios.scenario_stochastic_waiting.scenario_specific_hooks import (
    add_orders_hook, picker_arrival_hook,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "scenario_stochastic_waiting" / "config"


@pytest.mark.parametrize(
    ("arrivals", "pick_time"),
    [([0.0, 1.0, 2.0, 12.0], 0.0), ([0.0, 2.5, 5.5, 12.0], 2.0)],
)
def test_configured_active_tour_boundary(tmp_path, arrivals, pick_time):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="stochastic_waiting_config",
            overrides=[
                "cosy_repo=no_waiting",
                f"simulation.arrival_times_s={arrivals}",
                f"simulation.pick_time_s={pick_time}",
                f"instances_base={tmp_path.as_posix()}",
                f"cache_base={(tmp_path / 'cache').as_posix()}",
                f"experiment.output_dir={tmp_path.as_posix()}",
            ],
        )
    card = load_and_flatten_data_card(cfg.data_card)
    sim = setup_scenario(cfg)
    decision_engine = setup_decision_engine(cfg, card)
    sim.reset(hooks=[add_orders_hook, picker_arrival_hook])

    insertion_origins = []
    insertion_times = []
    checked_unstarted_projection = False
    while True:
        done, snapshot = sim.run()
        if done:
            break
        now = sim.state.current_time
        assert all(order.order_date <= now for order in snapshot.orders.orders)
        assert snapshot.warehouse_info is not sim.state.warehouse_info
        for order in snapshot.orders.orders:
            live = sim.state.order_manager._order_buffer.get(order.order_id)
            if live is not None:
                assert order is not live

        events, solution = decision_engine.on_trigger(snapshot)
        dynamic = snapshot.dynamic_warehouse_info
        if dynamic.active_tour_id is not None:
            assert isinstance(solution, CombinedRoutingSolution)
            insertion_origins.append(dynamic.routing_origin)
            insertion_times.append(now)
            tour = sim.state.tour_manager.get_tour(dynamic.active_tour_id)
            before = (tour.cursor, list(tour.remaining_picks), tour.route_version)
            sim.step(events, snapshot.problem_class, solution, snapshot)
            assert tour.route_version == before[2] + 1
            assert NodeArrival(now, tour.tour_id, before[2]).handle(sim.state) == []
            assert (tour.cursor, tour.remaining_picks, tour.route_version) == (
                0, list(solution.routes[0].batch.pick_positions), before[2] + 1,
            )
            assert any(isinstance(event, TravelEvent) and event.time == now
                       and event.route_version == tour.route_version for event in sim.events)
        else:
            sim.step(events, snapshot.problem_class, solution, snapshot)
            if not checked_unstarted_projection and sim.state.tour_manager.all_tours:
                tour = next(iter(sim.state.tour_manager.all_tours.values()))
                assert tour.status == TourStates.SCHEDULED
                before_buffer = sim.state.order_manager.get_pick_list_buffer()
                projected = ReORSPAdapter().transform_state(sim.state, "RORSP")
                assert tour.status == TourStates.SCHEDULED
                assert sim.state.order_manager.get_pick_list_buffer() == before_buffer
                assert tour.batch in projected.dynamic_warehouse_info.buffered_batches
                checked_unstarted_projection = True

    assert insertion_origins
    assert checked_unstarted_projection
    tours = list(sim.state.tour_manager.all_tours.values())
    picked = [pick.order_number for tour in tours for pick in tour.completed_picks]
    assert sorted(picked) == [0, 1, 2, 3]
    assert all(not tour.remaining_picks for tour in tours)
    if pick_time:
        assert 2.5 not in insertion_times
        assert any(2.5 < time < 5.5 for time in insertion_times)
    else:
        assert {origin.edge_destination is None for origin in insertion_origins} == {True, False}
