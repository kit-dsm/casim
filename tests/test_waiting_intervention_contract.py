"""Operational boundary checks; the stochastic numerical model is tested elsewhere."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from ware_ops_algos.algorithms import (
    AdmissionInput, Batching, CombinedRoutingSolution, PickPosition,
    RemainingRouteAdmission, WarehouseOrder,
)
from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards
from ware_ops_algos.domain_algo_mapper.domain_algo_mapper import DomainAlgorithmMapper
from ware_ops_algos.domain_models import DimensionType, PickCart

from casim.events.operational_events import ActiveTourOpportunity, NodeArrival, TravelEvent
from casim.domain_objects.tour_model import TourStates
from casim.pipelines.problem_based_template import (
    AbstractAdmission, AbstractBatching, AbstractBatchProvider,
    AdmittedTourBatch, traverse_pipeline,
)
from casim.pipelines.subproblems.admission import RemainingRouteAdmissionNode
from casim.pipelines.taxonomy import TAXONOMY
from casim.simulation_engine.state_adapter import ActiveTourAdapter, HennWaitingAdapter, ReORSPAdapter
from scenarios.experiment_commons import (
    load_and_flatten_data_card, setup_decision_engine, setup_scenario,
)
from scenarios.scenario_stochastic_waiting.scenario_specific_hooks import (
    add_orders_hook, picker_arrival_hook,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "scenario_stochastic_waiting" / "config"


def test_admission_card_matches_the_active_tour_problem():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="stochastic_waiting_config")
    card = load_and_flatten_data_card(cfg.data_card)
    card.problem_class = "OBRP"
    admission = next(card for card in load_packaged_algo_cards()
                     if card.algo_name == "RemainingRouteAdmission")
    assert admission.problem_type == "admission"
    assert DomainAlgorithmMapper(TAXONOMY).filter([admission], card) == [admission]


@pytest.mark.parametrize(
    ("arrivals", "pick_time", "waiting_repo", "active_expected", "admission_expected"),
    [
        ([0.0, 1.0, 2.0, 12.0], 0.0, "no_waiting", False, False),
        ([0.0, 2.5, 5.5, 12.0], 2.0, "no_waiting", True, False),
        ([0.0, 1.0, 2.0, 12.0], 0.0, "analytic_waiting", True, True),
        ([0.0, 1.0, 2.0, 5.0], 4.0, "no_waiting", True, False),
    ],
)
def test_configured_active_tour_boundary(tmp_path, arrivals, pick_time,
                                         waiting_repo, active_expected, admission_expected):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="stochastic_waiting_config",
            overrides=[
                f"cosy_repo={waiting_repo}",
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
    active_tasks = traverse_pipeline(decision_engine.get_solver("OBRP").pipelines)
    assert any(isinstance(task, RemainingRouteAdmissionNode) for task in active_tasks)
    assert any(isinstance(task, AdmittedTourBatch) for task in active_tasks)
    assert issubclass(RemainingRouteAdmissionNode, AbstractAdmission)
    assert not issubclass(RemainingRouteAdmissionNode, AbstractBatching)
    assert issubclass(AdmittedTourBatch, AbstractBatchProvider)
    assert not issubclass(AdmittedTourBatch, AbstractBatching)
    sim.reset(hooks=[add_orders_hook, picker_arrival_hook])

    insertion_origins = []
    insertion_times = []
    admitted = 0
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
            live_before_projection = (tour.cursor, tuple(tour.remaining_picks), tour.route_version)
            projected = ActiveTourAdapter().transform_state(sim.state, "OBRP")
            assert projected.dynamic_warehouse_info.active_tours[0] is not tour
            assert projected.dynamic_warehouse_info.active_candidate_ids == dynamic.active_candidate_ids
            assert (tour.cursor, tuple(tour.remaining_picks), tour.route_version) == live_before_projection
            before = (tour.cursor, list(tour.remaining_picks), tour.route_version)
            sim.step(events, snapshot.problem_class, solution, snapshot)
            if solution.routes:
                admitted += 1
                assert tour.route_version == before[2] + 1
                assert NodeArrival(now, tour.tour_id, before[2]).handle(sim.state) == []
                assert ActiveTourOpportunity(now, tour.tour_id, before[2]).is_stale(sim.state)
                assert (tour.cursor, tour.remaining_picks, tour.route_version) == (
                    0, list(solution.routes[0].batch.pick_positions), before[2] + 1,
                )
                assert any(isinstance(event, TravelEvent) and event.time == now
                           and event.route_version == tour.route_version for event in sim.events)
            else:
                assert (tour.cursor, tour.remaining_picks, tour.route_version) == tuple(before)
                assert all((tour.tour_id, order_id) in sim.state.considered_active_orders
                           for order_id in dynamic.active_candidate_ids)
        else:
            sim.step(events, snapshot.problem_class, solution, snapshot)
            if not checked_unstarted_projection and sim.state.tour_manager.all_tours:
                tour = next(iter(sim.state.tour_manager.all_tours.values()))
                assert tour.status == TourStates.SCHEDULED
                assert not sim.state.available_for_planning(0)
                assert not HennWaitingAdapter().transform_state(sim.state, "OBRSPW").resources.resources
                before_buffer = sim.state.order_manager.get_pick_list_buffer()
                projected = ReORSPAdapter().transform_state(sim.state, "RORSP")
                assert tour.status == TourStates.SCHEDULED
                assert sim.state.order_manager.get_pick_list_buffer() == before_buffer
                assert tour.batch in projected.dynamic_warehouse_info.buffered_batches
                checked_unstarted_projection = True

    assert bool(insertion_origins) == active_expected
    assert bool(admitted) == admission_expected
    assert checked_unstarted_projection
    tours = list(sim.state.tour_manager.all_tours.values())
    picked = [pick.order_number for tour in tours for pick in tour.completed_picks]
    assert sorted(picked) == [0, 1, 2, 3]
    assert all(not tour.remaining_picks for tour in tours)
    if arrivals[-1] == 5.0:
        assert all(time > 5.0 for time in insertion_times)
    if waiting_repo == "analytic_waiting":
        assert any(origin.edge_destination is not None for origin in insertion_origins)


def test_same_time_arrivals_are_visible_before_one_waiting_decision(tmp_path):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="stochastic_waiting_config", overrides=[
            "cosy_repo=no_waiting",
            "simulation.arrival_times_s=[0.0,0.0,0.0,12.0]",
            f"instances_base={tmp_path.as_posix()}",
            f"cache_base={(tmp_path / 'cache').as_posix()}",
            f"experiment.output_dir={tmp_path.as_posix()}",
        ])
    sim = setup_scenario(cfg)
    sim.reset(hooks=[add_orders_hook, picker_arrival_hook])
    done, snapshot = sim.run()
    assert not done
    assert snapshot.problem_class == "OBRSPW"
    assert snapshot.dynamic_warehouse_info.time == 0.0
    assert {order.order_id for order in snapshot.orders.orders} == {0, 1, 2}


def test_invalid_opportunity_ownership_and_fifo_admission_fail_at_load(tmp_path):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="stochastic_waiting_config", overrides=[
            f"instances_base={tmp_path.as_posix()}",
            f"cache_base={(tmp_path / 'cache').as_posix()}",
            f"experiment.output_dir={tmp_path.as_posix()}",
        ])
    cfg.engines.simulation_engine.problems.OBRP.triggers = ["WaitingOpportunity"]
    with pytest.raises(ValueError, match="assigned to both"):
        setup_scenario(cfg)

    cfg.engines.simulation_engine.problems.OBRP.triggers = ["ActiveTourOpportunity"]
    cfg.insertion_repo.components[3] = "casim.pipelines.subproblems.batching.FiFo"
    sim = setup_scenario(cfg)
    with pytest.raises(ValueError, match="admission node"):
        sim.reset(hooks=[add_orders_hook, picker_arrival_hook])


def test_remaining_route_admission_respects_path_and_cart_capacity():
    assert not isinstance(RemainingRouteAdmission(), Batching)
    cart = PickCart(n_dimension=1, capacities=[1], dimensions=[DimensionType.ORDERS],
                    n_boxes=4, box_can_mix_orders=False)
    candidate = WarehouseOrder(
        order_id=7, order_date=1.0,
        pick_positions=(PickPosition(order_number=7, article_id=7, amount=1,
                                     pick_node=(2, 1), in_store=1),),
    )

    def solve(remaining_route, occupied_bins):
        return RemainingRouteAdmission().solve(AdmissionInput(
            orders=(candidate,), pick_cart=cart,
            active_order_ids=frozenset(), candidate_order_ids=frozenset({7}),
            remaining_route=remaining_route, occupied_bins=occupied_bins,
        )).accepted_order_ids

    assert solve(((2, 1),), 3) == (7,)
    assert solve(((3, 1),), 3) == ()
    assert solve(((2, 1),), 4) == ()
