"""Operational boundary checks; the stochastic numerical model is tested elsewhere."""

import copy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from ware_ops_algos.algorithms import (
    AdmissionInput, AdmissionTour, Batching,
    PickPosition, RemainingRouteAdmission, RoutingOrigin, WarehouseOrder,
)
from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards
from ware_ops_algos.domain_algo_mapper.domain_algo_mapper import DomainAlgorithmMapper
from ware_ops_algos.domain_models import DimensionType, PickCart
from ware_ops_algos.planning.lifecycle import TourStates
from ware_ops_algos.planning.commands import ReplaceSuffix

from casim.events.operational_events import ActiveTourOpportunity, NodeArrival, OrderArrival, TravelEvent
from casim.pipelines.admission_view import to_admission_tour
from casim.pipelines.problem_based_template import (
    AbstractAdmission, AbstractBatching, AbstractBatchProvider,
    AdmittedTourBatch, traverse_pipeline,
)
from casim.pipelines.subproblems.admission import RemainingRouteAdmissionNode
from casim.pipelines.taxonomy import TAXONOMY, variant
from casim.config.data_card import load_and_flatten_data_card
from casim.run import prepare
from tests.scenario_configs import analytic_config


def test_admission_card_matches_the_active_tour_problem():
    cfg = analytic_config(None)
    card = load_and_flatten_data_card(cfg.data_card)
    card.problem_class = "OBRSP"
    card.decision_features = {"type": "decision", "features": {
        "incumbent": False, "routing_origin": "live"}}
    cards = load_packaged_algo_cards()
    admission = next(model for model in cards if model.algo_name == "RemainingRouteAdmission")
    reroutable = next(model for model in cards if model.algo_name == "ReroutableSShape")
    ordinary = next(model for model in cards if model.algo_name == "SShape")
    fifo = next(model for model in cards if model.algo_name == "FiFo")
    assert admission.problem_type == "admission"
    assert reroutable.problem_type == "active_tour_routing"
    live_stages = variant("OBRSP", card.decision_features["features"])
    assert [model for model in DomainAlgorithmMapper(TAXONOMY).filter(
        [admission, reroutable, ordinary, fifo], card)
        if model.problem_type in live_stages] == [admission, reroutable]
    for feature, unsupported in (
        ("capacities", [0]),
        ("box_can_mix_orders", True),
        ("n_boxes", 0),
        ("dimensions_type", ["items"]),
    ):
        incompatible = copy.deepcopy(card)
        incompatible.resources["features"][feature] = unsupported
        assert DomainAlgorithmMapper(TAXONOMY).filter([admission], incompatible) == []
    card.problem_class = "OBRP"
    assert DomainAlgorithmMapper(TAXONOMY).filter(
        [admission, reroutable, ordinary, fifo], card
    ) == [ordinary, fifo]


@pytest.mark.parametrize(
    ("arrivals", "pick_time", "waiting_repo", "active_expected", "admission_expected"),
    [
        ([0.0, 1.0, 2.0, 12.0], 0.0, "no_waiting", True, False),
        ([0.0, 2.5, 5.5, 12.0], 2.0, "no_waiting", True, False),
        ([0.0, 1.0, 2.0, 12.0], 0.0, "analytic_waiting", True, True),
        ([0.0, 1.0, 2.0, 5.0], 4.0, "no_waiting", True, False),
    ],
)
def test_configured_active_tour_boundary(tmp_path, arrivals, pick_time,
                                         waiting_repo, active_expected, admission_expected):
    cfg = analytic_config(tmp_path, "decisions=remaining_route_admission", f"cosy_repo={waiting_repo}", f"source.arrival_times_s={arrivals}", f"source.pick_time_s={pick_time}")
    prepared = prepare(cfg)
    sim, decision_engine = prepared.sim, prepared.engine
    active_tasks = traverse_pipeline(decision_engine.solver_map["active_tour_admission"].pipelines)
    assert any(isinstance(task, RemainingRouteAdmissionNode) for task in active_tasks)
    assert any(isinstance(task, AdmittedTourBatch) for task in active_tasks)
    assert issubclass(RemainingRouteAdmissionNode, AbstractAdmission)
    assert not issubclass(RemainingRouteAdmissionNode, AbstractBatching)
    assert issubclass(AdmittedTourBatch, AbstractBatchProvider)
    assert not issubclass(AdmittedTourBatch, AbstractBatching)
    prepared.reset()

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
            if sim.state.order_manager.has_buffered_order(order.order_id):
                live = next(item for item in sim.state.order_manager.get_order_buffer()
                            if item.order_id == order.order_id)
                assert order is not live

        result = decision_engine.on_trigger(snapshot)
        plan = snapshot.dynamic_warehouse_info.plan
        decision_name = snapshot.dynamic_warehouse_info.context.decision
        if decision_name == "active_tour_admission" and not plan.tours:
            assert not snapshot.dynamic_warehouse_info.context.solve
            sim.step(result, snapshot)
            continue
        if decision_name == "active_tour_admission":
            admission_tour = to_admission_tour(plan.tours[0])
            insertion_origins.append(admission_tour.routing_origin)
            insertion_times.append(now)
            tour = sim.state.tour_manager.get_tour(admission_tour.tour_id)
            live_before_projection = (tour.cursor, tuple(tour.remaining_picks), tour.route_version)
            projected = sim.state_adapters["active_tour_admission"].transform_state(sim.state, "active_tour_admission", "OBRSP")
            assert projected.dynamic_warehouse_info.plan.tours[0].batch is not tour.batch
            assert {order.order_id for order in projected.dynamic_warehouse_info.plan.order_buffer} == {
                order.order_id for order in plan.order_buffer}
            assert (tour.cursor, tuple(tour.remaining_picks), tour.route_version) == live_before_projection
            before = (tour.cursor, list(tour.remaining_picks), tour.route_version)
            sim.step(result, snapshot)
            tour = sim.state.tour_manager.get_tour(admission_tour.tour_id)
            replacements = [command for command in result.commands
                            if isinstance(command, ReplaceSuffix)]
            if replacements:
                admitted += 1
                assert tour.route_version == before[2] + 1
                assert NodeArrival(now, tour.tour_id, before[2]).handle(sim.state) == []
                assert ActiveTourOpportunity(now, tour.tour_id, before[2]).is_stale(sim.state)
                assert (tour.cursor, tour.remaining_picks, tour.route_version) == (
                    0, list(replacements[0].route.batch.pick_positions), before[2] + 1)
                assert any(isinstance(event, TravelEvent) and event.time == now
                           and event.route_version == tour.route_version for event in sim.events)
            else:
                assert (tour.cursor, tour.remaining_picks, tour.route_version) == tuple(before)
                assert all((tour.tour_id, order_id) in
                           sim.state.decision_considered_active_orders["active_tour_admission"]
                           for tour_id, order_id in result.considered_active_orders
                           if tour_id == tour.tour_id)
        else:
            sim.step(result, snapshot)
            if not checked_unstarted_projection and sim.state.tour_manager.all_tours:
                tour = next(iter(sim.state.tour_manager.all_tours.values()))
                assert tour.status in (TourStates.PLANNED, TourStates.SCHEDULED)
                assert not sim.state.available_for_planning(0)
                before_buffer = sim.state.order_manager.get_pick_list_buffer()
                projected = sim.state_adapters["OBRSP"].transform_state(sim.state, "OBRSP", "OBRSP")
                assert sim.state.order_manager.get_pick_list_buffer() == before_buffer
                if tour.status == TourStates.PLANNED:
                    assert tour.batch == projected.dynamic_warehouse_info.plan.tours[0].batch
                checked_unstarted_projection = True

    assert bool(insertion_origins) == active_expected
    assert bool(admitted) == admission_expected
    assert checked_unstarted_projection
    tours = list(sim.state.tour_manager.all_tours.values())
    picked = [pick.order_number for tour in tours for pick in tour.completed_picks]
    assert sorted(picked) == [0, 1, 2, 3]
    assert all(not tour.remaining_picks for tour in tours)
    if arrivals[-1] == 5.0:
        assert all(time >= 5.0 for time in insertion_times)
    if waiting_repo == "analytic_waiting":
        assert any(origin.edge_destination is not None for origin in insertion_origins)


def test_same_time_arrivals_are_visible_before_one_waiting_decision(tmp_path):
    cfg = analytic_config(tmp_path, "cosy_repo=no_waiting", "source.arrival_times_s=[0.0,0.0,0.0,12.0]")
    prepared = prepare(cfg)
    sim = prepared.sim
    prepared.reset()
    done, snapshot = sim.run()
    assert not done
    assert snapshot.problem_class == "OBRSP"
    assert snapshot.dynamic_warehouse_info.plan.time == 0.0
    assert {order.order_id for order in snapshot.orders.orders} == {0, 1, 2}


def test_shared_trigger_and_invalid_admission_repository(tmp_path):
    cfg = analytic_config(tmp_path, "decisions=remaining_route_admission")
    cfg.decisions.active_tour_admission.triggers = ["OrderArrival"]
    sim = prepare(cfg).sim
    assert sim.triggers_map[OrderArrival] == ("active_tour_admission", "OBRSP")

    cfg.decisions.active_tour_admission.triggers = ["ActiveTourOpportunity"]
    cfg.decisions.active_tour_admission.solver.repo.components[3] = (
        "casim.pipelines.subproblems.admission.MissingAdmission")
    with pytest.raises(ImportError, match="MissingAdmission"):
        prepare(cfg)

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
            orders=(candidate,),
            tours=(AdmissionTour(
                tour_id=1, picker_id=0,
                pick_cart=cart, active_order_ids=frozenset(),
                remaining_route=remaining_route, occupied_bins=occupied_bins,
                routing_origin=RoutingOrigin(position=(0, 0)),
            ),),
            candidate_order_ids=frozenset({7}),
        )).assignments

    assert solve(((2, 1),), 3) == ((1, 7),)
    assert solve(((3, 1),), 3) == ()
    assert solve(((2, 1),), 4) == ()


def test_admission_greedily_assigns_orders_across_tours():
    cart = PickCart(n_dimension=1, capacities=[1], dimensions=[DimensionType.ORDERS],
                    n_boxes=2, box_can_mix_orders=False)
    orders = tuple(WarehouseOrder(
        order_id=order_id, order_date=float(order_id),
        pick_positions=(PickPosition(order_number=order_id, article_id=order_id,
                                     amount=1, pick_node=(2, 1), in_store=1),),
    ) for order_id in (7, 8))
    tours = tuple(AdmissionTour(
        tour_id=tour_id, picker_id=tour_id,
        pick_cart=cart, active_order_ids=frozenset(),
        remaining_route=((2, 1),), occupied_bins=1,
        routing_origin=RoutingOrigin(position=(0, 0)),
    ) for tour_id in (1, 2))
    solution = RemainingRouteAdmission().solve(AdmissionInput(
        orders=orders, tours=tours, candidate_order_ids=frozenset({7, 8}),
    ))
    assert solution.assignments == ((1, 7), (2, 8))


def test_two_active_tours_can_receive_different_orders(tmp_path):
    cfg = analytic_config(tmp_path, "decisions=remaining_route_admission", "cosy_repo=no_waiting", "source.arrival_times_s=[0.0,0.1,1.0,1.0]", "+source.pick_locations=[[2,1],[6,1],[2,1],[6,1]]")
    cfg.decisions.OBRSP.conditions = [
        cfg.decisions.OBRSP.conditions[0]
    ]
    prepared = prepare(cfg)
    sim, source = prepared.sim, prepared.source
    original_initial_state = source.initial_state

    def initial_state_with_two_pickers():
        domain = original_initial_state()
        second = copy.deepcopy(domain.resources.resources[0])
        second.id = 1
        domain.resources.resources.append(second)
        return domain

    source.initial_state = initial_state_with_two_pickers
    decision_engine = prepare(cfg).engine
    prepared.reset()
    multi_tour_decisions = []
    while True:
        done, snapshot = sim.run()
        if done:
            break
        result = decision_engine.on_trigger(snapshot)
        plan = snapshot.dynamic_warehouse_info.plan
        if (snapshot.dynamic_warehouse_info.context.decision == "active_tour_admission"
                and len(plan.tours) == 2):
            multi_tour_decisions.append(tuple(command for command in result.commands
                                              if isinstance(command, ReplaceSuffix)))
        sim.step(result, snapshot)

    assert any(len(commands) == 2 for commands in multi_tour_decisions)
    tours = sim.state.tour_manager.all_tours
    assert {0, 2} <= set(tours[1].order_numbers)
    assert {1, 3} <= set(tours[2].order_numbers)
    assert sum(len(tour.completed_picks) for tour in tours.values()) == 4
