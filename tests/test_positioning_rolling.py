"""Focused execution contracts for positioning and rolling release."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from ware_ops_algos.algorithms import (
    AdmissionInput, AdmissionTour, Job, PickPosition, RoutingOrigin,
    ScheduledJob, WarehouseOrder,
)
from ware_ops_algos.algorithms.admission import CenterPositionAdmission
from ware_ops_algos.algorithms.routing.patrol import all_aisles_patrol_route
from ware_ops_algos.planning.lifecycle import Accepts, ResourceStates, TimeAttribute, TourStates
from ware_ops_algos.planning.scope import RevisionScope, StateInformation, Window
from ware_ops_algos.planning.commands import CancelTour, CommitTour, ReplaceSuffix, ViolationCode
from ware_ops_algos.planning.validation import check as validate_commands

from casim.decision_engine.commitment_policies import WindowCommitmentPolicy
from casim.decision_engine.decision_engine import DecisionEngine
from casim.domain_objects.sim_domain import ReconsiderReason
from casim.events.operational_events import OrderStreamClosed, PickerArrival
from casim.simulation_engine.conditions import VisiblePlanningWorkCondition
from casim.simulation_engine.state_adapter import StateAdapter
from scenarios.experiment_commons import load_and_flatten_data_card, setup_experiment


ROOT = Path(__file__).resolve().parents[1]
WALK_CONFIG = ROOT / "scenarios" / "scenario_walk_or_wait" / "config"


def _walk_sim(tmp_path, policy="center"):
    with initialize_config_dir(config_dir=str(WALK_CONFIG), version_base="1.3"):
        cfg = compose(config_name="walk_or_wait_config", overrides=[
            f"policy={policy}", "simulation=explicit_wait0",
            f"cosy_repo={'paper_center' if policy == 'center' else 'paper_patrol'}",
            "insertion_repo=center" if policy == "center" else
            "insertion_repo=paper_remaining_route",
            f"instances_base={(ROOT / 'scenarios').as_posix()}",
            f"cache_base={(tmp_path / 'cache').as_posix()}",
            f"experiment.output_dir={tmp_path.as_posix()}",
        ])
    return setup_experiment(cfg, load_and_flatten_data_card(cfg.data_card))


def test_center_holds_and_returns_unused_cart_at_closure(tmp_path):
    sim, engine = _walk_sim(tmp_path)
    sim.reset(hooks=[lambda sim, _: (
        sim.add_event(PickerArrival(0, 0, True)),
        sim.add_event(OrderStreamClosed(20)),
    )])
    saw_closure_route = False
    while True:
        done, snapshot = sim.run()
        if done:
            break
        result = engine.on_trigger(snapshot)
        if (snapshot.dynamic_warehouse_info.context.decision == "active_tour_admission"
                and "OrderStreamClosed" in snapshot.dynamic_warehouse_info.context.cause):
            saw_closure_route |= any(isinstance(command, ReplaceSuffix)
                                     and not command.added_order_ids
                                     for command in result.commands)
        sim.step(result, snapshot)
    tours = list(sim.state.tour_manager.all_tours.values())
    assert saw_closure_route
    assert len(tours) == 1
    assert tours[0].status == TourStates.DONE
    assert tours[0].completed_picks == []
    assert sim.state.resource_manager.get_resource(0).current_location.position == (
        sim.state.layout_manager.get_layout().graph_data.end_location)


def test_center_admission_respects_cart_capacity(tmp_path):
    sim, _ = _walk_sim(tmp_path)
    sim.reset()
    domain = sim._initial_domain
    cart = domain.resources.resources[0].pick_cart
    cart.n_boxes = 1
    cart.capacities = [1]
    orders = tuple(WarehouseOrder(order_id=index, order_date=0,
                                  pick_positions=(PickPosition(
                                      order_number=index, article_id=0,
                                      amount=1, pick_node=(2, 3), in_store=1),))
                   for index in (1, 2))
    tour = AdmissionTour(tour_id=1, picker_id=0,
                         pick_cart=cart, active_order_ids=frozenset(),
                         remaining_route=(), occupied_bins=0,
                         routing_origin=RoutingOrigin((4, 8)))
    decision = CenterPositionAdmission(domain.articles).solve(AdmissionInput(
        orders=orders, tours=(tour,), candidate_order_ids=frozenset({1, 2})))
    assert decision.assignments == ((1, 1),)


def test_due_window_keeps_overdue_and_opens_without_arrival(tmp_path):
    sim, _ = _walk_sim(tmp_path)
    sim.reset()
    order = sim._initial_domain.orders.orders[0]
    order.due_date = 5400
    sim.state.order_manager.add_order_to_buffer(order)
    adapter = StateAdapter(
        StateInformation(orders=True, window=Window(TimeAttribute.DUE_DATE, 3600)),
        RevisionScope(TourStates.SCHEDULED, Accepts.NONE))
    assert adapter.transform_state(sim.state, "rolling", "OBRSP").dynamic_warehouse_info.plan.order_buffer == ()
    sim.state.current_time = 1800
    assert len(adapter.transform_state(sim.state, "rolling", "OBRSP").dynamic_warehouse_info.plan.order_buffer) == 1
    sim.state.current_time = 6000
    assert len(adapter.transform_state(sim.state, "rolling", "OBRSP").dynamic_warehouse_info.plan.order_buffer) == 1
    order.due_date = 100000
    sim.state.done_flag = True
    assert len(adapter.transform_state(sim.state, "rolling", "OBRSP").dynamic_warehouse_info.plan.order_buffer) == 1


def test_solver_free_decision_releases_existing_draft(tmp_path):
    sim, _ = _walk_sim(tmp_path)
    sim.reset()
    sim.state.resource_manager.set_picker_available(0)
    route = all_aisles_patrol_route(sim.state.layout_manager.get_layout())
    job = Job(job_id=-1, processing_time=route.distance, release_time=0,
              due_date=float("inf"), n_picks=0, route=route, batch=route.batch)
    tour_id = sim.state.add_sequencing_to_planning_state(ScheduledJob(
        job=job, picker_id=0, start_time=500, end_time=500 + route.distance))
    adapter = StateAdapter(
        StateInformation(tours=frozenset({TourStates.PLANNED}),
                         resources=frozenset({ResourceStates.IDLE, ResourceStates.RESERVED})),
        RevisionScope(TourStates.SCHEDULED, Accepts.NONE))
    decision = DecisionEngine({}, {}, {"release_only": WindowCommitmentPolicy(300)})
    sim.state_adapters["release_only"] = adapter
    sim.precedence["release_only"] = 0
    early = adapter.transform_state(sim.state, "release_only", "OBRSP")
    result = decision.on_trigger(early)
    assert not result.commands
    assert result.reconsider_at == 200
    sim.step(result, early)
    sim.state.current_time = 200
    due = adapter.transform_state(sim.state, "release_only", "OBRSP")
    released = decision.on_trigger(due)
    assert [type(c) for c in released.commands] == [CommitTour]
    sim.step(released, due)
    assert sim.state.tour_manager.get_tour(tour_id).status == TourStates.SCHEDULED


def test_periodic_grid_opens_horizon_without_another_arrival(tmp_path):
    sim, engine = _walk_sim(tmp_path)
    sim.periods = {"OBRSP": 600}
    sim.state_adapters["OBRSP"] = StateAdapter(
        StateInformation(orders=True, window=Window(TimeAttribute.DUE_DATE, 3600)),
        RevisionScope(TourStates.SCHEDULED, Accepts.NONE))
    sim.conditions_map["OBRSP"] = (VisiblePlanningWorkCondition(),)

    def seed(sim, domain):
        order = domain.orders.orders[0]
        order.due_date = 5400
        sim.state.order_manager.add_order_to_buffer(order)
        sim.add_event(OrderStreamClosed(1900))

    sim.reset(hooks=[seed])
    while True:
        done, snapshot = sim.run()
        assert not done
        plan = snapshot.dynamic_warehouse_info.plan
        context = snapshot.dynamic_warehouse_info.context
        if plan.time == 1800:
            assert len(plan.order_buffer) == 1
            assert ReconsiderReason.PERIODIC in context.reconsider
            break
        sim.step(engine.on_trigger(snapshot), snapshot)


def test_fixed_tour_availability_changes_with_actual_completion(tmp_path):
    sim, _ = _walk_sim(tmp_path)
    sim.reset()
    sim.state.resource_manager.set_picker_available(0)
    route = all_aisles_patrol_route(sim.state.layout_manager.get_layout())
    job = Job(job_id=-1, processing_time=500, release_time=0,
              due_date=float("inf"), n_picks=0, route=route, batch=route.batch)
    tour_id = sim.state.add_sequencing_to_planning_state(ScheduledJob(
        job=job, picker_id=0, start_time=0, end_time=500))
    tour = sim.state.tour_manager.get_tour(tour_id)
    sim.state.tour_manager.advance(tour_id, TourStates.SCHEDULED)
    adapter = StateAdapter(
        StateInformation(tours=frozenset({TourStates.SCHEDULED, TourStates.STARTED}),
                         resources=frozenset({ResourceStates.IDLE, ResourceStates.RESERVED,
                                              ResourceStates.BUSY})),
        RevisionScope(TourStates.SCHEDULED, Accepts.NONE))
    sim.state.current_time = 100
    at_100 = adapter.transform_state(sim.state, "rolling", "OBRSP")
    plan = at_100.dynamic_warehouse_info.plan
    assert plan.available_from[0] == 500
    assert plan.tours[0].accepts is Accepts.NONE
    assert validate_commands((CancelTour(tour_id),), plan)[0].code is ViolationCode.ACCEPTS
    assert tour.status == TourStates.SCHEDULED
    sim.state.tour_manager.advance(tour_id, TourStates.PENDING)
    sim.state.tour_manager.advance(tour_id, TourStates.STARTED)
    sim.state.tour_manager.advance(tour_id, TourStates.DONE)
    sim.state.current_time = 300
    assert adapter.transform_state(sim.state, "rolling", "OBRSP").dynamic_warehouse_info.plan.available_from[0] == 300
    # A running tour keeps its picker unavailable until its planned end.
    sim.state.tour_manager.create_tour(route, 500)
    running_id = sim.state.tour_manager.get_all_tours()[-1].tour_id
    sim.state.tour_manager.assign_tour(running_id, 0)
    sim.state.tour_manager.schedule_tour(running_id, 0, 500)
    sim.state.tour_manager.advance(running_id, TourStates.SCHEDULED)
    sim.state.tour_manager.advance(running_id, TourStates.PENDING)
    sim.state.tour_manager.advance(running_id, TourStates.STARTED)
    sim.state.current_time = 700
    assert adapter.transform_state(sim.state, "rolling", "OBRSP").dynamic_warehouse_info.plan.available_from[0] == 700
