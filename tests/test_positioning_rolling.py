"""Focused execution contracts for positioning and rolling release."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from ware_ops_algos.algorithms import (
    AdmissionInput, AdmissionTour, Job, PickPosition, RoutingOrigin,
    ScheduledJob, WarehouseOrder,
)
from ware_ops_algos.algorithms.admission.center_position import CenterPositionAdmission
from ware_ops_algos.algorithms.routing.patrol import all_aisles_patrol_route
from ware_ops_algos.domain_models import TourStates

from casim.config.schema import CommitmentConfig, WaitingConfig
from casim.decision_engine.commands import (
    CancelTour, CommandViolation, RecallPositioning, ReleaseTour, Wait,
)
from casim.decision_engine.decision_engine import DecisionEngine
from casim.events.operational_events import OrderStreamClosed, PickerArrival, Reconsider
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
    saw_recall = False
    while True:
        done, snapshot = sim.run()
        if done:
            break
        result = engine.on_trigger(snapshot)
        saw_recall |= any(isinstance(command, RecallPositioning)
                          for command in result.commands)
        sim.step(result, snapshot)
    tours = list(sim.state.tour_manager.all_tours.values())
    assert saw_recall
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
    adapter = StateAdapter(selection={
        "orders": ["buffered"], "window": {"by": "due_date", "horizon_s": 3600}
    })
    assert adapter.transform_state(sim.state, "rolling").plan.orders == ()
    sim.state.current_time = 1800
    assert len(adapter.transform_state(sim.state, "rolling").plan.orders) == 1
    sim.state.current_time = 6000
    assert len(adapter.transform_state(sim.state, "rolling").plan.orders) == 1
    order.due_date = 100000
    sim.state.done_flag = True
    assert len(adapter.transform_state(sim.state, "rolling").plan.orders) == 1


def test_solver_free_decision_releases_existing_draft(tmp_path):
    sim, _ = _walk_sim(tmp_path)
    sim.reset()
    sim.state.resource_manager.set_picker_available(0)
    route = all_aisles_patrol_route(sim.state.layout_manager.get_layout())
    job = Job(job_id=-1, processing_time=route.distance, release_time=0,
              due_date=float("inf"), n_picks=0, route=route, batch=route.batch)
    tour_id = sim.state.add_sequencing_to_planning_state(ScheduledJob(
        job=job, picker_id=0, start_time=500, end_time=500 + route.distance))
    adapter = StateAdapter(selection={"tours": ["scheduled"],
                                      "resources": ["idle", "reserved"]},
                           scope={"freeze_from": "released"})
    decision = DecisionEngine(solver_map={},
                              commitment={"release_only": CommitmentConfig(
                                  rule="release_within", within_s=300)},
                              waiting={"release_only": WaitingConfig()})
    sim.state_adapters["release_only"] = adapter
    sim.precedence["release_only"] = 0
    early = adapter.transform_state(sim.state, "release_only")
    result = decision.on_trigger(early)
    assert not any(isinstance(c, ReleaseTour) for c in result.commands)
    assert any(isinstance(c, Wait) and c.reason == "release_window"
               and c.reconsider_at == 200 for c in result.commands)
    sim.step(result, early)
    sim.state.current_time = 200
    due = adapter.transform_state(sim.state, "release_only")
    released = decision.on_trigger(due)
    assert [type(c) for c in released.commands] == [ReleaseTour]
    sim.step(released, due)
    assert sim.state.tour_manager.get_tour(tour_id).status == TourStates.RELEASED


def test_periodic_grid_opens_horizon_without_another_arrival(tmp_path):
    sim, _ = _walk_sim(tmp_path)
    sim.periods = {"OBRSPW": 600}
    sim.state_adapters["OBRSPW"] = StateAdapter(selection={
        "orders": ["buffered"], "window": {"by": "due_date", "horizon_s": 3600}
    })
    sim.conditions_map["OBRSPW"] = (VisiblePlanningWorkCondition(),)

    def seed(sim, domain):
        order = domain.orders.orders[0]
        order.due_date = 5400
        sim.state.order_manager.add_order_to_buffer(order)
        sim.add_event(OrderStreamClosed(1900))

    sim.reset(hooks=[seed])
    done, snapshot = sim.run()
    assert not done
    assert snapshot.context.time == 1800
    assert len(snapshot.plan.orders) == 1
    assert "Reconsider:periodic" in snapshot.context.cause


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
    tour.status = TourStates.RELEASED
    adapter = StateAdapter(selection={"tours": ["released", "started"],
                                      "resources": ["idle", "reserved", "busy"]},
                           scope={"freeze_from": "released"})
    sim.state.current_time = 100
    at_100 = adapter.transform_state(sim.state, "rolling")
    assert at_100.plan.available_from[0] == 500
    assert at_100.plan.tours[0].accepts.value == "none"
    with pytest.raises(CommandViolation):
        sim.state.apply_all("rolling", (CancelTour(
            expected_decision_version=0, tour_id=tour_id,
            expected_route_version=0),), adapter.selection, adapter.scope)
    assert tour.status == TourStates.RELEASED
    tour.status = TourStates.DONE
    sim.state.current_time = 300
    assert adapter.transform_state(sim.state, "rolling").plan.available_from[0] == 300
    tour.status = TourStates.STARTED
    sim.state.current_time = 700
    assert adapter.transform_state(sim.state, "rolling").plan.available_from[0] == 700
