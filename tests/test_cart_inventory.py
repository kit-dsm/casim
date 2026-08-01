from copy import deepcopy
from pathlib import Path

import pytest
from ware_ops_algos.algorithms import (
    BatchObject,
    BatchingSolution,
    GreedyItemAssignment,
    Job,
    NodeType,
    PickPosition,
    Route,
    RouteNode,
    ScheduledJob,
    SchedulingSolution,
)

from casim.decision_engine.commitment_policies import SchedulingCommitmentPolicy
from casim.simulation_engine.state_adapter import OrderWindowAdapter
from casim.simulation_engine.conditions import NbrPickersCondition
from casim.state import State
from casim.state.storage_manager import StorageManager
from scenarios.scenario_reopt.loader import ReoptDataLoader


ROOT = Path(__file__).parents[1]
SCENARIO = ROOT / "scenarios" / "scenario_reopt"


def _domain():
    return ReoptDataLoader(SCENARIO).load(
        ROOT / "tests" / "data" / "bin_insertion_scattered.json"
    )


def _route(batch, start, end):
    sequence = [pick.pick_node for pick in batch.pick_positions]
    return Route(
        distance=0.0,
        item_sequence=sequence,
        batch=batch,
        annotated_route=[
            RouteNode(start, NodeType.ROUTE),
            *[
                RouteNode(node, NodeType.PICK)
                for node in sequence
            ],
            RouteNode(end, NodeType.ROUTE),
        ],
    )


def _scheduled(route, picker_id=0):
    job = Job(
        job_id=1,
        processing_time=1.0,
        release_time=0.0,
        due_date=float("inf"),
        n_picks=len(route.item_sequence),
        route=route,
    )
    return ScheduledJob(job, picker_id, 0.0, 1.0)


def test_scheduling_commitment_uses_time_and_per_picker_prefixes():
    jobs = []
    for job_id, picker_id, start in [
        (1, 0, 2),
        (2, 0, 5),
        (3, 1, 1),
        (4, 1, 7),
    ]:
        job = Job(job_id, 1, 0, 20, 0)
        jobs.append(ScheduledJob(job, picker_id, start, start + 1))
    solution = SchedulingSolution(jobs=jobs)
    snapshot = type("Snapshot", (), {
        "dynamic_warehouse_info": type("Info", (), {"time": 0})()
    })()
    selected = SchedulingCommitmentPolicy(
        max_jobs_per_picker=1,
        planning_horizon_s=6,
    ).apply(solution, snapshot)
    assert [value.job.job_id for value in selected.jobs] == [3, 1]
    assert [value.job.job_id for value in solution.jobs] == [1, 2, 3, 4]


def test_batch_commitment_rejects_a_stale_repeat_without_mutation():
    domain = _domain()
    state = State(
        domain.layout,
        domain.articles,
        domain.storage,
        domain.resources,
        domain.objective,
    )
    for order in domain.orders.orders[:2]:
        state.receive_order(order)
    assigned = GreedyItemAssignment(state.get_storage()).solve(
        domain.orders.orders[:2]
    ).resolved_orders
    solution = BatchingSolution(batches=[BatchObject(1, assigned)])
    state.commit_batching_solution(solution)
    before = state.unfinished_work()
    with pytest.raises(ValueError, match="no longer buffered"):
        state.commit_batching_solution(solution)
    assert state.unfinished_work() == before


def test_scattered_fixture_has_static_cart_spec_and_inventory():
    domain = _domain()
    cart = domain.resources.resources[0].pick_cart
    assert cart.n_boxes == 3
    assert cart.capacities == [1.0]
    assert cart.box_can_mix_orders is False
    assert {
        (location.article_id, location.x, location.y, location.amount)
        for location in domain.storage.locations
    } == {
        (101, 1, 2, 3.0),
        (101, 3, 10, 2.0),
        (102, 2, 4, 3.0),
        (102, 3, 8, 2.0),
        (103, 1, 11, 2.0),
        (103, 2, 8, 2.0),
    }


def test_inventory_reservation_confirmation_and_release_are_exact():
    domain = _domain()
    manager = StorageManager(domain.articles, domain.storage)
    first = PickPosition(1, 101, 2, (1, 2), 2)
    second = PickPosition(2, 101, 1, (1, 2), 1)

    manager.reserve(1, [first])
    manager.reserve(2, [second])
    available = {
        (location.article_id, location.x, location.y): location.amount
        for location in manager.planning_snapshot().locations
    }
    assert available[(101, 1, 2)] == 0
    with pytest.raises(ValueError, match="Insufficient available inventory"):
        manager.reserve(3, [second])
    assert manager.reservation_for(3) == {}

    manager.confirm_pick(1, first)
    with pytest.raises(ValueError, match="not reserved"):
        manager.confirm_pick(1, first)
    manager.release_reservation(2)
    available = {
        (location.article_id, location.x, location.y): location.amount
        for location in manager.planning_snapshot().locations
    }
    assert available[(101, 1, 2)] == 1


def test_inventory_aggregates_duplicate_article_locations():
    domain = _domain()
    storage = deepcopy(domain.storage)
    duplicate = deepcopy(storage.locations[0])
    duplicate.amount = 2
    storage.locations.append(duplicate)
    manager = StorageManager(domain.articles, storage)
    amounts = {
        (location.article_id, location.x, location.y): location.amount
        for location in manager.planning_snapshot().locations
    }
    key = (duplicate.article_id, duplicate.x, duplicate.y)
    assert amounts[key] == domain.storage.locations[0].amount + 2


def test_planning_projection_does_not_share_mutable_operational_objects():
    domain = _domain()
    state = State(
        domain.layout,
        domain.articles,
        domain.storage,
        domain.resources,
        domain.objective,
    )
    for order in domain.orders.orders[:2]:
        state.receive_order(order)
    snapshot = OrderWindowAdapter().transform_state(state, "OBRSP")

    assert snapshot.orders.orders[0] is not state.order_manager.get_order_buffer()[0]
    assert snapshot.resources.resources[0] is not (
        state.resource_manager.get_resource(0)
    )
    assert snapshot.storage is not state.storage_manager.get_storage()
    snapshot.orders.orders[0].order_positions[0].amount = 99
    snapshot.resources.resources[0].occupied = True
    snapshot.storage.locations[0].amount = 99
    assert state.order_manager.get_order_buffer()[0].order_positions[0].amount == 1
    assert state.resource_manager.get_resource(0).occupied is False
    assert state.storage_manager.planning_snapshot().locations[0].amount != 99


def test_order_window_excludes_picker_with_a_queued_tour():
    domain = _domain()
    state = State(
        domain.layout,
        domain.articles,
        domain.storage,
        domain.resources,
        domain.objective,
    )
    raw = {order.order_id: order for order in domain.orders.orders}
    state.receive_order(raw[1])
    resolved = GreedyItemAssignment(state.get_storage()).solve(
        [raw[1]]
    ).resolved_orders
    batch = BatchObject(1, resolved)
    start = domain.layout.layout_network.start_node
    end = domain.layout.layout_network.end_node
    state.commit_scheduled_job(
        _scheduled(_route(batch, start, end), picker_id=0)
    )
    state.receive_order(raw[2])

    snapshot = OrderWindowAdapter().transform_state(state, "OBRSP")

    assert [picker.id for picker in snapshot.resources.resources] == [1]
    assert NbrPickersCondition(1).get_decision(snapshot)
    assert not NbrPickersCondition(2).get_decision(snapshot)

    state.resource_manager.set_picker_unavailable(1)
    snapshot = OrderWindowAdapter().transform_state(state, "OBRSP")
    assert snapshot.resources.resources == []
    assert not NbrPickersCondition(1).get_decision(snapshot)


def test_active_plan_insertion_is_atomic_and_locks_on_first_pick():
    domain = _domain()
    state = State(
        domain.layout,
        domain.articles,
        domain.storage,
        domain.resources,
        domain.objective,
    )
    raw = {order.order_id: order for order in domain.orders.orders}
    state.receive_order(raw[1])
    state.receive_order(raw[2])
    resolved = GreedyItemAssignment(state.get_storage()).solve(
        [raw[1], raw[2]]
    ).resolved_orders
    initial_batch = BatchObject(1, resolved)
    start = domain.layout.layout_network.start_node
    end = domain.layout.layout_network.end_node
    tour_id = state.commit_scheduled_job(
        _scheduled(_route(initial_batch, start, end))
    )
    state.tour_manager.start_tour(tour_id, 0.0)
    tour = state.tour_manager.get_tour(tour_id)
    assert state.tour_manager.bin_order_ids(tour_id) == ((1,), (2,), ())
    assert state.tour_manager.locked_bin_ids(tour_id) == frozenset()

    state.receive_order(raw[5])
    inserted = GreedyItemAssignment(state.get_storage()).solve(
        [raw[5]]
    ).resolved_orders[0]
    merged = BatchObject(
        1,
        [*deepcopy(tour.batch.orders), inserted],
        bin_assignments={0: (1,), 1: (2,), 2: (5,)},
    )
    replacement = _route(merged, start, end)

    bad_pick = PickPosition(
        order_number=5,
        article_id=inserted.pick_positions[0].article_id,
        amount=999,
        pick_node=inserted.pick_positions[0].pick_node,
        in_store=999,
    )
    bad_order = deepcopy(inserted)
    bad_order.pick_positions = (bad_pick,)
    bad_batch = BatchObject(
        1,
        [*deepcopy(tour.batch.orders), bad_order],
        bin_assignments={0: (1,), 1: (2,), 2: (5,)},
    )
    before_reservation = state.storage_manager.reservation_for(tour_id)
    before_bins = deepcopy(tour.cart_bins)
    before_version = tour.route_version
    state.active_batch_insertion_enabled = True
    with pytest.raises(ValueError, match="Insufficient available inventory"):
        state.commit_active_plan(
            tour_id,
            _route(bad_batch, start, end),
            picker_id=0,
            expected_version=tour.route_version,
            time=0.0,
            resumes_execution=False,
        )
    assert state.storage_manager.reservation_for(tour_id) == before_reservation
    assert tour.cart_bins == before_bins
    assert tour.route_version == before_version
    assert [order.order_id for order in state.order_manager.get_order_buffer()] == [5]

    state.commit_active_plan(
        tour_id,
        replacement,
        picker_id=0,
        expected_version=tour.route_version,
        time=0.0,
        resumes_execution=False,
    )
    assert state.tour_manager.bin_order_ids(tour_id) == ((1,), (2,), (5,))
    assert state.order_manager.get_order_buffer() == []
    first_pick = state.confirm_pick(
        tour_id,
        RouteNode(tour.remaining_picks[0].pick_node, NodeType.PICK),
    )
    owner_bin = next(
        cart_bin
        for cart_bin in tour.cart_bins
        if first_pick.order_number in cart_bin.order_ids
    )
    assert owner_bin.locked is True
    assert owner_bin.picked_load == [1.0]

    state.cancel_tour(tour_id)
    assert state.storage_manager.reservation_for(tour_id) == {}
    available = {
        (location.article_id, location.x, location.y): location.amount
        for location in state.storage_manager.planning_snapshot().locations
    }
    assert available[
        (
            first_pick.article_id,
            first_pick.pick_node[0],
            first_pick.pick_node[1],
        )
    ] == 2.0
