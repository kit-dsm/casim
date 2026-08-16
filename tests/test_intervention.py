from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
from ware_ops_algos.algorithms import (
    BatchObject,
    NodeType,
    PickPosition,
    RouteNode,
    WarehouseOrder,
)
from ware_ops_algos.algorithms.algorithm_cards import (
    load_packaged_algo_cards,
)
from ware_ops_algos.algorithms.routing.routing import (
    ExactTSPRoutingDistance,
)

from casim.events.operational_events import (
    InterventionRequest,
    NodeArrival,
    OrderArrival,
    PickComplete,
    TourEnd,
    TravelEvent,
)
from casim.simulation_engine.state_adapter import _residual_batch
from scenarios.scenario_reopt.loader import ReoptDataLoader


ROOT = Path(__file__).parents[1]
SCENARIO = ROOT / "scenarios" / "scenario_reopt"


def test_two_picker_loader_uses_real_multiblock_layout_and_distinct_carts():
    domain = ReoptDataLoader(SCENARIO).load(
        ROOT / "tests" / "data" / "two_picker_intervention.json"
    )
    assert len(domain.orders.orders) == 8
    assert len(domain.resources.resources) == 2
    assert domain.resources.resources[0].pick_cart is not (
        domain.resources.resources[1].pick_cart
    )
    assert domain.layout.graph_data.n_blocks == 2
    assert domain.layout.graph_data.n_aisles == 3
    assert not domain.layout.layout_network.graph.is_directed()
    assert (3, 12) in domain.layout.layout_network.graph


def test_intervention_algorithms_declare_capabilities():
    cards = load_packaged_algo_cards()
    cards_with_capabilities = [
        card for card in cards if card.capabilities
    ]
    assert sorted(card.algo_name for card in cards_with_capabilities) == [
        "EDDScheduler",
        "ExactSolving",
        "NearestNeighbourhood",
        "ResidualFiFo",
    ]
    exact_card = next(
        card
        for card in cards_with_capabilities
        if card.algo_name == "ExactSolving"
    )
    assert exact_card.capabilities == {
        "arbitrary_origin_node": True,
        "arbitrary_origin_edge": True,
        "arbitrary_end_node": True,
        "residual_picks": True,
        "partial_batch": False,
        "locked_orders": False,
        "active_job_replacement": False,
    }


def test_stale_active_tour_events_are_no_ops():
    tour = SimpleNamespace(route_version=4)
    state = SimpleNamespace(
        tour_manager=SimpleNamespace(get_tour=lambda _tour_id: tour),
        accept_intervention_request=lambda _tour_id, version: version == 4,
        tour_event_is_stale=lambda _tour_id, version: version != 4,
    )
    before = vars(tour).copy()
    events = [
        TravelEvent(2.0, 1, 3),
        NodeArrival(2.0, 1, 3),
        PickComplete(2.0, 1, pick_start=1.0, route_version=3),
        TourEnd(2.0, 1, 3),
    ]
    assert all(event.handle(state) == [] for event in events)
    request = InterventionRequest(2.0, 1, 0, 3)
    assert request.handle(state) == []
    assert request.cancelled is True
    assert vars(tour) == before


def test_arrivals_deduplicate_during_travel_and_wait_for_atomic_pick():
    buffered = []
    tour = SimpleNamespace(
        tour_id=1,
        assigned_resource=0,
        route_version=0,
        status="started",
        replan_requested=False,
        intervention_event_pending=False,
        is_picking=False,
        is_travelling=True,
        edge_arrives_at=10.0,
    )
    state = SimpleNamespace(
        intervention_enabled=True,
        active_batch_insertion_enabled=True,
        receive_order=buffered.append,
        order_manager=SimpleNamespace(
            add_order_to_buffer=buffered.append,
        ),
        tour_manager=SimpleNamespace(
            active_tours=lambda: [tour],
            empty_bin_ids=lambda _tour_id: [0],
        ),
    )
    def request_arrival_intervention(_time):
        if tour.replan_requested or tour.intervention_event_pending:
            return None
        tour.replan_requested = True
        if tour.is_picking:
            return None
        tour.intervention_event_pending = True
        return tour.tour_id, tour.assigned_resource, tour.route_version
    state.request_arrival_intervention = request_arrival_intervention
    order = WarehouseOrder(order_id=9)
    first = OrderArrival(3.0, order).handle(state)
    second = OrderArrival(3.0, WarehouseOrder(order_id=10)).handle(state)
    assert len(first) == 1
    assert isinstance(first[0], InterventionRequest)
    assert second == []

    pick = PickPosition(1, 1, 1, (1, 2), 1)
    pick_tour = SimpleNamespace(
        tour_id=1,
        route_version=0,
        remaining_picks=[pick],
        completed_picks=[],
        replan_requested=True,
        intervention_event_pending=False,
        assigned_resource=0,
        pick_started_at=2.0,
        pick_ends_at=6.0,
        at_end=lambda: False,
    )
    def confirm_pick_operation(_tour_id, _pick_start, _time):
        completed_pick = pick_tour.remaining_picks.pop(0)
        pick_tour.completed_picks.append(completed_pick)
        pick_tour.pick_started_at = None
        pick_tour.pick_ends_at = None
        return "intervention", []

    pick_state = SimpleNamespace(
        confirm_pick_operation=confirm_pick_operation,
        tour_event_is_stale=lambda _tour_id, _version: False,
        tour_target=lambda _tour_id: (0, 0),
    )
    emitted = PickComplete(
        6.0,
        1,
        pick_start=2.0,
        route_version=0,
    ).handle(pick_state)
    assert pick_tour.completed_picks == [pick]
    assert pick_tour.remaining_picks == []
    assert len(emitted) == 1
    assert isinstance(emitted[0], InterventionRequest)


def test_residual_batch_keeps_locked_active_order_membership():
    completed = PickPosition(1, 1, 1, (1, 2), 1)
    remaining = PickPosition(2, 2, 1, (2, 3), 1)
    tour = SimpleNamespace(
        batch=BatchObject(
            1,
            [
                WarehouseOrder(1, pick_positions=(completed,)),
                WarehouseOrder(2, pick_positions=(remaining,)),
            ],
        ),
        remaining_picks=[remaining],
    )
    residual = _residual_batch(tour)
    assert residual.order_numbers == frozenset({1, 2})
    assert residual.orders[0].pick_positions == ()
    assert residual.orders[1].pick_positions == (remaining,)


def test_exact_tsp_residual_distance_matches_exhaustive_enumeration():
    domain = ReoptDataLoader(SCENARIO).load(
        ROOT / "tests" / "data" / "two_picker_intervention.json"
    )
    network = domain.layout.layout_network
    picks = [
        PickPosition(index, index, 1, node, 1)
        for index, node in enumerate(
            [(1, 2), (2, 4), (3, 5)],
            start=1,
        )
    ]
    router = ExactTSPRoutingDistance(
        start_node=network.start_node,
        end_node=network.end_node,
        closest_node_to_start=network.closest_node_to_start,
        min_aisle_position=network.min_aisle_position,
        max_aisle_position=network.max_aisle_position,
        distance_matrix=network.distance_matrix,
        predecessor_matrix=network.predecessor_matrix,
        picker=domain.resources.resources,
        gen_tour=True,
        gen_item_sequence=True,
        node_list=network.node_list,
        node_to_idx={
            node: index
            for index, node in enumerate(network.graph.nodes)
        },
        idx_to_node={
            index: node
            for index, node in enumerate(network.graph.nodes)
        },
        set_time_limit=60,
    )
    picks_before = list(picks)
    solution = router.solve(picks)
    repeated = router.solve(picks)
    matrix = network.distance_matrix
    expected = min(
        matrix.at[network.start_node, permutation[0]]
        + sum(
            matrix.at[left, right]
            for left, right in zip(permutation, permutation[1:])
        )
        + matrix.at[permutation[-1], network.end_node]
        for permutation in itertools.permutations(
            [pick.pick_node for pick in picks]
        )
    )
    assert solution.route.distance == expected
    assert repeated.route.distance == expected
    assert repeated.route.item_sequence == solution.route.item_sequence
    assert repeated.route.annotated_route == solution.route.annotated_route
    assert picks == picks_before
    assert solution.is_optimal is True
    assert solution.solver_status == "optimal"
    empty = router.solve([])
    assert empty.route.item_sequence == []
    assert empty.route.annotated_route[0].position == network.start_node
    assert empty.route.annotated_route[-1].position == network.end_node
    assert empty.is_optimal is True
