from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from ware_ops_algos.algorithms import (
    BatchObject,
    CombinedRoutingSolution,
    PickPosition,
    Route,
    WarehouseOrder,
)
from ware_ops_algos.domain_models import Resource, Resources, ResourceType

from casim.events.operational_events import OrderArrival
from scenarios.scenario_henn.algorithm import (
    HennWakeUp,
    decide_henn,
    route_service_time,
)
from scenarios.scenario_henn.loader import HennDataLoader, load_manifest
from scenarios.scenario_henn.reference import (
    compare_benchmarks,
    load_references,
)


SCENARIO_ROOT = (
    Path(__file__).parents[1] / "scenarios" / "scenario_henn"
).resolve()
MANIFEST = SCENARIO_ROOT / "reproduction" / "manifest.yaml"


def _domain(instance_id="H_abc1_40_29"):
    return HennDataLoader(SCENARIO_ROOT).load(
        MANIFEST,
        instance_id,
    )


def _picker():
    return Resource(
        id=0,
        capacity=30,
        speed=0.8,
        time_per_pick=10.0,
        tour_setup_time=180.0,
        available=True,
        occupied=False,
    )


def _route(
    batch_id: int,
    order_id: int,
    distance: float,
    arrival: float = 10.0,
    amount: int = 1,
) -> Route:
    position = PickPosition(
        order_number=order_id,
        article_id=order_id,
        amount=amount,
        pick_node=(1, 1),
        in_store=amount,
    )
    order = WarehouseOrder(
        order_id=order_id,
        order_date=arrival,
        pick_positions=(position,),
    )
    return Route(
        distance=distance,
        route=[(0, 0), (1, 1), (0, 0)],
        item_sequence=[(1, 1)] * amount,
        batch=BatchObject(batch_id=batch_id, orders=[order]),
    )


def _snapshot():
    return SimpleNamespace(
        resources=Resources(
            tpe=ResourceType.HUMAN,
            resources=[_picker()],
        )
    )


def test_manifest_and_raw_schemas_cover_all_64_instances():
    loader = HennDataLoader(SCENARIO_ROOT)
    entries = load_manifest(MANIFEST)
    assert len(entries) == 64
    for instance_id, paths in entries.items():
        instance_path, settings_path, arrival_path = [
            SCENARIO_ROOT / "data" / path for path in paths
        ]
        settings = loader._parse_settings(settings_path)
        orders = loader._parse_instance(instance_path)
        arrivals = loader._parse_arrivals(arrival_path)
        loader._validate(
            instance_id,
            instance_path,
            settings,
            orders,
            arrivals,
        )


def test_loader_preserves_faces_units_capacity_and_metadata():
    domain = _domain()
    assert domain.problem_class == "OBRP"
    assert len(domain.orders.orders) == 40
    assert len(domain.articles.articles) == 900
    assert len(domain.storage.locations) == 900
    assert domain.storage.get_locations_by_article_id(0)[0].x == 1
    assert domain.storage.get_locations_by_article_id(45)[0].x == 1
    picker = domain.resources.resources[0]
    assert picker.capacity == 30
    assert picker.speed == 0.8
    assert picker.time_per_pick == 10.0
    assert picker.tour_setup_time == 180.0
    assert all(order.due_date is None for order in domain.orders.orders)
    assert domain.henn_metadata["effective_routing"] == "s"


def test_reference_loader_maps_all_rows_and_preserves_precision():
    references = load_references(
        SCENARIO_ROOT / "data" / "results" / "experiments.xlsx"
    )
    assert len(references) == 64
    gil = references["H_abc1_40_29"]["gil_2020_grasp_vnd"]
    best = references["H_abc1_40_29"]["best_known"]
    assert gil["completion_time_s"] == Decimal("21109.376")
    assert gil["max_turnover_time_s"] == Decimal("10165.468")
    assert best["max_turnover_time_s"] == Decimal("9848.194")
    assert gil["benchmark_only"] and best["benchmark_only"]


def test_service_time_uses_the_actual_route_and_setup_once():
    route = _route(0, 0, distance=8.0, amount=2)
    assert route_service_time(route, _picker()) == 210.0


@pytest.mark.parametrize(
    ("selector", "expected_batch"),
    [
        ("first", 1),
        ("short", 2),
        ("long", 1),
        ("sav", 2),
    ],
)
def test_all_selection_rules(selector, expected_batch):
    short = _route(2, 0, distance=8.0)
    long = _route(1, 1, distance=16.0)
    decision = decide_henn(
        CombinedRoutingSolution(routes=[short, long]),
        _snapshot(),
        current_time=20.0,
        next_arrival=30.0,
        stream_exhausted=False,
        selector=selector,
        single_services={0: 260.0, 1: 220.0},
    )
    assert decision.action == "dispatch"
    assert decision.solution.jobs[0].job.route.batch.batch_id == expected_batch


def test_one_batch_wait_is_pure_and_natural_arrival_needs_no_wakeup():
    route = _route(0, 0, distance=8.0)
    candidate = CombinedRoutingSolution(routes=[route])
    snapshot = _snapshot()
    candidate_before = deepcopy(candidate)
    picker_before = deepcopy(snapshot.resources.resources[0])

    decision = decide_henn(
        candidate,
        snapshot,
        current_time=20.0,
        next_arrival=50.0,
        stream_exhausted=False,
        selector="short",
        single_services={0: 250.0},
    )

    # threshold = 2*10 + 250 - 200 = 70; arrival at 50 triggers naturally.
    assert decision.action == "wait"
    assert decision.wait_until is None
    assert candidate == candidate_before
    assert snapshot.resources.resources[0] == picker_before


def test_timed_wait_and_equality_dispatch():
    route = _route(0, 0, distance=8.0)
    candidate = CombinedRoutingSolution(routes=[route])
    snapshot = _snapshot()

    waiting = decide_henn(
        candidate,
        snapshot,
        current_time=20.0,
        next_arrival=100.0,
        stream_exhausted=False,
        selector="short",
        single_services={0: 250.0},
    )
    assert waiting.action == "wait"
    assert waiting.wait_until == 70.0

    dispatch = decide_henn(
        candidate,
        snapshot,
        current_time=70.0,
        next_arrival=100.0,
        stream_exhausted=False,
        selector="short",
        single_services={0: 250.0},
    )
    assert dispatch.action == "dispatch"
    job = dispatch.solution.jobs[0]
    assert job.start_time == 250.0
    assert job.end_time == 270.0


def test_final_arrival_dispatches_all_batches_sequentially():
    first = _route(0, 0, distance=8.0)
    second = _route(1, 1, distance=16.0)
    decision = decide_henn(
        CombinedRoutingSolution(routes=[first, second]),
        _snapshot(),
        current_time=20.0,
        next_arrival=None,
        stream_exhausted=True,
        selector="short",
        single_services={0: 200.0, 1: 210.0},
    )
    jobs = decision.solution.jobs
    assert decision.reason == "final_arrival_dispatch_all"
    assert len(jobs) == 2
    assert jobs[1].start_time == jobs[0].end_time + 180.0


def test_wakeup_has_lower_priority_than_equal_time_arrival():
    route_order = SimpleNamespace(order_id=0)
    arrival = OrderArrival(10.0, route_order)
    wake = HennWakeUp(10.0)
    assert arrival < wake


def test_cross_algorithm_comparator_never_claims_reproduction_match():
    actual = {
        "instance_id": "H_abc1_40_29",
        "algorithm_id": "henn_4_1+fcfs+s_shape+short",
        "completion_time_s": 100.0,
        "max_turnover_time_s": 50.0,
    }
    references = {
        "gil": {
            "completion_time_s": Decimal("100.000"),
            "max_turnover_time_s": Decimal("49.000"),
            "source": "test",
            "benchmark_only": True,
        }
    }
    report = compare_benchmarks(actual, references, 0.001, 1e-12)[0]
    assert report["benchmark_only"]
    assert report["completion_time"]["status"] == "EQUAL"
    assert report["max_turnover_time"]["status"] == "WORSE"
    assert report["batch_membership"] == "NOT_AVAILABLE"
