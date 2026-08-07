from __future__ import annotations

import subprocess

import pytest

from scenarios.scenario_runtime_audit.cli import (
    _run_worker_process,
    execute_worker,
    summarize_repetitions,
)


def _row(wall_time, events_per_s):
    return {
        "status": "success",
        "subprocess_wall_time_s": wall_time,
        "wall_time_s": wall_time / 2,
        "rates": {
            "events_per_s": events_per_s,
            "decisions_per_s": 2.0,
            "completed_orders_per_s": 3.0,
        },
        "phase_times_s": {"solver": wall_time / 2},
    }


def test_repetition_summary_keeps_variability_and_phase_data():
    summary = summarize_repetitions([_row(1.0, 10.0), _row(2.0, 5.0)])

    assert summary["successful"] == 2
    assert summary["wall_time_s"]["median"] == 1.5
    assert summary["wall_time_s"]["high_variability"]
    assert summary["median_rates"]["events_per_s"] == 7.5
    assert summary["median_phase_times_s"]["solver"] == 0.75


def test_worker_timeout_is_reported(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 0.01)

    monkeypatch.setattr(subprocess, "run", timeout)
    result = _run_worker_process(
        "no_wait", "small", tmp_path, timeout_s=0.01
    )

    assert result["status"] == "timeout"
    assert result["timeout_s"] == 0.01


def test_no_wait_worker_emits_complete_audit_schema(tmp_path):
    result = execute_worker("no_wait", "small", tmp_path)

    assert result["status"] == "success"
    assert result["outcomes"]["completion_reason"] == "drained"
    assert result["counts"]["events"] > 0
    assert result["counts"]["decisions"] > 0
    assert result["rates"]["events_per_s"] > 0
    assert result["phase_times_s"]["solver"] > 0
    assert result["existing_timing"]["decision_elapsed_s"] > 0


def test_field_aware_route_clone_detaches_mutable_plan_ownership():
    from ware_ops_algos.algorithms import (
        BatchObject,
        NodeType,
        PickPosition,
        Route,
        RouteNode,
        WarehouseOrder,
    )
    from casim.state.state import _clone_route_plan

    pick = PickPosition(1, 2, 1, (3, 4), 1)
    order = WarehouseOrder(order_id=1, pick_positions=(pick,))
    route = Route(
        distance=5.0,
        route=[(0, 0), (3, 4)],
        item_sequence=[(3, 4)],
        batch=BatchObject(0, [order], {0: (1,)}),
        annotated_route=[
            RouteNode((0, 0), NodeType.ROUTE),
            RouteNode((3, 4), NodeType.PICK),
        ],
    )

    detached = _clone_route_plan(route)
    route.route.append((9, 9))
    route.item_sequence.append((9, 9))
    route.annotated_route.clear()
    route.batch.orders[0].order_id = 99
    route.batch.bin_assignments[0] = (99,)

    assert detached.route == [(0, 0), (3, 4)]
    assert detached.item_sequence == [(3, 4)]
    assert len(detached.annotated_route) == 2
    assert detached.batch.orders[0].order_id == 1
    assert detached.batch.bin_assignments == {0: (1,)}
    assert detached.batch.orders[0].pick_positions[0] is pick
