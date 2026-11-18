"""The facts tables of a run, on the hand-derived minimal scenario (scenarios/minimal/README.md).

The tour leaves the depot at t = 0 over (1, 0) and (2, 0) up aisle 2 to the
pick at (2, 3): 1 + 1 + 3 = 5 unit edges at speed 1, so the walk is 5 long and
ends at t = 5; the pick takes 1 s (5 to 6). The tour walks 38 units in all and
ends at 38 + 1 = 39, which completes order 0 (arrival 0, due 400). There is no
setup, so the tour's start and setup end coincide; the one decision commits
it at t = 0.
"""

from pathlib import Path
import json

import pandas as pd
import pytest

from casim.run import compose_scenario, run
from casim.outputs.service import service_measures

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("minimal")
    run(compose_scenario(ROOT / "scenarios" / "minimal", output_dir=output_dir))
    return output_dir


def test_pick_table_has_the_walk_and_the_pick(outputs):
    picks = pd.read_csv(outputs / "facts" / "picks.csv")
    assert len(picks) == 1
    pick = picks.iloc[0]
    assert (pick.order_id, pick.picker_id, pick.node_x, pick.node_y) == (0, 0, 2, 3)
    assert (pick.walk_start, pick.walk_distance, pick.pick_start, pick.pick_end) == (0, 5, 5, 6)


def test_tour_table_has_setup_route_start_end_and_distance(outputs):
    tours = pd.read_csv(outputs / "facts" / "tours.csv", dtype={"order_ids": str})
    assert len(tours) == 1
    tour = tours.iloc[0]
    assert (tour.commit_time, tour.start, tour.setup_end, tour.end, tour.distance, tour.lines) == (
        0, 0, 0, 39, 38, 1)
    assert tour.order_ids == "0"


def test_order_table_has_tour_completion_and_span(outputs):
    order = pd.read_csv(outputs / "facts" / "orders.csv").iloc[0]
    assert (order.order_id, order.release_time, order.due_date, order.lines) == (0, 0, 400, 1)
    tour_id = pd.read_csv(outputs / "facts" / "tours.csv").tour_id.iloc[0]
    assert (order.tour_id, order.tour_start, order.completion, order.picker_id) == (tour_id, 0, 39, 0)
    assert (order.first_pick_start, order.last_pick_end) == (5, 6)
    assert not bool(order.affected)


def test_decision_log_and_service_measures(outputs):
    decisions = json.loads((outputs / "decisions.json").read_text())
    assert decisions["decisions"] == 1
    entry = decisions["decision_log"][0]
    assert entry["chosen_pipeline"] in {row["pipeline"] for row in entry["candidates"]}
    assert entry["chosen_score"] is not None
    assert entry["created_tours"] == 1
    assert entry["cancelled_tours"] == entry["changed_tours"] == 0
    assert entry["solve_seconds"] >= 0
    service = json.loads((outputs / "service.json").read_text())
    assert service["orders"]["all"]["completed_orders"] == 1
    assert service["orders"]["all"]["on_time_rate"] == 1
    assert service["orders"]["affected"]["orders"] == 0
    assert service["orders"]["unaffected"]["orders"] == 1
    assert sum(row["busy_s"] + row["idle_s"] for row in service["pickers"]) > 0


def test_source_marked_order_is_affected(tmp_path, monkeypatch):
    import casim.run as run_module

    original = run_module.build_source

    def marked_source(*args, **kwargs):
        source = original(*args, **kwargs)
        source.affected_order_ids = {0}
        return source

    monkeypatch.setattr(run_module, "build_source", marked_source)
    run(compose_scenario(ROOT / "scenarios" / "minimal", output_dir=tmp_path))
    assert pd.read_csv(tmp_path / "facts" / "orders.csv").iloc[0].affected
    service = json.loads((tmp_path / "service.json").read_text())
    assert service["orders"]["affected"]["completed_orders"] == 1
    assert service["orders"]["unaffected"]["orders"] == 0


def test_service_denominators_and_picker_presence():
    class Tracker:
        busy_time_by_picker = {0: 60}
        idle_time_by_picker = {0: 40}
        absent_time_by_picker = {0: 20}

    rows = [
        {"affected": True, "completion": 90, "due_date": 80},
        {"affected": False, "completion": 70, "due_date": 80},
        {"affected": False, "completion": None, "due_date": 80},
    ]
    measures = service_measures(rows, Tracker())
    assert measures["orders"]["all"]["on_time_rate"] == 0.5
    assert measures["orders"]["all"]["open_orders"] == 1
    assert measures["orders"]["affected"]["total_tardiness_s"] == 10
    assert measures["pickers"][0]["busy_share_of_presence"] == 0.6
    assert measures["pickers"][0]["absent_s"] == 20
