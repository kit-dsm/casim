"""The facts tables of a run, on the hand-derived minimal scenario (scenarios/minimal/README.md).

The tour leaves the depot at t = 0 over (1, 0) and (2, 0) up aisle 2 to the
pick at (2, 3): 1 + 1 + 3 = 5 unit edges at speed 1, so the walk is 5 long and
ends at t = 5; the pick takes 1 s (5 to 6). The tour walks 38 units in all and
ends at 38 + 1 = 39, which completes order 0 (arrival 0, due 400). There is no
setup, so the tour's start and setup end coincide; the one decision commits
it at t = 0.
"""

from pathlib import Path

import pandas as pd
import pytest

from casim.run import compose_scenario, run

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
