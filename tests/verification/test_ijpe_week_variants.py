"""The disruption scenarios of the IJPE week (scenarios/ijpe_week/variants).

The rules select orders by the warehouse's lists, so the tests need the
confidential IJPE data.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from casim.events.operational_events import OrderArrival, OrderStreamClosed, TruckDisruption
from casim.run import compose_scenario
from casim.sources.base import build_source
from casim.sources.exogenous import ExogenousSource

ROOT = Path(__file__).resolve().parents[2]
WEEK = ROOT / "scenarios" / "ijpe_week"
TABLES = ROOT / "scenarios" / "ijpe" / "data" / "canonical"
pytestmark = pytest.mark.skipif(not (TABLES / "lines.csv").exists(),
                                reason="confidential IJPE data")
H, DAY = 3600.0, 86400.0
MONDAY, WEDNESDAY, THURSDAY = 2 * DAY, 4 * DAY, 5 * DAY     # epoch Saturday 2026-04-11


def _week(variant=None):
    cfg = compose_scenario(WEEK, [f"variants={variant}"] if variant else [])
    source = build_source(OmegaConf.to_container(cfg.source, resolve=True), WEEK)
    source = ExogenousSource(source, OmegaConf.to_container(cfg.exogenous, resolve=True),
                             WEEK / "data")
    source.initial_state()
    return source, list(source.events())


def _listed(name):
    lines = (TABLES / "lists.csv").read_text(encoding="utf-8").splitlines()[1:]
    return {int(order) for sheet, order in (line.split(",") for line in lines) if sheet == name}


def _arrivals(events):
    return {e.order.order_id: e for e in events if isinstance(e, OrderArrival)}


@pytest.fixture(scope="module")
def plain():
    return _week()


@pytest.mark.parametrize("variant, at, due, sheet, count", [
    ("scenario1", MONDAY + 7.5 * H, MONDAY + 10 * H, "S1", 30),
    ("scenario2", THURSDAY + 10.5 * H, THURSDAY + 11 * H, "S2", 27),
])
def test_a_moved_truck_advances_only_the_later_deadlines_of_its_list(plain, variant, at, due,
                                                                     sheet, count):
    source, events = _week(variant)
    [truck] = [e for e in events if isinstance(e, TruckDisruption)]
    assert (truck.time, truck.due_date, len(truck.order_ids)) == (at, due, count)
    assert set(truck.order_ids) == set(source.affected_order_ids) <= _listed(sheet)
    before = {i: e.order.due_date for i, e in _arrivals(plain[1]).items()}
    assert all(before[i] > due for i in truck.order_ids)
    day = (at // DAY) * DAY
    kept = {i for i in _listed(sheet) & set(before) if day <= before[i] < day + DAY}
    kept -= set(truck.order_ids)
    assert kept and all(before[i] <= due for i in kept)


def test_a_new_store_order_arrives_copied_on_wednesday(plain):
    source, events = _week("scenario3")
    added = set(_arrivals(events)) - set(_arrivals(plain[1]))
    assert added == set(source.affected_order_ids) and len(added) == 3
    arrivals = _arrivals(events)
    assert {(arrivals[i].time, arrivals[i].order.due_date) for i in added} == {
        (WEDNESDAY + 8.5 * H, WEDNESDAY + 10 * H)}
    assert len({len(arrivals[i].order.order_positions) for i in added}) == 1
    assert [e.time for e in events if isinstance(e, OrderStreamClosed)] == [
        e.time for e in plain[1] if isinstance(e, OrderStreamClosed)]


def test_thursday_orders_move_to_wednesday_morning(plain):
    source, events = _week("scenario4")
    moved = set(source.affected_order_ids)
    assert len(moved) == 63 and moved <= _listed("S5")
    before, after = _arrivals(plain[1]), _arrivals(events)
    assert all(THURSDAY <= before[i].time < THURSDAY + DAY for i in moved)
    assert {(after[i].time, after[i].order.due_date) for i in moved} == {
        (WEDNESDAY + 6.5 * H, WEDNESDAY + 10 * H)}
    assert set(after) == set(before)


def test_scenarios_two_and_four_combine():
    source, events = _week("scenario2_4")
    [truck] = [e for e in events if isinstance(e, TruckDisruption)]
    assert truck.time == THURSDAY + 10.5 * H
    moved, _ = _week("scenario4")
    assert set(source.affected_order_ids) == set(moved.affected_order_ids) | set(truck.order_ids)
