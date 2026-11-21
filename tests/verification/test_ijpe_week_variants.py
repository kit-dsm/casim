"""The disruption scenarios of the IJPE week (scenarios/ijpe_week/variants, plan P2.1).

The variants select whole stores from the lists that scripts/ijpe/disruptions.py
writes into the confidential canonical tables, so the tests need the IJPE data.
"""

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from casim.events.operational_events import OrderArrival, OrderStreamClosed, TruckDisruption
from casim.run import compose_scenario
from casim.sources.base import build_source
from scripts.ijpe.disruptions import AREA_SHARE, S3, STORE_SCENARIOS, TE_M3, slot_hour

ROOT = Path(__file__).resolve().parents[2]
WEEK = ROOT / "scenarios" / "ijpe_week"
TABLES = ROOT / "scenarios" / "ijpe" / "data" / "canonical"
pytestmark = pytest.mark.skipif(
    not (TABLES / "templates" / "S3_v1_fixed.csv").exists(),
    reason="confidential IJPE data with the P2.1 selections (python -m scripts.ijpe.disruptions)")
H, DAY = 3600.0, 86400.0
MONDAY, WEDNESDAY, THURSDAY = 2 * DAY, 4 * DAY, 5 * DAY     # epoch Saturday 2026-04-11


def _week(variant=None):
    cfg = compose_scenario(WEEK, [f"variants={variant}"] if variant else [])
    source = build_source(OmegaConf.to_container(cfg.source, resolve=True), WEEK)
    source.initial_state()
    return source, list(source.events())


def _arrivals(events):
    return {e.order.order_id: e for e in events if isinstance(e, OrderArrival)}


def _orders():
    return pd.read_csv(TABLES / "orders.csv")


def _listed(name):
    lists = pd.read_csv(TABLES / "lists.csv")
    return set(lists.loc[lists["list"] == name, "order_id"].astype(int))


@pytest.fixture(scope="module")
def plain():
    return _week()


def _whole_stores_of_the_slot(ids, scenario):
    """The orders are every order of their stores on the day, from the scenario's slot."""
    day, hour, _ = STORE_SCENARIOS[scenario]
    orders = _orders()
    today = orders[orders["day"] == day]
    stores = set(today.loc[today["order_id"].isin(ids), "customer"])
    assert set(today.loc[today["customer"].isin(stores), "order_id"]) == set(ids)
    assert set(slot_hour(pd.Series(sorted(stores)))) == {hour}


@pytest.mark.parametrize("variant, scenario, at, due", [
    ("scenario1", "S1", MONDAY + 7.5 * H, MONDAY + 9.5 * H),
    ("scenario2", "S2", THURSDAY + 9.5 * H, THURSDAY + 10.5 * H),
    ("scenario2_short", "S2", THURSDAY + 10.5 * H, THURSDAY + 10.5 * H),
])
def test_an_advanced_departure_moves_the_deadline_of_whole_stores(plain, variant, scenario, at,
                                                                  due):
    source, events = _week(variant)
    [truck] = [e for e in events if isinstance(e, TruckDisruption)]
    assert (truck.time, truck.due_date) == (at, due)
    assert set(truck.order_ids) == set(source.affected_order_ids) == _listed(f"{scenario}_v1_fixed")
    before = {i: e.order.due_date for i, e in _arrivals(plain[1]).items()}
    # The 14:00 slot is due at 13:30, so the new deadline advances every selected order.
    assert {before[i] for i in truck.order_ids} == {(at // DAY) * DAY + 13.5 * H}
    _whole_stores_of_the_slot(truck.order_ids, scenario)


def test_a_new_store_order_of_the_target_volume_arrives_on_wednesday(plain):
    source, events = _week("scenario3")
    added = set(_arrivals(events)) - set(_arrivals(plain[1]))
    assert added == set(source.affected_order_ids) == {2604159001}
    [order] = [_arrivals(events)[i] for i in added]
    assert (order.time, order.order.due_date) == (WEDNESDAY + 8.5 * H, WEDNESDAY + 9.5 * H)
    template = pd.read_csv(TABLES / "templates" / "S3_v1_fixed.csv")
    assert len(order.order.order_positions) == len(template)
    target = S3[1] * AREA_SHARE * TE_M3
    assert abs(template["volume"].sum() / target - 1) <= 0.15
    assert [e.time for e in events if isinstance(e, OrderStreamClosed)] == [
        e.time for e in plain[1] if isinstance(e, OrderStreamClosed)]


def test_thursday_stores_of_the_ten_oclock_slot_move_to_wednesday(plain):
    source, events = _week("scenario4")
    moved = set(source.affected_order_ids)
    assert moved == _listed("S4_v1_fixed")
    before, after = _arrivals(plain[1]), _arrivals(events)
    assert all(THURSDAY <= before[i].time < THURSDAY + DAY for i in moved)
    assert {(after[i].time, after[i].order.due_date) for i in moved} == {
        (WEDNESDAY + 6.5 * H, WEDNESDAY + 9.5 * H)}
    assert set(after) == set(before)
    _whole_stores_of_the_slot(moved, "S4")


def test_scenarios_two_and_four_combine_without_a_shared_order():
    source, events = _week("scenario2_4")
    [truck] = [e for e in events if isinstance(e, TruckDisruption)]
    moved, _ = _week("scenario4")
    assert not set(moved.affected_order_ids) & set(truck.order_ids)
    assert set(source.affected_order_ids) == set(moved.affected_order_ids) | set(truck.order_ids)


@pytest.mark.parametrize("scenario", sorted(STORE_SCENARIOS))
def test_random_selections_meet_the_target_within_fifteen_percent(scenario):
    day, _, mail = STORE_SCENARIOS[scenario]
    lists = pd.read_csv(TABLES / "lists.csv")
    volume = pd.read_csv(TABLES / "lines.csv").groupby("order_id")["volume"].sum() / TE_M3
    for v in (0.5, 1, 2):
        names = [n for n in lists["list"].unique() if n.startswith(f"{scenario}_v{v:g}_r")]
        assert names
        for name in names:
            te = volume[lists.loc[lists["list"] == name, "order_id"]].sum()
            assert abs(te / (v * mail * AREA_SHARE) - 1) <= 0.15, name
