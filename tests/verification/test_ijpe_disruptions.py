"""The disruption scenarios of the IJPE day (scenarios/ijpe/source.py, DISRUPTIONS).

They read the warehouse's lists, so they need the confidential IJPE data.
"""

from pathlib import Path

import pytest

from casim.events.operational_events import OrderArrival, OrderStreamClosed, TruckDisruption
from casim.sources.wms_export import disruption_orders
from scenarios.ijpe.source import STAGING_MINUTES, ScenarioSource, fixed_departure_hour

DATA_DIR = Path(__file__).resolve().parents[2] / "scenarios" / "ijpe" / "data"
pytestmark = pytest.mark.skipif(not (DATA_DIR / "layout.csv").exists(),
                                reason="confidential IJPE data")
H = 3600.0
SETTINGS = dict(shift_start_s=19800.0, pause_threshold_s=300.0, speed=2306.0, time_per_pick=16.26,
                tour_setup_time=215.0, presence="arrival", due_plan="fixed_dispatch")


def _listed(sheet):
    frame = disruption_orders(DATA_DIR / "cache")
    return {int(order) for order in frame.loc[frame["sheet"] == sheet, "order"]}


def test_a_moved_truck_advances_the_deadline_of_its_listed_orders_only():
    source = ScenarioSource(DATA_DIR, day="2026-04-13", disruption="scenario1", **SETTINGS)
    domain = source.initial_state()
    events = list(source.events())
    [change] = [e for e in events if isinstance(e, TruckDisruption)]
    assert (change.time, change.due_date) == (7.5 * H, 10 * H)
    assert set(change.order_ids) == set(source.affected_order_ids) <= _listed("S1")
    # Every affected order was due later; listed orders due earlier keep their deadline.
    due = {o.order_id: o.due_date for o in domain.orders.orders}
    assert all(due[o] > 10 * H for o in change.order_ids)
    kept = (_listed("S1") & set(due)) - set(change.order_ids)
    assert kept and all(due[o] <= 10 * H for o in kept)
    assert due[change.order_ids[0]] == fixed_departure_hour(
        source.customer_by_order[str(change.order_ids[0])]) * H - STAGING_MINUTES * 60


def test_a_new_store_order_arrives_copied_by_its_transport_units():
    plain = ScenarioSource(DATA_DIR, day="2026-04-15", **SETTINGS)
    source = ScenarioSource(DATA_DIR, day="2026-04-15", disruption="scenario3", **SETTINGS)
    domain = source.initial_state()
    events = list(source.events())
    arrivals = [e for e in events if isinstance(e, OrderArrival) and e.time > SETTINGS["shift_start_s"]]
    assert len(arrivals) == 3                                   # ceil(6 TU / 2.6)
    assert {e.order.order_id for e in arrivals} == set(source.affected_order_ids)
    assert all(e.time == 8.5 * H and e.order.due_date == 10 * H for e in arrivals)
    assert len({len(e.order.order_positions) for e in arrivals}) == 1
    # The stream closes once, after the last arrival; the recorded plan holds
    # one tour per added order, without a recorded picker.
    assert [e.time for e in events if isinstance(e, OrderStreamClosed)] == [8.5 * H]
    tours = domain.information.processes["recorded_plan"].tours
    added = [t for t in tours if t.picker_id == -1]
    assert sorted(t.order_ids[0] for t in added) == sorted(source.affected_order_ids)
    assert len(domain.orders.orders) == len(plain.initial_state().orders.orders)


def test_thursday_orders_move_to_wednesday_and_leave_thursday():
    wednesday = ScenarioSource(DATA_DIR, day="2026-04-15", disruption="scenario4", **SETTINGS)
    wednesday.initial_state()
    moved = set(wednesday.affected_order_ids)
    held = {int(o) for o in wednesday.customer_by_order}
    assert moved and moved <= _listed("S5") and not moved & held
    arrivals = [e for e in wednesday.events() if isinstance(e, OrderArrival)
                and e.order.order_id in moved]
    assert len(arrivals) == len(moved) and {e.time for e in arrivals} == {6.5 * H}
    thursday = ScenarioSource(DATA_DIR, day="2026-04-16", disruption="scenario4_next_day",
                              **SETTINGS)
    left = {int(o) for o in thursday.customer_by_order}
    assert not left & _listed("S5") and not thursday.affected_order_ids


def test_a_disruption_runs_on_its_day_with_the_dispatch_plan():
    with pytest.raises(ValueError, match="runs on"):
        ScenarioSource(DATA_DIR, day="2026-04-14", disruption="scenario1", **SETTINGS)
    with pytest.raises(ValueError, match="fixed_dispatch"):
        ScenarioSource(DATA_DIR, day="2026-04-13", disruption="scenario1",
                       **{**SETTINGS, "due_plan": None})
    with pytest.raises(ValueError, match="unknown"):
        ScenarioSource(DATA_DIR, day="2026-04-13", disruption="scenario9", **SETTINGS)
