"""Exogenous rules on a canonical order stream (casim.sources.exogenous, order_stream)."""

import json

import pytest

from casim.events.operational_events import (
    OrderArrival, OrderStreamClosed, PickerArrival, PickerDeparture, TruckDisruption,
)
from casim.sources.exogenous import ExogenousSource, slot, validate
from casim.sources.order_stream import OrderStreamSource

H = 3600.0
A, B = (7500.0, 1200.0), (11350.0, 2000.0)      # pick nodes of the drawing below


@pytest.fixture
def data(tmp_path):
    (tmp_path / "layout.csv").write_text(
        "-90;-90;-90;-90;-90;-90\n1001;-1;1002;1003;-2;1004\n"
        "1005;-1;1006;1007;-2;1008\n-90;-90;-90;-90;-90;-90\n", encoding="utf-8")
    (tmp_path / "orders.csv").write_text(
        "order_id,release,customer\n"
        "1,2026-04-13 05:30,north\n2,2026-04-13 05:30,south\n"
        "3,2026-04-14 05:30,north\n4,2026-04-14 05:30,east\n", encoding="utf-8")
    (tmp_path / "lines.csv").write_text(
        "order_id,article,quantity,x,y\n"
        f"1,10,5,{A[0]},{A[1]}\n1,11,30,{B[0]},{B[1]}\n2,10,2,{A[0]},{A[1]}\n"
        f"3,12,7,{B[0]},{B[1]}\n4,10,1,{A[0]},{A[1]}\n", encoding="utf-8")
    (tmp_path / "lists.csv").write_text("list,order_id\nS1,1\nS1,2\nS5,3\nS5,99\n",
                                        encoding="utf-8")
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "S3.csv").write_text(
        f"article,quantity,x,y\n20,4,{B[0]},{B[1]}\n10,3,{A[0]},{A[1]}\n", encoding="utf-8")
    (tmp_path / "fit.json").write_text(json.dumps({
        "quantity_bounds": [6, 12, 24], "line_s": [[10.0], [20.0], [30.0], [40.0]],
        "setup_s": [100.0, 300.0], "order_sd": 0.0}), encoding="utf-8")
    return tmp_path


def _source(data):
    return OrderStreamSource(data, epoch="2026-04-13",
                             pickers={"count": 3, "speed": 1000.0, "time_per_pick": 15.0,
                                      "tour_setup_time": 200.0})


def test_the_stream_releases_each_order_at_its_time_and_knows_none_before(data):
    source = _source(data)
    domain = source.initial_state()
    events = list(source.events())
    assert domain.orders.orders == [] and len(domain.resources.resources) == 3
    assert [(e.order.order_id, e.time) for e in events if isinstance(e, OrderArrival)] == [
        (1, 5.5 * H), (2, 5.5 * H), (3, 29.5 * H), (4, 29.5 * H)]
    assert [e.time for e in events if isinstance(e, OrderStreamClosed)] == [29.5 * H]
    assert source.order_attributes[4] == {"customer": "east"}
    tours = domain.information.processes["recorded_plan"].tours
    assert [line.pick_node for line in tours[0].lines] == [A, B]


def test_rules_move_inject_set_due_dates_and_roster_in_their_phases(data):
    rules = {
        # Listed out of phase order: the phases decide, not the listing.
        "truck": {"kind": "due_change", "orders": {"list": "S1"}, "at": "2026-04-13 07:30",
                  "due": "2026-04-13 10:00"},
        "slots": {"kind": "dispatch_slots", "hours": [10, 12, 14, 16], "staging_minutes": 30,
                  "attribute": "customer"},
        "roster": {"kind": "roster", "days": ["2026-04-13", "2026-04-14"],
                   "shift": ["05:30", "14:30"], "pickers": [2, 1], "breaks": [["08:30", "09:00"]]},
        "move": {"kind": "order_move", "orders": {"list": "S5"}, "at": "2026-04-13 06:30",
                 "due": "2026-04-13 10:00"},
        "extra": {"kind": "order_injection", "template": "S3", "copies": 2, "first_id": 900,
                  "at": "2026-04-13 08:30", "due": "2026-04-13 10:00"},
        "times": {"kind": "process_times", "fit": "fit.json", "seed": 3, "first_setup_s": 215},
    }
    source = ExogenousSource(_source(data), rules, data)
    domain = source.initial_state()
    events = list(source.events())
    assert [e.time for e in events] == sorted(e.time for e in events)
    arrivals = {e.order.order_id: e for e in events if isinstance(e, OrderArrival)}
    # Order 3 moved from Tuesday to Monday 06:30 with its new due date; 99 is not in the stream.
    assert arrivals[3].time == 6.5 * H and arrivals[3].order.due_date == 10 * H
    # Two copies of the template arrive at 08:30; their articles are stored.
    assert arrivals[900].time == arrivals[901].time == 8.5 * H
    assert len(arrivals[900].order.order_positions) == 2
    assert 20 in {a.article_id for a in domain.articles.articles}
    # Slots: the remaining orders get their store's slot minus staging on their day.
    due = {i: e.order.due_date for i, e in arrivals.items()}
    hours = [10, 12, 14, 16]
    assert due[4] == 24 * H + hours[slot("east", 4)] * H - 1800
    # The truck takes only the listed orders due later than 10:00.
    [truck] = [e for e in events if isinstance(e, TruckDisruption)]
    expected = [i for i in (1, 2) if due[i] > 10 * H]
    assert (truck.time, truck.due_date, truck.order_ids) == (7.5 * H, 10 * H, expected)
    assert source.affected_order_ids == frozenset({3, 900, 901, *expected})
    # Roster: two pickers on Monday, one on Tuesday, each with the break and the shift end.
    monday = [e for e in events if isinstance(e, PickerArrival) and e.time < 24 * H]
    assert sorted(e.picker_id for e in monday) == [0, 1]
    tuesday = [e for e in events if isinstance(e, PickerArrival) and e.time >= 24 * H]
    assert [e.picker_id for e in tuesday] == [0]
    leaves = [e for e in events if isinstance(e, PickerDeparture) and e.picker_id == 0]
    assert [(e.time, e.duration) for e in leaves] == [
        (8.5 * H, 1800.0), (14.5 * H, None), (32.5 * H, 1800.0), (38.5 * H, None)]
    assert [e.time for e in events if isinstance(e, OrderStreamClosed)] == [29.5 * H]
    # Times: the truth covers the injected orders; the pickers plan with the means.
    assert domain.truth.setup_s[0][0] == 215 and (900, 20, 4) in domain.truth.line_s
    assert domain.resources.resources[0].tour_setup_time == 200.0


def test_unknown_kinds_and_fields_are_refused():
    with pytest.raises(ValueError, match="unknown kind"):
        validate({"x": {"kind": "earthquake"}})
    with pytest.raises(ValueError, match="lacks"):
        validate({"x": {"kind": "roster", "days": []}})
    with pytest.raises(ValueError, match="unknown"):
        validate({"x": {"kind": "order_move", "orders": {"ids": [1]}, "at": "2026-04-13 06:00",
                        "when": 1}})
