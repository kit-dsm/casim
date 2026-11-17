"""NbrOrdersCondition counts the work open to the decision (F67, Task 4.15).

Open work is the order buffer and the orders of planned tours the decision
may rebuild. A waiting batch that sits in a planned tour while the buffer is
empty still counts, so the decision at the stream's end solves and its
waiting rule can release the batch.
"""

from types import SimpleNamespace

from ware_ops_algos.planning.lifecycle import Accepts, TourStates

from casim.simulation_engine.conditions import NbrOrdersCondition


def _snapshot(buffer=0, tours=(), input_closed=False):
    plan = SimpleNamespace(
        order_buffer=tuple(range(buffer)),
        tours=tuple(SimpleNamespace(status=status, accepts=accepts,
                                    batch=SimpleNamespace(order_numbers=list(range(n))))
                    for status, accepts, n in tours))
    context = SimpleNamespace(input_closed=input_closed)
    return SimpleNamespace(dynamic_warehouse_info=SimpleNamespace(plan=plan, context=context))


def test_buffered_orders_count():
    assert NbrOrdersCondition(2).get_decision(_snapshot(buffer=2), None)
    assert not NbrOrdersCondition(3).get_decision(_snapshot(buffer=2), None)


def test_orders_of_a_rebuildable_planned_tour_count():
    snapshot = _snapshot(tours=[(TourStates.PLANNED, Accepts.REBUILD, 2)])
    assert NbrOrdersCondition(1).get_decision(snapshot, None)
    assert NbrOrdersCondition(2).get_decision(snapshot, None)
    assert not NbrOrdersCondition(3).get_decision(snapshot, None)


def test_buffer_and_planned_tours_add_up():
    snapshot = _snapshot(buffer=1, tours=[(TourStates.PLANNED, Accepts.REBUILD, 2)])
    assert NbrOrdersCondition(3).get_decision(snapshot, None)


def test_committed_frozen_and_started_tours_do_not_count():
    snapshot = _snapshot(tours=[(TourStates.SCHEDULED, Accepts.REBUILD, 2),
                                (TourStates.PLANNED, Accepts.NONE, 2),
                                (TourStates.STARTED, Accepts.INSERT, 2)])
    assert not NbrOrdersCondition(1).get_decision(snapshot, None)


def test_allow_when_done_counts_open_work():
    snapshot = _snapshot(tours=[(TourStates.PLANNED, Accepts.REBUILD, 1)], input_closed=True)
    assert NbrOrdersCondition(5, allow_when_done=True).get_decision(snapshot, None)
    assert not NbrOrdersCondition(5, allow_when_done=True).get_decision(
        _snapshot(input_closed=True), None)
