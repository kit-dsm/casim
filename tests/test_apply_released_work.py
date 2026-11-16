"""F62 on the live side: orders released by a cancel in the same decision are free for ReplaceSuffix.

Draft tour 1 holds order 10, draft tour 2 holds order 11 and came from a pick list, so
cancelling it returns the pick list {11, 12} to the pick-list buffer. The decision cancels
tour 2 and extends tour 1 by order 11. The live check accepts it; applying it dissolves
the pick list: order 11 goes to tour 1 and order 12 back to the order buffer, so every
order keeps exactly one place.
"""

from types import SimpleNamespace

import pytest
from ware_ops_algos.algorithms import BatchObject
from ware_ops_algos.domain_models import Order
from ware_ops_algos.planning.commands import CancelTour, ReplaceSuffix, ViolationCode
from ware_ops_algos.planning.lifecycle import TourStates

from casim.state.apply import CommandViolation, _take_from_pick_lists, check
from casim.state.order_manager import OrderManager


def _order(order_id):
    return Order(order_id=order_id, order_date=0)


class _Tours:
    def __init__(self, tours):
        self.tours = tours

    def has_tour(self, tour_id):
        return tour_id in self.tours

    def get_tour(self, tour_id):
        return self.tours[tour_id]


def _state():
    orders = OrderManager()
    for order_id in (10, 11, 12, 13):
        orders.add_order_to_history(_order(order_id))
    tours = _Tours({
        1: SimpleNamespace(tour_id=1, status=TourStates.PLANNED, order_numbers=[10]),
        2: SimpleNamespace(tour_id=2, status=TourStates.PLANNED, order_numbers=[11]),
    })
    return SimpleNamespace(order_manager=orders, tour_manager=tours, current_time=0.0)


def _extend(*order_ids):
    return ReplaceSuffix(1, SimpleNamespace(routing_origin=None), tuple(order_ids))


def test_released_orders_are_free_additions():
    check(_state(), (CancelTour(2), _extend(11)))


def test_orders_of_a_tour_that_stays_are_not_free():
    with pytest.raises(CommandViolation) as error:
        check(_state(), (_extend(11),))
    assert error.value.code is ViolationCode.STATE_INFORMATION


def test_orders_nobody_released_are_not_free():
    with pytest.raises(CommandViolation):
        check(_state(), (CancelTour(2), _extend(13)))


def test_taking_part_of_a_pick_list_returns_the_rest_to_the_order_buffer():
    state = _state()
    state.order_manager.add_pick_list_to_buffer(BatchObject(7, [_order(11), _order(12)]))
    _take_from_pick_lists(state, (11,))
    state.order_manager.clear_order_buffer_by_ids([11])
    assert state.order_manager.get_pick_list_buffer() == []
    assert [order.order_id for order in state.order_manager.get_order_buffer()] == [12]
