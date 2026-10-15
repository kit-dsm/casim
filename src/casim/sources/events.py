"""Initial exogenous events of a replayed instance.

Events at the same time and priority are processed in creation order (the
event id), so each function creates its events in one fixed order and yields
them sorted by time with a stable sort: ties keep that order.
"""

from __future__ import annotations

from collections.abc import Iterator

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.events.operational_events import OrderArrival, OrderStreamClosed, PickerArrival


def _in_time_order(events: list[Event]) -> Iterator[Event]:
    return iter(sorted(events, key=lambda event: event.time))


def _orders(domain: SimWarehouseDomain, closure_s: float | None) -> list[Event]:
    orders = domain.orders.orders
    closure = max(order.order_date for order in orders) if closure_s is None else closure_s
    return [*(OrderArrival(order.order_date, order) for order in orders),
            OrderStreamClosed(closure)]


def _pickers(domain: SimWarehouseDomain, time: float) -> list[Event]:
    return [PickerArrival(time=time, picker_id=resource.id, picker_available=True)
            for resource in domain.resources.resources]


def first_arrival_release(domain: SimWarehouseDomain,
                          closure_s: float | None = None) -> Iterator[Event]:
    """Orders, the stream's closure (default: at the last arrival), then every
    picker at the first arrival."""
    first = min(order.order_date for order in domain.orders.orders)
    return _in_time_order([*_orders(domain, closure_s), *_pickers(domain, first)])


def shift_start_release(domain: SimWarehouseDomain, shift_start_s: float,
                        closure_s: float | None = None) -> Iterator[Event]:
    """Every picker at the shift start, then the orders and the stream's closure."""
    return _in_time_order([*_pickers(domain, shift_start_s), *_orders(domain, closure_s)])
