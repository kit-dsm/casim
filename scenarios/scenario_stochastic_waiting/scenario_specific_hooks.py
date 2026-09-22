"""Study-specific initial events. The forecast is never read from this queue."""

from casim.events.operational_events import OrderStreamClosed, PickerArrival


def add_orders_hook(sim, domain):
    for order in domain.orders.orders:
        sim.add_order(order)
    sim.add_event(OrderStreamClosed(max(order.order_date for order in domain.orders.orders)))


def picker_arrival_hook(sim, domain):
    first_arrival = min(order.order_date for order in domain.orders.orders)
    for picker in domain.resources.resources:
        sim.add_event(PickerArrival(first_arrival, picker.id, picker_available=True))
