"""Seed the operational stream for the configured paper demonstration."""

from casim.events.operational_events import OrderStreamClosed, PickerArrival


def seed_orders(sim, domain):
    for order in domain.orders.orders:
        sim.add_order(order)
    sim.add_event(OrderStreamClosed(max(order.order_date for order in domain.orders.orders)))


def seed_pickers(sim, domain):
    shift_start = float(sim.data_loader.cfg.simulation.shift_start_s)
    for picker in domain.resources.resources:
        sim.add_event(PickerArrival(shift_start, picker.id, picker_available=True))
