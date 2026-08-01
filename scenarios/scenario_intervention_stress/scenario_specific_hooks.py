from casim.events.operational_events import FlushRemainingOrders


def add_orders_hook(simulation, domain) -> None:
    """Add the fixture's online order stream to the event queue."""
    orders = sorted(
        domain.orders.orders,
        key=lambda value: (
            float(value.order_date or 0.0),
            int(value.order_id),
        ),
    )
    for order in orders:
        simulation.add_order(order)
    close_time = max(
        (float(order.order_date or 0.0) for order in orders),
        default=0.0,
    )
    simulation.add_event(FlushRemainingOrders(close_time))
