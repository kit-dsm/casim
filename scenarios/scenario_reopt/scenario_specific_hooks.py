from __future__ import annotations

from casim.events.operational_events import FlushRemainingOrders


def add_orders_hook(simulation, domain) -> None:
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


def build_sim_hooks() -> list:
    return [add_orders_hook]
