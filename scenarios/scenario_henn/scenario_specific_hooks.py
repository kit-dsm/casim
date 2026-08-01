from __future__ import annotations

from casim.events.operational_events import FlushRemainingOrders
from scenarios.scenario_henn.algorithm import HennWakeUp


def add_orders_hook(simulation, domain) -> None:
    """Install the immutable arrival stream on every simulation reset."""
    orders = sorted(
        domain.orders.orders,
        key=lambda value: (value.order_date, value.order_id),
    )
    for order in orders:
        simulation.add_order(order)
    close_time = max(
        (float(order.order_date or 0.0) for order in orders),
        default=0.0,
    )
    simulation.add_event(FlushRemainingOrders(close_time))


def add_henn_wakeup_trigger_hook(problem_class: str):
    def add_trigger(simulation, domain) -> None:
        simulation.triggers_map[HennWakeUp] = problem_class

    return add_trigger


def build_sim_hooks(cfg) -> list:
    return [
        add_orders_hook,
        add_henn_wakeup_trigger_hook(cfg.data_card.problem_class),
    ]
