from __future__ import annotations

from casim.events.operational_events import add_orders_hook


def build_sim_hooks() -> list:
    return [add_orders_hook]
