from __future__ import annotations

from casim.events.operational_events import add_orders_hook
from scenarios.scenario_henn.algorithm import HennWakeUp


def add_henn_wakeup_trigger_hook(binding: tuple[str, str]):
    def add_trigger(simulation, domain) -> None:
        simulation.triggers_map[HennWakeUp] = binding

    return add_trigger


def build_sim_hooks(cfg) -> list:
    binding = (str(cfg.data_card.problem_class), "none")
    return [
        add_orders_hook,
        add_henn_wakeup_trigger_hook(binding),
    ]
