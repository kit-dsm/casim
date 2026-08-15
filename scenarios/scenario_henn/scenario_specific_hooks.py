from __future__ import annotations

from casim.events.operational_events import add_orders_hook
from scenarios.scenario_henn.algorithm import HennWakeUp


def add_henn_wakeup_trigger_hook(problem_class: str):
    def add_trigger(simulation, domain) -> None:
        simulation.triggers_map[HennWakeUp] = problem_class

    return add_trigger


def build_sim_hooks(cfg) -> list:
    return [
        add_orders_hook,
        add_henn_wakeup_trigger_hook(cfg.data_card.problem_class),
    ]
