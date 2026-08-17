"""Study composition for generated/Henn order-batching learning runs."""

from ware_ops_algos.domain_models import DataCard

from casim.decision_engine import DecisionEngine
from casim.solvers.fixed_route_scheduler import FixedRouteScheduler
from scenarios.scenario_henn_rl.order_batching import OrderBatchingEnv
from casim.events.decision_events import PickListDone
from casim.events.operational_events import (
    FlushRemainingOrders,
    OrderArrival,
    PickerIdle,
    add_orders_hook,
)
from casim.simulation_engine.simulation_engine import SimulationEngine
from casim.simulation_engine.state_adapter import StateAdapter


def build_environment(*, data_loader, objective_scale=1.0) -> OrderBatchingEnv:
    """Compose the study's controlled OBP and fixed downstream ORSP."""
    downstream_binding = ("ORSP", "none")
    solver = FixedRouteScheduler(problem_class="ORSP")
    solver.prepare(
        DataCard(
            name="structured_batching",
            problem_class="ORSP",
            objective="distance",
            layout={},
            articles={},
            orders={},
            resources={},
            storage={},
            warehouse_info={},
        )
    )
    decision_engine = DecisionEngine(solver_map={downstream_binding: solver})
    simulation = SimulationEngine(
        state_adapters={
            ("OBP", "none"): StateAdapter(
                problem_class="OBP",
                replanning="none",
                orders={"source": "buffered"},
                resources={
                    "source": "dispatchable",
                    "scope": "trigger_if_present",
                },
            ),
            downstream_binding: StateAdapter(
                problem_class="ORSP",
                replanning="none",
                batches={"source": "buffered"},
                resources={"source": "nonactive"},
            ),
        },
        data_loader=data_loader,
        loader_kwargs={"instance_id": ""},
        triggers_map={
            OrderArrival: ("OBP", "none"),
            PickerIdle: ("OBP", "none"),
            FlushRemainingOrders: ("OBP", "none"),
            PickListDone: downstream_binding,
        },
        conditions_map={
            ("OBP", "none"): {"pickers": 1, "orders": 1},
            downstream_binding: {"pickers": 1, "batches": 1},
        },
        event_loggers=[],
        completion_mode="drain",
        horizon_time=None,
    )
    return OrderBatchingEnv(
        simulation,
        decision_engine,
        controlled_problem="OBP",
        reset_hooks=(add_orders_hook,),
        objective_scale=objective_scale,
    )
