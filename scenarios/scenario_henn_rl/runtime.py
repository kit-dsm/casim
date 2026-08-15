"""Study composition for generated/Henn order-batching learning runs."""

from ware_ops_algos.domain_models import DataCard

from casim.decision_engine.decision_engine import DecisionEngine
from casim.decision_engine.fixed_route_scheduler import FixedRouteScheduler
from casim.envs.order_batching import OrderBatchingEnv
from casim.events.decision_events import PickListDone
from casim.events.operational_events import (
    FlushRemainingOrders,
    OrderArrival,
    PickerIdle,
    add_orders_hook,
)
from casim.simulation_engine.simulation_engine import (
    NbrBatchesCondition,
    NbrOrdersCondition,
    NbrPickersCondition,
    SimulationEngine,
)
from casim.simulation_engine.state_adapter import (
    OrderWindowAdapter,
    ORSPAdapter,
)


def build_environment(*, data_loader, objective_scale=1.0) -> OrderBatchingEnv:
    """Compose the study's controlled OBP and fixed downstream ORSP."""
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
    decision_engine = DecisionEngine(solver_map={"ORSP": solver})
    simulation = SimulationEngine(
        state_adapters={
            "OBP": OrderWindowAdapter(max_pickers=1),
            "ORSP": ORSPAdapter(),
        },
        data_loader=data_loader,
        loader_kwargs={"instance_id": ""},
        triggers_map={
            OrderArrival: "OBP",
            PickerIdle: "OBP",
            FlushRemainingOrders: "OBP",
            PickListDone: "ORSP",
        },
        conditions_map={
            "OBP": [NbrPickersCondition(1), NbrOrdersCondition(1)],
            "ORSP": [NbrPickersCondition(1), NbrBatchesCondition(1)],
        },
        event_loggers=[],
        completion_mode="drain",
        horizon_time=None,
        show_progress=False,
    )
    return OrderBatchingEnv(
        simulation,
        decision_engine,
        controlled_problem="OBP",
        reset_hooks=(add_orders_hook,),
        objective_scale=objective_scale,
    )
