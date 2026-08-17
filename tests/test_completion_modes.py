from pathlib import Path
from types import SimpleNamespace

from ware_ops_algos.algorithms import BatchObject, GreedyItemAssignment
from ware_ops_algos.domain_models import Order

from casim.simulation_engine.simulation_engine import SimulationEngine
from casim.events.operational_events import OrderArrival
from casim.state.order_manager import OrderManager
from scenarios.scenario_reopt.loader import ReoptDataLoader


ROOT = Path(__file__).parents[1]
SCENARIO = ROOT / "scenarios" / "scenario_reopt"


def _engine(hook):
    domain = ReoptDataLoader(SCENARIO).load(
        SCENARIO / "data" / "paper_example.json"
    )
    engine = SimulationEngine(
        state_adapters={},
        triggers_map={},
        conditions_map={},
        loader_kwargs={},
        data_loader=SimpleNamespace(load=lambda **_: domain),
        event_loggers=[],
        completion_mode="horizon",
        horizon_time=2,
    )
    engine.reset(hooks=[hook])
    return engine


def test_horizon_reports_orders_that_have_not_arrived():
    def add_future_order(engine, domain):
        order = domain.orders.orders[0]
        order.order_date = 5
        engine.add_order(order)

    engine = _engine(add_future_order)
    done, _ = engine.run()

    assert done
    assert engine.state.completion_reason == "horizon_complete"
    assert engine.state.current_time == 2
    pending_order_arrival_ids = sorted(
        event.order_id
        for event in engine.events
        if isinstance(event, OrderArrival)
    )
    assert pending_order_arrival_ids == [1]


def test_horizon_reports_prebatched_work():
    def add_prebatched_work(engine, domain):
        order = GreedyItemAssignment(
            engine.state.storage_manager.planning_snapshot()
        ).solve(
            [domain.orders.orders[0]]
        ).resolved_orders[0]
        engine.state.order_manager.add_pick_list_to_buffer(
            BatchObject(1, [order])
        )

    engine = _engine(add_prebatched_work)
    done, _ = engine.run()

    assert done
    assert engine.state.completion_reason == "horizon_complete"
    unfinished = engine.state.unfinished_work()
    assert unfinished["buffered_batch_order_ids"] == [1]


def test_rescheduled_order_rejects_its_stale_arrival_version():
    manager = OrderManager()
    order = Order(order_id="future", order_date=10, due_date=20)
    assert manager.register_order(order) == 0

    replacement = manager.reschedule_unreleased_orders(
        from_due=20,
        new_due=15,
        new_release=5,
    )

    assert replacement == [("future", 5.0, 1)]
    assert manager.release_order("future", 0) is None
    assert manager.release_order("future", 1) is order
