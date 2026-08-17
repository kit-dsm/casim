"""Run CASIM until one externally controlled order-batching decision."""

from ware_ops_algos.algorithms import (
    BatchObject,
    BatchingSolution,
    GreedyItemAssignment,
)
from ware_ops_algos.algorithms.batching.batching_utils import CapacityChecker


class OrderBatchingEnv:
    """Expose one batching snapshot and commit selected order IDs.

    The supplied simulation and decision engine are already configured. CASIM
    solves every non-controlled problem normally; this object only pauses at
    the configured batching problem.
    """

    def __init__(
        self,
        simulation,
        decision_engine,
        *,
        controlled_problem="OBP",
        reset_hooks=(),
        objective_scale=1.0,
    ):
        self.simulation = simulation
        self.decision_engine = decision_engine
        self.controlled_problem = str(controlled_problem)
        self.reset_hooks = tuple(reset_hooks)
        self.objective_scale = float(objective_scale)
        self.current = None
        self.order_count = 0
        self.episode_horizon = 1.0
        self._previous_flow_time = 0.0

    def reset(self, instance_id):
        """Reset one simulation realization and return `(snapshot, orders)`."""
        self.simulation.loader_kwargs["instance_id"] = str(instance_id)
        domain = self.simulation.reset(hooks=list(self.reset_hooks))
        self.order_count = len(domain.orders.orders)
        last_arrival = max(
            (float(order.order_date or 0.0) for order in domain.orders.orders),
            default=0.0,
        )
        self.episode_horizon = max(1.0, last_arrival + 24 * 60 * 60)
        self._previous_flow_time = 0.0
        if self._advance():
            raise RuntimeError("Simulation completed before its first batch decision")
        return self.current

    def _advance(self):
        while True:
            done, snapshot = self.simulation.run()
            if done:
                self.current = None
                return True
            if snapshot.problem_class == self.controlled_problem:
                assignment = GreedyItemAssignment(snapshot.storage).solve(
                    snapshot.orders.orders
                )
                self.current = (snapshot, tuple(assignment.resolved_orders))
                return False
            events, solution = self.decision_engine.on_trigger(snapshot)
            self.simulation.step(events)

    def step(self, order_ids):
        """Commit one feasible batch and advance to the next batching trigger."""
        if self.current is None:
            raise RuntimeError("No batching decision is waiting")
        snapshot, orders = self.current
        requested = {int(order_id) for order_id in order_ids}
        selected = [
            order for order in orders if int(order.order_id) in requested
        ]
        if not selected or len(selected) != len(requested):
            raise ValueError("Batch contains unavailable order IDs")
        picker = snapshot.resources.resources[0]
        if not CapacityChecker(picker.pick_cart, snapshot.articles).orders_fit(
            selected
        ):
            raise ValueError("Batch exceeds picker cart capacity")
        solution = BatchingSolution(
            algo_name="ExternalOrderBatching",
            batches=[BatchObject(batch_id=0, orders=selected)],
        )
        events, committed = self.decision_engine.commit(snapshot, solution)
        self.simulation.step(events)
        done = self._advance()
        current_flow_time = self.simulation.state.tracker.accrued_flow_time(
            self.simulation.state.current_time
        )
        reward = -(
            current_flow_time - self._previous_flow_time
        ) / (max(1, self.order_count) * self.objective_scale)
        self._previous_flow_time = current_flow_time
        return self.current, reward, done
