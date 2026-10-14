"""CoSy nodes for the warehouse waiting subproblem."""

from ware_ops_algos.algorithms import (
    AnalyticStochasticWaiting, HennWaiting, NoWaiting, OrderCountWaiting,
    Waiting,
)

from casim.pipelines.problem_based_template import AbstractWaiting


class NoWaitingNode(AbstractWaiting):
    def _get_inited_waiter(self) -> Waiting:
        return NoWaiting()


class OrderCountWaitingNode(AbstractWaiting):
    """CoSy component shared by the configured order-count policies."""

    min_orders: int

    def _get_inited_waiter(self) -> Waiting:
        return OrderCountWaiting(self.min_orders)


class HennWaitingNode(AbstractWaiting):
    def _get_inited_waiter(self) -> Waiting:
        routing_task = self.requires()["scheduling_sol"].requires()["routing_sol"]
        return HennWaiting(routing_task._get_inited_router())


class AnalyticWaitingNode(AbstractWaiting):
    def _get_inited_waiter(self) -> Waiting:
        return AnalyticStochasticWaiting()
