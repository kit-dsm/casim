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


class WaitForOne(OrderCountWaitingNode):
    min_orders = 1


class WaitForTwo(OrderCountWaitingNode):
    min_orders = 2


class WaitForThree(OrderCountWaitingNode):
    min_orders = 3


class WaitForFour(OrderCountWaitingNode):
    min_orders = 4


# CoSy currently synthesizes task classes without Luigi value parameters.
# These aliases let policy configuration state only the numeric threshold.
WaitFor0 = NoWaitingNode
WaitFor1 = WaitForOne
WaitFor2 = WaitForTwo
WaitFor3 = WaitForThree
WaitFor4 = WaitForFour


class HennWaitingNode(AbstractWaiting):
    def _get_inited_waiter(self) -> Waiting:
        routing_task = self.requires()["scheduling_sol"].requires()["routing_sol"]
        return HennWaiting(routing_task._get_inited_router())


class AnalyticWaitingNode(AbstractWaiting):
    def _get_inited_waiter(self) -> Waiting:
        return AnalyticStochasticWaiting()
