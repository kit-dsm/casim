"""CoSy components for the three current waiting policies."""

from ware_ops_algos.algorithms import NoWaiting, OrderCountWaiting, StartImmediatelyWaiting, HennWaiting, AnalyticStochasticWaiting

from casim.pipelines.problem_based_template import AbstractWaiting


class NoWaitingNode(AbstractWaiting):
    def _get_inited_waiter(self):
        return NoWaiting()


class StartImmediatelyNode(AbstractWaiting):
    def _get_inited_waiter(self):
        return StartImmediatelyWaiting()


class WaitForOne(AbstractWaiting):
    def _get_inited_waiter(self):
        return OrderCountWaiting(1)


class WaitForTwo(AbstractWaiting):
    def _get_inited_waiter(self):
        return OrderCountWaiting(2)


class WaitForThree(AbstractWaiting):
    def _get_inited_waiter(self):
        return OrderCountWaiting(3)


class WaitForFour(AbstractWaiting):
    def _get_inited_waiter(self):
        return OrderCountWaiting(4)


class HennWaitingNode(AbstractWaiting):
    def _get_inited_waiter(self):
        return HennWaiting()


class AnalyticWaitingNode(AbstractWaiting):
    def _get_inited_waiter(self):
        return AnalyticStochasticWaiting()
