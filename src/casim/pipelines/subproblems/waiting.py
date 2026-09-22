"""CoSy components for the three current waiting policies."""

from ware_ops_algos.algorithms import NoWaiting, HennWaiting, AnalyticStochasticWaiting

from casim.pipelines.problem_based_template import AbstractWaiting


class NoWaitingNode(AbstractWaiting):
    def _get_inited_waiter(self):
        return NoWaiting()


class HennWaitingNode(AbstractWaiting):
    def _get_inited_waiter(self):
        return HennWaiting()


class AnalyticWaitingNode(AbstractWaiting):
    def _get_inited_waiter(self):
        return AnalyticStochasticWaiting()
