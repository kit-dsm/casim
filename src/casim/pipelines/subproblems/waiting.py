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

    def _single_order_service_times(self, scheduled, picker):
        if not scheduled.jobs:
            return None
        routing_task = self.requires()["scheduling_sol"].requires()["routing_sol"]
        router = routing_task._get_inited_router()
        services = {}
        for order in scheduled.jobs[0].job.route.batch.orders:
            router.reset_parameters()
            route = router.solve(order.pick_positions).route
            services[order.order_id] = (
                picker.tour_setup_time + route.distance / picker.speed
                + sum(p.in_store for p in order.pick_positions) * picker.time_per_pick
            )
        return services


class AnalyticWaitingNode(AbstractWaiting):
    def _get_inited_waiter(self):
        return AnalyticStochasticWaiting()
