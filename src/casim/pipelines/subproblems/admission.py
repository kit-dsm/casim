"""Adapt an admission decision to the existing CoSy batching output."""

from ware_ops_algos.algorithms import (
    AdmissionInput, BatchObject, BatchingSolution, RemainingRouteAdmission,
)
from cosy_luigi import CoSyLuigiTaskParameter

from casim.pipelines.problem_based_template import (
    AbstractBatching, AbstractItemAssignment, dump_pickle, load_pickle,
)


class RemainingRouteAdmissionNode(AbstractBatching):
    item_assignment_sol = CoSyLuigiTaskParameter(AbstractItemAssignment)

    def run(self):
        dynamic = load_pickle(self.input()["instance"]["dynamic_warehouse_info"].path)
        resources = self._get_resources()
        if len(resources.resources) != 1 or dynamic.active_tour_id is None:
            raise ValueError("Remaining-route admission needs one active picker")
        assigned = load_pickle(self.input()["item_assignment_sol"]["item_assignment_sol"].path)
        orders = tuple(assigned.resolved_orders)
        decision = RemainingRouteAdmission().solve(AdmissionInput(
            orders=orders,
            pick_cart=resources.resources[0].pick_cart,
            active_order_ids=dynamic.active_order_ids,
            candidate_order_ids=dynamic.active_candidate_ids,
            remaining_route=dynamic.remaining_route_positions,
            occupied_bins=dynamic.occupied_bins,
        ))
        by_id = {order.order_id: order for order in orders}
        active = [order for order in orders if order.order_id in dynamic.active_order_ids]
        accepted = [by_id[order_id] for order_id in decision.accepted_order_ids]
        batches = [BatchObject(batch_id=0, orders=active + accepted)] if accepted else []
        dump_pickle(self.output()["batching_sol"].path, BatchingSolution(
            algo_name=decision.algo_name,
            execution_time=decision.execution_time,
            batches=batches,
        ))
