"""Configured active-tour admission algorithm."""

from ware_ops_algos.algorithms import (
    AdmissionInput, RemainingRouteAdmission,
)
from cosy_luigi import CoSyLuigiTaskParameter

from casim.pipelines.problem_based_template import (
    AbstractAdmission, AbstractItemAssignment, dump_pickle, load_pickle,
)


class RemainingRouteAdmissionNode(AbstractAdmission):
    item_assignment_sol = CoSyLuigiTaskParameter(AbstractItemAssignment)

    def run(self):
        dynamic = load_pickle(self.input()["instance"]["dynamic_warehouse_info"].path)
        resources = load_pickle(self.input()["instance"]["resources"].path)
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
        dump_pickle(self.output()["admission_sol"].path, decision)
