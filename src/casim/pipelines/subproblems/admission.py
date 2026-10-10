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
        assigned = load_pickle(self.input()["item_assignment_sol"]["item_assignment_sol"].path)
        orders = tuple(assigned.resolved_orders)
        decision = RemainingRouteAdmission().solve(AdmissionInput(
            orders=orders,
            tours=dynamic.admission_tours,
            candidate_order_ids=dynamic.active_candidate_ids,
            considered_pairs=dynamic.considered_active_orders,
        ))
        dump_pickle(self.output()["admission_sol"].path, decision)
