"""Configured active-tour admission algorithm."""

from ware_ops_algos.algorithms import (
    AdmissionInput, RemainingRouteAdmission, CenterPositionAdmission,
)
from cosy_luigi import CoSyLuigiTaskParameter

from casim.pipelines.problem_based_template import (
    AbstractAdmission, AbstractItemAssignment, dump_pickle, load_pickle,
)
from casim.pipelines.admission_view import admitting_tours


class _AdmissionNode(AbstractAdmission):
    item_assignment_sol = CoSyLuigiTaskParameter(AbstractItemAssignment)

    def _admission_input(self) -> AdmissionInput:
        plan = load_pickle(self.input()["instance"]["plan"].path)
        memory = load_pickle(self.input()["instance"]["memory"].path)
        context = load_pickle(self.input()["instance"]["context"].path)
        assigned = load_pickle(self.input()["item_assignment_sol"]["item_assignment_sol"].path)
        return AdmissionInput(
            orders=tuple(assigned.resolved_orders),
            tours=admitting_tours(plan),
            candidate_order_ids=frozenset(order.order_id for order in plan.order_buffer),
            considered_pairs=memory.considered_active_orders,
            input_closed=context.input_closed,
        )


class RemainingRouteAdmissionNode(_AdmissionNode):
    def run(self):
        decision = RemainingRouteAdmission().solve(self._admission_input())
        dump_pickle(self.output()["admission_sol"].path, decision)


class CenterPositionAdmissionNode(_AdmissionNode):
    def run(self):
        articles = load_pickle(self.input()["instance"]["articles"].path)
        decision = CenterPositionAdmission(articles).solve(self._admission_input())
        dump_pickle(self.output()["admission_sol"].path, decision)
