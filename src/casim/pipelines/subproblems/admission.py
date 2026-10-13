"""Configured active-tour admission algorithm."""

from ware_ops_algos.algorithms import (
    AdmissionInput, RemainingRouteAdmission, CenterPositionAdmission,
)
from cosy_luigi import CoSyLuigiTaskParameter

from casim.pipelines.problem_based_template import (
    AbstractAdmission, AbstractItemAssignment, dump_pickle, load_pickle,
)
from casim.pipelines.admission_view import to_admission_tour


class RemainingRouteAdmissionNode(AbstractAdmission):
    item_assignment_sol = CoSyLuigiTaskParameter(AbstractItemAssignment)

    def run(self):
        plan = load_pickle(self.input()["instance"]["plan"].path)
        memory = load_pickle(self.input()["instance"]["memory"].path)
        assigned = load_pickle(self.input()["item_assignment_sol"]["item_assignment_sol"].path)
        orders = tuple(assigned.resolved_orders)
        decision = RemainingRouteAdmission().solve(AdmissionInput(
            orders=orders,
            tours=tuple(to_admission_tour(view) for view in plan.tours
                        if view.status.value == "started"
                        and view.accepts.value in ("insert", "revise")),
            candidate_order_ids=frozenset(order.order_id
                                          for order in plan.order_buffer),
            considered_pairs=memory.considered_active_orders,
        ))
        dump_pickle(self.output()["admission_sol"].path, decision)


class CenterPositionAdmissionNode(AbstractAdmission):
    item_assignment_sol = CoSyLuigiTaskParameter(AbstractItemAssignment)

    def run(self):
        plan = load_pickle(self.input()["instance"]["plan"].path)
        memory = load_pickle(self.input()["instance"]["memory"].path)
        articles = load_pickle(self.input()["instance"]["articles"].path)
        assigned = load_pickle(self.input()["item_assignment_sol"]["item_assignment_sol"].path)
        decision = CenterPositionAdmission(articles).solve(AdmissionInput(
            orders=tuple(assigned.resolved_orders),
            tours=tuple(to_admission_tour(view) for view in plan.tours
                        if view.status.value == "started"
                        and view.positioning_mode == "center_relocation"
                        and not view.active_order_ids),
            candidate_order_ids=frozenset(order.order_id
                                          for order in plan.order_buffer),
            considered_pairs=memory.considered_active_orders,
        ))
        dump_pickle(self.output()["admission_sol"].path, decision)
