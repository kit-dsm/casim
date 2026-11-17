"""Configured active-tour admission algorithm."""

from ware_ops_algos.algorithms import (
    AdmissionInput, RemainingRouteAdmission, CenterPositionAdmission,
)
from cosy_luigi import CoSyLuigiTaskParameter

from casim.pipelines.problem_based_template import (
    AbstractAdmission, AbstractItemAssignment, dump_pickle, load_pickle,
)
from casim.pipelines.tour_filters import admitting_tours


def _admission_input(task: AbstractAdmission) -> AdmissionInput:
    """The admission input of a decision: buffered orders and the admitting tours."""
    inputs = task.input()
    plan = load_pickle(inputs["instance"]["plan"].path)
    memory = load_pickle(inputs["instance"]["memory"].path)
    context = load_pickle(inputs["instance"]["context"].path)
    assigned = load_pickle(inputs["item_assignment_sol"]["item_assignment_sol"].path)
    return AdmissionInput(
        orders=tuple(assigned.resolved_orders),
        tours=admitting_tours(plan),
        candidate_order_ids=frozenset(order.order_id for order in plan.order_buffer),
        considered_pairs=memory.considered_active_orders,
        input_closed=context.input_closed,
    )


class RemainingRouteAdmissionNode(AbstractAdmission):
    item_assignment_sol = CoSyLuigiTaskParameter(AbstractItemAssignment)

    def run(self):
        decision = RemainingRouteAdmission().solve(_admission_input(self))
        dump_pickle(self.output()["admission_sol"].path, decision)


class CenterPositionAdmissionNode(AbstractAdmission):
    item_assignment_sol = CoSyLuigiTaskParameter(AbstractItemAssignment)

    def run(self):
        articles = load_pickle(self.input()["instance"]["articles"].path)
        decision = CenterPositionAdmission(articles).solve(_admission_input(self))
        dump_pickle(self.output()["admission_sol"].path, decision)
