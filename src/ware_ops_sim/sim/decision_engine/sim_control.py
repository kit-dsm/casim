from __future__ import annotations
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from experiments.experiment_commons import OnlineRunner

from ware_ops_algos.algorithms import Job, Batching, Routing, GreedyItemAssignment

from ware_ops_sim.sim.events.events import Event
from ware_ops_sim.sim.conditions import Condition
from ware_ops_pipes.utils.experiment_utils import RankingEvaluator


class DecisionEngine:
    def __init__(self,
                 execution: OnlineRunner,
                 selector: RankingEvaluator,
                 requirements_policies: dict[str, Condition],
                 triggers: dict[Type[Event], str],
                 control_mode="cls",
                 learnable_problems: list[str] | None = None,
                 fixed_routing_policy=None,
                 fixed_batching_policy=None,
                 fixed_sequencing_policy=None,
                 fixed_item_assignment_policy=None,
                 implementation_modules: dict = None,
                 strategies: dict = None):

        self.execution = execution
        self.selector = selector
        self.requirements_policies = requirements_policies  # TODO Needs to map to Trigger event as well?
        self.triggers = triggers
        self.learnable_problems = learnable_problems
        self.implementation_modules = {}
        self.fixed_item_assigner: GreedyItemAssignment = fixed_item_assignment_policy
        self.fixed_routing_policy: Routing | None = fixed_routing_policy
        self.fixed_batching_policy: Batching | None = fixed_batching_policy
        self.fixed_sequencing_policy: Job | None = fixed_sequencing_policy
        self.control_mode = control_mode
        self.implementation_modules = implementation_modules
        self.strategies = strategies
        assert control_mode in ["cls", "fixed"]
        # if control_mode == "cls":
        #     assert self.implementation_modules
        # elif control_mode == "fixed":
        #     assert fixed_routing_policy and fixed_sequencing_policy and fixed_batching_policy

    def register_trigger(self, trigger_type: Type[Event], problem: str):
        self.triggers[trigger_type] = problem

    def register_requirements_policy(self, problem: str, policy: Condition):
        self.requirements_policies[problem] = policy

    def get_strategy(self, problem: str):
        return self.strategies[problem]

    def get_execution(self) -> OnlineRunner:
        return self.execution
