"""Choose a feasible decision candidate and its plan commitments."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from time import perf_counter

from ware_ops_algos.planning.commands import Command
from ware_ops_algos.planning.conversion import decision_effects, to_commands
from ware_ops_algos.planning.validation import check as validate_commands

from casim.decision_engine.commitment_policies import CommitmentPolicy
from casim.decision_engine.interfaces import DecisionSolver
from casim.decision_engine.positioning import closure_commands, positioning_commands
from casim.decision_engine.solution_ranker import SolutionRanker
from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.trackers import DecisionTracker


@dataclass(frozen=True)
class DecisionResult:
    commands: tuple[Command, ...]
    waiting: bool = False
    reconsider_at: float | None = None
    considered_active_orders: frozenset[tuple[int, int]] = frozenset()
    solver_name: str | None = None
    objective_value: float | None = None
    solve_seconds: float = 0.0
    conversion_seconds: float = 0.0


class DecisionEngine:
    def __init__(self, solver_map: dict[str, DecisionSolver],
                 solution_rankers: dict[str, SolutionRanker],
                 commitment_policies: dict[str, CommitmentPolicy]):
        if set(solver_map) != set(solution_rankers):
            raise ValueError("solver_map and solution_rankers must have the same decision names")
        self.solver_map = solver_map
        self.solution_rankers = solution_rankers
        self.commitment_policies = commitment_policies
        self.decision_tracker = DecisionTracker()
        self._accepted_choices: dict[str, tuple[object, str]] = {}

    def on_trigger(self, state_snapshot: SimWarehouseDomain, action=None) -> DecisionResult:
        dynamic = state_snapshot.dynamic_warehouse_info
        decision = dynamic.context.decision
        plan = dynamic.plan
        recalls = closure_commands(state_snapshot)
        if recalls:
            return DecisionResult(commands=recalls)
        commands: tuple[Command, ...] = ()
        waiting = dynamic.memory.waiting
        reconsider_at = dynamic.memory.reconsider_at
        considered = dynamic.memory.considered_active_orders
        chosen_name = None
        objective_value = None
        solve_seconds = 0.0
        conversion_seconds = 0.0

        if dynamic.context.solve and decision in self.solver_map:
            started = perf_counter()
            candidates = self.solver_map[decision].solve(state_snapshot)
            solve_seconds = perf_counter() - started
            conversion_start = perf_counter()
            feasible = {}
            for name, solution in candidates.items():
                proposal = to_commands(solution, plan)
                if not validate_commands(proposal, plan):
                    feasible[name] = (solution, proposal)
            conversion_seconds = perf_counter() - conversion_start
            if not feasible:
                raise ValueError(f"{decision}: no feasible candidate solution")
            if action is None:
                if len(feasible) == 1:
                    chosen_name = next(iter(feasible))
                    solution, commands = feasible[chosen_name]
                    try:
                        objective_value = self.solution_rankers[decision].score(
                            solution, commands)
                    except ValueError:
                        objective_value = None
                else:
                    solution, commands, chosen_name, objective_value = (
                        self.solution_rankers[decision].select_best(feasible))
            else:
                names = tuple(sorted(feasible))
                chosen_name = names[action] if isinstance(action, int) else action
                if chosen_name not in feasible:
                    raise ValueError(f"{decision}: action {action!r} is not a feasible candidate")
                solution, commands = feasible[chosen_name]
                try:
                    objective_value = self.solution_rankers[decision].score(
                        solution, commands)
                except ValueError:
                    objective_value = None
            waiting, reconsider_at, selected_considered = decision_effects(solution)
            considered = considered | selected_considered
            commands = (*commands, *positioning_commands(solution, commands))
            self._accepted_choices[decision] = solution, chosen_name

        if not waiting and decision in self.commitment_policies:
            policy_snapshot = copy(state_snapshot)
            policy_snapshot.dynamic_warehouse_info = replace(
                dynamic, memory=replace(dynamic.memory, waiting=False))
            commits, policy_reconsider = self.commitment_policies[decision].apply(
                commands, policy_snapshot)
            commands = (*commands, *commits)
            if policy_reconsider is not None:
                reconsider_at = policy_reconsider
        return DecisionResult(
            commands=commands, waiting=waiting, reconsider_at=reconsider_at,
            considered_active_orders=considered, solver_name=chosen_name,
            objective_value=objective_value, solve_seconds=solve_seconds,
            conversion_seconds=conversion_seconds,
        )

    def on_accepted(self, result: DecisionResult,
                    snapshot: SimWarehouseDomain) -> None:
        decision = snapshot.dynamic_warehouse_info.context.decision
        choice = self._accepted_choices.pop(decision, None)
        if choice is None:
            return
        solution, name = choice
        ids = {order_id for job in getattr(solution, "jobs", ())
               for order_id in job.order_numbers}
        ids.update(order_id for batch in getattr(solution, "batches", ())
                   for order_id in batch.order_numbers)
        ids.update(order_id for route in getattr(solution, "routes", ())
                   for order_id in route.batch.order_numbers)
        self.decision_tracker.on_decision(
            problem_class=decision, input_ids=len(ids), selected_pipeline=name,
            kpi_value=result.objective_value, kpi=snapshot.objective,
            runtime=getattr(solution, "execution_time", 0.0),
            elapsed=result.solve_seconds,
        )
