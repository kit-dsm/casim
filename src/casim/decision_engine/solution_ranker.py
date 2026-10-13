"""Rank typed solver results, with an optional cost for editing a live plan."""

from ware_ops_algos.algorithms import (
    BatchingSolution, CombinedRoutingSolution, SchedulingSolution,
)

from casim.decision_engine.commands import (
    CancelTour, CreateTour, InsertOrders, Reassign, ReplaceSuffix, Reschedule,
)
from casim.decision_engine.conversion import to_commands


CHANGE_WEIGHTS = {
    CancelTour: 4,
    CreateTour: 4,
    InsertOrders: 1,
    ReplaceSuffix: 2,
    Reschedule: 1,
    Reassign: 1,
}


class SolutionRanker:
    def __init__(self, objective: str = "distance", change_penalty: float = 0.0):
        self.objective = objective
        self.change_penalty = change_penalty

    def _schedule_cost(self, solution: SchedulingSolution) -> float:
        jobs = solution.jobs
        if self.objective == "distance":
            return sum(job.job.distance for job in jobs)
        if self.objective == "makespan":
            return max((job.end_time for job in jobs), default=0.0)
        if self.objective == "mean_order_completion_time":
            n_orders = sum(len(job.order_numbers) for job in jobs)
            return (sum(job.end_time * len(job.order_numbers) for job in jobs)
                    / n_orders if n_orders else 0.0)
        if self.objective == "tardiness":
            return sum(job.tardiness for job in jobs)
        raise ValueError(f"Unsupported objective {self.objective}")

    def select_best(self, solutions: dict[str, object], problem_class: str,
                    snapshot=None, candidate_commands=None):
        if not solutions:
            raise ValueError(f"No candidate solution for {problem_class}")
        first = next(iter(solutions.values()))
        if isinstance(first, BatchingSolution):
            key = next(reversed(solutions))
            return solutions[key], key, None

        if isinstance(first, SchedulingSolution):
            ranked = []
            for key, solution in solutions.items():
                objective = self._schedule_cost(solution)
                changes = 0
                if self.change_penalty:
                    if snapshot is None:
                        raise ValueError("Change penalty requires a decision snapshot")
                    commands = (candidate_commands.get(key)
                                if candidate_commands is not None else None)
                    if commands is None:
                        commands = to_commands(solution, snapshot)
                    changes = sum(CHANGE_WEIGHTS.get(type(command), 0)
                                  for command in commands)
                ranked.append((objective + self.change_penalty * changes,
                               changes, key))
            score, _, key = min(ranked)
            return solutions[key], key, score

        ranked = []
        for key, solution in solutions.items():
            routes = (solution.routes if isinstance(solution, CombinedRoutingSolution)
                      else [item.route for item in solution])
            ranked.append((sum(route.distance for route in routes), key))
        score, key = min(ranked)
        chosen = solutions[key]
        if not isinstance(chosen, CombinedRoutingSolution):
            chosen = CombinedRoutingSolution(routes=[item.route for item in chosen])
        return chosen, key, score
