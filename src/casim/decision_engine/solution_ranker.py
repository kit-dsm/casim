"""Score feasible plan candidates using the warehouse objective and edit cost."""

from __future__ import annotations

from ware_ops_algos.planning.commands import CancelTour, ReplaceSuffix, Reschedule


class SolutionRanker:
    def __init__(self, objective: str, change_penalty: float = 0.0):
        if change_penalty < 0:
            raise ValueError("change_penalty must be nonnegative")
        self.objective = objective
        self.change_penalty = change_penalty

    def objective_value(self, solution) -> float:
        jobs = getattr(solution, "jobs", None)
        if jobs is not None:
            if self.objective == "distance":
                return sum(job.job.distance for job in jobs)
            if self.objective == "makespan":
                return max((job.end_time for job in jobs), default=0.0)
            if self.objective == "mean_order_completion_time":
                count = sum(len(job.order_numbers) for job in jobs)
                return (sum(job.end_time * len(job.order_numbers) for job in jobs)
                        / count if count else 0.0)
            if self.objective == "tardiness":
                return sum(job.tardiness for job in jobs)
        routes = getattr(solution, "routes", None)
        if routes is not None and self.objective == "distance":
            return sum(route.distance for route in routes)
        raise ValueError(f"No {self.objective} objective for {type(solution).__name__}")

    @staticmethod
    def changed(commands) -> int:
        return len({command.tour_id for command in commands
                    if isinstance(command, (CancelTour, ReplaceSuffix, Reschedule))})

    def score(self, solution, commands) -> float:
        return self.objective_value(solution) + self.change_penalty * self.changed(commands)

    def select_best(self, feasible: dict[str, tuple[object, tuple]]):
        if not feasible:
            raise ValueError("No feasible candidate solution")
        scores = {name: self.score(solution, commands)
                  for name, (solution, commands) in feasible.items()}
        name = min(scores, key=lambda candidate: (scores[candidate], candidate))
        solution, commands = feasible[name]
        return solution, commands, name, scores[name]
