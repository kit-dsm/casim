import pandas as pd
from ware_ops_algos.algorithms import CombinedRoutingSolution, SchedulingSolution, BatchingSolution


class SolutionRanker:
    def __init__(self, objective="distance"):
        self.objective = objective

    def select_best(self, solutions: dict[str, object], problem_class: str):
        best_key = None
        best_kpi_value = None
        solution_object = None

        if problem_class in ["OBP", "OSBP"]:
            # No KPI — just take any (last) key
            for k in solutions:
                best_key = k
            solution_object = solutions[best_key]

        elif problem_class in ["ORP", "OBRP", "BSRP"]:
            best_kpi_value = float("inf")
            for k, sol in solutions.items():
                # sol is either list[RoutingSolution] or CombinedRoutingSolution
                if isinstance(sol, CombinedRoutingSolution):
                    dist = sum(r.distance for r in sol.routes)
                else:
                    dist = sum(r.route.distance for r in sol)
                if dist < best_kpi_value:
                    best_key, best_kpi_value = k, dist

            sol = solutions[best_key]
            if isinstance(sol, CombinedRoutingSolution):
                solution_object = sol
            else:
                solution_object = CombinedRoutingSolution(routes=[r.route for r in sol])

        elif problem_class in ["OBRSP", "ORSP", "RORSP"]:
            best_kpi_value = float("inf")
            for k, sol in solutions.items():
                sol: SchedulingSolution
                if self.objective == "distance":
                    kpi = sum(j.job.distance for j in sol.jobs)
                elif self.objective == "makespan":
                    kpi = max(j.end_time for j in sol.jobs)
                elif self.objective == "tardiness":
                    kpi = sum(
                        max(0, j.end_time - j.job.route.batch.earliest_due_date)
                        for j in sol.jobs
                    )
                else:
                    raise ValueError(f"Not a valid objective: {self.objective}")
                if kpi < best_kpi_value:
                    best_key, best_kpi_value = k, kpi
            solution_object = solutions[best_key]
        else:
            raise ValueError(f"No a known problem class: {problem_class}")

        return solution_object, best_key, best_kpi_value