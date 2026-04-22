# import pandas as pd
# from ware_ops_algos.algorithms import Job, PlanningState, CombinedRoutingSolution
#
#
# class ObjectiveEvaluator:
#     def __init__(self, objective="distance"):
#         self.objective = objective
#
#     def select_best(self, plans, problem_class):
#         solution_object = None
#         best_key = None
#         best_kpi_value = None
#
#         if problem_class in ["OBP"]:
#             best_key = None
#             for k, plan in plans.items():
#                 best_key = k
#             solution_object = plans[best_key].batching_solutions
#         elif problem_class in ["ORP", "OBRP", "BSRP"]:
#             best_key, best_kpi_value = None, float("inf")
#             for k, plan in plans.items():
#                 plan: PlanningState
#                 dist = sum(r.route.distance for r in plan.routing_solutions)
#                 if dist < best_kpi_value:
#                     best_key, best_kpi_value = k, dist
#             solution = plans[best_key].routing_solutions
#             routes = []
#             for r in solution:
#                 routes.append(r.route)
#             solution_object = CombinedRoutingSolution(routes=routes)
#
#         elif problem_class in ["OBRSP"]:
#             best_key, best_kpi_value = None, float("inf")
#             for k, plan in plans.items():
#                 plan: PlanningState
#                 orders = []
#                 for j in plan.sequencing_solutions.jobs:
#                     for o in j.route.pick_list.orders:
#                         orders.append(o)
#                 eval_due_date = self._evaluate_due_dates(plan.sequencing_solutions.jobs)
#                 makespan = eval_due_date["completion_time"].min()
#                 dist = sum(a.distance for a in plan.sequencing_solutions.jobs)
#                 if self.objective == "distance":
#                     kpi = dist
#                 elif self.objective == "makespan":
#                     kpi = makespan
#                 else:
#                     raise ValueError(f"Not a valid objective {self.objective}")
#                 if kpi < best_kpi_value:
#                     best_key, best_kpi_value = k, kpi
#             solution_object = plans[best_key].sequencing_solutions
#         return solution_object, best_key, best_kpi_value
#
#     @staticmethod
#     def _evaluate_due_dates(assignments: list[Job]):
#         orders = []
#         for job in assignments:
#             for o in job.route.pick_list.orders:
#                 orders.append(o)
#
#         order_by_id = {o.order_id: o for o in orders}
#         records = []
#         for ass in assignments:
#             end_time = ass.end_time
#             for on in ass.route.pick_list.order_numbers:
#                 o = order_by_id.get(on)
#                 if o is None:
#                     continue
#                 if o.due_date is None:
#                     due_date = 999999999
#                     on_time = True
#                 else:
#                     due_date = o.due_date
#                     on_time = end_time <= due_date
#                 arrival_time = o.order_date
#                 start_time = ass.start_time
#                 lateness = end_time - due_date
#                 records.append({
#                     "order_number": on,
#                     "arrival_time": arrival_time,
#                     "start_time": start_time,
#                     "batch_idx": ass.batch_idx,
#                     "picker_id": ass.picker_id,
#                     "completion_time": end_time,
#                     "due_date": due_date,
#                     "lateness": lateness,
#                     "tardiness": max(0, lateness),
#                     "on_time": on_time,
#                 })
#         return pd.DataFrame(records)

import pandas as pd
from ware_ops_algos.algorithms import CombinedRoutingSolution, SchedulingSolution, BatchingSolution


class SolutionRanker:
    def __init__(self, objective="distance"):
        self.objective = objective

    def select_best(self, solutions: dict[str, object], problem_class: str):
        best_key = None
        best_kpi_value = None
        solution_object = None

        if problem_class in ["OBP"]:
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

        elif problem_class in ["OBRSP", "ORSP"]:
            best_kpi_value = float("inf")
            for k, sol in solutions.items():
                sol: SchedulingSolution
                if self.objective == "distance":
                    kpi = sum(j.distance for j in sol.jobs)
                elif self.objective == "makespan":
                    kpi = max(j.end_time for j in sol.jobs)
                elif self.objective == "tardiness":
                    kpi = sum(
                        max(0, j.end_time - j.route.pick_list.earliest_due_date)
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