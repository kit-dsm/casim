import pandas as pd
from ware_ops_algos.algorithms import Job, PlanningState, CombinedRoutingSolution


class ObjectiveEvaluator:
    def __init__(self, objective="distance"):
        self.objective = objective

    def select_best(self, plans, problem_class):
        solution_object = None
        best_key = None

        if problem_class in ["OBP"]:
            best_key = None
            for k, plan in plans.items():
                best_key = k
            solution_object = plans[best_key].batching_solutions
        elif problem_class in ["ORP", "OBRP", "BSRP"]:
            best_key, best_dist = None, float("inf")
            for k, plan in plans.items():
                plan: PlanningState
                dist = sum(r.route.distance for r in plan.routing_solutions)
                if dist < best_dist:
                    best_key, best_dist = k, dist
            solution = plans[best_key].routing_solutions
            routes = []
            for r in solution:
                routes.append(r.route)
            solution_object = CombinedRoutingSolution(routes=routes)

        elif problem_class in ["OBRSP"]:
            best_key, best_kpi = None, float("inf")
            for k, plan in plans.items():
                plan: PlanningState
                orders = []
                for j in plan.sequencing_solutions.jobs:
                    for o in j.route.pick_list.orders:
                        orders.append(o)
                eval_due_date = self._evaluate_due_dates(plan.sequencing_solutions.jobs)
                makespan = eval_due_date["completion_time"].min()
                # print("Mean lateness", eval_due_date["lateness"].mean())
                # print("Max lateness", eval_due_date["lateness"].max())
                # print("# On time", eval_due_date["on_time"].sum())
                # print("Makespan", makespan)
                dist = sum(a.distance for a in plan.sequencing_solutions.jobs)
                # print("Distance", dist)
                # kpi = eval_due_date["tardiness"].max()
                if self.objective == "distance":
                    kpi = dist
                elif self.objective == "makespan":
                    kpi = makespan
                else:
                    raise ValueError(f"Not a valid objective {self.objective}")
                if kpi < best_kpi:
                    best_key, best_kpi = k, dist
            solution_object = plans[best_key].sequencing_solutions
        return solution_object, best_key

    @staticmethod
    def _evaluate_due_dates(assignments: list[Job]):
        orders = []
        for job in assignments:
            for o in job.route.pick_list.orders:
                orders.append(o)

        order_by_id = {o.order_id: o for o in orders}
        records = []
        for ass in assignments:
            end_time = ass.end_time
            for on in ass.route.pick_list.order_numbers:
                o = order_by_id.get(on)
                if o is None:
                    continue
                if o.due_date is None:
                    due_date = 999999999
                    on_time = True
                else:
                    due_date = o.due_date
                    on_time = end_time <= due_date
                arrival_time = o.order_date
                start_time = ass.start_time
                lateness = end_time - due_date
                records.append({
                    "order_number": on,
                    "arrival_time": arrival_time,
                    "start_time": start_time,
                    "batch_idx": ass.batch_idx,
                    "picker_id": ass.picker_id,
                    "completion_time": end_time,
                    "due_date": due_date,
                    "lateness": lateness,
                    "tardiness": max(0, lateness),
                    "on_time": on_time,
                })
        return pd.DataFrame(records)