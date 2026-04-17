import logging
import time
from collections import defaultdict

from ware_ops_algos.algorithms import AlgorithmSolution, CombinedRoutingSolution, \
    SchedulingSolution, BatchingSolution

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.events.decision_events import SequencingDone, RoutingDone, PickListDone
from casim.pipelines.objective_evaluator import ObjectiveEvaluator
from casim.pipelines.pipeline_runner import CoSyRunner
from casim.trackers import DecisionTracker

logger = logging.getLogger(__name__)


class CommitmentPolicy:
    def apply(self, solution: AlgorithmSolution) -> AlgorithmSolution:
        raise NotImplementedError


class SchedulingCommitmentPolicy(CommitmentPolicy):
    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    def apply(self, solution: SchedulingSolution) -> SchedulingSolution:
        committed = sorted(solution.jobs, key=lambda j: j.start_time)[:self.n_jobs]
        return SchedulingSolution(jobs=committed, execution_time=solution.execution_time)


class CommitAllPolicy(CommitmentPolicy):
    def apply(self, solution: AlgorithmSolution) -> AlgorithmSolution:
        return solution


class DecisionEngine:
    def __init__(self,
                 execution_map: dict[str, CoSyRunner],
                 evaluator: ObjectiveEvaluator,
                 commitment_policies: dict[str, CommitmentPolicy] | None = {"OBRSP": SchedulingCommitmentPolicy()},
                 learnable_problems: list[str] | None = None
                 ):

        self.execution_map = execution_map
        self.learnable_problems = learnable_problems or []
        self.evaluator = evaluator
        self.commitment_policies = commitment_policies
        self.selected_pipelines = defaultdict(dict)
        self.decision_tracker = DecisionTracker()

    def get_execution(self, problem: str) -> CoSyRunner:
        return self.execution_map[problem]

    def on_trigger(self, state_snapshot: SimWarehouseDomain):
        problem = state_snapshot.problem_class
        runner = self.get_execution(problem)
        start_time_sim = state_snapshot.dynamic_warehouse_info.time
        start_time = time.perf_counter()
        solutions = runner.solve(state_snapshot)
        elapsed = time.perf_counter() - start_time

        solution = self.select_strategy(solutions, problem)
        if solution:
            if self.commitment_policies:
                policy = self.commitment_policies[problem]
                if policy:
                    solution = policy.apply(solution)
            return self.solution_to_events(solution, start_time_sim), solution
        return None

    def select_strategy(self, solutions, problem):
        # evaluator = self.evaluators_map[problem]
        best_solution, best_key, best_kpi_value = self.evaluator.select_best(solutions, problem)
        if isinstance(best_solution, CombinedRoutingSolution):
            order_ids = [o for r in best_solution.routes for o in r.pick_list.order_numbers]
            self.decision_tracker.on_decision(
                problem_class=problem,
                input_ids=order_ids,
                selected_pipeline=best_key,
                kpi_value=best_kpi_value,
                kpi=self.evaluator.objective,
                runtime=best_solution.execution_time
            )
        elif isinstance(best_solution, SchedulingSolution):
            order_ids = [o for j in best_solution.jobs for o in j.route.pick_list.order_numbers]
            self.decision_tracker.on_decision(
                problem_class=problem,
                input_ids=order_ids,
                selected_pipeline=best_key,
                kpi_value=best_kpi_value,
                kpi=self.evaluator.objective,
                runtime=best_solution.execution_time
            )

        else:
            raise ValueError(type(best_solution))
        return best_solution

    def solution_to_events(self, solution: AlgorithmSolution, finish_time):
        logger.info("Solution type", type(solution))

        if isinstance(solution, CombinedRoutingSolution):
            events_to_return = self._routes_to_events(solution, finish_time)

        elif isinstance(solution, SchedulingSolution):
            events_to_return = self._schedules_to_events(solution, finish_time)

        elif isinstance(solution, BatchingSolution):
            events_to_return = self._batches_to_events(solution, finish_time)

        else:
            raise Exception("Not a known solution", type(solution))

        return events_to_return

    # These functions return ProcessEvents that add solution objects to state
    @staticmethod
    def _schedules_to_events(sequencing_sol: SchedulingSolution, finish_time):
        """
        Turn sequencing solution into TourStart events.
        """
        events_to_return: list[Event] = []
        # jobs = [sequencing_sol.jobs[0]]
        jobs = sequencing_sol.jobs # TODO How to window?
        assignments = sorted(jobs, key=lambda a: (a.picker_id, a.start_time))
        for a in assignments:
            events_to_return.append(SequencingDone(finish_time, a))
        return events_to_return

    @staticmethod
    def _routes_to_events(routing_solution: CombinedRoutingSolution, finish_time) -> list[RoutingDone]:
        events_to_return = []
        routes = routing_solution.routes
        for r in routes:
            events_to_return.append(RoutingDone(finish_time, r))
        return events_to_return

    @staticmethod
    def _batches_to_events(batching_solution: BatchingSolution, finish_time):
        events_to_return = []
        pls = batching_solution.pick_lists
        for pl in pls:
            events_to_return.append(PickListDone(finish_time, pl))
        return events_to_return