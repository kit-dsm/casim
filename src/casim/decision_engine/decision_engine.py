import logging
import time
from collections import defaultdict

from ware_ops_algos.algorithms import AlgorithmSolution, CombinedRoutingSolution, \
    SchedulingSolution, BatchingSolution

from casim.decision_engine.commitment_policies import CommitmentPolicy, CommitAllPolicy
from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.events.decision_events import SequencingDone, RoutingDone, PickListDone
from casim.pipelines.pipeline_runner import CoSySolver
from casim.trackers import DecisionTracker
logger = logging.getLogger(__name__)


class DecisionEngine:
    def __init__(self,
                 solver_map: dict[str, CoSySolver],
                 commitment_policies: dict[str, CommitmentPolicy],
                 learnable_problems: list[str] | None = None,
                 event_map: dict[str, Event] | None = None,
                 ):

        self.solver_map = solver_map
        self.learnable_problems = learnable_problems or []
        self.commitment_policies = commitment_policies
        self.selected_pipelines = defaultdict(dict)
        self.decision_tracker = DecisionTracker()
        self.event_map = event_map or {}

    def get_solver(self, problem: str) -> CoSySolver:
        return self.solver_map[problem]

    def on_trigger(self, state_snapshot: SimWarehouseDomain, action=None):
        problem = state_snapshot.problem_class
        runner = self.get_solver(problem)
        start_time_sim = state_snapshot.dynamic_warehouse_info.time
        start_time = time.perf_counter()
        result = runner.solve(state_snapshot, action)
        if result is None:
            return None
        solution, solver_name, objective_value = result
        elapsed = time.perf_counter() - start_time
        if solution:
            self.on_solution(
                solution,
                solver_name,
                objective_value,
                state_snapshot.objective,
                state_snapshot.problem_class,
                elapsed)

            policy = self.commitment_policies.get(problem) or CommitAllPolicy()
            full_solution = solution
            solution = policy.apply(solution, state_snapshot)
            self.decision_tracker.on_commitment(
                returned=self._solution_size(full_solution),
                committed=self._solution_size(solution),
                policy=policy.__class__.__name__,
            )
            return (
                self.solution_to_events(
                    solution,
                    start_time_sim,
                    state_snapshot,
                ),
                solution,
            )
        return None

    @staticmethod
    def _solution_size(solution: AlgorithmSolution) -> int:
        if isinstance(solution, SchedulingSolution):
            return len(solution.jobs)
        if isinstance(solution, BatchingSolution):
            return len(solution.batches)
        if isinstance(solution, CombinedRoutingSolution):
            return len(solution.routes)
        return 0

    def on_solution(self, best_solution: AlgorithmSolution, solver_name, objective_value, objective, problem, elapsed):
        if isinstance(best_solution, CombinedRoutingSolution):
            order_ids = [o for r in best_solution.routes for o in r.batch.order_numbers]
        elif isinstance(best_solution, SchedulingSolution):
            order_ids = [o for j in best_solution.jobs for o in j.job.route.batch.order_numbers]
        elif isinstance(best_solution, BatchingSolution):
            order_ids = [o_id for b in best_solution.batches for o_id in b.order_numbers]
        else:
            raise ValueError(type(best_solution))

        self.decision_tracker.on_decision(
            problem_class=problem,
            input_ids=len(order_ids),
            selected_pipeline=solver_name,
            kpi_value=objective_value,
            kpi=objective,
            runtime=best_solution.execution_time,
            elapsed=elapsed
        )

    def solution_to_events(
        self,
        solution: AlgorithmSolution,
        finish_time,
        state_snapshot: SimWarehouseDomain | None = None,
    ):
        logger.info("Solution type: %s", type(solution))

        if isinstance(solution, CombinedRoutingSolution):
            events_to_return = self._routes_to_events(
                solution,
                finish_time,
                state_snapshot,
            )

        elif isinstance(solution, SchedulingSolution):
            events_to_return = self._schedules_to_events(
                solution,
                finish_time,
                state_snapshot,
            )

        elif isinstance(solution, BatchingSolution):
            events_to_return = self._batches_to_events(solution, finish_time)

        else:
            raise Exception("Not a known solution", type(solution))

        return events_to_return

    # These functions return ProcessEvents that add solution objects to state
    def _schedules_to_events(
        self,
        sequencing_sol: SchedulingSolution,
        finish_time,
        state_snapshot: SimWarehouseDomain | None,
    ):
        """
        Turn sequencing solution into TourStart events.
        """
        cls = self.event_map.get("SequencingDone", SequencingDone)
        replace_tour_ids = ()
        if state_snapshot is not None:
            replace_tour_ids = tuple(
                int(tour.tour_id)
                for tour in (
                    state_snapshot.dynamic_warehouse_info.replannable_tours or []
                )
                if state_snapshot.problem_class == "RORSP"
            )
        return [
            cls(
                finish_time,
                sequencing_sol,
                replace_tour_ids=replace_tour_ids,
            )
        ]

    def _routes_to_events(
        self,
        routing_solution: CombinedRoutingSolution,
        finish_time,
        state_snapshot: SimWarehouseDomain | None,
    ) -> list[RoutingDone]:
        events_to_return = []
        routes = routing_solution.routes
        cls = self.event_map.get("RoutingDone", RoutingDone)
        dynamic = (
            state_snapshot.dynamic_warehouse_info
            if state_snapshot is not None
            else None
        )
        active_tour_id = (
            dynamic.active_tour_id if dynamic is not None else None
        )
        route_version = (
            dynamic.route_version if dynamic is not None else None
        )
        resumes_execution = (
            dynamic.intervention_resumes_execution
            if dynamic is not None
            else False
        )
        picker_id = (
            dynamic.current_picker.id
            if dynamic is not None
            and dynamic.current_picker is not None
            else None
        )
        for r in routes:
            events_to_return.append(
                cls(
                    finish_time,
                    r,
                    picker_id=picker_id,
                    tour_id=active_tour_id,
                    expected_route_version=route_version,
                    resumes_execution=resumes_execution,
                )
            )
        return events_to_return


    def _batches_to_events(self, batching_solution: BatchingSolution, finish_time):
        cls = self.event_map.get("PickListDone", PickListDone)
        return [cls(finish_time, batching_solution)]
