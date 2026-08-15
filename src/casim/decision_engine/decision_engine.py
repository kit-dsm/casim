import logging
import time
from dataclasses import replace

from ware_ops_algos.algorithms import AlgorithmSolution, CombinedRoutingSolution, \
    SchedulingSolution, BatchingSolution

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.operational_events import Event
from casim.events.decision_events import SequencingDone, RoutingDone, PickListDone
from casim.trackers import DecisionTracker
logger = logging.getLogger(__name__)


class SchedulingCommitmentPolicy:
    """Select a deterministic executable prefix from a full schedule."""

    def __init__(
        self,
        n_jobs: int | None = None,
        max_jobs_per_picker: int | None = None,
        planning_horizon_s: float | None = None,
    ):
        self.n_jobs = n_jobs
        self.max_jobs_per_picker = max_jobs_per_picker
        self.planning_horizon_s = planning_horizon_s

    def apply(
        self,
        solution: SchedulingSolution,
        state_snapshot: SimWarehouseDomain,
    ) -> SchedulingSolution:
        committed = sorted(
            solution.jobs,
            key=lambda job: (
                job.start_time,
                job.end_time,
                int(job.picker_id),
                job.job.job_id,
            ),
        )
        if self.planning_horizon_s is not None:
            now = float(state_snapshot.dynamic_warehouse_info.time or 0.0)
            limit = now + float(self.planning_horizon_s)
            committed = [
                job for job in committed if job.start_time <= limit
            ]
        if self.max_jobs_per_picker is not None:
            per_picker: dict[int, int] = {}
            selected = []
            for job in committed:
                picker_id = int(job.picker_id)
                count = per_picker.get(picker_id, 0)
                if count >= self.max_jobs_per_picker:
                    continue
                selected.append(job)
                per_picker[picker_id] = count + 1
            committed = selected
        if self.n_jobs is not None:
            committed = committed[: self.n_jobs]
        return replace(solution, jobs=committed)


class DecisionEngine:
    def __init__(self,
                 solver_map: dict[str, object],
                 commitment_policies: dict[str, SchedulingCommitmentPolicy] | None = None,
                 event_map: dict[str, Event] | None = None,
                 ):

        self.solver_map = solver_map
        self.commitment_policies = commitment_policies or {}
        self.decision_tracker = DecisionTracker()
        self.event_map = event_map or {}

    def solve(self, state_snapshot: SimWarehouseDomain, action=None):
        """Execute the already-prepared solver and record its decision."""
        problem = state_snapshot.problem_class
        solver = self.solver_map[problem]
        start_time = time.perf_counter()
        result = solver.solve(state_snapshot, action)
        if result is None:
            return None
        solution, solver_name, objective_value = result
        elapsed = time.perf_counter() - start_time
        if not solution:
            return None
        self._record_solution(
            solution,
            solver_name,
            objective_value,
            state_snapshot.objective,
            problem,
            elapsed,
        )
        return solution, solver_name, objective_value

    def commit(
        self,
        state_snapshot: SimWarehouseDomain,
        solution: AlgorithmSolution,
    ):
        """Apply commitment policy and convert a semantic solution to events."""
        full_solution = solution
        policy = self.commitment_policies.get(state_snapshot.problem_class)
        if policy is not None:
            solution = policy.apply(solution, state_snapshot)
        self.decision_tracker.on_commitment(
            returned=self._solution_size(full_solution),
            committed=self._solution_size(solution),
            policy=(
                policy.__class__.__name__
                if policy is not None
                else "CommitAllPolicy"
            ),
        )
        return (
            self.solution_to_events(
                solution,
                state_snapshot.dynamic_warehouse_info.time,
                state_snapshot,
            ),
            solution,
        )

    def on_trigger(self, state_snapshot: SimWarehouseDomain, action=None):
        """Solve and commit one normal runtime decision."""
        result = self.solve(state_snapshot, action)
        if result is None:
            return None
        solution, _, _ = result
        return self.commit(state_snapshot, solution)

    @staticmethod
    def _solution_size(solution: AlgorithmSolution) -> int:
        if isinstance(solution, SchedulingSolution):
            return len(solution.jobs)
        if isinstance(solution, BatchingSolution):
            return len(solution.batches)
        if isinstance(solution, CombinedRoutingSolution):
            return len(solution.routes)
        return 0

    def _record_solution(self, best_solution: AlgorithmSolution, solver_name, objective_value, objective, problem, elapsed):
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
            cls = self.event_map.get("PickListDone", PickListDone)
            events_to_return = [cls(finish_time, solution)]

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
