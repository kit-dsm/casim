import logging
import time
from dataclasses import dataclass, replace

from ware_ops_algos.algorithms import AlgorithmSolution, CombinedRoutingSolution, \
    SchedulingSolution, PositioningSolution, ActiveTourRoutingSolution, BatchingSolution, Job, ScheduledJob

from casim.config.schema import CommitmentConfig, ReleaseGates, WaitingConfig
from ware_ops_algos.domain_models import SimWarehouseDomain
from casim.events.base_events import Event
from casim.events.decision_events import RoutingDone
from casim.trackers import DecisionTracker
from casim.decision_engine.commands import Command, Wait, RecallPositioning
from casim.decision_engine.conversion import to_commands
from casim.decision_engine.waiting import release_decision
from casim.interfaces import DecisionSolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionResult:
    commands: tuple[Command, ...]
    events: tuple[Event, ...] = ()
    solution: AlgorithmSolution | None = None
    conversion_seconds: float = 0.0
    solver_name: str | None = None
    objective_value: float | None = None
    solve_seconds: float = 0.0


class DecisionEngine:
    def __init__(self,
                 solver_map: dict[str, DecisionSolver],
                 commitment: dict[str, CommitmentConfig],
                 waiting: dict[str, WaitingConfig],
                 release_gates: dict[str, ReleaseGates] | None = None,
                 learnable_problems: list[str] | None = None,
                 ):

        self.solver_map = solver_map
        self.learnable_problems = learnable_problems or []
        self.commitment = commitment
        self.waiting = waiting
        self.release_gates = release_gates or {}
        self.decision_tracker = DecisionTracker()

    def get_solver(self, problem: str) -> DecisionSolver:
        return self.solver_map[problem]

    def on_trigger(self, state_snapshot: SimWarehouseDomain, action=None):
        decision = state_snapshot.context.decision

        if state_snapshot.context.input_closed and not state_snapshot.plan.orders:
            recalls = self._recall_commands(state_snapshot)
            if recalls:
                return DecisionResult(recalls)
        if not state_snapshot.context.solve or decision not in self.solver_map:
            start = time.perf_counter()
            solution = self._draft_solution(state_snapshot)
            solution, commands = self._schedule_commands(solution, state_snapshot)
            return DecisionResult(commands, (), solution,
                                  time.perf_counter() - start)
        runner = self.get_solver(decision)
        start_time = time.perf_counter()
        solution, solver_name, objective_value, commands = runner.solve(
            state_snapshot, action,
            evaluate=lambda candidate: self._evaluate_candidate(candidate,
                                                                 state_snapshot),
        )
        elapsed = time.perf_counter() - start_time
        if solution:
            events = ()
            if isinstance(solution, CombinedRoutingSolution) and not hasattr(solution, "assignments"):
                events = tuple(RoutingDone(state_snapshot.context.time, route)
                               for route in solution.routes)
            return DecisionResult(commands, events, solution,
                                  runner.last_conversion_seconds,
                                  solver_name, objective_value,
                                  max(0.0, elapsed - runner.last_conversion_seconds))
        return DecisionResult(())

    def _evaluate_candidate(self, solution: AlgorithmSolution,
                            snapshot: SimWarehouseDomain):
        if isinstance(solution, SchedulingSolution):
            return self._schedule_commands(solution, snapshot)
        if (isinstance(solution, ActiveTourRoutingSolution)
                and snapshot.context.input_closed and not solution.assignments):
            return solution, (*to_commands(solution, snapshot),
                              *self._recall_commands(snapshot))
        return solution, to_commands(solution, snapshot)

    @staticmethod
    def _recall_commands(snapshot: SimWarehouseDomain) -> tuple[RecallPositioning, ...]:
        return tuple(RecallPositioning(
            expected_decision_version=snapshot.context.version,
            tour_id=tour.tour_id, expected_route_version=tour.route_version,
            expected_cursor=tour.cursor,
        ) for tour in snapshot.plan.tours
            if tour.status.value == "started"
            and tour.positioning_mode == "center_relocation"
            and not tour.batch.order_numbers)

    def on_accepted(self, result: DecisionResult,
                    snapshot: SimWarehouseDomain) -> None:
        if result.solver_name is None or result.solution is None:
            return
        self.on_solution(
            result.solution, result.solver_name, result.objective_value,
            snapshot.objective, snapshot.context.decision, result.solve_seconds,
        )

    @staticmethod
    def _draft_solution(snapshot: SimWarehouseDomain) -> SchedulingSolution:
        jobs = []
        modes = {}
        for tour in snapshot.plan.tours:
            if tour.status.value != "scheduled":
                continue
            route = tour.route
            job = Job(
                job_id=tour.tour_id, processing_time=tour.processing_time,
                release_time=route.arrival_time, due_date=route.earliest_due_date,
                n_picks=len(route.item_sequence), route=route, batch=route.batch,
            )
            jobs.append(ScheduledJob(job=job, picker_id=tour.picker_id,
                                     start_time=tour.start_time,
                                     end_time=tour.end_time))
            if tour.positioning_mode is not None:
                modes[job.job_id] = tour.positioning_mode
        return PositioningSolution(jobs=jobs, modes=modes)

    def _schedule_commands(self, solution: SchedulingSolution,
                           snapshot: SimWarehouseDomain):
        release = release_decision(
            solution.jobs, snapshot,
            self.waiting[snapshot.context.decision],
            self.commitment[snapshot.context.decision],
            self.release_gates.get(snapshot.context.decision),
        )
        planned = replace(solution, jobs=list(release.planned_jobs))
        commands = to_commands(planned, snapshot, released_jobs=release.jobs)
        for reconsider_at, reason in release.reconsiderations:
            commands = (*commands, Wait(
                expected_decision_version=snapshot.context.version,
                reconsider_at=reconsider_at, reason=reason,
            ))
        return planned, commands

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
