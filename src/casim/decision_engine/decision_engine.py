import logging
import time
from collections import defaultdict
from typing import Type

from ware_ops_algos.algorithms import AlgorithmSolution, PickListSelectionSolution, CombinedRoutingSolution, \
    SchedulingSolution, BatchingSolution

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.events.decision_events import SequencingDone, RoutingDone, PickListDone
from casim.pipelines.objective_evaluator import ObjectiveEvaluator
from casim.pipelines.pipeline_runner import CoSyRunner
from casim.state.conditions import Condition

logger = logging.getLogger(__name__)


class DecisionEngine:
    def __init__(self,
                 execution_map: dict[str, CoSyRunner],
                 # conditions_map: dict[str, Condition],
                 # trigger_map: dict[Type[Event], str],
                 # evaluators_map: dict[str, ObjectiveEvaluator],
                 evaluator: ObjectiveEvaluator,
                 learnable_problems: list[str] | None = None
                 ):

        self.execution_map = execution_map
        # self.conditions_map = conditions_map
        # self.triggers = trigger_map
        self.learnable_problems = learnable_problems or []
        # self.evaluators_map = evaluators_map
        self.evaluator = evaluator
        # self.commit_window =
        self.selected_pipelines = defaultdict(dict)

    def get_execution(self, problem: str) -> CoSyRunner:
        return self.execution_map[problem]

    def get_solution(self, state_snapshot: SimWarehouseDomain):
        problem = state_snapshot.problem_class
        runner = self.get_execution(problem)
        start_time_sim = state_snapshot.dynamic_warehouse_info.time
        start_time = time.perf_counter()
        solutions = runner.solve(state_snapshot)
        elapsed = time.perf_counter() - start_time

        solution, best_key = self.select_strategy(solutions, problem)
        for o in state_snapshot.orders.orders:
            self.selected_pipelines[problem][o.order_id] = best_key
        return solution

    def on_trigger(self, state_snapshot: SimWarehouseDomain):
        problem = state_snapshot.problem_class
        runner = self.get_execution(problem)
        start_time_sim = state_snapshot.dynamic_warehouse_info.time
        start_time = time.perf_counter()
        solutions = runner.solve(state_snapshot)
        elapsed = time.perf_counter() - start_time

        solution, best_key = self.select_strategy(solutions, problem)
        for o in state_snapshot.orders.orders:
            self.selected_pipelines[problem][o.order_id] = best_key
        if solution:
            events_to_return = self.solution_to_events(solution, start_time_sim)
            return events_to_return
        else:
            return None

    def select_strategy(self, solution, problem):
        # evaluator = self.evaluators_map[problem]
        best_strategy = self.evaluator.select_best(solution, problem)
        return best_strategy

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
    def _schedules_to_events(self, sequencing_sol: SchedulingSolution, finish_time):
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

    def _routes_to_events(self, routing_solution: CombinedRoutingSolution, finish_time) -> list[RoutingDone]:
        events_to_return = []
        routes = routing_solution.routes
        for r in routes:
            events_to_return.append(RoutingDone(finish_time, r))
        return events_to_return

    def _batches_to_events(self, batching_solution: BatchingSolution, finish_time):
        events_to_return = []
        pls = batching_solution.pick_lists
        for pl in pls:
            events_to_return.append(PickListDone(finish_time, pl))
        return events_to_return