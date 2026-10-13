from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
from typing import Callable

from cosy.maestro import Maestro
from cosy_luigi import CoSyLuigiRepo
from hydra.utils import get_class
from luigi.configuration import get_config
from ware_ops_algos.algorithms import AlgorithmSolution
from ware_ops_algos.domain_models import BaseWarehouseDomain, DataCard
from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards

from casim.pipelines.in_memory_dag import InMemoryDagExecutor
from casim.pipelines.solution_ranker import SolutionRanker
from casim.pipelines.problem_based_template import (
    PlanCandidateEndpoint, BatchingPlanCandidate, RoutingPlanCandidate,
    SchedulingPlanCandidate, SequencingPlanCandidate, PositioningPlanCandidate,
    clear_store, dump_pickle, load_pickle,
)

PLAN_SOURCES = {
    "batching": BatchingPlanCandidate,
    "routing": RoutingPlanCandidate,
    "scheduling": SchedulingPlanCandidate,
    "positioning": PositioningPlanCandidate,
    "sequencing": SequencingPlanCandidate,
}


class CoSySolver:
    def __init__(
        self,
        instances_dir: Path,
        cache_dir: Path,
        output_dir: Path,
        instance_name: str,
        solution_ranker: SolutionRanker,
        output: str | None = None,
        endpoint=None,
        problem_class=None,
        verbose: bool = False,
        luigi_cfg = None,
        repo=None
    ):
        self.instance_name = instance_name
        self.instances_dir = Path(instances_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path: Path = self.cache_dir / "dynamic_info.pkl"
        self.verbose = verbose
        self.endpoint = endpoint
        self.output = output
        self.problem_class = problem_class
        self.pipelines = None
        self.solution_ranker = solution_ranker
        self.luigi_cfg = luigi_cfg
        self.repo_cfg = repo
        self.luigi_logging_opts =  SimpleNamespace(
            background=luigi_cfg.background,
            logdir=luigi_cfg.logdir,
            logging_conf_file=luigi_cfg.logging_conf_file,
            log_level=luigi_cfg.log_level,
        )
        self.output_folder = Path(output_dir) / "cosy"
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.algorithm_cards = load_packaged_algo_cards()
        if self.verbose:
            print(f"Loaded {len(self.algorithm_cards)} model cards")

    def dump_domain(self, dynamic_domain: BaseWarehouseDomain):
        dump_pickle(str(self.cache_path), dynamic_domain)

    def build_pipelines(self, data_card: DataCard):
        config = get_config()
        config.set('PipelineParams', 'output_folder', str(self.output_folder))
        config.set('PipelineParams', 'domain_path', str(self.cache_path))
        problem = data_card.problem_class
        if self.output not in PLAN_SOURCES:
            raise ValueError(f"Unknown plan output stage {self.output!r}")
        PlanCandidateEndpoint.configure(data_card, self.algorithm_cards)

        if not self.pipelines:
            if self.verbose:
                print("Building pipelines")
            repo_classes = [get_class(path) for path in self.repo_cfg.components]
            repo_classes.extend((PLAN_SOURCES[self.output], PlanCandidateEndpoint))
            repo = CoSyLuigiRepo(*repo_classes)
            maestro = Maestro(repo.cls_repo, repo.taxonomy)
            self.pipelines = list(maestro.query(PlanCandidateEndpoint.target()))
            if not self.pipelines:
                raise ValueError(
                    f"No CoSy pipeline for {problem}: check the configured components "
                    "and their algorithm-card requirements"
                )
            if self.verbose:
                print(f"Found {len(self.pipelines)} pipelines")
        else:
            if self.verbose:
                print("Using cached pipelines")

    def solve(self, dynamic_domain: BaseWarehouseDomain, action=None,
              evaluate: Callable[[AlgorithmSolution], tuple[AlgorithmSolution, tuple]] | None = None
              ) -> tuple[AlgorithmSolution, str, float, tuple | None]:
        self.dump_domain(dynamic_domain)
        if not self.pipelines:
            raise RuntimeError(f"No configured CoSy pipeline for {dynamic_domain.problem_class}")

        selected = (list(enumerate(self.pipelines)) if action is None else
                    [(action, self.pipelines[action])])
        solutions = {}
        candidate_commands = {}
        self.last_conversion_seconds = 0.0
        for index, pipeline in selected:
            self.dump_domain(dynamic_domain)
            InMemoryDagExecutor().execute_many([pipeline])
            output_path = pipeline.output()["plan_candidate"].path
            key = f"{index}:{Path(output_path).stem}"
            solution = load_pickle(output_path)
            if evaluate is not None:
                started = perf_counter()
                solution, commands = evaluate(solution)
                self.last_conversion_seconds += perf_counter() - started
                candidate_commands[key] = commands
            solutions[key] = solution
            self._cleanup_after_solution(self.output_folder)
        best_solution, best_key, best_kpi_value = self.select_strategy(
            solutions, dynamic_domain.problem_class, dynamic_domain,
            candidate_commands)
        return (best_solution, best_key, best_kpi_value,
                candidate_commands.get(best_key))

    def select_strategy(self, solutions, problem, domain=None,
                        candidate_commands=None):
        best_solution, best_key, best_kpi_value = self.solution_ranker.select_best(
            solutions, problem, domain, candidate_commands)
        return best_solution, best_key, best_kpi_value

    @staticmethod
    def _cleanup_after_solution(output_folder: Path):
        clear_store()
