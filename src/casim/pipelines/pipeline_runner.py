from pathlib import Path
from types import SimpleNamespace

from cosy.maestro import Maestro
from cosy_luigi import CoSyLuigiRepo
from hydra.utils import get_class
from luigi.configuration import get_config
from ware_ops_algos.algorithms import AlgorithmSolution
from ware_ops_algos.domain_models import BaseWarehouseDomain, DataCard
from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards

from casim.pipelines.in_memory_dag import InMemoryDagExecutor
from casim.pipelines.problem_based_template import (
    PlanCandidateEndpoint, BatchingPlanCandidate, RoutingPlanCandidate,
    SchedulingPlanCandidate, PositioningPlanCandidate,
    WaitingPlanCandidate,
    AbstractBatching, AbstractPickerRouting, AbstractScheduling,
    AbstractWaiting, AbstractPositioning,
    clear_store, dump_pickle, load_pickle, traverse_pipeline,
)
from casim.pipelines.taxonomy import variant

PLAN_SOURCES = {
    "batching": BatchingPlanCandidate,
    "routing": RoutingPlanCandidate,
    "scheduling": SchedulingPlanCandidate,
    "waiting": WaitingPlanCandidate,
    "positioning": PositioningPlanCandidate,
}


class CoSySolver:
    def __init__(
        self,
        instances_dir: Path,
        cache_dir: Path,
        output_dir: Path,
        instance_name: str,
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
        self.output = None
        self.problem_class = problem_class
        self.pipelines = None
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
        repo_classes = [get_class(path) for path in self.repo_cfg.components]
        stage_classes = {
            "batching": AbstractBatching,
            "routing": AbstractPickerRouting,
            "active_tour_routing": AbstractPickerRouting,
            "scheduling": AbstractScheduling,
            "waiting": AbstractWaiting,
            "positioning": AbstractPositioning,
        }
        features = data_card.decision_features["features"]
        available = [stage for stage in variant(problem, features)
                     if stage in stage_classes and any(
                         issubclass(component, stage_classes[stage])
                         for component in repo_classes)]
        if not available:
            raise ValueError(f"No configured terminal subproblem for {problem}")
        self.output = available[-1]
        terminal = "routing" if self.output == "active_tour_routing" else self.output
        PlanCandidateEndpoint.configure(data_card, self.algorithm_cards)

        if not self.pipelines:
            if self.verbose:
                print("Building pipelines")
            repo_classes.extend((PLAN_SOURCES[terminal], PlanCandidateEndpoint))
            repo = CoSyLuigiRepo(*repo_classes)
            maestro = Maestro(repo.cls_repo, repo.taxonomy)
            configured_classes = set(repo_classes)
            self.pipelines = [pipeline for pipeline in
                              maestro.query(PlanCandidateEndpoint.target())
                              if all(type(task) in configured_classes
                                     for task in traverse_pipeline([pipeline]))]
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

    def solve(self, dynamic_domain: BaseWarehouseDomain) -> dict[str, AlgorithmSolution]:
        self.dump_domain(dynamic_domain)
        if not self.pipelines:
            raise RuntimeError(f"No configured CoSy pipeline for {dynamic_domain.problem_class}")

        solutions = {}
        for index, pipeline in enumerate(self.pipelines):
            self.dump_domain(dynamic_domain)
            InMemoryDagExecutor().execute_many([pipeline])
            output_path = pipeline.output()["plan_candidate"].path
            key = f"{index}:{Path(output_path).stem}"
            solution = load_pickle(output_path)
            solutions[key] = solution
            self._cleanup_after_solution(self.output_folder)
        return solutions

    @staticmethod
    def _cleanup_after_solution(output_folder: Path):
        clear_store()
