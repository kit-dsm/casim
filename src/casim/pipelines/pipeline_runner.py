from pathlib import Path
from types import SimpleNamespace

import logging
import luigi
from cosy.maestro import Maestro
from cosy_luigi import CoSyLuigiRepo
from hydra.utils import get_class
from luigi.configuration import get_config
from ware_ops_algos.algorithms import CombinedRoutingSolution, SchedulingSolution, BatchingSolution, AlgorithmSolution
from ware_ops_algos.domain_models import BaseWarehouseDomain, DataCard
from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards
from ware_ops_algos.domain_algo_mapper.domain_algo_mapper import (
    DomainAlgorithmMapper,
)

from casim.pipelines.in_memory_dag import InMemoryDagExecutor
from casim.pipelines.solution_ranker import SolutionRanker
from casim.pipelines.problem_based_template import (
    AbstractItemAssignment,
    AbstractScheduling,
    BatchingNode,
    InstanceLoader,
    OrderSplitter,
    PickerRouting,
    PickListProvider,
    clear_store, iter_store, dump_pickle,
)

from casim.pipelines.taxonomy import TAXONOMY

logger = logging.getLogger(__name__)

class CoSySolver:
    def __init__(
        self,
        instances_dir: Path,
        cache_dir: Path,
        output_dir: Path,
        instance_name: str,
        solution_ranker: SolutionRanker,
        endpoint=None,
        problem_class=None,
        verbose: bool = False,
        luigi_cfg = None,
        repo=None,
        executor: str = "luigi",
    ):
        self.instance_name = instance_name
        self.instances_dir = Path(instances_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path: Path = self.cache_dir / "dynamic_info.pkl"
        self.verbose = verbose
        self.endpoint = endpoint
        self.problem_class = problem_class
        self.pipelines = None
        self.solution_ranker = solution_ranker
        self.luigi_cfg = luigi_cfg
        self.repo_cfg = repo
        self.executor = str(executor)
        if self.luigi_cfg is not None:
            self.luigi_logging_opts = SimpleNamespace(
                background=luigi_cfg.background,
                logdir=luigi_cfg.logdir,
                logging_conf_file=luigi_cfg.logging_conf_file,
                log_level=luigi_cfg.log_level,
            )
        else:
            self.luigi_logging_opts = None
        self.output_folder = Path(output_dir) / "cosy"
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.algorithm_cards = load_packaged_algo_cards()
        if self.verbose:
            logger.info(
                "Loaded %d model cards", len(self.algorithm_cards)
            )

    def dump_domain(self, dynamic_domain: BaseWarehouseDomain):
        dump_pickle(str(self.cache_path), dynamic_domain)

    def prepare(self, data_card: DataCard):
        config = get_config()
        config.set('PipelineParams', 'output_folder', str(self.output_folder))
        config.set('PipelineParams', 'domain_path', str(self.cache_path))
        if self.luigi_cfg.get("runtime") is not None:
            config.set(
                "PipelineParams",
                "runtime",
                str(self.luigi_cfg.runtime),
            )
        problem = data_card.problem_class
        endpoint_cls = get_class(TAXONOMY[problem]["endpoint"])
        endpoint_cls.configure(data_card, self.algorithm_cards)

        if not self.pipelines:
            if self.verbose:
                logger.info("Building pipelines")
            repo_classes = [get_class(path) for path in self.repo_cfg.components]
            mapper = DomainAlgorithmMapper(TAXONOMY)
            applicable_cards = mapper.filter(
                self.algorithm_cards,
                data_card,
                verbose=False,
            )
            applicable_ids = {id(card) for card in applicable_cards}
            algorithm_task_types = (
                AbstractItemAssignment,
                BatchingNode,
                PickerRouting,
                AbstractScheduling,
                OrderSplitter,
            )
            filtered_repo_classes = []
            for component_cls in repo_classes:
                matching_cards = [
                    card
                    for card in self.algorithm_cards
                    if component_cls.__name__
                    in {
                        card.algo_name,
                        card.implementation.get("component_name"),
                    }
                ]
                is_algorithm_task = issubclass(
                    component_cls,
                    algorithm_task_types,
                )
                if is_algorithm_task and not matching_cards:
                    raise ValueError(
                        "Configured algorithm component has no matching "
                        f"AlgorithmCard: {component_cls.__name__}"
                    )
                if matching_cards and not any(
                    id(card) in applicable_ids for card in matching_cards
                ):
                    if self.verbose:
                        logger.info(
                            "Excluding inapplicable component %s",
                            component_cls.__name__,
                        )
                    continue
                filtered_repo_classes.append(component_cls)

            configured_algorithms = [
                cls.__name__
                for cls in repo_classes
                if issubclass(cls, algorithm_task_types)
            ]
            retained_algorithms = [
                cls.__name__
                for cls in filtered_repo_classes
                if issubclass(cls, algorithm_task_types)
            ]
            if configured_algorithms and not retained_algorithms:
                raise ValueError(
                    "No configured algorithm component is applicable to "
                    f"{problem}: {configured_algorithms}"
                )
            repo = CoSyLuigiRepo(*filtered_repo_classes)
            maestro = Maestro(repo.cls_repo, repo.taxonomy)
            self.pipelines = list(maestro.query(endpoint_cls.target()))
            if not self.pipelines:
                raise ValueError(
                    "The applicable configured components cannot form a "
                    f"pipeline for {problem}"
                )
            if self.verbose:
                logger.info("Found %d pipelines", len(self.pipelines))
        else:
            if self.verbose:
                logger.info("Using cached pipelines")

    def solve(self, dynamic_domain: BaseWarehouseDomain, action: None) -> tuple[AlgorithmSolution, str, float] | None:
        self.dump_domain(dynamic_domain)
        if not self.pipelines:
            logger.error("No valid pipelines found")
            return None

        if self.executor == "memory":
            executor = InMemoryDagExecutor()
            executor.execute_many(self.pipelines)
        else:
            luigi.interface.InterfaceLogging.setup(self.luigi_logging_opts)
            if not action and not action == 0:
                luigi.build(self.pipelines, local_scheduler=True)
            else:
                luigi.build(self.pipelines[action], local_scheduler=True)

        solutions = self._load_solutions(dynamic_domain.problem_class)
        self._cleanup_after_solution(self.output_folder)
        best_solution, best_key, best_kpi_value = self.select_strategy(solutions, dynamic_domain.problem_class)
        return best_solution, best_key, best_kpi_value

    def select_strategy(self, solutions, problem):
        best_solution, best_key, best_kpi_value = self.solution_ranker.select_best(solutions, problem)
        return best_solution, best_key, best_kpi_value

    @staticmethod
    def _load_solutions(problem_class: str) -> dict:
        suffix = {
            "OBRSP": "scheduling_sol.pkl",
            "ORSP": "scheduling_sol.pkl",
            "OBP": "batching_sol.pkl",
            "OSBP": "batching_sol.pkl",
            "ORP": "routing_sol.pkl", "OBRP": "routing_sol.pkl",
            "BSRP": "routing_sol.pkl",
        }[problem_class]
        return {Path(p).stem: obj for p, obj in iter_store(f"*{suffix}")}

    @staticmethod
    def _cleanup_after_solution(output_folder: Path):
        clear_store()
