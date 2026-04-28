from pathlib import Path
from types import SimpleNamespace

import luigi
import ware_ops_algos
from cosy.maestro import Maestro
from cosy_luigi import CoSyLuigiRepo
from hydra.utils import get_class
from luigi.configuration import get_config
from ware_ops_algos.algorithms import CombinedRoutingSolution, SchedulingSolution, BatchingSolution, AlgorithmSolution
from ware_ops_algos.domain_models import BaseWarehouseDomain, DataCard
from ware_ops_algos.utils.general_functions import load_model_cards

from casim.pipelines.solution_ranker import SolutionRanker
from casim.pipelines.problem_based_template import (
    InstanceLoader, PickListProvider, ResultAggregationBatching, ResultAggregationRouting, ResultAggregationScheduling,
    clear_store, iter_store, dump_pickle, ResultAggregationSequencing,
)

from casim.pipelines.taxonomy import TAXONOMY

ENDPOINT_REGISTRY = {
    "ResultAggregationRouting":   ResultAggregationRouting,
    "ResultAggregationBatching":  ResultAggregationBatching,
    "ResultAggregationScheduling": ResultAggregationScheduling,
    "ResultAggregationSequencing": ResultAggregationSequencing,
}


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
        repo=None
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
        self.luigi_logging_opts =  SimpleNamespace(
            background=luigi_cfg.background,
            logdir=luigi_cfg.logdir,
            logging_conf_file=luigi_cfg.logging_conf_file,
            log_level=luigi_cfg.log_level,
        )
        self.output_folder = Path(output_dir) / "cosy"
        self.output_folder.mkdir(parents=True, exist_ok=True)

        pkg_dir = Path(ware_ops_algos.__file__).parent
        model_cards_path = pkg_dir / "algorithms" / "algorithm_cards"
        self.algorithm_cards = load_model_cards(str(model_cards_path))
        if self.verbose:
            print(f"Loaded {len(self.algorithm_cards)} model cards")

    def dump_domain(self, dynamic_domain: BaseWarehouseDomain):
        dump_pickle(str(self.cache_path), dynamic_domain)

    def build_pipelines(self, data_card: DataCard):
        config = get_config()
        config.set('PipelineParams', 'output_folder', str(self.output_folder))
        config.set('PipelineParams', 'domain_path', str(self.cache_path))
        problem = data_card.problem_class
        endpoint_str = TAXONOMY[problem]["endpoint"]
        endpoint_cls = ENDPOINT_REGISTRY[endpoint_str]
        endpoint_cls.configure(data_card, self.algorithm_cards)

        if not self.pipelines:
            if self.verbose:
                print("Building pipelines")
            repo_classes = [get_class(path) for path in self.repo_cfg.components]
            repo = CoSyLuigiRepo(*repo_classes)
            maestro = Maestro(repo.cls_repo, repo.taxonomy)
            self.pipelines = list(maestro.query(endpoint_cls.target()))
            if self.verbose:
                print(f"✓ Found {len(self.pipelines)} pipelines")
        else:
            if self.verbose:
                print("Using cached pipelines")

    def solve(self, dynamic_domain: BaseWarehouseDomain) -> tuple[AlgorithmSolution, str, float] | None:
        self.dump_domain(dynamic_domain)
        if not self.pipelines:
            print("⚠ No valid pipelines found!")
            return None

        luigi.interface.InterfaceLogging.setup(self.luigi_logging_opts)
        luigi.build(self.pipelines, local_scheduler=True)

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
            "OBRSP": "sequencing_sol.pkl",
            "ORSP": "scheduling_sol.pkl",
            "OBP": "pick_list_sol.pkl",
            "ORP": "routing_sol.pkl", "OBRP": "routing_sol.pkl",
            "BSRP": "routing_sol.pkl",
        }[problem_class]
        return {Path(p).stem: obj for p, obj in iter_store(f"*{suffix}")}

    @staticmethod
    def _cleanup_after_solution(output_folder: Path):
        clear_store()

