import pickle
from pathlib import Path

import luigi
import ware_ops_algos
from cosy.maestro import Maestro
from cosy_luigi import CoSyLuigiRepo
from luigi.configuration import get_config
from ware_ops_algos.domain_models import BaseWarehouseDomain, DataCard
from ware_ops_algos.utils.general_functions import load_model_cards

from casim.pipelines.problem_based_template import (
    InstanceLoader, PickListProvider, ResultAggregationPickList, ResultAggregationRouting, ResultAggregationScheduling,
    clear_store, iter_store, dump_pickle,
)
from casim.pipelines.subproblems.item_assingment import GreedyIA
from casim.pipelines.subproblems.batching import ClarkAndWrightNN, OrderNrFiFo
from casim.pipelines.subproblems.picker_routing import SShape, Return, LargestGap, Midpoint, NearestNeighbourhood
from casim.pipelines.subproblems.scheduling import LPTScheduler, SPTScheduler, EDDScheduler


ENDPOINT_REGISTRY = {
    "ResultAggregationRouting":   ResultAggregationRouting,
    "ResultAggregationPickList":  ResultAggregationPickList,
    "ResultAggregationScheduling": ResultAggregationScheduling,
}


class CoSyRunner:
    def __init__(
        self,
        instance_set_name: str,
        instances_dir: Path,
        cache_dir: Path,
        output_dir: Path,
        instance_name: str,
        endpoint=None,
        problem_class=None,
        verbose: bool = False,
    ):
        self.instance_set_name = instance_set_name
        self.instance_name = instance_name
        self.instances_dir = Path(instances_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path: Path = self.cache_dir / "dynamic_info.pkl"
        self.verbose = verbose
        self.endpoint = endpoint
        self.problem_class = problem_class
        self.pipelines = None

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

        endpoint_cls = ENDPOINT_REGISTRY[self.endpoint]
        endpoint_cls.configure(data_card, self.algorithm_cards,{}
                               # {"GreedyIA",
                               #  "Midpoint",
                               #  "LargestGap",
                               #  "NearestNeighbourhood",
                               #  "SShape",
                               #  "ClarkAndWrightNN",
                               #  "LPTScheduler"
                               #  }
                               )

        if not self.pipelines:
            if self.verbose:
                print("Building pipelines")

            repo = CoSyLuigiRepo(
                InstanceLoader,
                GreedyIA,
                # FiFo,
                OrderNrFiFo,
                # DueDate,
                # LSBatchingNNFiFo,
                # ClarkAndWrightSShape,
                # ClarkAndWrightNN,
                SShape,
                Return,
                LargestGap,
                Midpoint,
                NearestNeighbourhood,
                # SPTScheduler,
                # LPTScheduler,
                EDDScheduler,
                PickListProvider,
                # ResultAggregationPickList,
                # ResultAggregationRouting,
                ResultAggregationScheduling,
            )
            maestro = Maestro(repo.cls_repo, repo.taxonomy)
            self.pipelines = list(maestro.query(endpoint_cls.target()))
            if self.verbose:
                print(f"✓ Found {len(self.pipelines)} pipelines")
        else:
            if self.verbose:
                print("Using cached pipelines")

    def solve(self, dynamic_domain: BaseWarehouseDomain) -> dict:
        self.dump_domain(dynamic_domain)
        if not self.pipelines:
            print("⚠ No valid pipelines found!")
            return {}

        luigi.interface.InterfaceLogging.setup(type('opts',
                                                    (),
                                                    {'background': None,
                                                     'logdir': None,
                                                     'logging_conf_file': None,
                                                     'log_level': 'CRITICAL'
                                                     }))
        luigi.build(self.pipelines, local_scheduler=True)

        solutions = self._load_solutions(dynamic_domain.problem_class)
        self._cleanup_after_solution(self.output_folder)
        return solutions

    @staticmethod
    def _load_solutions(problem_class: str) -> dict:
        suffix = {
            "OBRSP": "scheduling_sol.pkl", "ORSP": "scheduling_sol.pkl",
            "OBP": "pick_list_sol.pkl",
            "ORP": "routing_sol.pkl", "OBRP": "routing_sol.pkl",
            "BSRP": "routing_sol.pkl",
        }[problem_class]
        return {Path(p).stem: obj for p, obj in iter_store(f"*{suffix}")}

    @staticmethod
    def _cleanup_after_solution(output_folder: Path):
        clear_store()