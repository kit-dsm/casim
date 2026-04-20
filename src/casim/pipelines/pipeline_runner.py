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
)
from casim.pipelines.subproblems.item_assingment import GreedyIA
from casim.pipelines.subproblems.batching import ClarkAndWrightNN
from casim.pipelines.subproblems.picker_routing import SShape
from casim.pipelines.subproblems.scheduling import LPTScheduler
from casim.io_helpers import dump_pickle


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
                # OrderNrFiFo,
                # DueDate,
                # LSBatchingNNFiFo,
                # ClarkAndWrightSShape,
                ClarkAndWrightNN,
                SShape,
                # Return,
                # LargestGap,
                # Midpoint,
                # NearestNeighbourhood,
                # SPTScheduler,
                LPTScheduler,
                # EDDScheduler,
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

    def _load_solutions(self, problem_class: str) -> dict:
        """
        Load raw solution objects from output files.
        Keys are the task_id-based filenames, values are solution objects.
        """
        if problem_class in ["OBRSP", "ORSP"]:
            pattern = "*scheduling_sol.pkl"
        elif problem_class in ["OBP"]:
            pattern = "*pick_list_sol.pkl"
        elif problem_class in ["ORP", "OBRP", "BSRP"]:
            pattern = "*routing_sol.pkl"
        else:
            raise ValueError(f"Unknown problem class: {problem_class}")

        solutions = {}
        for f in self.output_folder.glob(pattern):
            with open(f, "rb") as fh:
                try:
                    solutions[f.stem] = pickle.load(fh)
                except Exception as e:
                    print(f"❌ Failed to load {f}: {e}")
        return solutions

    def _cleanup_after_solution(self, output_folder: Path):
        for file_path in output_folder.glob("*"):
            if file_path.is_file():
                file_path.unlink()