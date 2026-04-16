import pickle
from pathlib import Path

import luigi
import ware_ops_algos
from cosy.maestro import Maestro
from cosy_luigi.combinatorics import CoSyLuigiRepo
from luigi.configuration import get_config
from ware_ops_algos.domain_models import BaseWarehouseDomain, DataCard
from ware_ops_algos.utils.general_functions import load_model_cards

from casim.pipelines.pipeline_template import InstanceLoader, GreedyIA, FiFo, OrderNrFiFo, DueDate, LSBatchingNNFiFo, \
    ClarkAndWrightNN, ClarkAndWrightSShape, LSBatchingSShapeFiFo, PickListProvider, SShape, Return, LargestGap, \
    Midpoint, NearestNeighbourhood, RatliffRosenthal, TSPRouting, FiFoBatchSelection, LPTScheduler, EDDScheduler, \
    SPTScheduler, ERDScheduler, EvaluationPickList, EvaluationRouting, EvaluationScheduling
from scenarios.io_helpers import dump_pickle


ENDPOINT_REGISTRY = {
    "EvaluationRouting": EvaluationRouting,
    "EvaluationPickList": EvaluationPickList,
    "EvaluationScheduling": EvaluationScheduling
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
        print(self.cache_path)
        dump_pickle(str(self.cache_path), dynamic_domain)

    def build_pipelines(self, data_card: DataCard):
        config = get_config()
        config.set('PipelineParams', 'output_folder', str(self.output_folder))
        config.set('PipelineParams', 'instance_set_name', self.instance_set_name)
        config.set('PipelineParams', 'instance_name', self.instance_name)
        config.set('PipelineParams', 'instance_path', str(self.instances_dir / self.instance_set_name / self.instance_name))
        config.set('PipelineParams', 'domain_path', str(self.cache_path))
        config.set('PipelineParams', 'pick_lists_path', str(self.output_folder))

        # Evaluation.configure(data_card, self.models)
        endpoint = ENDPOINT_REGISTRY[self.endpoint]
        endpoint.configure(data_card, self.algorithm_cards)
        if not self.pipelines:
            if self.verbose:
                print("Building pipelines")
            repo = CoSyLuigiRepo(InstanceLoader,
                                 GreedyIA,
                                 # FiFo,
                                 # OrderNrFiFo,
                                 # DueDate,
                                 # LSBatchingNNFiFo,
                                 ClarkAndWrightNN,
                                 # ClarkAndWrightSShape,
                                 # LSBatchingSShapeFiFo,
                                 # PickListProvider,
                                 # SShape,
                                 # Return,
                                 # LargestGap,
                                 # Midpoint,
                                 NearestNeighbourhood,
                                 # RatliffRosenthal,
                                 # TSPRouting,
                                 # FiFoBatchSelection,
                                 # LPTScheduler,
                                 # EDDScheduler,
                                 SPTScheduler,
                                 # ERDScheduler,
                                 EvaluationPickList,
                                 EvaluationRouting,
                                 EvaluationScheduling)

            maestro = Maestro(repo.cls_repo, repo.taxonomy)
            self.pipelines = list(maestro.query(endpoint.target()))
            if self.verbose:
                print(f"✓ Found {len(self.pipelines)} pipelines")
        else:
            if self.verbose:
                print("Using cached pipelines")

    def solve(self, dynamic_domain: BaseWarehouseDomain):
        self.dump_domain(dynamic_domain)
        if not self.pipelines:
            print("⚠ No valid pipelines found!")
            return
        luigi.interface.InterfaceLogging.setup(type('opts',
                                                    (),
                                                    {'background': None,
                                                     'logdir': None,
                                                     'logging_conf_file': None,
                                                     'log_level': 'CRITICAL'
                                                     }))
        luigi.build(self.pipelines, local_scheduler=True)
        # self._cleanup(self.output_folder)
        plans = None
        if dynamic_domain.problem_class in ["OBRSP"]:
            plans = self.load_sequencing_solutions(str(self.output_folder))
        elif dynamic_domain.problem_class in ["OBP"]:
            plans = self.load_batching_solutions(str(self.output_folder))
        elif dynamic_domain.problem_class in ["OBRP"]:
            plans = self.load_routing_solutions(str(self.output_folder))
        self._cleanup_after_solution(self.output_folder)
        return plans

    def _cleanup(self, output_folder: Path):
        """Clean up intermediate files"""
        try:
            for file_path in output_folder.glob("InstanceLoader__*.pkl"):
                file_path.unlink()
        except Exception as e:
            print(f"⚠ Cleanup failed: {e}")

    def _cleanup_after_solution(self, output_folder: Path):
        """Clean up intermediate files"""
        luigi.mock.MockFileSystem().get_all_data().clear()
        for file_path in output_folder.glob("*"):
            file_path.unlink()

    @staticmethod
    def load_routing_solutions(base_dir: str):
        sol_files = Path(base_dir).glob("**/*routing_plan.pkl")
        solutions = {}
        for f in sol_files:
            with open(f, "rb") as fh:
                try:
                    solutions[f.name] = pickle.load(fh)
                except Exception as e:
                    print(f"❌ Failed to load {f}: {e}")
        return solutions

    @staticmethod
    def load_batching_solutions(base_dir: str):
        sol_files = Path(base_dir).glob("**/*pick_list_plan.pkl")
        solutions = {}
        for f in sol_files:
            with open(f, "rb") as fh:
                try:
                    solutions[f.name] = pickle.load(fh)
                except Exception as e:
                    print(f"❌ Failed to load {f}: {e}")
        return solutions

    @staticmethod
    def load_sequencing_solutions(base_dir: str):
        sol_files = Path(base_dir).glob("**/*scheduling_plan.pkl")
        solutions = {}
        for f in sol_files:
            with open(f, "rb") as fh:
                try:
                    solutions[f.name] = pickle.load(fh)
                except Exception as e:
                    print(f"❌ Failed to load {f}: {e}")
        return solutions

    @staticmethod
    def load_pl_selection_solutions(base_dir: str):
        sol_files = Path(base_dir).glob("**/*pick_list_plan.pkl")
        solutions = {}
        for f in sol_files:
            with open(f, "rb") as fh:
                try:
                    solutions[f.name] = pickle.load(fh)
                except Exception as e:
                    print(f"❌ Failed to load {f}: {e}")
        return solutions