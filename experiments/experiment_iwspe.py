import csv
import pickle
import time
from abc import ABC

from pathlib import Path
from typing import Tuple

import hydra
import luigi
from hydra.utils import instantiate
from omegaconf import DictConfig

from ware_ops_algos.algorithms import PlanningState, CombinedRoutingSolution, AssignmentSolution, RoundRobinAssigner, \
    PickList, AlgorithmSolution, Algorithm, I, O
from ware_ops_algos.domain_models import BaseWarehouseDomain, datacard_from_instance
from ware_ops_algos.domain_models.taxonomy import SUBPROBLEMS
from ware_ops_pipes.pipelines import set_pipeline_params

from experiments.experiment_commons import build_state_transformers, build_trigger_map, build_req_policy, \
    dump_pipelines_csv, OnlineRunner, build_data_loader
from ware_ops_sim.data_loaders import IWSPELoader
from ware_ops_sim.sim import WarehouseSimulation, SimWarehouseDomain
from ware_ops_sim.sim.conditions import ShiftStartNumbOrdersCondition, NumberOrdersCondition
from ware_ops_sim.sim.decision_engine.sim_control import DecisionEngine
from ware_ops_sim.sim.events.events import OrderArrival, RoutingDone, OrderPriorityChange, PickerArrival, PickerTourQuery, \
    TourEnd, BreakStart
from ware_ops_sim.sim.state.state_transformer import OnlineStateTransformer
from ware_ops_algos.algorithms.algorithm_filter import AlgorithmFilter
from ware_ops_pipes.utils.experiment_utils import PipelineRunner, RankingEvaluatorDistance
from ware_ops_algos.utils.io_helpers import load_pickle, dump_pickle, dump_json


# class OnlineRunner(PipelineRunner):
#     """Runner for IOPVRP instances (paired files)"""
#     def __init__(self, instance_set_name: str, instances_dir: Path, cache_dir: Path,
#                  project_root: Path, instance_name: str, **loader_kwargs):
#         runner_kwargs = {k: v for k, v in loader_kwargs.items()
#                          if k in ['max_pipelines', 'verbose', 'cleanup']}
#
#         super().__init__(instance_set_name, instances_dir, cache_dir, project_root, **runner_kwargs)
#         self.instance_name = instance_name
#         self.run_id = -1
#         self.used_pipelines = {}
#         self.implementation_module = {
#             "GreedyIA": "ware_ops_pipes.pipelines.components.item_assignment.greedy_item_assignment",
#             # "NNIA": "ware_ops_pipes.pipelines.components.item_assignment.nn_item_assignment",
#             # "SinglePosIA": "ware_ops_pipes.pipelines.components.item_assignment.single_pos_item_assignment",
#             # "MinMinIA": "ware_ops_pipes.pipelines.components.item_assignment.min_min_item_assignment",
#             # "MinMaxIA": "ware_ops_pipes.pipelines.components.item_assignment.min_max_item_assignment",
#             "DummyOS": "ware_ops_pipes.pipelines.components.order_selection.dummy_order_selection",
#             # "MinMaxArticlesOS": "ware_ops_pipes.pipelines.components.order_selection.min_max_articles_os",
#             # "MinMaxAislesOS": "ware_ops_pipes.pipelines.components.order_selection.min_max_aisles_os",
#             "GreedyOS": "ware_ops_pipes.pipelines.components.order_selection.greedy_order_selection",
#             # "MinAisleConflictsOS": "ware_ops_pipes.pipelines.components.order_selection.min_aisle_conflicts_os",
#             # "MinDistOS": "ware_ops_pipes.pipelines.components.order_selection.min_dist_os",
#             # "MinSharedAislesOS": "ware_ops_pipes.pipelines.components.order_selection.min_shared_aisles_os",
#             # "SShape": "ware_ops_pipes.pipelines.components.routing.s_shape",
#             # "NearestNeighbourhood": "ware_ops_pipes.pipelines.components.routing.nn",
#             "PLRouting": "ware_ops_pipes.pipelines.components.routing.pl",
#             # "LargestGap": "ware_ops_pipes.pipelines.components.routing.largest_gap",
#             # "Midpoint": "ware_ops_pipes.pipelines.components.routing.midpoint",
#             # "Return": "ware_ops_pipes.pipelines.components.routing.return_algo",
#             # "ExactSolving": "ware_ops_pipes.pipelines.components.routing.exact_algo",
#             # "RatliffRosenthal": "ware_ops_pipes.pipelines.components.routing.sprp",
#             # "FiFo": "ware_ops_pipes.pipelines.components.batching.fifo",
#             # "OrderNrFiFo": "ware_ops_pipes.pipelines.components.batching.order_nr_fifo",
#             # "DueDate": "ware_ops_pipes.pipelines.components.batching.due_date",
#             # "Random": "ware_ops_pipes.pipelines.components.batching.random",
#             # "CombinedBatchingRoutingAssigning": "ware_ops_pipes.pipelines.components.routing.joint_batching_routing_assigning",
#             # "ClosestDepotMinDistanceSeedBatching": "ware_ops_pipes.pipelines.components.batching.seed",
#             # "ClosestDepotMaxSharedArticlesSeedBatching": "ware_ops_pipes.pipelines.components.batching.seed_shared_articles",
#             # "ClarkAndWrightSShape": "ware_ops_pipes.pipelines.components.batching.clark_and_wright_sshape",
#             # "ClarkAndWrightNN": "ware_ops_pipes.pipelines.components.batching.clark_and_wright_nn",
#             # "ClarkAndWrightRR": "ware_ops_pipes.pipelines.components.batching.clark_and_wright_rr",
#             # "LSBatchingRR": "ware_ops_pipes.pipelines.components.batching.ls_rr",
#             # "LSBatchingNNRand": "ware_ops_pipes.pipelines.components.batching.ls_nn_rand",
#             # "LSBatchingNNDueDate": "ware_ops_pipes.pipelines.components.batching.ls_nn_due",
#             # "LSBatchingNNFiFo": "ware_ops_pipes.pipelines.components.batching.ls_nn_fifo",
#             # "SPTScheduling": "ware_ops_pipes.pipelines.components.sequencing.spt_scheduling",
#             # "LPTScheduling": "ware_ops_pipes.pipelines.components.sequencing.lpt_scheduling",
#             "EDDScheduling": "ware_ops_pipes.pipelines.components.sequencing.edd_scheduling",
#             # "EDDSequencing": "ware_ops_pipes.pipelines.components.sequencing.edd_sequencing",
#             # "RRAssigner": "ware_ops_pipes.pipelines.components.picker_assignment.round_robin_assignment"
#         }
#
#     def dump_domain(self, dynamic_domain: BaseWarehouseDomain):
#         dump_pickle(str(self.cache_path), dynamic_domain)
#
#     def load_domain(self, instance_name: str, file_paths: list[Path]) -> BaseWarehouseDomain:
#         return load_pickle(str(self.cache_path))
#
#     def discover_instances(self) -> list[Tuple[str, list[Path]]]:
#         pass
#
#     def solve(self, dynamic_domain: BaseWarehouseDomain) -> CombinedRoutingSolution:
#         # instance_name = "sim_experiment"
#         self.dump_domain(dynamic_domain)
#
#         print("models:")
#         for m in self.models:
#             print(m)
#
#         # Filter applicable algorithms
#         algo_filter = AlgorithmFilter(SUBPROBLEMS)
#         models_applicable = algo_filter.filter(
#             algorithms=self.models,
#             instance=dynamic_domain,
#             verbose=self.verbose
#         )
#         self.run_id += 1
#         self._import_models(models_applicable)
#         dynamic_instance_name = self.instance_name + "_" + str(self.run_id)
#         output_folder = (
#                 self.project_root / "experiments" / "online" / "output"
#                 / dynamic_instance_name
#         )
#         output_folder.mkdir(parents=True, exist_ok=True)
#
#         set_pipeline_params(
#             output_folder=str(output_folder),
#             instance_set_name=self.instance_set_name,
#             instance_name=dynamic_instance_name,
#             instance_path="",
#             domain_path=str(self.cache_path)
#         )
#
#         # Build and run pipelines
#         pipelines = self._build_pipelines()
#
#         if pipelines:
#             print(f"\n✓ Running {len(pipelines)} pipelines...\n")
#             luigi.interface.InterfaceLogging.setup(type('opts',
#                                                         (),
#                                                         {'background': None,
#                                                          'logdir': None,
#                                                          'logging_conf_file': None,
#                                                          'log_level': 'CRITICAL'  # <<<<<<<<<<
#                                                          }))
#             luigi.build(pipelines, local_scheduler=True)
#
#             if self.cleanup:
#                 self._cleanup(output_folder)
#
#             sequencing_plans = self.load_sequencing_solutions(output_folder)
#             routing_plans = self.load_routing_solutions(output_folder)
#
#             # based on the resulting plans retrieve the best solution via simple ranking best -> worst
#             if dynamic_domain.problem_class in ["OBSRP", "OSRP"]:
#                 best_key, best_dist = None, float("inf")
#                 for k, plan in sequencing_plans.items():
#                     plan: PlanningState
#                     dist = sum(a.distance for a in plan.sequencing_solutions.jobs)
#                     if dist < best_dist:
#                         best_key, best_dist = k, dist
#                 print(sequencing_plans)
#                 print("best key", best_key)
#                 # print(sequencing_plans[best_key].provenance["instance_solving"]["algo"])
#                 solution_object = sequencing_plans[best_key].sequencing_solutions
#                 for o in dynamic_domain.orders.orders:
#                     self.used_pipelines[o.order_id] = sequencing_plans[best_key].provenance
#
#             elif dynamic_domain.problem_class in ["OBRP", "SPRP", "ORP"]:
#                 best_key, best_dist = None, float("inf")
#                 for k, plan in routing_plans.items():
#                     plan: PlanningState
#                     dist = sum(r.route.distance for r in plan.routing_solutions)
#                     if dist < best_dist:
#                         best_key, best_dist = k, dist
#                 solution = routing_plans[best_key].routing_solutions
#                 routes = []
#                 for r in solution:
#                     routes.append(r.route)
#                 solution_object = CombinedRoutingSolution(routes=routes)
#                 for o in dynamic_domain.orders.orders:
#                     self.used_pipelines[o.order_id] = routing_plans[best_key].provenance
#             else:
#                 raise ValueError
#             return solution_object
#         else:
#             print("⚠ No valid pipelines found!")
#
#     @staticmethod
#     def load_routing_solutions(base_dir: str):
#         sol_files = Path(base_dir).glob("**/*routing_plan.pkl")
#         solutions = {}
#         for f in sol_files:
#             with open(f, "rb") as fh:
#                 try:
#                     solutions[f.name] = pickle.load(fh)
#                 except Exception as e:
#                     print(f"❌ Failed to load {f}: {e}")
#         return solutions
#
#     @staticmethod
#     def load_sequencing_solutions(base_dir: str):
#         sol_files = Path(base_dir).glob("**/*scheduling_plan.pkl")
#         solutions = {}
#         for f in sol_files:
#             with open(f, "rb") as fh:
#                 try:
#                     solutions[f.name] = pickle.load(fh)
#                 except Exception as e:
#                     print(f"❌ Failed to load {f}: {e}")
#         return solutions


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

    # Configuration
    project_root = Path(cfg.project_root)
    instances_dir = Path(cfg.instances_base) / cfg.data_cards.name
    cache_dir = Path(cfg.cache_base) / cfg.data_cards.name
    cache_path = Path(cfg.cache_base) / "dynamic_info.pkl"

    cache_dir.mkdir(parents=True, exist_ok=True)

    orders_path = instances_dir / "orders.csv"
    layout_path = instances_dir / "layout.csv"

    # loader = IWSPELoader(
    #     instances_dir=instances_dir,
    #     cfg=cfg,
    # )
    # _ = loader.load(orders_path=orders_path, layout_path=layout_path, use_cache=False)

    state_transformers = build_state_transformers(cfg)

    ranker = RankingEvaluatorDistance(output_dir="", instance_name="")

    runner = OnlineRunner(instance_set_name=cfg.data_cards.name,
                          instances_dir=instances_dir,
                          cache_dir=cache_dir,
                          project_root=project_root,
                          instance_name=cfg.experiment.instance_name,
                          verbose=True)

    loader_kwargs = {k: instances_dir / v
                     for k, v in cfg.data_cards.source.items()
                     if k.endswith("path")}

    loader = build_data_loader(cfg)

    # domain = loader.load(orders_path=orders_path, layout_path=layout_path, use_cache=False)
    domain = loader.load(**loader_kwargs)
    dc = datacard_from_instance(domain, "initial_dc")

    print(dc)


    runner.build_pipelines(data_card=dc)

    trigger_map = build_trigger_map(cfg)
    req_policy = build_req_policy(cfg)
    learnable_problems = cfg.simulation.learnable_problems

    sim_control = DecisionEngine(execution=runner,
                                 selector=ranker,
                                 requirements_policies=req_policy,
                                 triggers=trigger_map,
                                 learnable_problems=learnable_problems
                                 )

    sim = WarehouseSimulation(
        state_transformers=state_transformers,
        control=sim_control,
        data_loader=loader,
        domain_cache_path=str(cache_path),
        order_list_path=orders_path,
        order_line_path=layout_path,
        loader_kwargs=loader_kwargs
    )

    def picker_arrival_hook(sim: WarehouseSimulation,
                            domain: SimWarehouseDomain):
        print("n_pickers", domain.resources.resources)
        for resource in domain.resources.resources:
            sim.add_event(PickerArrival(time=0,
                                        picker_id=resource.id))

    def scheduled_breaks_hook(sim: WarehouseSimulation,
                              domain: SimWarehouseDomain):
        # 09:00 break (2h after 07:00 start)
        sim.add_event(BreakStart(time=7200, duration=1800))

        # 12:00 break (5h after 07:00 start)
        sim.add_event(BreakStart(time=18000, duration=1800))

    sim.reset(hooks=[picker_arrival_hook])
    start = time.time()
    sim.run()
    end = time.time()
    print("runtime", end-start)


    # import cProfile
    # import pstats
    #
    # with cProfile.Profile() as pr:
    #     sim.run()
    # stats = pstats.Stats(pr)
    # stats.sort_stats('cumulative')
    # stats.print_stats(20)




    # dump_pipelines_csv("./used_pipelines.csv", runner.used_pipelines)
    # dump_json("./used_pipelines.json", runner.used_pipelines)


if __name__ == "__main__":
    main()
