import csv
import json
import pickle
from pathlib import Path

import luigi
import pandas as pd
from cosy.maestro import Maestro
from luigi.configuration import get_config
from luigi.mock import MockFileSystem
from luigi.tools.deps_tree import print_tree

from hydra.utils import instantiate
from omegaconf import DictConfig

from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls_luigi.inhabitation_task import RepoMeta
from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator
from ware_ops_pipes.pipelines.templates import AbstractPickerRouting, AbstractPickListGeneration, RawPickListGeneration

from ware_ops_sim.data_loaders import IWSPELoader, IOPVRPLoader, IBRSPLoader, HennOnlineLoader
from cosy_luigi.combinatorics import CoSyLuigiRepo
from ware_ops_sim.pipelines.cosy_template_2 import InstanceLoader, GreedyIA, FiFo, OrderNrFiFo, DueDate, \
    PickListProvider, SShape, \
    RatliffRosenthal, NearestNeighbourhood, Evaluation, EvaluationPickList, EvaluationRouting, TSPRouting, Return, \
    LargestGap, Midpoint, LSBatchingNNFiFo, ClarkAndWrightNN, ClarkAndWrightSShape, LSBatchingSShapeFiFo, \
    FiFoBatchSelection, LPTScheduler, EvaluationScheduling, AbstractScheduling, EDDScheduler, SPTScheduler, ERDScheduler

from ware_ops_sim.sim.events.events import OrderArrival, RoutingDone, OrderPriorityChange, PickerArrival, \
    PickerTourQuery, OrderSelectionDone, TourEnd, PickerIdle, ShiftStart, PickListSelectionDone, FlushRemainingOrders, \
    PickListDone
import ware_ops_algos
from ware_ops_algos.algorithms import PlanningState, CombinedRoutingSolution, AssignmentSolution, RoundRobinAssigner, \
    PickList, AlgorithmSolution, Algorithm, I, O, GreedyItemAssignment, PickListRouting, WarehouseOrder, Job, NodeType, \
    NearestNeighbourhoodRouting
from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import BaseWarehouseDomain, DataCard
from ware_ops_algos.domain_models.taxonomy import SUBPROBLEMS
from ware_ops_algos.algorithms.algorithm_filter import AlgorithmFilter
from ware_ops_algos.utils.io_helpers import load_pickle, dump_pickle, dump_json
from ware_ops_algos.utils.general_functions import load_model_cards, import_model_class
from ware_ops_pipes import inhabit, set_pipeline_params
from ware_ops_pipes.utils.experiment_utils import RankingEvaluatorDistance



EVENT_REGISTRY = {
    "OrderArrival": OrderArrival,
    "RoutingDone": RoutingDone,
    "OrderPriorityChange": OrderPriorityChange,
    "PickerArrival": PickerArrival,
    "PickerTourQuery": PickerTourQuery,
    "PickerIdle": PickerIdle,
    "TourEnd": TourEnd,
    "OrderSelectionDone": OrderSelectionDone,
    "PickListSelectionDone": PickListSelectionDone,
    "ShiftStart": ShiftStart,
    "FlushRemainingOrders": FlushRemainingOrders,
    "PickListDone": PickListDone
}

LOADER_REGISTRY = {
    "IWSPELoader": IWSPELoader,
    "IOPVRPLoader": IOPVRPLoader,
    "IBRSPLoader": IBRSPLoader,
    "HennOnlineLoader": HennOnlineLoader
}

ENDPOINT_REGISTRY = {
    "AbstractPickerRouting": AbstractPickerRouting,
    "AbstractPickListGeneration": AbstractPickListGeneration,
    "RawPickListGeneration": RawPickListGeneration,
    "AbstractScheduling": AbstractScheduling,
    "EvaluationRouting": EvaluationRouting,
    "EvaluationPickList": EvaluationPickList,
    "EvaluationScheduling": EvaluationScheduling
}


def build_trigger_map(cfg: DictConfig) -> dict:
    """Build trigger_map from config: {EventClass -> policy_key}"""
    return {
        EVENT_REGISTRY[event_name]: policy_key
        for event_name, policy_key in cfg.decision_engine.triggers.items()
    }


def build_req_policy(cfg: DictConfig) -> dict:
    """Build req_policy from config: {policy_key -> Condition instance}"""
    return {
        policy_key: instantiate(condition_cfg)
        for policy_key, condition_cfg in cfg.decision_engine.conditions.items()
    }


def build_state_transformers(cfg: DictConfig) -> dict:
    return {
        problem_key: instantiate(st_cfg,
                                 domain_cache_path=Path(cfg.cache_base))
        for problem_key, st_cfg in cfg.decision_engine.state_transformer.items()
    }

def build_data_loader(cfg: DictConfig) -> DataLoader:
    data_loader_cls = cfg.data_card.source.data_loader
    print(data_loader_cls)
    data_loader = LOADER_REGISTRY[data_loader_cls](
        instances_dir=Path(cfg.instances_base) /
                      cfg.data_card.name,
                      cfg=cfg)
    return data_loader


def make_execution(cfg):
    return {problem_key: instantiate(
        st_cfg,
        instance_set_name=cfg.data_card.name,
        instances_dir=Path(cfg.instances_base),
        cache_dir=Path(cfg.cache_base) / cfg.data_card.name,
        project_root=Path(cfg.project_root),
        instance_name=cfg.experiment.instance_name,
        verbose=True) for problem_key, st_cfg in cfg.decision_engine.execution.items()
    }


def dump_pipelines_csv(filepath, pipelines):
    rows = []
    for order_id, stages in pipelines.items():
        row = {"order_id": order_id}
        for stage_name, stage_data in stages.items():
            row[f"{stage_name}_algo"] = stage_data.get("algo", "")
            row[f"{stage_name}_time"] = stage_data.get("time", 0)
        rows.append(row)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _build_pick_lists(orders: list[WarehouseOrder]):
    # build pick lists
    pick_positions = []
    for order in orders:
        for pos in order.pick_positions:
            pick_positions.append(pos)

    pick_list = PickList(
        pick_positions=pick_positions,
        release=0,
        earliest_due_date=0,
        orders=orders
    )
    return pick_list


class FixedRunnerSPRP:
    def __init__(self,
                 instance_set_name: str,
                 instances_dir: Path,
                 cache_dir: Path,
                 project_root: Path,
                 instance_name: str,
                 max_pipelines: int = 10,
                 verbose: bool = False,
                 cleanup: bool = True,
                 endpoint=None,
                 problem_class=None,
                 ranker=RankingEvaluatorDistance  # TODO USE THIS!!!
                 ):
        self.instance_set_name = instance_set_name
        self.instance_name = instance_name
        self.instances_dir = Path(instances_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path: Path = self.cache_dir / "dynamic_info.pkl"
        self.project_root = Path(project_root)
        self.verbose = verbose
        self.endpoint = endpoint
        self.problem_class = problem_class
        self.pipelines = None

        self.output_folder = (
                self.project_root / "experiments" / "output" / "cosy"
                / self.instance_set_name / self.instance_name
        )
        self.output_folder.mkdir(parents=True, exist_ok=True)

        pkg_dir = Path(ware_ops_algos.__file__).parent
        model_cards_path = pkg_dir / "algorithms" / "algorithm_cards"
        self.models = load_model_cards(str(model_cards_path))
        if self.verbose:
            print(f"Loaded {len(self.models)} model cards")

    def solve(self, dynamic_domain: BaseWarehouseDomain) -> AlgorithmSolution:
        layout = dynamic_domain.layout
        current_picker = dynamic_domain.warehouse_info.current_picker
        pick_list: PickList = dynamic_domain.warehouse_info.buffered_pick_lists[0]
        # print(pick_list.order_numbers)
        layout_network = layout.layout_network

        router = NearestNeighbourhoodRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=[current_picker],
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
            fixed_depot=True  # todo needs to come from config
        )
        routing_solution = router.solve(pick_list.pick_positions)
        routing_solution.route.pick_list = pick_list
        # plan.routing_solutions.append(routing_solution.routes[0])
        routes = []
        routes.append(routing_solution.route)
        solution_object = CombinedRoutingSolution(routes=routes)
        return solution_object


class OnlineRunner:
    def __init__(self,
                 instance_set_name: str,
                 instances_dir: Path,
                 cache_dir: Path,
                 project_root: Path,
                 instance_name: str,
                 max_pipelines: int = 10,
                 verbose: bool = False,
                 cleanup: bool = True,
                 endpoint=None,
                 problem_class=None,
                 ranker=RankingEvaluatorDistance  # TODO USE THIS!!!
                 ):

        self.instance_name = instance_name
        self.instance_set_name = instance_set_name
        self.instances_dir = Path(instances_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path: Path | None = None

        self.project_root = Path(project_root)
        self.src_dir = project_root / "src" / "warehouse_algos"
        self.max_pipelines = max_pipelines
        self.verbose = verbose
        self.cleanup = cleanup
        self.pipeline_runtimes = {}
        self.run_id = -1
        self.used_pipelines = {}
        self.endpoint = endpoint
        self.problem_class = problem_class
        self.implementation_module = {
            "GreedyIA": "ware_ops_pipes.pipelines.components.item_assignment.greedy_item_assignment",
            # "NNIA": "ware_ops_pipes.pipelines.components.item_assignment.nn_item_assignment",
            # "SinglePosIA": "ware_ops_pipes.pipelines.components.item_assignment.single_pos_item_assignment",
            # "MinMinIA": "ware_ops_pipes.pipelines.components.item_assignment.min_min_item_assignment",
            # "MinMaxIA": "ware_ops_pipes.pipelines.components.item_assignment.min_max_item_assignment",
            "DummyOS": "ware_ops_pipes.pipelines.components.order_selection.dummy_order_selection",
            # "MinMaxArticlesOS": "ware_ops_pipes.pipelines.components.order_selection.min_max_articles_os",
            # "MinMaxAislesOS": "ware_ops_pipes.pipelines.components.order_selection.min_max_aisles_os",
            # "GreedyOS": "ware_ops_pipes.pipelines.components.order_selection.greedy_order_selection",
            # "MinAisleConflictsOS": "ware_ops_pipes.pipelines.components.order_selection.min_aisle_conflicts_os",
            # "MinDistOS": "ware_ops_pipes.pipelines.components.order_selection.min_dist_os",
            # "MinSharedAislesOS": "ware_ops_pipes.pipelines.components.order_selection.min_shared_aisles_os",
            # "SShape": "ware_ops_pipes.pipelines.components.routing.s_shape",
            "NearestNeighbourhood": "ware_ops_pipes.pipelines.components.routing.nn",
            # "PLRouting": "ware_ops_pipes.pipelines.components.routing.pl",
            # "LargestGap": "ware_ops_pipes.pipelines.components.routing.largest_gap",
            # "Midpoint": "ware_ops_pipes.pipelines.components.routing.midpoint",
            # "Return": "ware_ops_pipes.pipelines.components.routing.return_algo",
            # "ExactSolving": "ware_ops_pipes.pipelines.components.routing.exact_algo",
            # "RatliffRosenthal": "ware_ops_pipes.pipelines.components.routing.sprp",
            # "FiFo": "ware_ops_pipes.pipelines.components.batching.fifo",
            # "OrderNrFiFo": "ware_ops_pipes.pipelines.components.batching.order_nr_fifo",
            "DueDate": "ware_ops_pipes.pipelines.components.batching.due_date",
            # "Random": "ware_ops_pipes.pipelines.components.batching.random",
            # "CombinedBatchingRoutingAssigning": "ware_ops_pipes.pipelines.components.routing.joint_batching_routing_assigning",
            # "ClosestDepotMinDistanceSeedBatching": "ware_ops_pipes.pipelines.components.batching.seed",
            # "ClosestDepotMaxSharedArticlesSeedBatching": "ware_ops_pipes.pipelines.components.batching.seed_shared_articles",
            # "ClarkAndWrightSShape": "ware_ops_pipes.pipelines.components.batching.clark_and_wright_sshape",
            # "ClarkAndWrightNN": "ware_ops_pipes.pipelines.components.batching.clark_and_wright_nn",
            # "ClarkAndWrightRR": "ware_ops_pipes.pipelines.components.batching.clark_and_wright_rr",
            # "LSBatchingRR": "ware_ops_pipes.pipelines.components.batching.ls_rr",
            # "LSBatchingNNRand": "ware_ops_pipes.pipelines.components.batching.ls_nn_rand",
            # "LSBatchingNNDueDate": "ware_ops_pipes.pipelines.components.batching.ls_nn_due",
            # "LSBatchingNNFiFo": "ware_ops_pipes.pipelines.components.batching.ls_nn_fifo",
            "SPTScheduling": "ware_ops_pipes.pipelines.components.sequencing.spt_scheduling",
            "LPTScheduling": "ware_ops_pipes.pipelines.components.sequencing.lpt_scheduling",
            "EDDScheduling": "ware_ops_pipes.pipelines.components.sequencing.edd_scheduling",
            # "EDDSequencing": "ware_ops_pipes.pipelines.components.sequencing.edd_sequencing",
            # "RRAssigner": "ware_ops_pipes.pipelines.components.picker_assignment.round_robin_assignment"
        }
        self.pipelines = None
        self.loader: DataLoader | None = None
        self.ranker = ranker
        pkg_dir = Path(ware_ops_algos.__file__).parent
        model_cards_path = pkg_dir / "algorithms" / "algorithm_cards"
        self.models = load_model_cards(str(model_cards_path))
        if self.verbose:
            print(f"Loaded {len(self.models)} model cards")

        self.dynamic_instance_name = self.instance_name + "_" + str(self.run_id)
        self.output_folder = (
                self.project_root / "experiments" / "online" / "output"
                / self.dynamic_instance_name
        )
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def dump_domain(self, dynamic_domain: BaseWarehouseDomain):
        print(self.cache_path)
        dump_pickle(str(self.cache_path), dynamic_domain)

    def build_pipelines(self, data_card: DataCard):
        # Filter applicable algorithms
        algo_filter = AlgorithmFilter(SUBPROBLEMS)
        models_applicable = algo_filter.filter(
            algorithms=self.models,
            instance=data_card,
            verbose=self.verbose
        )
        self.run_id += 1
        self._import_models(models_applicable)

        set_pipeline_params(
            output_folder=str(self.output_folder),
            instance_set_name=self.instance_set_name,
            instance_name=self.dynamic_instance_name,
            instance_path="",
            domain_path=str(self.cache_path)
        )

        # Build and run pipelines
        if not self.pipelines:
            print("Building pipelines")
            self.pipelines = self._build_pipelines()
            for p in self.pipelines:
                print(p)
            # print(self.pipelines)
        else:
            print("Using cached pipelines")

    def _build_pick_lists(self, orders: list[WarehouseOrder]):
        # build pick lists
        pick_positions = []
        for order in orders:
            for pos in order.pick_positions:
                pick_positions.append(pos)

        pick_list = PickList(
            pick_positions=pick_positions,
            release=0,
            earliest_due_date=0,
            orders=orders
        )
        return pick_list

    def _evaluate_due_dates(self, assignments: list[Job], orders: list[WarehouseOrder]):
        order_by_id = {o.order_id: o for o in orders}
        records = []
        for ass in assignments:
            end_time = ass.end_time
            for on in ass.route.pick_list.order_numbers:
                o = order_by_id.get(on)
                if o is None:
                    continue
                if o.due_date is None:
                    continue  # skip if no due date
                arrival_time = o.order_date
                start_time = ass.start_time
                due_ts = o.due_date  # .timestamp()
                lateness = end_time - due_ts
                records.append({
                    "order_number": on,
                    "arrival_time": arrival_time,
                    "start_time": start_time,
                    "batch_idx": ass.batch_idx,
                    "picker_id": ass.picker_id,
                    "completion_time": end_time,
                    "due_date": o.due_date,
                    "lateness": lateness,
                    "tardiness": max(0, lateness),
                    "on_time": end_time <= due_ts,
                })
        return pd.DataFrame(records)

    # def solve(self, dynamic_domain: BaseWarehouseDomain, action=None) -> AlgorithmSolution:
    #
    #     orders = dynamic_domain.orders
    #     layout = dynamic_domain.layout
    #     resources = dynamic_domain.resources
    #     articles = dynamic_domain.articles
    #     storage_locations = dynamic_domain.storage
    #     current_picker = dynamic_domain.warehouse_info.current_picker
    #
    #     layout_network = layout.layout_network
    #
    #     orders_selected = [orders.orders[0]]
    #
    #     selector = GreedyItemAssignment(storage_locations)
    #     ia_sol = selector.solve(orders_selected)
    #
    #     # pick_list = [pos for pos in orders.orders[0].pick_positions]
    #
    #     pick_lists = []
    #     for order in ia_sol.resolved_orders:
    #         pl = self._build_pick_lists([order])
    #         pick_lists.append(pl)
    #
    #     router = PickListRouting(
    #         start_node=layout_network.start_node,
    #         end_node=layout_network.end_node,
    #         closest_node_to_start=layout_network.closest_node_to_start,
    #         min_aisle_position=layout_network.min_aisle_position,
    #         max_aisle_position=layout_network.max_aisle_position,
    #         distance_matrix=layout_network.distance_matrix,
    #         predecessor_matrix=layout_network.predecessor_matrix,
    #         picker=[current_picker],
    #         gen_tour=True,
    #         gen_item_sequence=True,
    #         node_list=layout_network.node_list,
    #         node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
    #         idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
    #         fixed_depot=True  # todo needs to come from config
    #     )
    #     routing_solution = router.solve(pick_lists[0].pick_positions)
    #     routing_solution.route.pick_list = pick_lists[0]
    #     # plan.routing_solutions.append(routing_solution.routes[0])
    #     routes = []
    #     routes.append(routing_solution.route)
    #     solution_object = CombinedRoutingSolution(routes=routes)
    #     return solution_object

    def solve(self, dynamic_domain: BaseWarehouseDomain, action=None) -> AlgorithmSolution:
        self.dump_domain(dynamic_domain)
        if self.pipelines:
            luigi.interface.InterfaceLogging.setup(type('opts',
                                                        (),
                                                        {'background': None,
                                                         'logdir': None,
                                                         'logging_conf_file': None,
                                                         'log_level': 'DEBUG'
                                                         }))
            if action == None:
                # print(f"\n✓ Running {len(self.pipelines)} pipelines...\n")
                luigi.build(self.pipelines, local_scheduler=True)
            else:
                luigi.build([self.pipelines[action]], local_scheduler=True)

            if self.cleanup:
                self._cleanup(self.output_folder)

            sequencing_plans = self.load_sequencing_solutions(str(self.output_folder))
            routing_plans = self.load_routing_solutions(str(self.output_folder))
            pick_list_plans = self.load_batching_solutions(str(self.output_folder))

            # based on the resulting plans retrieve the best solution via simple ranking best -> worst
            if dynamic_domain.problem_class in ["OBP"]:
                print(type(pick_list_plans))
                best_key = None
                for k, plan in pick_list_plans.items():
                    print(k)
                    print()
                    best_key = k
                solution_object = pick_list_plans[best_key].batching_solutions

            elif dynamic_domain.problem_class in ["OBSRP", "OSRP"]:
                best_key, best_kpi = None, float("inf")
                print("plans", sequencing_plans)
                for k, plan in sequencing_plans.items():
                    plan: PlanningState
                    orders = []
                    for j in plan.sequencing_solutions.jobs:
                        for o in j.route.pick_list.orders:
                            orders.append(o)
                    eval_due_date = self._evaluate_due_dates(plan.sequencing_solutions.jobs, orders)
                    makespan = eval_due_date["completion_time"].min()
                    print(k)
                    print("Mean lateness", eval_due_date["lateness"].mean())
                    print("Max lateness", eval_due_date["lateness"].max())
                    print("# On time", eval_due_date["on_time"].sum())
                    dist = sum(a.distance for a in plan.sequencing_solutions.jobs)
                    print("Distance", dist)
                    # kpi = eval_due_date["tardiness"].max()
                    kpi = dist
                    # kpi = eval_due_date["on_time"].sum()
                    # kpi = makespan
                    test = plan.sequencing_solutions.jobs[0].route
                    pick_nodes = [n for n in test.annotated_route if n.node_type == NodeType.PICK]
                    print("From sol", len(pick_nodes))
                    if kpi < best_kpi:
                        best_key, best_kpi = k, dist
                # print(sequencing_plans)
                # print("best key", best_key)
                # print(sequencing_plans[best_key].provenance["instance_solving"]["algo"])
                solution_object = sequencing_plans[best_key].sequencing_solutions
                for o in dynamic_domain.orders.orders:
                    self.used_pipelines[o.order_id] = sequencing_plans[best_key].provenance


            elif dynamic_domain.problem_class in ["OBRP", "SPRP", "ORP"]:
                best_key, best_dist = None, float("inf")
                for k, plan in routing_plans.items():
                    plan: PlanningState
                    dist = sum(r.route.distance for r in plan.routing_solutions)
                    if dist < best_dist:
                        best_key, best_dist = k, dist
                solution = routing_plans[best_key].routing_solutions
                routes = []
                for r in solution:
                    routes.append(r.route)
                solution_object = CombinedRoutingSolution(routes=routes)
                for o in dynamic_domain.orders.orders:
                    self.used_pipelines[o.order_id] = routing_plans[best_key].provenance
            else:
                raise ValueError
            self._cleanup_after_solution(self.output_folder)
            return solution_object
        else:
            print("⚠ No valid pipelines found!")

    def _import_models(self, models_applicable):
        """Import applicable model implementations"""
        for model in models_applicable:
            model_name = model.model_name
            if model_name not in self.implementation_module:
                if self.verbose:
                    print(f"⚠ Unknown model: {model_name}, skipping...")
                continue

            try:
                module_path = self.implementation_module[model_name]
                cls = import_model_class(model_name, module_path)
                if self.verbose:
                    print(f"✅ {model_name}")
            except Exception as e:
                if self.verbose:
                    print(f"❌ Failed to import {model_name}: {e}")

    def _build_pipelines(self):
        """Build valid pipelines using inhabitation"""
        from ware_ops_pipes.pipelines.templates.template_1 import (
            InstanceLoader, AbstractItemAssignment, AbstractPickListGeneration,
            BatchedPickListGeneration, AbstractPickerAssignment, AbstractPickerRouting, AbstractSequencing,
            AbstractScheduling, AbstractResultAggregation
        )

        if self.endpoint:
            endpoint = ENDPOINT_REGISTRY[self.endpoint]
        else:
            endpoint = AbstractResultAggregation
        repository = RepoMeta.repository
        fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
        inhabitation_result, inhabitation_size = inhabit(endpoint)

        max_results = self.max_pipelines if inhabitation_size == 0 else inhabitation_size

        validator = UniqueTaskPipelineValidator([
            InstanceLoader,
            AbstractItemAssignment,
            AbstractPickListGeneration,
            # AbstractOrderSelection,
            BatchedPickListGeneration,
            AbstractPickerAssignment,
            AbstractPickerRouting,
            AbstractSequencing,
            AbstractScheduling,
            AbstractResultAggregation
        ])

        print(f"Enumerating up to {max_results} pipelines...")
        pipelines = [
            t() for t in inhabitation_result.evaluated[0:max_results]
            if validator.validate(t())
        ]

        if self.verbose and pipelines:
            print(f"✓ Found {len(pipelines)} valid pipelines")
            for i, pipeline in enumerate(pipelines[:3], 1):  # Show first 3
                print(f"\nPipeline {i}:")
                print(print_tree(pipeline))
        return pipelines

    def _cleanup(self, output_folder: Path):
        """Clean up intermediate files"""
        try:
            for file_path in output_folder.glob("InstanceLoader__*.pkl"):
                file_path.unlink()
                # if self.verbose:
                #     print(f"🗑 Deleted {file_path.name}")
        except Exception as e:
            print(f"⚠ Cleanup failed: {e}")

    def _cleanup_after_solution(self, output_folder: Path):
        """Clean up intermediate files"""
        for file_path in output_folder.glob("*"):
            file_path.unlink()
            # if self.verbose:
            #     print(f"🗑 Deleted {file_path.name}")

    def create_ranking(self, instance_name: str, output_folder: Path):
        """Create ranking for this instance"""
        try:
            # ranker = RankingEvaluatorDistance(
            #     output_dir=str(output_folder),
            #     instance_name=instance_name
            # )

            ranker = self.ranker(
                output_dir=str(output_folder),
                instance_name=instance_name
            )
            # Rank by distance
            df = ranker.evaluate("tours_summary.total_distance", minimize=True)

            # Best pipeline is first row
            if not df.empty:
                best = df.iloc[0]
                print(f"Best: {best['pipeline_id']} = {best['value']:.2f}")
                return best

        except Exception as e:
            print(f"⚠ Ranking error: {e}")

    def save_runtimes(self):
        output_folder = (
                self.project_root / "experiments" / "output"
                / "runtimes"
        )
        output_folder.mkdir(parents=True, exist_ok=True)
        with open(output_folder / f"{self.instance_set_name}.json", "w") as f:
            json.dump(self.pipeline_runtimes, f, indent=2)

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
        print("Base dir", base_dir)
        # sol_files = Path(base_dir).glob("**/*pick_list_plan.pkl")
        sol_files = Path(base_dir).glob("**/*pick_list_sol.pkl")
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
        # sol_files = Path(base_dir).glob("**/*scheduling_plan.pkl")
        sol_files = Path(base_dir).glob("**/*scheduling_sol.pkl")
        solutions = {}
        for f in sol_files:
            with open(f, "rb") as fh:
                try:
                    solutions[f.name] = pickle.load(fh)
                except Exception as e:
                    print(f"❌ Failed to load {f}: {e}")
        return solutions


class CoSyRunner:
    def __init__(
        self,
        instance_set_name: str,
        instances_dir: Path,
        cache_dir: Path,
        project_root: Path,
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
        self.project_root = Path(project_root)
        self.verbose = verbose
        self.endpoint = endpoint
        self.problem_class = problem_class
        self.pipelines = None

        self.output_folder = (
            self.project_root / "experiments" / "output" / "cosy"
            / self.instance_set_name / self.instance_name
        )
        self.output_folder.mkdir(parents=True, exist_ok=True)

        pkg_dir = Path(ware_ops_algos.__file__).parent
        model_cards_path = pkg_dir / "algorithms" / "algorithm_cards"
        self.models = load_model_cards(str(model_cards_path))
        if self.verbose:
            print(f"Loaded {len(self.models)} model cards")

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
        endpoint.configure(data_card, self.models)
        if not self.pipelines:
            if self.verbose:
                print("Building pipelines")
            repo = CoSyLuigiRepo(InstanceLoader,
                                 GreedyIA,
                                 # FiFo,
                                 # OrderNrFiFo,
                                 # DueDate,
                                 LSBatchingNNFiFo,
                                 ClarkAndWrightNN,
                                 ClarkAndWrightSShape,
                                 LSBatchingSShapeFiFo,
                                 PickListProvider,
                                 SShape,
                                 Return,
                                 LargestGap,
                                 Midpoint,
                                 NearestNeighbourhood,
                                 # RatliffRosenthal,
                                 # TSPRouting,
                                 # FiFoBatchSelection,
                                 LPTScheduler,
                                 EDDScheduler,
                                 SPTScheduler,
                                 ERDScheduler,
                                 EvaluationPickList,
                                 EvaluationRouting,
                                 EvaluationScheduling)
            maestro = Maestro(repo.cls_repo, repo.taxonomy)
            # results = maestro.query(Evaluation.target())
            self.pipelines = list(maestro.query(endpoint.target()))
            if self.verbose:
                print(f"✓ Found {len(self.pipelines)} pipelines")
        else:
            if self.verbose:
                print("Using cached pipelines")

    def solve(self, dynamic_domain: BaseWarehouseDomain, action=None):
        self.dump_domain(dynamic_domain)
        if not self.pipelines:
            print("⚠ No valid pipelines found!")
            return
        luigi.interface.InterfaceLogging.setup(type('opts',
                                                    (),
                                                    {'background': None,
                                                     'logdir': None,
                                                     'logging_conf_file': None,
                                                     'log_level': 'DEBUG'
                                                     }))
        luigi.build(self.pipelines, local_scheduler=True)
        self._cleanup(self.output_folder)
        # print("Output", self.output_folder)
        sequencing_plans = self.load_sequencing_solutions(str(self.output_folder))
        routing_plans = self.load_routing_solutions(str(self.output_folder))
        pick_list_plans = self.load_batching_solutions(str(self.output_folder))
        pl_selection_plans = self.load_pl_selection_solutions(str(self.output_folder))

        # based on the resulting plans retrieve the best solution via simple ranking best -> worst
        if dynamic_domain.problem_class in ["OBP"]:
            print(type(pick_list_plans))
            best_key = None
            for k, plan in pick_list_plans.items():
                print(k)
                print()
                best_key = k
            solution_object = pick_list_plans[best_key].batching_solutions
        elif dynamic_domain.problem_class in ["ORP", "OBRP", "BSRP"]:
            best_key, best_dist = None, float("inf")
            for k, plan in routing_plans.items():
                plan: PlanningState
                dist = sum(r.route.distance for r in plan.routing_solutions)
                if dist < best_dist:
                    best_key, best_dist = k, dist
            solution = routing_plans[best_key].routing_solutions
            routes = []
            for r in solution:
                routes.append(r.route)
            solution_object = CombinedRoutingSolution(routes=routes)

        elif dynamic_domain.problem_class in ["OBRSP"]:
            best_key, best_kpi = None, float("inf")
            print("plans", sequencing_plans)
            for k, plan in sequencing_plans.items():
                plan: PlanningState
                orders = []
                for j in plan.sequencing_solutions.jobs:
                    for o in j.route.pick_list.orders:
                        orders.append(o)
                eval_due_date = self._evaluate_due_dates(plan.sequencing_solutions.jobs, orders)
                makespan = eval_due_date["completion_time"].min()
                print(k)
                print("Mean lateness", eval_due_date["lateness"].mean())
                print("Max lateness", eval_due_date["lateness"].max())
                print("# On time", eval_due_date["on_time"].sum())
                print("Makespan", makespan)
                dist = sum(a.distance for a in plan.sequencing_solutions.jobs)
                print("Distance", dist)
                # kpi = eval_due_date["tardiness"].max()
                kpi = dist
                # kpi = eval_due_date["on_time"].sum()
                # kpi = makespan
                if kpi < best_kpi:
                    best_key, best_kpi = k, dist
            solution_object = sequencing_plans[best_key].sequencing_solutions
                # for o in dynamic_domain.orders.orders:
                #     self.used_pipelines[o.order_id] = sequencing_plans[best_key].provenance
        else:
            raise ValueError
        self._cleanup_after_solution(self.output_folder)
        return solution_object

    def _cleanup(self, output_folder: Path):
        """Clean up intermediate files"""
        try:
            for file_path in output_folder.glob("InstanceLoader__*.pkl"):
                file_path.unlink()
                # if self.verbose:
                #     print(f"🗑 Deleted {file_path.name}")
        except Exception as e:
            print(f"⚠ Cleanup failed: {e}")

    def _cleanup_after_solution(self, output_folder: Path):
        """Clean up intermediate files"""
        luigi.mock.MockFileSystem().get_all_data().clear()
        for file_path in output_folder.glob("*"):
            file_path.unlink()
            # if self.verbose:
            #     print(f"🗑 Deleted {file_path.name}")

    def _evaluate_due_dates(self, assignments: list[Job], orders: list[WarehouseOrder]):
        order_by_id = {o.order_id: o for o in orders}
        records = []
        for ass in assignments:
            end_time = ass.end_time
            for on in ass.route.pick_list.order_numbers:
                o = order_by_id.get(on)
                if o is None:
                    continue
                if o.due_date is None:
                    due_date = 999999999
                    on_time = True
                else:
                    due_date = o.due_date
                    on_time = end_time <= due_date
                arrival_time = o.order_date
                start_time = ass.start_time
                lateness = end_time - due_date
                records.append({
                    "order_number": on,
                    "arrival_time": arrival_time,
                    "start_time": start_time,
                    "batch_idx": ass.batch_idx,
                    "picker_id": ass.picker_id,
                    "completion_time": end_time,
                    "due_date": due_date,
                    "lateness": lateness,
                    "tardiness": max(0, lateness),
                    "on_time": on_time,
                })
        return pd.DataFrame(records)

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