import os
import time
from dataclasses import dataclass
from os.path import join as pjoin
import pickle
from pathlib import Path
from typing import Sequence, Callable, Iterable, Mapping

import luigi
import ware_ops_algos

from cosy.maestro import Maestro
from luigi import LocalTarget
from luigi.configuration import get_config
from luigi.mock import MockTarget
from luigi.tools import deps_tree
from ware_ops_algos.algorithms.algorithm_filter import ConstraintEvaluator, AlgorithmFilter
# from ware_ops_algos.domain_models.taxonomy import SUBPROBLEMS
from ware_ops_algos.utils.general_functions import load_model_cards, ModelCard

from ware_ops_sim.sim.sim_domain import DynamicInfo
from cosy_luigi.combinatorics import CoSyLuigiTask, CoSyLuigiTaskParameter, CoSyLuigiRepo
from ware_ops_algos.data_loaders import HesslerIrnichLoader
from ware_ops_algos.algorithms import (GreedyItemAssignment,
                                       DummyOrderSelection,
                                       SShapeRouting,
                                       NearestNeighbourhoodRouting,
                                       ExactTSPRoutingDistance,
                                       FifoBatching,
                                       PlanningState,
                                       OrderNrFifoBatching, Routing, WarehouseOrder, PickList, BatchingSolution,
                                       RatliffRosenthalRouting, DueDateBatching, Batching, ExactCombinedBatchingRouting,
                                       LargestGapRouting, MidpointRouting, ReturnRouting, LocalSearchBatching,
                                       ClarkAndWrightBatching, SchedulingInput, CombinedRoutingSolution,
                                       OrderSelectionSolution, PickListSelectionSolution, PriorityScheduling,
                                       LPTScheduling, EDDScheduling, SPTScheduling, ERDScheduling, EDDSequencer)
from ware_ops_algos.domain_models import BaseWarehouseDomain, Articles, Resources, StorageLocations, LayoutData, \
    DataCard, datacard_from_instance, OrdersDomain

from ware_ops_pipes.utils.io_helpers import load_pickle, dump_pickle

SUBPROBLEMS = {"OBRP": {"variables": ["item_assignment", "batching", "routing"]},
               "SPRP": {"variables": ["routing"]},
               "ORP": {
                   "objectives": ["tardiness", "picking_time", "cost", "completion_time"],
                   "variables": ["routing"]
               },
               "OBP": {
                    "objectives": ["tardiness", "picking_time", "cost"],
                    "variables": ["item_assignment", "order_selection", "batching"]
                    },
               "BSRP": {
                   "objectives": [],
                   "variables": ["batching", "routing"]
               },
               "OBRSP": {
                   "objectives": [],
                   "variables": ["item_assignment", "batching", "routing", "sequencing"]
               }
               }


@dataclass
class WarehouseContext:
    problem: str
    features: dict[str, dict]
    objective: str

    @classmethod
    def from_domain(cls, domain: BaseWarehouseDomain):
        features = {}
        for domain_name in ["layout", "resources", "orders", "storage", "articles"]:
            domain_obj = getattr(domain, domain_name)
            features[domain_name] = {
                "tpe": domain_obj.get_type_value(),
                "features": domain_obj.get_features(),
            }
        return cls(
            problem=domain.problem_class,
            features=features,
            objective=domain.objective,
        )


class PipelineParams(luigi.Config):
    output_folder = luigi.Parameter(default=pjoin(os.getcwd(), "outputs"))
    seed = luigi.IntParameter(default=42)

    instance_set_name = luigi.Parameter(default=None)
    instance_name = luigi.Parameter(default=None)
    instance_path = luigi.Parameter(default=None)
    domain_path = luigi.Parameter(default=None)
    pick_lists_path = luigi.Parameter(default=None)
    runtime = luigi.IntParameter(default=300)


class BaseComponent(CoSyLuigiTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pipeline_params = PipelineParams()
        os.makedirs(self.pipeline_params.output_folder, exist_ok=True)

    def get_luigi_local_target_with_task_id(
            self,
            out_name
    ) -> LocalTarget:
        return LocalTarget(
            pjoin(self.pipeline_params.output_folder,
                  out_name)
        )
        # return LocalTarget(out_name)


########## loading ###############

class InstanceLoader(BaseComponent):
    def output(self):
        return {
            "domain": self.get_luigi_local_target_with_task_id("domain.pkl"),
            "orders": self.get_luigi_local_target_with_task_id("orders.pkl"),
            "resources": self.get_luigi_local_target_with_task_id("resources.pkl"),
            "layout": self.get_luigi_local_target_with_task_id("layout.pkl"),
            "articles": self.get_luigi_local_target_with_task_id("articles.pkl"),
            "storage": self.get_luigi_local_target_with_task_id("storage.pkl"),
            "warehouse_info": self.get_luigi_local_target_with_task_id("warehouse_info.pkl"),
        }

    def run(self):
        domain_path = self.pipeline_params.domain_path
        if not domain_path:
            raise ValueError("Pipeline parameter 'domain_path' is not set.")

        # Load cached domain object
        domain: BaseWarehouseDomain = load_pickle(domain_path)
        print("Orders run", len(domain.orders.orders))
        # for target in self.output().values():
        #     os.makedirs(os.path.dirname(target.path), exist_ok=True)
        dump_pickle(self.output()["domain"].path, domain)
        dump_pickle(self.output()["orders"].path, domain.orders)
        dump_pickle(self.output()["resources"].path, domain.resources)
        dump_pickle(self.output()["layout"].path, domain.layout)
        dump_pickle(self.output()["articles"].path, domain.articles)
        dump_pickle(self.output()["storage"].path, domain.storage)
        dump_pickle(self.output()["warehouse_info"].path, domain.warehouse_info)


class IA(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)


class GreedyIA(IA):

    def output(self):
        return {
            "item_assignment_plan": self.get_luigi_local_target_with_task_id(
                self.__class__.__name__ + "-" + "item_assignment_plan.pkl"
            )
        }

    def run(self):
        orders_domain = load_pickle(self.input()["instance"]["orders"].path)
        storage: StorageLocations = load_pickle(self.input()["instance"]["storage"].path)
        selector = GreedyItemAssignment(storage)
        ia_sol = selector.solve(orders_domain.orders)
        orders_domain.orders = ia_sol.resolved_orders
        plan = PlanningState(
            item_assignment=ia_sol,
        )

        algo_name = selector.__class__.__name__

        plan.provenance["item_assignment"] = {
            "algo": algo_name,
            "time": ia_sol.execution_time,
        }
        dump_pickle(self.output()["item_assignment_plan"].path, plan)


########## batching ##############

class AbstractBatching(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)

    def _get_variant_label(self):
        pass

    def output(self):
        return {
            "pick_list_plan": self.get_luigi_local_target_with_task_id(
                self._get_variant_label() + "-" + self.__class__.__name__ + "-" + "pick_list_plan.pkl"
            )
        }


class PickListProvider(AbstractBatching):
    instance = CoSyLuigiTaskParameter(InstanceLoader)

    def _get_variant_label(self):
        return "BatchProvider"

    def _load_warehouse_info(self) -> DynamicInfo:
        return load_pickle(self.input()["instance"]["warehouse_info"].path)

    def run(self):
        # pick_lists_path = self.pipeline_params.pick_lists_path
        # plan = load_pickle(Path(pick_lists_path))
        warehouse_info = self._load_warehouse_info()
        pick_lists = warehouse_info.buffered_pick_lists
        batching_sol = BatchingSolution(pick_lists=pick_lists)
        plan = PlanningState(
            batching_solutions=batching_sol
        )
        dump_pickle(self.output()["pick_list_plan"].path, plan)


class FiFoBatchSelection(AbstractBatching):
    # Takes a list of batches and returns which batch to run next
    instance = CoSyLuigiTaskParameter(InstanceLoader)

    def _get_variant_label(self):
        return "OSBatchProvider"

    def _load_warehouse_info(self) -> DynamicInfo:
        return load_pickle(self.input()["instance"]["warehouse_info"].path)

    def run(self):
        warehouse_info = self._load_warehouse_info()
        pick_lists = warehouse_info.buffered_pick_lists
        batching_sol = BatchingSolution(pick_lists=[pick_lists[0]])
        plan = PlanningState(
            batching_solutions=batching_sol
        )
        dump_pickle(self.output()["pick_list_plan"].path, plan)


class BatchingNode(AbstractBatching):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    item_assignment_plan = CoSyLuigiTaskParameter(GreedyIA)

    def _get_variant_label(self):
        return Path(self.input()["item_assignment_plan"]["item_assignment_plan"].path).stem

    def _get_inited_batcher(self) -> Batching:
        ...

    @staticmethod
    def _latest_order_arrival(orders: list[WarehouseOrder]) -> float:
        if any(o.order_date is not None for o in orders):
            arrivals = [o.order_date for o in orders]
            return max(arrivals) if arrivals else 0.0
        else:
            return 0.0

    @staticmethod
    def _first_due_date(orders: list[WarehouseOrder]) -> float:
        if any(o.order_date is not None for o in orders):
            due_dates = [o.order_date for o in orders]
            return min(due_dates) if due_dates else float("inf")
        else:
            return 0.0

    def _build_pick_lists(self, orders: list[WarehouseOrder]):
        # build pick lists
        pick_positions = []
        for order in orders:
            for pos in order.pick_positions:
                pick_positions.append(pos)

        pick_list = PickList(
            pick_positions=pick_positions,
            release=self._latest_order_arrival(orders),
            earliest_due_date=self._first_due_date(orders),
            orders=orders
        )
        return pick_list

    def run(self):
        batcher: Batching = self._get_inited_batcher()
        plan: PlanningState = load_pickle(self.input()["item_assignment_plan"]["item_assignment_plan"].path)
        resolved_orders = plan.item_assignment.resolved_orders
        batching_sol = batcher.solve(resolved_orders)

        batches = batching_sol.batches
        pick_lists = []
        for batch in batches:
            pl = self._build_pick_lists(batch.orders)
            pick_lists.append(pl)

        batching_sol.pick_lists = pick_lists

        plan.batching_solutions = batching_sol

        if batcher.__class__.__name__ in ["SeedBatching", "ClarkAndWrightBatching", "LocalSearchBatching"]:
            batching_algo_name = batcher.algo_name
        else:
            batching_algo_name = batcher.__class__.__name__

        plan.provenance["routing_input"] = {
            "algo": batching_algo_name,
            "time": batcher.execution_time,
        }

        dump_pickle(self.output()["pick_list_plan"].path, plan)


# class RawPickListGeneration(BatchingNode):
#     def run(self):
#         plan: PlanningState = load_pickle(self.input()["item_assignment_plan"]["item_assignment_plan"].path)
#         orders = plan.item_assignment.resolved_orders
#         pick_lists = []
#         for order in orders:
#             pl = self._build_pick_lists([order])
#             pick_lists.append(pl)
#
#         batching_solution = BatchingSolution(pick_lists=pick_lists)
#
#         plan.batching_solutions = batching_solution
#
#         plan.provenance["routing_input"] = {
#             "algo": "RawInput",
#             "time": 0,
#         }
#
#         dump_pickle(self.output()["pick_list_plan"].path, plan)


class OrderNrFiFo(BatchingNode):  # -> BatchedPickListGeneration
    def _get_inited_batcher(self):
        articles: Articles = load_pickle(self.input()["instance"]["articles"].path)
        resources: Resources = load_pickle(self.input()["instance"]["resources"].path)
        batcher = OrderNrFifoBatching(
            pick_cart=resources.resources[0].pick_cart,
            articles=articles,
        )
        return batcher


class FiFo(BatchingNode):  # -> BatchedPickListGeneration
    def _get_inited_batcher(self):
        articles: Articles = load_pickle(self.input()["instance"]["articles"].path)
        resources: Resources = load_pickle(self.input()["instance"]["resources"].path)
        batcher = FifoBatching(
            pick_cart=resources.resources[0].pick_cart,
            articles=articles,
        )
        return batcher


class DueDate(BatchingNode):
    def _get_inited_batcher(self):
        articles: Articles = load_pickle(self.input()["instance"]["articles"].path)
        resources: Resources = load_pickle(self.input()["instance"]["resources"].path)
        batcher = DueDateBatching(
            pick_cart=resources.resources[0].pick_cart,
            articles=articles,
        )
        return batcher

class LSBatchingNNFiFo(BatchingNode):
    def _get_inited_batcher(self):
        articles: Articles = load_pickle(self.input()["instance"]["articles"].path)
        resources: Resources = load_pickle(self.input()["instance"]["resources"].path)
        layout: LayoutData = load_pickle(self.input()["instance"]["layout"].path)
        layout_network = layout.layout_network
        routing_kwargs = {
            "start_node": layout_network.start_node,
            "end_node": layout_network.end_node,
            "closest_node_to_start": layout_network.closest_node_to_start,
            "min_aisle_position": layout_network.min_aisle_position,
            "max_aisle_position": layout_network.max_aisle_position,
            "distance_matrix": layout_network.distance_matrix,
            "predecessor_matrix": layout_network.predecessor_matrix,
            "picker": resources.resources,
            "gen_tour": True,
            "gen_item_sequence": True,
            "node_list": layout_network.node_list,
            "node_to_idx": {node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            "idx_to_node": {idx: node for idx, node in enumerate(list(layout_network.graph.nodes))}
        }

        batcher = LocalSearchBatching(  # capacity=resources.resources[0].capacity,
            pick_cart=resources.resources[0].pick_cart,
            articles=articles,
            routing_class=NearestNeighbourhoodRouting,
            routing_class_kwargs=routing_kwargs,
            start_batching_class=FifoBatching,
            time_limit=self.pipeline_params.runtime)
        return batcher


class LSBatchingSShapeFiFo(BatchingNode):
    def _get_inited_batcher(self):
        articles: Articles = load_pickle(self.input()["instance"]["articles"].path)
        resources: Resources = load_pickle(self.input()["instance"]["resources"].path)
        layout: LayoutData = load_pickle(self.input()["instance"]["layout"].path)
        layout_network = layout.layout_network
        routing_kwargs = {
            "start_node": layout_network.start_node,
            "end_node": layout_network.end_node,
            "closest_node_to_start": layout_network.closest_node_to_start,
            "min_aisle_position": layout_network.min_aisle_position,
            "max_aisle_position": layout_network.max_aisle_position,
            "distance_matrix": layout_network.distance_matrix,
            "predecessor_matrix": layout_network.predecessor_matrix,
            "picker": resources.resources,
            "gen_tour": True,
            "gen_item_sequence": True,
            "node_list": layout_network.node_list,
            "node_to_idx": {node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            "idx_to_node": {idx: node for idx, node in enumerate(list(layout_network.graph.nodes))}
        }

        batcher = LocalSearchBatching(  # capacity=resources.resources[0].capacity,
            pick_cart=resources.resources[0].pick_cart,
            articles=articles,
            routing_class=SShapeRouting,
            routing_class_kwargs=routing_kwargs,
            start_batching_class=FifoBatching,
            time_limit=self.pipeline_params.runtime)
        return batcher


class ClarkAndWrightNN(BatchingNode):
    def _get_inited_batcher(self):
        articles: Articles = load_pickle(self.input()["instance"]["articles"].path)
        resources: Resources = load_pickle(self.input()["instance"]["resources"].path)
        layout: LayoutData = load_pickle(self.input()["instance"]["layout"].path)
        layout_network = layout.layout_network
        routing_kwargs = {"start_node": layout_network.start_node,
                         "end_node": layout_network.end_node,
                         "closest_node_to_start": layout_network.closest_node_to_start,
                         "min_aisle_position": layout_network.min_aisle_position,
                         "max_aisle_position": layout_network.max_aisle_position,
                         "distance_matrix": layout_network.distance_matrix,
                         "predecessor_matrix": layout_network.predecessor_matrix,
                         "picker": resources.resources,
                         "gen_tour": False,
                         "gen_item_sequence": False,
                         }

        batcher = ClarkAndWrightBatching(
            pick_cart=resources.resources[0].pick_cart,
            articles=articles,
            routing_class=NearestNeighbourhoodRouting,
            routing_class_kwargs=routing_kwargs,
            time_limit=self.pipeline_params.runtime
        )
        return batcher


class ClarkAndWrightSShape(BatchingNode):
    def _get_inited_batcher(self):
        articles: Articles = load_pickle(self.input()["instance"]["articles"].path)
        resources: Resources = load_pickle(self.input()["instance"]["resources"].path)
        layout: LayoutData = load_pickle(self.input()["instance"]["layout"].path)
        layout_network = layout.layout_network
        routing_kwargs = {"start_node": layout_network.start_node,
                         "end_node": layout_network.end_node,
                         "closest_node_to_start": layout_network.closest_node_to_start,
                         "min_aisle_position": layout_network.min_aisle_position,
                         "max_aisle_position": layout_network.max_aisle_position,
                         "distance_matrix": layout_network.distance_matrix,
                         "predecessor_matrix": layout_network.predecessor_matrix,
                         "picker": resources.resources,
                         "gen_tour": False,
                         "gen_item_sequence": False,
                         }

        batcher = ClarkAndWrightBatching(
            pick_cart=resources.resources[0].pick_cart,
            articles=articles,
            routing_class=SShapeRouting,
            routing_class_kwargs=routing_kwargs,
            time_limit=self.pipeline_params.runtime
        )
        return batcher


class CombinedBatchingRouting(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    item_assignment_plan = CoSyLuigiTaskParameter(GreedyIA)

    def _load_resources(self) -> Resources:
        return load_pickle(self.input()["instance"]["resources"].path)

    def _load_routing_data(self):
        return load_pickle(self.input()["instance"]["routing_input"].path)

    def _load_layout(self) -> LayoutData:
        return load_pickle(self.input()["instance"]["layout"].path)

    def _load_articles(self) -> Articles:
        return load_pickle(self.input()["instance"]["articles"].path)

    def run(self):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        graph = layout_network.graph

        plan: PlanningState = load_pickle(self.input()["item_assignment_plan"].path)
        resolved_orders = plan.item_assignment.resolved_orders

        pick_list_combined = []
        for order in resolved_orders:
            for pos in order.pick_positions:
                pick_list_combined.append(pos)

        router = ExactCombinedBatchingRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            big_m=1000,
            time_limit=self.pipeline_params.runtime,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(graph.nodes))},
        )

        sol = router.solve(pick_list_combined)


class PickerAssignment(BaseComponent):
    # decides for a SINGLE batch and a list of pickers which picker should pick it
    instance = CoSyLuigiTaskParameter(InstanceLoader)

    pass


class PickerRouting(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    pick_list_plan = CoSyLuigiTaskParameter(AbstractBatching)

    def _get_variant_label(self):
        return Path(self.input()["pick_list_plan"]["pick_list_plan"].path).stem

    def _get_inited_router(self) -> Routing:
        pass

    def _load_resources(self) -> Resources:
        return load_pickle(self.input()["instance"]["resources"].path)

    def _load_routing_data(self):
        return load_pickle(self.input()["instance"]["routing_input"].path)

    def _load_layout(self) -> LayoutData:
        return load_pickle(self.input()["instance"]["layout"].path)

    def _load_articles(self) -> Articles:
        return load_pickle(self.input()["instance"]["articles"].path)

    def run(self):
        router: Routing = self._get_inited_router()
        plan: PlanningState = load_pickle(self.input()["pick_list_plan"]["pick_list_plan"].path)
        pick_lists = plan.batching_solutions.pick_lists
        for i, pl in enumerate(pick_lists):
            routing_solution = router.solve(pl.pick_positions)
            routing_solution.route.pick_list = pl
            # plan.routing_solutions.append(routing_solution.routes[0])
            plan.routing_solutions.append(routing_solution)

            router.reset_parameters()
        plan.provenance["instance_solving"] = {
            "algo": router.__class__.__name__,
        }

        dump_pickle(self.output()["routing_plan"].path, plan)

    def output(self):
        return {
            "routing_plan": self.get_luigi_local_target_with_task_id(
                self._get_variant_label() + "-" + self.__class__.__name__ + "-" + "routing_plan.pkl"
            )
        }


class SShape(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        router = SShapeRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))}
        )

        return router


class LargestGap(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        router = LargestGapRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))}
        )

        return router


class Midpoint(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        router = MidpointRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))}
        )

        return router


class Return(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        router = ReturnRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))}
        )

        return router


class NearestNeighbourhood(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        router = NearestNeighbourhoodRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))}
        )

        return router


class TSPRouting(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        layout_network = layout.layout_network
        router = ExactTSPRoutingDistance(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
            set_time_limit=self.pipeline_params.runtime
        )

        return router


class RatliffRosenthal(PickerRouting):
    def _get_inited_router(self):
        resources = self._load_resources()
        layout = self._load_layout()
        graph_params = layout.graph_data
        layout_network = layout.layout_network

        rr_routing = RatliffRosenthalRouting(
            start_node=layout.graph_data.start_connection_point,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            n_aisles=graph_params.n_aisles,
            n_pick_locations=graph_params.n_pick_locations,
            dist_aisle=graph_params.dist_aisle,
            dist_pick_locations=graph_params.dist_pick_locations,
            dist_aisle_location=graph_params.dist_bottom_to_pick_location,
            dist_start=graph_params.dist_start,
            dist_end=graph_params.dist_end,
        )

        return rr_routing


class AbstractSequencing(BaseComponent):
    # Takes a list of batches and sequences them
    # -> returns an ordered list of batches with the first item to be considered first
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    routing_plan = CoSyLuigiTaskParameter(PickerRouting)

    def output(self):
        return {
            "sequencing_plan": self.get_luigi_local_target_with_task_id(
                self._get_variant_label() + "-" + self.__class__.__name__ + "-" + "sequencing_plan.pkl"
            )
        }

    def run(self):
        plan: PlanningState = load_pickle(self.input()["routing_plan"]["routing_plan"].path)
        routing_solutions = plan.routing_solutions
        orders = self._load_orders()
        resources = self._load_resources()

        if isinstance(routing_solutions, CombinedRoutingSolution):
            routes = routing_solutions.routes

            sequencing_inpt = SchedulingInput(routes=routes,
                                              orders=orders,
                                              resources=resources)
        else:
            routes = [route.route for route in routing_solutions]
            sequencing_inpt = SchedulingInput(routes=routes,
                                              orders=orders,
                                              resources=resources)

        sequencer = self._get_inited_sequencer()
        sequencing_solution = sequencer.solve(sequencing_inpt)

        plan.provenance["instance_solving"] = {
            "algo": sequencer.__class__.__name__,
        }

        plan.sequencing_solutions = sequencing_solution
        dump_pickle(self.output()["sequencing_plan"].path, plan)

    def _get_inited_sequencer(self) -> PriorityScheduling:
        ...

    def _load_resources(self) -> Resources:
        return load_pickle(self.input()["instance"]["resources"].path)

    def _load_orders(self) -> OrdersDomain:
        return load_pickle(self.input()["instance"]["orders"].path)

    def _load_routing_data(self):
        return load_pickle(self.input()["routing_input"]["routing_input"].path)

    def _load_routing_solution(self):
        return load_pickle(self.input()["routing_sol"]["routing_sol"].path)


class EDDSequencing(AbstractSequencing):
    def _get_inited_sequencer(self):
        orders = self._load_orders()
        return EDDSequencer(orders=orders)


class AbstractScheduling(BaseComponent):
    # Assignment + Sequencing integrated
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    routing_plan = CoSyLuigiTaskParameter(PickerRouting)

    def output(self):
        return {
            "scheduling_plan": self.get_luigi_local_target_with_task_id(
                self._get_variant_label() + "-" + self.__class__.__name__ + "-" + "scheduling_plan.pkl"
            )
        }

    def _get_variant_label(self):
        return Path(self.input()["routing_plan"]["routing_plan"].path).stem

    def run(self):
        plan: PlanningState = load_pickle(self.input()["routing_plan"]["routing_plan"].path)
        routing_solutions = plan.routing_solutions
        orders = self._load_orders()
        resources = self._load_resources()

        if isinstance(routing_solutions, CombinedRoutingSolution):
            routes = routing_solutions.routes

            sequencing_inpt = SchedulingInput(routes=routes,
                                              orders=orders,
                                              resources=resources)
        else:
            routes = [route.route for route in routing_solutions]
            sequencing_inpt = SchedulingInput(routes=routes,
                                              orders=orders,
                                              resources=resources)

        scheduler = self._get_inited_scheduler()
        scheduling_solution = scheduler.solve(sequencing_inpt)
        plan.provenance["instance_solving"] = {
            "algo": scheduler.__class__.__name__,
        }
        plan.sequencing_solutions = scheduling_solution
        dump_pickle(self.output()["scheduling_plan"].path, plan)

    def _get_inited_scheduler(self) -> PriorityScheduling:
        ...

    def _load_resources(self) -> Resources:
        return load_pickle(self.input()["instance"]["resources"].path)

    def _load_orders(self) -> OrdersDomain:
        return load_pickle(self.input()["instance"]["orders"].path)

    def _load_routing_data(self):
        return load_pickle(self.input()["routing_input"]["routing_input"].path)

    def _load_routing_solution(self):
        return load_pickle(self.input()["routing_sol"]["routing_sol"].path)


class EDDScheduler(AbstractScheduling):
    def _get_inited_scheduler(self):
        scheduler = EDDScheduling()
        return scheduler


class ERDScheduler(AbstractScheduling):
    def _get_inited_scheduler(self):
        scheduler = ERDScheduling()
        return scheduler


class LPTScheduler(AbstractScheduling):
    def _get_inited_scheduler(self):
        scheduler = LPTScheduling()
        return scheduler


class SPTScheduler(AbstractScheduling):
    def _get_inited_scheduler(self):
        scheduler = SPTScheduling()
        return scheduler


class Evaluation(BaseComponent):
    def _get_variant_label(self):
        pass

    def run(self):
        routing_plan = load_pickle(self.input()["routing_plan"].path)
        done = {"Done": 1}
        dump_pickle(self.output()["done"].path, done)

    def output(self):
        return {
            "done": self.get_luigi_local_target_with_task_id(
                self._get_variant_label() + "-" + self.__class__.__name__ + "-" + "done.pkl"
            )
        }

    # def output(self):
    #     return MockTarget("Finished")

    @classmethod
    def configure(cls, data_card: DataCard, models: list[ModelCard]):
        cls._data_card = data_card
        cls._models = models

    # Global? vs: vars (während der synthese zugewiesenen variablen)
    @classmethod
    def constraints(cls) -> Sequence[Callable[..., bool]]:
        return [
            lambda vs: problem_type_constraint(vs, SUBPROBLEMS, cls._data_card, cls._models),
            lambda vs: feature_constraint(vs, cls._data_card, cls._models),
            lambda vs: batching_loader_constraint(vs, SUBPROBLEMS, cls._data_card, PickListProvider),
            lambda vs: check_unique(vs, [Evaluation])
        ]


class EvaluationRouting(Evaluation):
    routing_plan = CoSyLuigiTaskParameter(PickerRouting)

    def _get_variant_label(self):
        return Path(self.input()["routing_plan"]["routing_plan"].path).stem

    def run(self):
        routing_plan = load_pickle(self.input()["routing_plan"]["routing_plan"].path)
        done = {"Done": 1}
        dump_pickle(self.output()["done"].path, done)
        # self.output().open("w").write("Ok.")

class EvaluationPickList(Evaluation):
    pick_list_plan = CoSyLuigiTaskParameter(AbstractBatching)

    def _get_variant_label(self):
        return Path(self.input()["pick_list_plan"]["pick_list_plan"].path).stem

    def run(self):
        routing_plan = load_pickle(self.input()["pick_list_plan"]["pick_list_plan"].path)
        done = {"Done": 1}
        dump_pickle(self.output()["done"].path, done)
        # self.output().open("w").write("Ok.")

class EvaluationScheduling(Evaluation):
    scheduling_plan = CoSyLuigiTaskParameter(AbstractScheduling)

    def _get_variant_label(self):
        return Path(self.input()["scheduling_plan"]["scheduling_plan"].path).stem

    def run(self):
        scheduling_plan = load_pickle(self.input()["scheduling_plan"]["scheduling_plan"].path)
        done = {"Done": 1}
        dump_pickle(self.output()["done"].path, done)
        # self.output().open("w").write("Ok.")


def traverse_pipeline(vs: Iterable[CoSyLuigiTask]) -> Iterable[CoSyLuigiTask]:
    result = [*vs]
    for v in result:
        req = v.requires()
        if isinstance(req, dict):
            req = v.requires().values()
        result.extend(traverse_pipeline(req))
    return result


def traverse_pipeline_2(vs: Iterable[CoSyLuigiTask]) -> Iterable[CoSyLuigiTask]:
    result = [*vs]
    for v in result:
        result.extend(traverse_pipeline(v.requires()))
    return result


def check_unique(vs: Mapping[str, CoSyLuigiTask], required_to_be_unique: Iterable[type[CoSyLuigiTask]]) -> bool:
    classes = [pc.__class__ for pc in traverse_pipeline(vs.values())]
    seen_subclasses = {}
    for c in classes:
        # print(c)
        for unique in required_to_be_unique:
            if issubclass(c, unique):
                if unique in seen_subclasses:
                    if seen_subclasses[unique] != c:
                        return False
                else:
                    seen_subclasses[unique] = c
    return True

# was prüfen die funktionen wirklich? Statt constraints
def batching_loader_constraint(vs, subproblems, data_card: DataCard, exclusive):
    classes = [pc.__class__ for pc in traverse_pipeline(vs.values())]
    problem = data_card.problem_class  # changed
    problems = subproblems[problem]["variables"]
    if "batching" in problems and exclusive in classes:
        return False
    return True


def problem_type_constraint(vs, subproblems, data_card: DataCard, models) -> bool:
    classes = [pc.__class__ for pc in traverse_pipeline(vs.values())]
    problem = data_card.problem_class  # changed
    problems = subproblems[problem]["variables"]
    for c in classes:
        for m in models:
            if m.implementation["class_name"] == c.__name__:
                if m.problem_type not in problems:
                    print(f"{m.model_name} not applicable {m.problem_type} not in {problems}")
                    return False
    return True


def feature_constraint(vs, data_card: DataCard, models) -> bool:
    classes = [pc.__class__ for pc in traverse_pipeline(vs.values())]

    # Build a flat features lookup from DataCard sections
    domain_sections = {
        "layout": data_card.layout,
        "articles": data_card.articles,
        "orders": data_card.orders,
        "resources": data_card.resources,
        "storage": data_card.storage,
    }

    for c in classes:
        for m in models:
            if m.model_name == c.__name__:
                # if m.model_name == "SShape":
                #     print()
                for domain, reqs in m.requirements.items():
                    section = domain_sections.get(domain)
                    if section is None:
                        continue

                    required_tpe = reqs["type"]
                    required_features = reqs.get("features", [])
                    required_features = [] if required_features in (None, [None]) else required_features
                    constraints = reqs.get("constraints", {})

                    domain_type = section["type"]
                    domain_features = []
                    for f in section["features"].keys():
                        if str(section["features"][f]) == "0":
                            domain_features.append(f)
                        if section["features"][f]:
                            domain_features.append(f)

                    if "any" not in required_tpe and domain_type not in required_tpe:
                        print(f"{m.model_name} not applicable")
                        return False

                    missing_features = [f for f in required_features if f not in domain_features]
                    if missing_features:
                        print(f"{m.model_name} not applicable, missing feature: {missing_features}")
                        return False

                    for feature_name, constraint in constraints.items():
                        if feature_name not in domain_features:
                            return False
                        evaluator = ConstraintEvaluator()
                        if not evaluator.evaluate(domain_features[feature_name], constraint):
                            return False
    return True





def main():
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

    instances_base = DATA_DIR / "instances"
    cache_base = DATA_DIR / "instances" / "caches"
    instance_set = "BahceciOencan"  # SPRP-SS
    cache_path = cache_base / instance_set

    instance_name = "Pr_20_1_20_Store1_01.txt"
    file_path = instances_base / instance_set / instance_name
    output_folder = (
            PROJECT_ROOT / "experiments" / "output" / "cosy"
            / instance_set / instance_name
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    #  DataCard rausschreiben

    loader = HesslerIrnichLoader(str(instances_base / instance_set), str(cache_base / instance_set))

    domain = loader.load(str(file_path))
    print("Orders initial", len(domain.orders.orders))

    dc = datacard_from_instance(domain, "test")
    dc.problem_class = "OBRP"
    pkg_dir = Path(ware_ops_algos.__file__).parent
    model_cards_path = pkg_dir / "algorithms" / "algorithm_cards"
    models = load_model_cards(str(model_cards_path))

    Evaluation.configure(dc, models)

    config = get_config()
    config.set('PipelineParams', 'output_folder', str(output_folder))
    config.set('PipelineParams', 'instance_set_name', instance_set)
    config.set('PipelineParams', 'instance_name', instance_name)
    config.set('PipelineParams', 'instance_path', str(file_path))
    config.set('PipelineParams', 'domain_path', str(loader.cache_path))
    config.set('PipelineParams', 'pick_lists_path', str(output_folder))


    repo = CoSyLuigiRepo(InstanceLoader,
                         GreedyIA,
                         FiFo,
                         OrderNrFiFo,
                         DueDate,
                         PickListProvider,
                         SShape,
                         RatliffRosenthal,
                         EvaluationPickList,
                         EvaluationRouting
                         )

    maestro = Maestro(repo.cls_repo, repo.taxonomy)
    results = maestro.query(EvaluationPickList.target())
    luigi.build(results, local_scheduler=True)

    print("OBP Done")
    dc.problem_class = "SPRP"
    maestro = Maestro(repo.cls_repo, repo.taxonomy)
    # for result in maestro.query(Evaluation.target()):
    #     print(deps_tree.print_tree(result))
    # pipelines = [r for r in maestro.query(Evaluation.target())]
    # print()
    results = maestro.query(EvaluationRouting.target())
    luigi.build(results, local_scheduler=True)

    # instance_name = "Pr_30_6_05_Store3_11.txt"
    # file_path = instances_base / instance_set / instance_name
    #
    # loader = HesslerIrnichLoader(str(instances_base / instance_set), str(cache_base / instance_set))
    # _ = loader.load(str(file_path))
    #
    # components = traverse_pipeline([pipelines[0]])
    # components_unique = list(set(components))
    #
    # luigi.build([pipelines[0]], local_scheduler=True)

    # results = maestro.query(Evaluation.target())
    # luigi.build(results, local_scheduler=True)


if __name__ == "__main__":
    main()