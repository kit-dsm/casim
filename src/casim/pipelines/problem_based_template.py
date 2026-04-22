import fnmatch
import os
import pickle
from os.path import join as pjoin
from pathlib import Path
from typing import Sequence, Callable, Iterable, Mapping


from cosy.maestro import Maestro

import luigi
from luigi.configuration import get_config


from cosy_luigi import CoSyLuigiTask, CoSyLuigiTaskParameter, CoSyLuigiRepo

from ware_ops_algos.algorithms import (
    Routing,
    WarehouseOrder,
    PickList,
    BatchingSolution,
    Batching,
    SchedulingInput,
    CombinedRoutingSolution,
    ItemAssignmentSolution,
    RoutingSolution,
    PriorityScheduling,
    ItemAssignment, )
from ware_ops_algos.algorithms.algorithm_filter import ConstraintEvaluator
from ware_ops_algos.domain_models import (
    Articles,
    Resources,
    LayoutData,
    DataCard,
    OrdersDomain,
)
from ware_ops_algos.utils.general_functions import ModelCard

from casim.domain_objects.sim_domain import DynamicInfo, SimWarehouseDomain
from casim.pipelines.taxonomy import TAXONOMY
from casim.io_helpers import dump_json


class PipelineParams(luigi.Config):
    output_folder = luigi.Parameter(default=pjoin(os.getcwd(), "outputs"))
    seed = luigi.IntParameter(default=42)
    domain_path = luigi.Parameter(default=None)
    runtime = luigi.IntParameter(default=300)

_STORE: dict[str, object] = {}

class MemoryTarget(luigi.Target):
    def __init__(self, path: str):
        self.path = path

    def exists(self):
        hit = self.path in _STORE
        return hit

def dump_pickle(path, obj):
    _STORE[str(path)] = obj

def load_pickle(path):
    path = str(path)
    if path in _STORE:
        return _STORE[path]
    with open(path, "rb") as f:
        return pickle.load(f)

def iter_store(pattern: str):
    for k, v in _STORE.items():
        if fnmatch.fnmatch(k, pattern):
            yield k, v

def clear_store(prefix: str | None = None):
    if prefix is None:
        _STORE.clear()
    else:
        for k in [k for k in _STORE if k.startswith(prefix)]:
            del _STORE[k]

#


class BaseComponent(CoSyLuigiTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline_params = PipelineParams()

    def get_luigi_local_target_with_task_id(self, out_name) -> MemoryTarget:
        return MemoryTarget(
            pjoin(self.pipeline_params.output_folder,
                  self.task_id + "_" + out_name)
        )

# ─────────────────────────── Loading ────────────────────────────────────────


class InstanceLoader(BaseComponent):
    def output(self):
        return {
            "domain":         self.get_luigi_local_target_with_task_id("domain.pkl"),
            "orders":         self.get_luigi_local_target_with_task_id("orders.pkl"),
            "resources":      self.get_luigi_local_target_with_task_id("resources.pkl"),
            "layout":         self.get_luigi_local_target_with_task_id("layout.pkl"),
            "articles":       self.get_luigi_local_target_with_task_id("articles.pkl"),
            "storage":        self.get_luigi_local_target_with_task_id("storage.pkl"),
            "dynamic_warehouse_info": self.get_luigi_local_target_with_task_id("dynamic_warehouse_info.pkl"),
        }

    def run(self):
        domain_path = self.pipeline_params.domain_path
        if not domain_path:
            raise ValueError("Pipeline parameter 'domain_path' is not set.")
        domain: SimWarehouseDomain = load_pickle(domain_path)
        dump_pickle(self.output()["domain"].path, domain)
        dump_pickle(self.output()["orders"].path, domain.orders)
        dump_pickle(self.output()["resources"].path, domain.resources)
        dump_pickle(self.output()["layout"].path, domain.layout)
        dump_pickle(self.output()["articles"].path, domain.articles)
        dump_pickle(self.output()["storage"].path, domain.storage)
        dump_pickle(self.output()["dynamic_warehouse_info"].path, domain.dynamic_warehouse_info)


# ─────────────────────────── Item Assignment ────────────────────────────────

class AbstractItemAssignment(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)

    def output(self):
        return {
            "item_assignment_sol": self.get_luigi_local_target_with_task_id("item_assignment_sol.pkl")
        }

    def run(self):
        orders_domain: OrdersDomain = load_pickle(self.input()["instance"]["orders"].path)
        item_assigner = self._get_inited_ia()
        ia_sol: ItemAssignmentSolution = item_assigner.solve(orders_domain.orders)
        dump_pickle(self.output()["item_assignment_sol"].path, ia_sol)

    def _get_storage(self):
        storage = load_pickle(self.input()["instance"]["storage"].path)
        return storage

    def _get_inited_ia(self) -> ItemAssignment:
        ...


# ─────────────────────────── Batching ───────────────────────────────────────

class AbstractBatching(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)

    def output(self):
        return {
            "pick_list_sol": self.get_luigi_local_target_with_task_id("pick_list_sol.pkl")
        }

    def _get_articles(self) -> Articles:
        articles = load_pickle(self.input()["instance"]["articles"].path)
        return articles

    def _get_resources(self) -> Resources:
        resources = load_pickle(self.input()["instance"]["resources"].path)
        return resources

    def _get_layout(self) -> LayoutData:
        layout = load_pickle(self.input()["instance"]["layout"].path)
        return layout


class PickListProvider(AbstractBatching):
    instance = CoSyLuigiTaskParameter(InstanceLoader)

    def _load_warehouse_info(self) -> DynamicInfo:
        return load_pickle(self.input()["instance"]["dynamic_warehouse_info"].path)

    def run(self):
        warehouse_info = self._load_warehouse_info()
        pick_lists = warehouse_info.buffered_pick_lists
        batching_sol = BatchingSolution(pick_lists=pick_lists)
        dump_pickle(self.output()["pick_list_sol"].path, batching_sol)


class BatchingNode(AbstractBatching):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    item_assignment_sol = CoSyLuigiTaskParameter(AbstractItemAssignment)

    def _get_inited_batcher(self) -> Batching:
        ...

    @staticmethod
    def _latest_order_arrival(orders: list[WarehouseOrder]) -> float:
        if any(o.order_date is not None for o in orders):
            arrivals = [o.order_date for o in orders]
            return max(arrivals) if arrivals else 0.0
        return 0.0

    @staticmethod
    def _first_due_date(orders: list[WarehouseOrder]) -> float:
        if any(o.order_date is not None for o in orders):
            due_dates = [o.order_date for o in orders]
            return min(due_dates) if due_dates else float("inf")
        return 0.0

    def _build_pick_lists(self, orders: list[WarehouseOrder]) -> PickList:
        pick_positions = [pos for order in orders for pos in order.pick_positions]
        return PickList(
            pick_positions=pick_positions,
            release=self._latest_order_arrival(orders),
            earliest_due_date=self._first_due_date(orders),
            orders=orders,
        )

    def run(self):
        batcher: Batching = self._get_inited_batcher()
        ia_sol: ItemAssignmentSolution = load_pickle(
            self.input()["item_assignment_sol"]["item_assignment_sol"].path
        )
        resolved_orders = ia_sol.resolved_orders
        batching_sol = batcher.solve(resolved_orders)

        pick_lists = [self._build_pick_lists(batch.orders) for batch in batching_sol.batches]
        batching_sol.pick_lists = pick_lists

        if batcher.__class__.__name__ in ["SeedBatching", "ClarkAndWrightBatching", "LocalSearchBatching"]:
            batching_sol.algo_name = batcher.algo_name
        else:
            batching_sol.algo_name = batcher.__class__.__name__

        dump_pickle(self.output()["pick_list_sol"].path, batching_sol)


# ─────────────────────────── Routing ────────────────────────────────────────

class PickerRouting(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    pick_list_sol = CoSyLuigiTaskParameter(AbstractBatching)

    def _get_inited_router(self) -> Routing:
        pass

    def _load_resources(self) -> Resources:
        return load_pickle(self.input()["instance"]["resources"].path)

    def _load_layout(self) -> LayoutData:
        return load_pickle(self.input()["instance"]["layout"].path)

    def _load_articles(self) -> Articles:
        return load_pickle(self.input()["instance"]["articles"].path)

    def output(self):
        return {
            "routing_sol": self.get_luigi_local_target_with_task_id("routing_sol.pkl")
        }

    def run(self):
        router: Routing = self._get_inited_router()
        pick_list_sol: BatchingSolution = load_pickle(
            self.input()["pick_list_sol"]["pick_list_sol"].path
        )
        routes = []
        algo_name = None
        execution_time = 0
        for pl in pick_list_sol.pick_lists:
            routing_solution: RoutingSolution = router.solve(pl.pick_positions)
            algo_name = routing_solution.algo_name
            routing_solution.route.pick_list = pl
            routes.append(routing_solution.route)
            execution_time += routing_solution.execution_time
            router.reset_parameters()

        combined_sol = CombinedRoutingSolution(
            algo_name=algo_name,
            execution_time=execution_time,
            routes=routes
        )
        dump_pickle(self.output()["routing_sol"].path, combined_sol)


# ─────────────────────────── Scheduling ─────────────────────────────────────

class AbstractScheduling(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    routing_sol = CoSyLuigiTaskParameter(PickerRouting)

    def output(self):
        return {
            "scheduling_sol": self.get_luigi_local_target_with_task_id("scheduling_sol.pkl")
        }

    def _get_inited_scheduler(self) -> PriorityScheduling:
        ...

    def _load_resources(self) -> Resources:
        return load_pickle(self.input()["instance"]["resources"].path)

    def _load_orders(self) -> OrdersDomain:
        return load_pickle(self.input()["instance"]["orders"].path)

    def run(self):
        routing_sol = load_pickle(self.input()["routing_sol"]["routing_sol"].path)
        orders = self._load_orders()
        resources = self._load_resources()

        if isinstance(routing_sol, CombinedRoutingSolution):
            routes = routing_sol.routes
        else:
            routes = [r.route for r in routing_sol]

        scheduling_input = SchedulingInput(routes=routes, orders=orders, resources=resources)
        scheduler = self._get_inited_scheduler()
        scheduling_sol = scheduler.solve(scheduling_input)
        dump_pickle(self.output()["scheduling_sol"].path, scheduling_sol)

# ─────────────────────────── Sequencing ─────────────────────────────────────


class AbstractSequencing(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    routing_sol = CoSyLuigiTaskParameter(PickerRouting)

    def output(self):
        return {
            "sequencing_sol": self.get_luigi_local_target_with_task_id("sequencing_sol.pkl")
        }

    def _get_inited_sequencer(self) -> PriorityScheduling:
        ...

    def _load_resources(self) -> Resources:
        return load_pickle(self.input()["instance"]["resources"].path)

    def _load_orders(self) -> OrdersDomain:
        return load_pickle(self.input()["instance"]["orders"].path)

    def run(self):
        routing_sols = load_pickle(self.input()["routing_sol"]["routing_sol"].path)
        orders = self._load_orders()
        resources = self._load_resources()

        if isinstance(routing_sols, CombinedRoutingSolution):
            routes = routing_sols.routes
        else:
            routes = [r.route for r in routing_sols]

        sequencing_input = SchedulingInput(routes=routes, orders=orders, resources=resources)
        sequencer = self._get_inited_sequencer()
        sequencing_sol = sequencer.solve(sequencing_input)
        dump_pickle(self.output()["sequencing_sol"].path, sequencing_sol)


# ─────────────────────────── Result Aggregation ──────────────────────────────

_PARAM_TO_STAGE = {
    "routing_sol":         "routing",
    "pick_list_sol":       "batching",
    "item_assignment_sol": "item_assignment",
    "scheduling_sol":      "scheduling",
    "sequencing_sol":      "sequencing",
}

def _collect_from_graph(task: CoSyLuigiTask) -> dict:
    collected = {}
    visited = set()

    def _walk(t):
        if id(t) in visited:
            return
        visited.add(id(t))
        req = t.requires()
        if not isinstance(req, dict):
            return
        for param_name, child in req.items():
            stage = _PARAM_TO_STAGE.get(param_name)
            if stage and stage not in collected:
                output_key = param_name  # output key matches param name
                sol = load_pickle(child.output()[output_key].path)
                collected[stage] = {
                    "task_class": type(child).__name__,
                    "algo":       getattr(sol, "algo_name", type(child).__name__),
                    "time":       getattr(sol, "execution_time", None),
                    "solution":   sol,
                }
            _walk(child)

    _walk(task)
    return collected


class ResultAggregation(BaseComponent):
    """
    Terminal task replacing all Evaluation* classes.
    Walks the task graph, loads solutions, computes KPIs, writes summary.json.
    """

    def output(self):
        return {
            "summary": self.get_luigi_local_target_with_task_id("summary.json")
        }

    @classmethod
    def configure(cls, data_card: DataCard, models: list[ModelCard], allowed_model_names: set):
        cls._data_card = data_card
        cls._models = models
        cls._allowed_model_names = allowed_model_names

    @classmethod
    def constraints(cls) -> Sequence[Callable[..., bool]]:
        return [
            lambda vs: problem_type_constraint(vs, TAXONOMY, cls._data_card, cls._models),
            lambda vs: feature_constraint(vs, cls._data_card, cls._models),
            lambda vs: batching_loader_constraint(vs, TAXONOMY, cls._data_card, PickListProvider),
            lambda vs: check_unique(vs, [ResultAggregation]),
            lambda vs: fixed_algorithms_constraint(vs, cls._allowed_model_names, cls._models)
        ]

    def _build_provenance(self, summary: dict, collected: dict) -> None:
        provenance_list = []
        for stage in ["item_assignment", "batching", "routing", "sequencing", "scheduling"]:
            if stage in collected:
                entry = collected[stage]
                provenance_list.append({
                    "stage": stage,
                    "algo": entry["algo"],
                    "time": entry["time"],
                    "task_class": entry["task_class"],
                })
                summary[f"{stage}_algo"] = entry["algo"]
                summary[f"{stage}_time"] = entry["time"]
        summary["provenance"] = provenance_list

    @staticmethod
    def _compute_routing_summary(routing_sols) -> dict:
        if isinstance(routing_sols, CombinedRoutingSolution):
            total = sum(r.distance for r in routing_sols.routes)
            per_tour = {f"tour_{i}_distance": r.distance for i, r in enumerate(routing_sols.routes)}
        else:
            total = sum(r.route.distance for r in routing_sols)
            per_tour = {f"tour_{i}_distance": r.route.distance for i, r in enumerate(routing_sols)}
        return {"total_distance": total, "tour_distances": per_tour}

    @staticmethod
    def _compute_scheduling_summary(scheduling_sol, orders: OrdersDomain) -> dict:
        order_by_id = {o.order_id: o for o in orders.orders}
        records = []
        for job in scheduling_sol.jobs:
            end_time = job.end_time
            for on in job.route.pick_list.order_numbers:
                o = order_by_id.get(on)
                if o is None:
                    continue
                due_date = o.due_date if o.due_date is not None else float("inf")
                lateness = end_time - due_date
                records.append({
                    "lateness": lateness,
                    "tardiness": max(0, lateness),
                    "on_time": end_time <= due_date,
                    "completion_time": end_time,
                })
        if not records:
            return {}
        import pandas as pd
        df = pd.DataFrame(records)
        makespan = df["completion_time"].max()
        return {
            "makespan": float(makespan),
            "on_time_rate": float(df["on_time"].mean() * 100),
            "avg_lateness": float(df["lateness"].mean()),
            "avg_tardiness": float(df["tardiness"].mean()),
            "max_lateness": float(df["lateness"].max()),
            "max_tardiness": float(df["tardiness"].max()),
        }

    def _run_impl(self):
        raise NotImplementedError


class ResultAggregationRouting(ResultAggregation):
    routing_sol = CoSyLuigiTaskParameter(PickerRouting)

    def run(self):
        collected = _collect_from_graph(self)
        summary = {}
        self._build_provenance(summary, collected)

        routing_entry = collected.get("routing")
        if routing_entry is None:
            raise ValueError("No routing solution in graph.")
        summary["routing_summary"] = self._compute_routing_summary(routing_entry["solution"])
        dump_json(self.output()["summary"].path, summary)

class ResultAggregationScheduling(ResultAggregation):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    scheduling_sol = CoSyLuigiTaskParameter(AbstractScheduling)

    def run(self):
        collected = _collect_from_graph(self)
        summary = {}
        self._build_provenance(summary, collected)
        routing_entry = collected.get("routing")
        if routing_entry is not None:
            summary["routing_summary"] = self._compute_routing_summary(routing_entry["solution"])
        scheduling_entry = collected.get("scheduling")
        if scheduling_entry is None:
            raise ValueError("No scheduling solution in graph.")
        orders: OrdersDomain = load_pickle(self.input()["instance"]["orders"].path)
        summary["scheduling_summary"] = self._compute_scheduling_summary(
            scheduling_entry["solution"], orders
        )
        dump_json(self.output()["summary"].path, summary)


class ResultAggregationBatching(ResultAggregation):
    pick_list_sol = CoSyLuigiTaskParameter(AbstractBatching)

    def run(self):
        collected = _collect_from_graph(self)
        summary = {}
        self._build_provenance(summary, collected)
        dump_json(self.output()["summary"].path, summary)


# ─────────────────────────── Graph Utilities ─────────────────────────────────

def traverse_pipeline(vs: Iterable[CoSyLuigiTask], visited=None) -> list[CoSyLuigiTask]:
    if visited is None:
        visited = set()
    result = []
    for v in vs:
        vid = id(v)
        if vid in visited:
            continue
        visited.add(vid)
        result.append(v)
        req = v.requires()
        if isinstance(req, dict):
            req = list(req.values())
        result.extend(traverse_pipeline(req, visited))
    return result


def check_unique(
    vs: Mapping[str, CoSyLuigiTask],
    required_to_be_unique: Iterable[type[CoSyLuigiTask]],
    get_classes=None,
) -> bool:
    classes = get_classes(vs) if get_classes else [pc.__class__ for pc in traverse_pipeline(vs.values())]
    seen_subclasses = {}
    for c in classes:
        for unique in required_to_be_unique:
            if issubclass(c, unique):
                if unique in seen_subclasses and seen_subclasses[unique] != c:
                    return False
                seen_subclasses[unique] = c
    return True


def batching_loader_constraint(vs, subproblems, data_card: DataCard, exclusive, get_classes=None):
    classes = get_classes(vs) if get_classes else [pc.__class__ for pc in traverse_pipeline(vs.values())]
    problem = data_card.problem_class
    problems = subproblems[problem]["variables"]
    if "batching" in problems and exclusive in classes:
        return False
    return True


def problem_type_constraint(vs, subproblems, data_card: DataCard, models, get_classes=None) -> bool:
    classes = get_classes(vs) if get_classes else [pc.__class__ for pc in traverse_pipeline(vs.values())]
    problem = data_card.problem_class
    problems = subproblems[problem]["variables"]
    for c in classes:
        for m in models:
            # if m.implementation["class_name"] == c.__name__:
            if m.model_name == c.__name__:
                if m.model_name == "ClarkAndWrightSShape":
                    print(m)
                if m.problem_type not in problems:
                    print(f"{m.model_name} not applicable {m.problem_type} not in {problems}")
                    return False
    return True


def feature_constraint(vs, data_card: DataCard, models, get_classes=None) -> bool:
    classes = get_classes(vs) if get_classes else [pc.__class__ for pc in traverse_pipeline(vs.values())]
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
                for domain, reqs in m.requirements.items():
                    section = domain_sections.get(domain)
                    if section is None:
                        continue
                    required_tpe = reqs["type"]
                    required_features = reqs.get("features", [])
                    required_features = [] if required_features in (None, [None]) else required_features
                    constraints = reqs.get("constraints", {})
                    domain_type = section["type"]
                    domain_features = [
                        f for f in section["features"]
                        if str(section["features"][f]) == "0" or section["features"][f]
                    ]
                    if "any" not in required_tpe and domain_type not in required_tpe:
                        print(f"{m.model_name} not applicable, {domain_type} not in {required_tpe}")
                        return False
                    missing_features = [f for f in required_features if f not in domain_features]
                    if missing_features:
                        print(f"{m.model_name} not applicable, missing feature: {missing_features}")
                        return False
                    for feature_name, constraint in constraints.items():
                        if feature_name not in domain_features:
                            return False
                        evaluator = ConstraintEvaluator()
                        if not evaluator.evaluate(feature_name, constraint):
                            return False
    return True


def fixed_algorithms_constraint(vs, allowed_names: set[str], models, get_classes=None) -> bool:
    if not allowed_names:
        return True
    necessary = {"InstanceLoader", "PickListProvider",
                  "ResultAggregationRouting", "ResultAggregationPickList",
                  "ResultAggregationScheduling"}
    # algo_names = {m.model_name for m in models}
    classes = get_classes(vs) if get_classes else [
        pc.__class__ for pc in traverse_pipeline(vs.values())
    ]

    for c in classes:
        if c.__name__ in necessary:
            continue
        if c.__name__ not in allowed_names:
            print(f"{c.__name__} not allowed")
            return False
    return True


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    import yaml

    import ware_ops_algos
    from ware_ops_algos.utils.general_functions import load_model_cards
    from ware_ops_algos.data_loaders import HesslerIrnichLoader
    from scenarios.experiment_commons import load_and_flatten_data_card

    from casim.pipelines.subproblems.item_assingment import GreedyIA
    from casim.pipelines.subproblems.batching import FiFo, OrderNrFiFo, DueDate
    from casim.pipelines.subproblems.picker_routing import SShape, RatliffRosenthal


    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

    instances_base = DATA_DIR / "instances"
    cache_base = DATA_DIR / "instances" / "caches"
    instance_set = "BahceciOencan"
    instance_name = "Pr_20_1_20_Store1_01.txt"
    file_path = instances_base / instance_set / instance_name
    output_folder = (
        PROJECT_ROOT / "experiments" / "output" / "cosy"
        / instance_set / instance_name
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    loader = HesslerIrnichLoader(str(instances_base / instance_set), str(cache_base / instance_set))
    domain = loader.load(str(file_path))
    print("Orders initial", len(domain.orders.orders))
    card_path = DATA_DIR / "data_cards/bahceci_oencan.yaml"
    with open(card_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    datacard = load_and_flatten_data_card(raw)
    pkg_dir = Path(ware_ops_algos.__file__).parent
    model_cards_path = pkg_dir / "algorithms" / "algorithm_cards"
    models = load_model_cards(str(model_cards_path))

    ResultAggregation.configure(datacard, models)

    config = get_config()
    config.set("PipelineParams", "output_folder", str(output_folder))
    config.set("PipelineParams", "domain_path", str(loader.cache_path))

    repo = CoSyLuigiRepo(
        InstanceLoader,
        GreedyIA,
        FiFo,
        OrderNrFiFo,
        DueDate,
        PickListProvider,
        SShape,
        RatliffRosenthal,
        ResultAggregationBatching,
        ResultAggregationRouting,
    )

    maestro = Maestro(repo.cls_repo, repo.taxonomy)

    results = maestro.query(ResultAggregationRouting.target())
    luigi.build(results, local_scheduler=True)
    # print("OBP Done")
    #
    # dc.problem_class = "SPRP"
    # maestro = Maestro(repo.cls_repo, repo.taxonomy)
    # results = maestro.query(ResultAggregationRouting.target())
    # luigi.build(results, local_scheduler=True)


if __name__ == "__main__":
    main()