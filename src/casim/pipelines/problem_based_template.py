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
    AdmissionSolution,
    BatchObject,
    BatchingSolution,
    Batching,
    CombinedRoutingSolution,
    ItemAssignmentSolution,
    RoutingSolution,
    ItemAssignment, Scheduler, Job, build_jobs)
from ware_ops_algos.algorithms import WaitingInput, WaitingSolution
from ware_ops_algos.domain_algo_mapper.domain_algo_mapper import DomainAlgorithmMapper
from ware_ops_algos.algorithms.order_splitting.order_splitting import OrderSplitting
from ware_ops_algos.domain_models import (
    Articles,
    Resources,
    LayoutData,
    DataCard,
    OrdersDomain,
)
from ware_ops_algos.algorithms.algorithm_cards import AlgorithmCard

from casim.domain_objects.sim_domain import (
    ActiveTourBatch, ActiveTourBatchingSolution, DynamicInfo, SimWarehouseDomain,
)
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


# ─────────────────────────── Order Splitting ────────────────────────────────

class AbstractOrderProvider(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)

    def output(self):
        return {
            "orders": self.get_luigi_local_target_with_task_id("orders.pkl")
        }

    def run(self):
        pass


class OrdersProvider(AbstractOrderProvider):
    def run(self):
        orders_domain: OrdersDomain = load_pickle(self.input()["instance"]["orders"].path)
        dump_pickle(self.output()["orders"].path, orders_domain)


class OrderSplitter(AbstractOrderProvider):
    def _get_order_splitter(self) -> OrderSplitting:
        ...

    def run(self):
        order_splitter = self._get_order_splitter()
        orders_domain: OrdersDomain = load_pickle(self.input()["instance"]["orders"].path)
        orders = orders_domain.orders

        solution = order_splitter.solve(orders)
        orders_domain.orders = solution.orders
        dump_pickle(self.output()["orders"].path, orders_domain)

# ─────────────────────────── Item Assignment ────────────────────────────────

class AbstractItemAssignment(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    orders = CoSyLuigiTaskParameter(AbstractOrderProvider)

    def output(self):
        return {
            "item_assignment_sol": self.get_luigi_local_target_with_task_id("item_assignment_sol.pkl")
        }

    def run(self):
        orders_domain: OrdersDomain = load_pickle(self.input()["orders"]["orders"].path)
        item_assigner = self._get_inited_ia()
        ia_sol: ItemAssignmentSolution = item_assigner.solve(orders_domain.orders)
        dump_pickle(self.output()["item_assignment_sol"].path, ia_sol)

    def _get_storage(self):
        storage = load_pickle(self.input()["instance"]["storage"].path)
        return storage

    def _get_inited_ia(self) -> ItemAssignment:
        ...


# ─────────────────────────── Admission ──────────────────────────────────────

class AbstractAdmission(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)

    def output(self):
        return {"admission_sol": self.get_luigi_local_target_with_task_id("admission_sol.pkl")}


# ─────────────────────────── Batching ───────────────────────────────────────

class AbstractBatchProvider(BaseComponent):
    """Supply routeable batches, whether produced by batching or admission."""

    instance = CoSyLuigiTaskParameter(InstanceLoader)

    def output(self):
        return {
            "batching_sol": self.get_luigi_local_target_with_task_id("batching_sol.pkl")
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


class AbstractBatching(AbstractBatchProvider):
    """The ordinary order-batching decision stage."""


class AdmittedTourBatch(AbstractBatchProvider):
    """Assemble route input from an admission decision; make no selection."""

    admission_sol = CoSyLuigiTaskParameter(AbstractAdmission)
    item_assignment_sol = CoSyLuigiTaskParameter(AbstractItemAssignment)

    def run(self):
        decision: AdmissionSolution = load_pickle(self.input()["admission_sol"]["admission_sol"].path)
        dynamic = load_pickle(self.input()["instance"]["dynamic_warehouse_info"].path)
        assigned = load_pickle(self.input()["item_assignment_sol"]["item_assignment_sol"].path)
        orders = assigned.resolved_orders
        by_id = {order.order_id: order for order in orders}
        accepted_by_tour = {}
        for tour_id, order_id in decision.assignments:
            accepted_by_tour.setdefault(tour_id, []).append(order_id)
        batches = []
        for tour in dynamic.admission_tours:
            accepted = accepted_by_tour.get(tour.tour_id, [])
            if accepted:
                active = [by_id[order_id] for order_id in sorted(tour.active_order_ids)]
                batches.append(ActiveTourBatch(
                    batch_id=tour.tour_id,
                    orders=active + [by_id[order_id] for order_id in accepted],
                    tour_id=tour.tour_id,
                ))
        dump_pickle(self.output()["batching_sol"].path, ActiveTourBatchingSolution(
            batches=batches,
            assignments=decision.assignments,
            considered_pairs=decision.considered_pairs,
        ))


class PickListProvider(AbstractBatching):
    instance = CoSyLuigiTaskParameter(InstanceLoader)

    def _load_warehouse_info(self) -> DynamicInfo:
        return load_pickle(self.input()["instance"]["dynamic_warehouse_info"].path)

    def run(self):
        warehouse_info = self._load_warehouse_info()
        batches = warehouse_info.buffered_batches
        batching_sol = BatchingSolution(batches=batches)
        dump_pickle(self.output()["batching_sol"].path, batching_sol)


class BatchingNode(AbstractBatching):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    item_assignment_sol = CoSyLuigiTaskParameter(AbstractItemAssignment)

    def _get_inited_batcher(self) -> Batching:
        ...

    @staticmethod
    def _latest_order_arrival(orders: list[WarehouseOrder]) -> float:
        arrivals = [o.order_date for o in orders]
        return max(arrivals) if arrivals else 0.0

    def run(self):
        batcher: Batching = self._get_inited_batcher()
        ia_sol: ItemAssignmentSolution = load_pickle(
            self.input()["item_assignment_sol"]["item_assignment_sol"].path
        )
        resolved_orders = ia_sol.resolved_orders
        batching_sol = batcher.solve(resolved_orders)

        # pick_lists = [self._build_pick_lists(batch.orders) for batch in batching_sol.batches]
        # batching_sol.pick_lists = pick_lists

        if batcher.__class__.__name__ in ["SeedBatching", "ClarkAndWrightBatching", "LocalSearchBatching"]:
            batching_sol.algo_name = batcher.algo_name
        else:
            batching_sol.algo_name = batcher.__class__.__name__

        dump_pickle(self.output()["batching_sol"].path, batching_sol)


# ─────────────────────────── Routing ────────────────────────────────────────

class AbstractPickerRouting(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    batching_sol = CoSyLuigiTaskParameter(AbstractBatchProvider)

    def _get_inited_router(self) -> Routing:
        pass

    def _load_resources(self) -> Resources:
        return load_pickle(self.input()["instance"]["resources"].path)

    def _load_dynamic_info(self) -> DynamicInfo:
        return load_pickle(self.input()["instance"]["dynamic_warehouse_info"].path)

    def _load_layout(self) -> LayoutData:
        return load_pickle(self.input()["instance"]["layout"].path)

    def _load_articles(self) -> Articles:
        return load_pickle(self.input()["instance"]["articles"].path)

    def output(self):
        return {
            "routing_sol": self.get_luigi_local_target_with_task_id("routing_sol.pkl")
        }


class PickerRouting(AbstractPickerRouting):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    batching_sol = CoSyLuigiTaskParameter(AbstractBatchProvider)

    def run(self):
        router: Routing = self._get_inited_router()
        batching_sol: BatchingSolution = load_pickle(
            self.input()["batching_sol"]["batching_sol"].path
        )
        routes = []
        algo_name = ""
        execution_time = 0
        for b in batching_sol.batches:
            routing_solution: RoutingSolution = router.solve(b.pick_positions)
            algo_name = routing_solution.algo_name
            routing_solution.route.batch = b
            routes.append(routing_solution.route)
            execution_time += routing_solution.execution_time
            router.reset_parameters()

        combined_sol = CombinedRoutingSolution(
            algo_name=algo_name,
            execution_time=execution_time,
            routes=routes
        )
        dump_pickle(self.output()["routing_sol"].path, combined_sol)


class AbstractScheduling(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    routing_sol = CoSyLuigiTaskParameter(AbstractPickerRouting)
    orders = CoSyLuigiTaskParameter(AbstractOrderProvider)

    def output(self):
        return {
            "scheduling_sol": self.get_luigi_local_target_with_task_id("scheduling_sol.pkl")
        }

    def _get_inited_scheduler(self) -> Scheduler:
        ...

    def _load_resources(self) -> Resources:
        return load_pickle(self.input()["instance"]["resources"].path)

    def _load_orders(self) -> OrdersDomain:
        return load_pickle(self.input()["orders"]["orders"].path)

    def _load_warehouse_info(self) -> DynamicInfo:
        return load_pickle(self.input()["instance"]["dynamic_warehouse_info"].path)

    def run(self):
        routing_sol = load_pickle(self.input()["routing_sol"]["routing_sol"].path)
        orders = self._load_orders()
        resources = self._load_resources()
        dynamic_info = self._load_warehouse_info()

        routes = None
        if isinstance(routing_sol, CombinedRoutingSolution):
            routes = routing_sol.routes
        else:
            routes = [r.route for r in routing_sol]

        jobs = build_jobs(routes, resources, release_time=dynamic_info.time)
        # scheduling_input = SchedulingInput(routes=routes, orders=orders, resources=resources)
        scheduler = self._get_inited_scheduler()
        scheduling_sol = scheduler.solve(jobs)
        dump_pickle(self.output()["scheduling_sol"].path, scheduling_sol)

# ─────────────────────────── Sequencing ─────────────────────────────────────


class AbstractSequencing(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    routing_sol = CoSyLuigiTaskParameter(AbstractPickerRouting)

    def output(self):
        return {
            "sequencing_sol": self.get_luigi_local_target_with_task_id(
                "sequencing_sol.pkl"
            )
        }

    def _get_inited_sequencer(self):
        raise NotImplementedError

    def _load_resources(self) -> Resources:
        return load_pickle(self.input()["instance"]["resources"].path)

    def run(self):
        routing_sol = load_pickle(
            self.input()["routing_sol"]["routing_sol"].path
        )
        resources = self._load_resources()

        if isinstance(routing_sol, CombinedRoutingSolution):
            routes = routing_sol.routes
        else:
            routes = [r.route for r in routing_sol]

        sequencing_input = SequencingInput(
            routes=routes,
            resources=resources,
        )

        sequencer = self._get_inited_sequencer()
        sequencing_sol = sequencer.solve(sequencing_input)

        dump_pickle(
            self.output()["sequencing_sol"].path,
            sequencing_sol,
        )


class AbstractWaiting(BaseComponent):
    """Apply one configured waiting algorithm to scheduled candidates."""

    instance = CoSyLuigiTaskParameter(InstanceLoader)
    scheduling_sol = CoSyLuigiTaskParameter(AbstractScheduling)

    def output(self):
        return {"waiting_sol": self.get_luigi_local_target_with_task_id("waiting_sol.pkl")}

    def _get_inited_waiter(self):
        raise NotImplementedError

    def _single_order_service_times(self, scheduled, picker):
        return None

    def run(self):
        domain = load_pickle(self.input()["instance"]["domain"].path)
        scheduled = load_pickle(self.input()["scheduling_sol"]["scheduling_sol"].path)
        if not domain.resources.resources:
            raise ValueError("Waiting needs an available picker")
        picker = domain.resources.resources[0]
        waiter = self._get_inited_waiter()
        decision = waiter.solve(WaitingInput(
            candidates=tuple(scheduled.jobs),
            current_time=domain.dynamic_warehouse_info.time,
            input_closed=domain.dynamic_warehouse_info.done,
            deadline_reached=domain.dynamic_warehouse_info.wait_expired,
            picker=picker,
            layout=domain.layout,
            information=domain.information,
            single_order_service_times=self._single_order_service_times(scheduled, picker),
        ))
        dump_pickle(self.output()["waiting_sol"].path, decision)

# ─────────────────────────── Result Aggregation ──────────────────────────────

_PARAM_TO_STAGE = {
    "waiting_sol":         "waiting",
    "routing_sol":         "routing",
    "admission_sol":       "admission",
    "batching_sol":       "batching",
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
            if isinstance(child, AdmittedTourBatch):
                stage = None
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
    def configure(cls, data_card: DataCard, models: list[AlgorithmCard]):
        cls._data_card = data_card
        cls._models = models

    @classmethod
    def constraints(cls) -> Sequence[Callable[..., bool]]:
        return [
            lambda vs: algorithm_applicability_constraint(vs, cls._data_card, cls._models),
            lambda vs: batching_loader_constraint(vs, TAXONOMY, cls._data_card, PickListProvider),
            lambda vs: orders_provider_constraint(vs, TAXONOMY, cls._data_card, OrdersProvider),
            lambda vs: check_unique(vs, [ResultAggregation]),
        ]

    def _build_provenance(self, summary: dict, collected: dict) -> None:
        provenance_list = []
        for stage in ["item_assignment", "batching", "admission", "routing", "sequencing", "scheduling", "waiting"]:
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
            for on in job.job.route.batch.order_numbers:
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
    batching_sol = CoSyLuigiTaskParameter(AbstractBatching)

    def run(self):
        collected = _collect_from_graph(self)
        summary = {}
        self._build_provenance(summary, collected)
        dump_json(self.output()["summary"].path, summary)


class ResultAggregationSequencing(ResultAggregation):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    scheduling_sol = CoSyLuigiTaskParameter(AbstractSequencing)

    def run(self):
        collected = _collect_from_graph(self)
        summary = {}
        self._build_provenance(summary, collected)
        routing_entry = collected.get("routing")
        if routing_entry is not None:
            summary["routing_summary"] = self._compute_routing_summary(routing_entry["solution"])
        sequencing_entry = collected.get("sequencing")
        if sequencing_entry is None:
            raise ValueError("No scheduling solution in graph.")
        orders: OrdersDomain = load_pickle(self.input()["instance"]["orders"].path)
        summary["scheduling_summary"] = self._compute_scheduling_summary(
            sequencing_entry["solution"], orders
        )
        dump_json(self.output()["summary"].path, summary)


class ResultAggregationWaiting(ResultAggregation):
    waiting_sol = CoSyLuigiTaskParameter(AbstractWaiting)

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
        print(f"Not valid, {exclusive} in {classes}")
        return False
    return True


def orders_provider_constraint(vs, subproblems, data_card: DataCard, exclusive, get_classes=None):
    classes = get_classes(vs) if get_classes else [pc.__class__ for pc in traverse_pipeline(vs.values())]
    problem = data_card.problem_class
    problems = subproblems[problem]["variables"]
    if "order_splitting" in problems and exclusive in classes:
        print(f"Not valid, {exclusive} in {classes}")
        return False
    return True


def algorithm_applicability_constraint(vs, data_card: DataCard, models, get_classes=None) -> bool:
    """Identify CoSy components; CASOP owns their applicability decision."""
    classes = get_classes(vs) if get_classes else [pc.__class__ for pc in traverse_pipeline(vs.values())]
    mapper = DomainAlgorithmMapper(TAXONOMY)
    for c in classes:
        for m in models:
            if m.implementation.get("component_name", m.algo_name) == c.__name__:
                if not mapper.filter([m], data_card):
                    return False
    return True


def main():
    import yaml

    import ware_ops_algos
    from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards
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
    models = load_packaged_algo_cards()

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
