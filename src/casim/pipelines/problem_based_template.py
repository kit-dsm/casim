import os
import pickle
from os.path import join as pjoin
from typing import Sequence, Callable, Iterable, Mapping

import luigi
from cosy_luigi import CoSyLuigiTask, CoSyLuigiTaskParameter

from ware_ops_algos.algorithms import (
    Routing,
    WarehouseOrder,
    ActiveTourRoutingSolution,
    AdmissionSolution,
    BatchObject,
    BatchingSolution,
    Batching,
    CombinedRoutingSolution,
    ItemAssignmentSolution,
    PositioningInput,
    Positioning,
    Waiting, WaitingInput,
    RoutingSolution,
    ItemAssignment, Scheduler, build_jobs)
from ware_ops_algos.domain_algo_mapper.domain_algo_mapper import DomainAlgorithmMapper
from ware_ops_algos.domain_models import (
    Articles,
    Resources,
    LayoutData,
    DataCard, OrdersDomain,
)
from ware_ops_algos.algorithms.algorithm_cards import AlgorithmCard

from casim.pipelines.taxonomy import TAXONOMY
from casim.pipelines.card_components import card_matches_component
from casim.pipelines.tour_filters import admitting_tours, rebuildable_tours
from casim.pipelines.subproblems.routers import build_router
from casim.domain_objects.sim_domain import ReconsiderReason
from casim.domain_objects.sim_domain import DecisionContext, SimWarehouseDomain
from ware_ops_algos.planning.models import PlanView


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
            "plan": self.get_luigi_local_target_with_task_id("plan.pkl"),
            "context": self.get_luigi_local_target_with_task_id("context.pkl"),
            "memory": self.get_luigi_local_target_with_task_id("memory.pkl"),
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
        dynamic = domain.dynamic_warehouse_info
        dump_pickle(self.output()["plan"].path, dynamic.plan)
        dump_pickle(self.output()["context"].path, dynamic.context)
        dump_pickle(self.output()["memory"].path, dynamic.memory)


class IncumbentProvider(BaseComponent):
    """Expose selected draft tours to repair stages as one typed input."""

    instance = CoSyLuigiTaskParameter(InstanceLoader)

    def output(self):
        return {"incumbent_tours":
                self.get_luigi_local_target_with_task_id("incumbent_tours.pkl")}

    def run(self):
        domain: SimWarehouseDomain = load_pickle(self.input()["instance"]["domain"].path)
        drafts = rebuildable_tours(domain.dynamic_warehouse_info.plan)
        dump_pickle(self.output()["incumbent_tours"].path, drafts)


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

class AbstractBatching(BaseComponent):
    """The ordinary order-batching decision stage."""

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


class AdmittedTourBatch(BaseComponent):
    """Assemble route input from an admission decision; make no selection.

    One batch per tour that admission gave orders or returns, in plan order,
    with ``batch_id`` the tour's ID: its unpicked orders, then the admitted
    ones. The admission decision is passed on unchanged as ``admission_sol``.
    """

    instance = CoSyLuigiTaskParameter(InstanceLoader)
    admission_sol = CoSyLuigiTaskParameter(AbstractAdmission)
    item_assignment_sol = CoSyLuigiTaskParameter(AbstractItemAssignment)

    def output(self):
        return {"batching_sol": self.get_luigi_local_target_with_task_id("batching_sol.pkl"),
                "admission_sol": self.get_luigi_local_target_with_task_id("admission_sol.pkl")}

    def run(self):
        decision: AdmissionSolution = load_pickle(self.input()["admission_sol"]["admission_sol"].path)
        plan: PlanView = load_pickle(self.input()["instance"]["plan"].path)
        assigned = load_pickle(self.input()["item_assignment_sol"]["item_assignment_sol"].path)
        by_id = {order.order_id: order for order in assigned.resolved_orders}
        accepted_by_tour = {}
        for tour_id, order_id in decision.assignments:
            accepted_by_tour.setdefault(tour_id, []).append(order_id)
        returned = set(decision.return_tour_ids)
        batches = []
        for view in admitting_tours(plan):
            accepted = accepted_by_tour.get(view.tour_id, [])
            if accepted or view.tour_id in returned:
                active = [by_id[order_id] for order_id in sorted(view.progress.active_order_ids)]
                batches.append(BatchObject(
                    batch_id=view.tour_id,
                    orders=active + [by_id[order_id] for order_id in accepted],
                ))
        dump_pickle(self.output()["batching_sol"].path, BatchingSolution(batches=batches))
        dump_pickle(self.output()["admission_sol"].path, decision)


class PickListProvider(AbstractBatching):
    instance = CoSyLuigiTaskParameter(InstanceLoader)

    def run(self):
        plan: PlanView = load_pickle(self.input()["instance"]["plan"].path)
        batches = [*plan.pick_list_buffer,
                   *(view.batch for view in rebuildable_tours(plan))]
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

        dump_pickle(self.output()["batching_sol"].path, batching_sol)


# ─────────────────────────── Routing ────────────────────────────────────────

class _RoutingStage(BaseComponent):
    """Inputs, output and router of the two routing stages."""

    instance = CoSyLuigiTaskParameter(InstanceLoader)
    router_class: type[Routing] | None = None

    def _get_inited_router(self, pickers=None, **options) -> Routing:
        """A router of ``router_class`` for ``pickers`` (default: all resources)."""
        if self.router_class is None:
            raise ValueError(f"{type(self).__name__} has no router_class")
        return build_router(self.router_class, self._load_layout(),
                            pickers if pickers is not None else self._load_resources().resources,
                            **options)

    def _load_resources(self) -> Resources:
        return load_pickle(self.input()["instance"]["resources"].path)

    def _load_plan(self) -> PlanView:
        return load_pickle(self.input()["instance"]["plan"].path)

    def _load_layout(self) -> LayoutData:
        return load_pickle(self.input()["instance"]["layout"].path)

    def _load_articles(self) -> Articles:
        return load_pickle(self.input()["instance"]["articles"].path)

    def output(self):
        return {
            "routing_sol": self.get_luigi_local_target_with_task_id("routing_sol.pkl")
        }


class AbstractPickerRouting(_RoutingStage):
    """Route the batches of the batching stage from the depot."""

    batching_sol = CoSyLuigiTaskParameter(AbstractBatching)


class PickerRouting(AbstractPickerRouting):
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


class AbstractActiveTourRouting(_RoutingStage):
    """Route each admitted tour from its own projected origin."""

    batching_sol = CoSyLuigiTaskParameter(AdmittedTourBatch)

    def run(self):
        batching_sol = load_pickle(self.input()["batching_sol"]["batching_sol"].path)
        admission_sol = load_pickle(self.input()["batching_sol"]["admission_sol"].path)
        # A started tour's picker comes with its tour view, not the decision's resources.
        tours = {view.tour_id: view for view in admitting_tours(self._load_plan())}
        routes = []
        execution_time = 0.0
        algo_name = ""
        for batch in batching_sol.batches:
            tour = tours[batch.batch_id]
            router = self._get_inited_router(
                pickers=[tour.picker],
                routing_origin=tour.progress.routing_origin,
            )
            routing_solution = router.solve(batch.pick_positions)
            route = routing_solution.route
            route.batch = batch
            routes.append(route)
            algo_name = routing_solution.algo_name
            execution_time += routing_solution.execution_time
        dump_pickle(self.output()["routing_sol"].path, ActiveTourRoutingSolution(
            algo_name=algo_name,
            execution_time=execution_time,
            routes=routes,
            assignments=admission_sol.assignments,
            considered_pairs=admission_sol.considered_pairs,
        ))


class AbstractScheduling(BaseComponent):
    instance = CoSyLuigiTaskParameter(InstanceLoader)
    routing_sol = CoSyLuigiTaskParameter(AbstractPickerRouting)
    orders = CoSyLuigiTaskParameter(AbstractOrderProvider)

    def output(self):
        return {
            "scheduling_sol": self.get_luigi_local_target_with_task_id("scheduling_sol.pkl")
        }

    def _get_inited_scheduler(self, available_from: dict[int, float]) -> Scheduler:
        ...

    def _load_resources(self) -> Resources:
        return load_pickle(self.input()["instance"]["resources"].path)

    def _load_orders(self) -> OrdersDomain:
        return load_pickle(self.input()["orders"]["orders"].path)

    def _load_context(self) -> DecisionContext:
        return load_pickle(self.input()["instance"]["context"].path)

    def _load_plan(self) -> PlanView:
        return load_pickle(self.input()["instance"]["plan"].path)

    def run(self):
        routing_sol = load_pickle(self.input()["routing_sol"]["routing_sol"].path)
        resources = self._load_resources()

        routes = None
        if isinstance(routing_sol, CombinedRoutingSolution):
            routes = routing_sol.routes
        else:
            routes = [r.route for r in routing_sol]

        plan = self._load_plan()
        jobs = build_jobs(routes, resources, release_time=plan.time)
        scheduler = self._get_inited_scheduler(dict(plan.available_from))
        scheduling_sol = scheduler.solve(jobs)
        dump_pickle(self.output()["scheduling_sol"].path, scheduling_sol)


class AbstractPositioning(BaseComponent):
    """Optional stage that adds empty movement jobs to a complete schedule."""

    instance = CoSyLuigiTaskParameter(InstanceLoader)
    scheduling_sol = CoSyLuigiTaskParameter(AbstractScheduling)

    def output(self):
        return {"positioning_sol":
                self.get_luigi_local_target_with_task_id("positioning_sol.pkl")}

    def _get_positioner(self) -> Positioning:
        raise NotImplementedError

    def run(self):
        inputs = self.input()
        plan: PlanView = load_pickle(inputs["instance"]["plan"].path)
        context: DecisionContext = load_pickle(inputs["instance"]["context"].path)
        schedule = load_pickle(inputs["scheduling_sol"]["scheduling_sol"].path)
        layout = load_pickle(inputs["instance"]["layout"].path)
        occupied = frozenset(view.picker_id for view in plan.tours
                             if view.picker_id is not None)
        result = self._get_positioner().solve(PositioningInput(
            schedule=schedule,
            resources=tuple(load_pickle(inputs["instance"]["resources"].path).resources),
            layout=layout,
            current_time=plan.time, input_closed=context.input_closed,
            occupied_picker_ids=occupied,
        ))
        dump_pickle(self.output()["positioning_sol"].path, result)


class AbstractWaiting(BaseComponent):
    """Apply a configured waiting algorithm to a complete schedule."""

    instance = CoSyLuigiTaskParameter(InstanceLoader)
    scheduling_sol = CoSyLuigiTaskParameter(AbstractScheduling)

    def output(self):
        return {"waiting_sol":
                self.get_luigi_local_target_with_task_id("waiting_sol.pkl")}

    # A waiter whose card requires one picker (n_resources = 1) times the
    # schedule with that picker's speed and pick time.
    single_picker: bool = False

    def _get_inited_waiter(self) -> Waiting:
        raise NotImplementedError

    def run(self):
        inputs = self.input()
        domain = load_pickle(inputs["instance"]["domain"].path)
        schedule = load_pickle(inputs["scheduling_sol"]["scheduling_sol"].path)
        dynamic = domain.dynamic_warehouse_info
        pickers = {picker.id: picker for picker in domain.resources.resources}
        if self.single_picker and len(pickers) != 1:
            raise ValueError(f"{type(self).__name__} requires exactly one picker, "
                             f"the decision sees {len(pickers)}")
        if not schedule.jobs:
            if not pickers:
                raise ValueError("Waiting requires a picker")
            picker = next(iter(pickers.values()))
        else:
            picker = pickers[schedule.jobs[0].picker_id]
        result = self._get_inited_waiter().solve(WaitingInput(
            candidates=tuple(schedule.jobs),
            current_time=dynamic.plan.time,
            input_closed=dynamic.context.input_closed,
            picker=picker,
            layout=domain.layout,
            information=domain.information,
            deadline_reached=ReconsiderReason.WAIT in dynamic.context.reconsider,
        ))
        dump_pickle(self.output()["waiting_sol"].path, result)

# ─────────────────────────── Plan candidate ────────────────────────────────

class AbstractPlanCandidate(BaseComponent):
    """A typed bridge from one CoSy solution stage to the decision boundary."""

    def output(self):
        return {"plan_candidate":
                self.get_luigi_local_target_with_task_id("plan_candidate.pkl")}


class BatchingPlanCandidate(AbstractPlanCandidate):
    source = CoSyLuigiTaskParameter(AbstractBatching)

    def run(self):
        dump_pickle(self.output()["plan_candidate"].path,
                    load_pickle(self.input()["source"]["batching_sol"].path))


class RoutingPlanCandidate(AbstractPlanCandidate):
    source = CoSyLuigiTaskParameter(AbstractPickerRouting)

    def run(self):
        dump_pickle(self.output()["plan_candidate"].path,
                    load_pickle(self.input()["source"]["routing_sol"].path))


class ActiveTourRoutingPlanCandidate(AbstractPlanCandidate):
    source = CoSyLuigiTaskParameter(AbstractActiveTourRouting)

    def run(self):
        dump_pickle(self.output()["plan_candidate"].path,
                    load_pickle(self.input()["source"]["routing_sol"].path))


class SchedulingPlanCandidate(AbstractPlanCandidate):
    source = CoSyLuigiTaskParameter(AbstractScheduling)

    def run(self):
        dump_pickle(self.output()["plan_candidate"].path,
                    load_pickle(self.input()["source"]["scheduling_sol"].path))


class WaitingPlanCandidate(AbstractPlanCandidate):
    source = CoSyLuigiTaskParameter(AbstractWaiting)

    def run(self):
        dump_pickle(self.output()["plan_candidate"].path,
                    load_pickle(self.input()["source"]["waiting_sol"].path))


class PositioningPlanCandidate(AbstractPlanCandidate):
    source = CoSyLuigiTaskParameter(AbstractPositioning)

    def run(self):
        dump_pickle(self.output()["plan_candidate"].path,
                    load_pickle(self.input()["source"]["positioning_sol"].path))


class PlanCandidateEndpoint(BaseComponent):
    """One endpoint for all solution families; the source remains typed."""

    candidate = CoSyLuigiTaskParameter(AbstractPlanCandidate)
    _data_card: DataCard
    _models: list[AlgorithmCard]

    def output(self):
        return {"plan_candidate":
                self.get_luigi_local_target_with_task_id("plan_candidate.pkl")}

    @classmethod
    def configure(cls, data_card: DataCard, models: list[AlgorithmCard]):
        cls._data_card = data_card
        cls._models = models

    @classmethod
    def constraints(cls) -> Sequence[Callable[..., bool]]:
        return [
            lambda vs: algorithm_applicability_constraint(vs, cls._data_card, cls._models),
            lambda vs: batching_loader_constraint(vs, TAXONOMY, cls._data_card, PickListProvider),
            lambda vs: check_unique(vs, [PlanCandidateEndpoint]),
        ]

    def run(self):
        dump_pickle(self.output()["plan_candidate"].path,
                    load_pickle(self.input()["candidate"]["plan_candidate"].path))


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


def algorithm_applicability_constraint(vs, data_card: DataCard, models, get_classes=None) -> bool:
    """Identify CoSy components; CASOP owns their applicability decision."""
    classes = get_classes(vs) if get_classes else [pc.__class__ for pc in traverse_pipeline(vs.values())]
    mapper = DomainAlgorithmMapper(TAXONOMY)
    for c in classes:
        for m in models:
            if card_matches_component(m, c):
                if not mapper.filter([m], data_card):
                    return False
    return True
