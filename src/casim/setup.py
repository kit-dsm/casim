"""Explicit pre-runtime construction for a CASIM run.

Hydra composition ends at :func:`compile_engine`, the single compilation
boundary.  Each entry of ``engines.problems`` is a runtime decision binding
named by the pair ``(problem_class, replanning)``: ``problem_class`` selects
the mathematical variables (taxonomy) and ``replanning`` (``none`` /
``unstarted`` / ``active``) selects how far into already-committed work those
variables may change.  The :class:`DecisionCard` for that pair is the
declarative twin of the old procedural exposure mapping; a
:class:`DecisionCompatibilityMapper` validates the user configuration against
the card and the existing taxonomy before any runtime object is built.

The resolved decision configuration produces the existing :class:`StateAdapter`
arguments, so the runtime projection stays unchanged:

    DecisionCard + user configuration
        → DecisionCompatibilityMapper
        → valid StateAdapter
        → ``projected_features()``
        → effective DataCard
        → ``DomainAlgorithmMapper``
        → applicable algorithms
        → CoSy / Maestro pipelines

:func:`build_runtime` instantiates the runtime from the compiled result
instead of interpreting raw Hydra configuration again.  The compiler only
resolves decision/engine semantics; loaders and visualization are not part of
the compilation boundary.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import DataCard, load_and_flatten_data_card

from casim.decision_card import (
    CONDITION_SPECS,
    DECISION_CARDS,
    CompiledCondition,
    DecisionCompatibilityMapper,
)
from casim.decision_engine import (
    DecisionEngine,
    SchedulingCommitmentPolicy,
)
from casim.events import operational_events
from casim.events.operational_events import Event, InterventionRequest
from casim.loggers import DashLogger, ProgressLogger
from casim.pipelines.solution_ranker import SolutionRanker
from casim.pipelines.taxonomy import TAXONOMY, solution_kind
from casim.simulation_engine.simulation_engine import SimulationEngine
from casim.simulation_engine.state_adapter import StateAdapter
from casim.solvers.cosy_solver import ApplicabilityReport, CoSySolver


_TRIGGER_EVENTS = {
    "OrderArrival": operational_events.OrderArrival,
    "PickerIdle": operational_events.PickerIdle,
    "FlushRemainingOrders": operational_events.FlushRemainingOrders,
    "WMSRun": operational_events.WMSRun,
    "PlanningRun": operational_events.PlanningRun,
    "ShiftStart": operational_events.ShiftStart,
    "InterventionRequest": operational_events.InterventionRequest,
    "TruckDisruption": operational_events.TruckDisruption,
    "VolumeShiftAcrossDay": operational_events.VolumeShiftAcrossDay,
    "OrderIngestion": operational_events.OrderIngestion,
}

_REPLANNING_VALUES = {"none", "unstarted", "active"}

# The decision space CASIM actually supports, expressed as the keys of the
# declarative :data:`DECISION_CARDS` registry.  Kept as a set for the public
# surface and for tests that enumerate the binding space.
SUPPORTED_DECISIONS = set(DECISION_CARDS)


def _resolve_trigger(name: str):
    if name in _TRIGGER_EVENTS:
        return _TRIGGER_EVENTS[name]
    event_cls = getattr(operational_events, name, None)
    if event_cls is None or not isinstance(event_cls, type):
        raise ValueError(f"Unknown trigger event: {name!r}")
    if not issubclass(event_cls, operational_events.Event):
        raise ValueError(f"Trigger {name!r} is not an Event subclass")
    return event_cls


def build_data_loader(cfg: DictConfig) -> DataLoader:
    loader_cfg = cfg.input.data_loader
    if not isinstance(loader_cfg, DictConfig) or not loader_cfg.get("_target_"):
        raise ValueError(
            "input.data_loader must be a Hydra object with an explicit "
            "_target_ class path"
        )
    return instantiate(loader_cfg, _recursive_=False)


# ─────────────────────────── compiled runtime ──────────────────────────


@dataclass(frozen=True)
class CompiledDecision:
    """One resolved decision binding before the event loop runs.

    Kept minimal: only the concrete objects CASIM needs at runtime.
    The binding key carries ``problem_class`` and ``replanning``; the
    card is not retained (its projection lives in the adapter).  Frozen
    so a compiled decision cannot be silently mutated after :func:`compile_engine`
    returns; solver preparation replaces the decision via :func:`dataclasses.replace`.
    """

    binding: tuple[str, str]
    adapter: StateAdapter
    trigger_classes: tuple[type[Event], ...]
    conditions: tuple[CompiledCondition, ...]
    solver: object
    policy: SchedulingCommitmentPolicy | None
    applicability: ApplicabilityReport | None = None


@dataclass(frozen=True)
class CompiledEngine:
    """The fully resolved decision/engine configuration returned by
    :func:`compile_engine`.
    """

    decisions: dict[tuple[str, str], CompiledDecision]
    triggers_map: dict[type, tuple[str, str]]
    data_card: DataCard

    @property
    def intervention_enabled(self) -> bool:
        return InterventionRequest in self.triggers_map

    @property
    def active_batch_insertion_enabled(self) -> bool:
        return _is_active_insertion(self.triggers_map.get(InterventionRequest))

    def explain(self, binding: tuple[str, str]) -> str:
        """Render the resolved semantics of one binding as plain text.

        Every line derives from a semantic owner: binding, adapter
        projection, taxonomy, ConditionSpec, commitment policy, and
        applicability report.  
        """
        if binding not in self.decisions:
            raise KeyError(f"No compiled decision for binding {binding}")
        decision = self.decisions[binding]
        problem_class, replanning = binding
        adapter = decision.adapter
        lines: list[str] = []
        lines.append(f"{problem_class} / {replanning}")
        lines.append("")
        lines.append("projection")
        for line in _describe_projection(adapter):
            lines.append(f"  {line}")
        lines.append("")
        lines.append("triggers")
        if decision.trigger_classes:
            for trigger_cls in decision.trigger_classes:
                lines.append(f"  {trigger_cls.__name__}")
        else:
            lines.append("  (none)")
        lines.append("")
        lines.append("conditions")
        for line in _describe_conditions(decision.conditions):
            lines.append(f"  {line}")
        lines.append("")
        lines.append("decision variables")
        for variable in TAXONOMY[problem_class]["variables"]:
            lines.append(f"  {variable}")
        lines.append("")
        lines.append("solution kind")
        lines.append(f"  {solution_kind(problem_class)}")
        lines.append("")
        lines.append("projected algorithm features")
        for feature in adapter.projected_features():
            lines.append(f"  {feature}")
        lines.append("")
        lines.append("commitment")
        for line in _describe_policy(decision.policy):
            lines.append(f"  {line}")
        lines.append("")
        if decision.applicability is not None:
            lines.append("applicability")
            for line in _describe_applicability(decision.applicability):
                lines.append(f"  {line}")
        else:
            lines.append("applicability")
            lines.append("  (direct solver; no CoSy applicability report)")
        return "\n".join(lines)


def _describe_projection(adapter: StateAdapter) -> list[str]:
    lines: list[str] = []
    has_active = adapter.active_tour_cfg is not None
    cfg = adapter.orders_cfg
    if cfg is not None:
        lines.append("orders: buffered")
        if cfg.get("limit") is not None:
            lines.append(f"  limit = {cfg['limit']}")
    cfg = adapter.batches_cfg
    if cfg is not None:
        source = str(cfg.get("source", "buffered"))
        label = {
            "buffered": "buffered",
            "buffered_and_replannable": "buffered + replannable",
        }.get(source, source)
        lines.append(f"batches: {label}")
        if cfg.get("due_horizon_s") is not None:
            lines.append(f"  due_horizon_s = {cfg['due_horizon_s']}")
        if cfg.get("limit") is not None:
            lines.append(f"  limit = {cfg['limit']}")
    if has_active:
        cfg = adapter.active_tour_cfg
        lines.append(f"active_tour: {cfg.get('source', 'residual')}")
        if float(cfg.get("congestion_penalty", 0.0)) > 0:
            lines.append(f"  congestion_penalty = {cfg['congestion_penalty']}")
        lines.append("resources: (active tour; single picker)")
    else:
        cfg = adapter.resources_cfg
        if cfg is not None:
            source = str(cfg.get("source", "all"))
            lines.append(f"resources: {source}")
    if not lines:
        lines.append("(no projection)")
    return lines


def _describe_conditions(conditions: tuple[CompiledCondition, ...]) -> list[str]:
    if not conditions:
        return ["(no conditions)"]
    return [cond.spec.render(cond.key, cond.value) for cond in conditions]


def _describe_policy(policy: SchedulingCommitmentPolicy | None) -> list[str]:
    if policy is None:
        return ["none"]
    parts = []
    if policy.n_jobs is not None:
        parts.append(f"n_jobs = {policy.n_jobs}")
    if policy.max_jobs_per_picker is not None:
        parts.append(f"max_jobs_per_picker = {policy.max_jobs_per_picker}")
    if policy.planning_horizon_s is not None:
        parts.append(f"planning_horizon_s = {policy.planning_horizon_s}")
    return parts if parts else ["commit all (policy has no limits)"]


def _describe_applicability(report: ApplicabilityReport) -> list[str]:
    return [
        f"considered components: {', '.join(report.considered_components) or '(none)'}",
        f"applicable algorithms: {', '.join(report.applicable_algorithms) or '(none)'}",
        f"excluded components: {', '.join(report.excluded_components) or '(none)'}",
        f"retained components: {', '.join(report.retained_components) or '(none)'}",
        f"pipelines: {len(report.pipelines)}",
    ]


# ─────────────────────────── compilation ────────────────────────────────


def _build_solver(cfg: DictConfig, problem_class: str, problem_cfg: DictConfig):
    solver_cfg = problem_cfg.solver
    solver_type = str(solver_cfg.get("type"))
    if solver_type == "cosy":
        return _build_cosy_solver(cfg, problem_class, solver_cfg)
    if solver_type == "direct":
        params = OmegaConf.to_container(solver_cfg, resolve=True)
        params.pop("type", None)
        return instantiate(params, _recursive_=False, problem_class=problem_class)
    raise ValueError(f"Unknown solver type: {solver_type!r}")


def _build_cosy_solver(cfg: DictConfig, problem_class: str, solver_cfg: DictConfig):
    objective = str(solver_cfg.get("objective", "distance"))
    ranker = SolutionRanker(objective=objective)
    repo_cfg = solver_cfg.get("repo")
    executor = str(solver_cfg.get("executor", "luigi"))
    working_dir = cfg.experiment.get("working_dir", cfg.experiment.output_dir)
    return CoSySolver(
        instances_dir=Path(cfg.instances_base),
        cache_dir=Path(cfg.cache_base) / cfg.data_card.name,
        output_dir=working_dir,
        instance_name=cfg.experiment.instance_name,
        solution_ranker=ranker,
        problem_class=problem_class,
        verbose=False,
        luigi_cfg=cfg.luigi,
        repo=repo_cfg,
        executor=executor,
    )


def _compile_bindings(
    cfg: DictConfig,
) -> tuple[dict[tuple[str, str], CompiledDecision], dict[type, tuple[str, str]]]:
    """Resolve and validate every decision binding, and build its adapter,
    triggers, conditions, commitment policy, and unprepared solver.

    This is the validation half of compilation: it rejects semantically
    meaningless configurations through :class:`DecisionCompatibilityMapper`
    and the global duplicate-binding/duplicate-trigger checks, then
    constructs the runtime objects that do not require the data card.  Solver
    preparation (applicability filtering and pipeline synthesis) happens
    later in :func:`compile_engine` once the effective data card is known.
    """
    mapper = DecisionCompatibilityMapper()
    decisions: dict[tuple[str, str], CompiledDecision] = {}
    triggers_map: dict[type, tuple[str, str]] = {}
    bindings: set[tuple[str, str]] = set()
    for problem_cfg in cfg.engines.problems:
        problem_class = str(problem_cfg.problem_class)
        replanning = str(problem_cfg.get("replanning", "none"))
        if replanning not in _REPLANNING_VALUES:
            raise ValueError(
                f"replanning must be one of {sorted(_REPLANNING_VALUES)}, got "
                f"{replanning!r}"
            )
        binding = (problem_class, replanning)
        if binding in bindings:
            raise ValueError(
                f"Duplicate decision binding {binding}; each "
                f"(problem_class, replanning) pair may appear at most once"
            )
        bindings.add(binding)
        trigger_classes = tuple(
            _resolve_trigger(str(name))
            for name in (problem_cfg.get("triggers") or [])
        )
        card = mapper.validate(
            problem_class,
            replanning,
            problem_cfg,
            trigger_classes=list(trigger_classes),
        )
        exposure = card.exposure_dict(
            due_horizon_s=problem_cfg.get("due_horizon_s"),
            limit=problem_cfg.get("limit"),
            congestion_penalty=problem_cfg.get("congestion_penalty", 0.0),
        )
        adapter = StateAdapter(
            problem_class=problem_class,
            replanning=replanning,
            **exposure,
        )
        for event_cls in trigger_classes:
            if event_cls in triggers_map:
                raise ValueError(
                    f"Event already bound to {triggers_map[event_cls]}, "
                    f"cannot also bind to {binding}"
                )
            triggers_map[event_cls] = binding
        conditions = _compile_conditions(problem_cfg)
        solver = _build_solver(cfg, problem_class, problem_cfg)
        policy = None
        commit_cfg = problem_cfg.get("commit")
        if commit_cfg is not None:
            policy = SchedulingCommitmentPolicy(
                n_jobs=commit_cfg.get("n_jobs"),
                max_jobs_per_picker=commit_cfg.get("max_jobs_per_picker"),
                planning_horizon_s=commit_cfg.get("planning_horizon_s"),
            )
        decisions[binding] = CompiledDecision(
            binding=binding,
            adapter=adapter,
            trigger_classes=trigger_classes,
            conditions=conditions,
            solver=solver,
            policy=policy,
            applicability=None,
        )
    return decisions, triggers_map


def _compile_conditions(problem_cfg: DictConfig) -> tuple[CompiledCondition, ...]:
    """Build compiled condition tuples from a ``requires:`` block."""
    req = problem_cfg.get("requires")
    if req is None:
        return ()
    resolved = OmegaConf.to_container(req, resolve=True)
    return tuple(
        CompiledCondition(key, CONDITION_SPECS[key], resolved[key])
        for key in resolved
    )


def _effective_data_card(
    data_card: DataCard,
    problem_class: str,
    adapter: StateAdapter,
) -> DataCard:
    """Copy the data card and add the adapter's projected features for the
    decision's problem class.  This is the existing bridge into
    :class:`DomainAlgorithmMapper`."""
    effective_card = deepcopy(data_card)
    effective_card.problem_class = problem_class
    features = dict((effective_card.warehouse_info or {}).get("features") or {})
    features.update(
        {feature: True for feature in adapter.projected_features()}
    )
    warehouse_info = dict(effective_card.warehouse_info or {})
    warehouse_info["features"] = features
    effective_card.warehouse_info = warehouse_info
    return effective_card


def _applicability_report(solver) -> ApplicabilityReport | None:
    """Surface the report :meth:`CoSySolver.prepare` cached.  Direct solvers
    return ``None`` from :meth:`applicability_report`."""
    if not isinstance(solver, CoSySolver):
        return None
    return solver.applicability_report()


def compile_engine(
    cfg: DictConfig,
    data_card: DataCard | None = None,
) -> CompiledEngine:
    """Compile the full decision/engine configuration from Hydra config.

    This is the single explicit compilation boundary.  It resolves each
    ``(problem_class, replanning)`` binding to its :class:`DecisionCard`,
    validates its configuration, builds the :class:`StateAdapter`, resolves
    triggers and conditions, validates and builds the commitment policy,
    builds the decision-specific effective :class:`DataCard`, prepares the
    solver and algorithm applicability, validates global constraints
    (duplicate bindings/triggers), and returns the resolved engine.
    """
    if data_card is None:
        data_card = load_and_flatten_data_card(
            OmegaConf.to_container(cfg.data_card, resolve=True)
        )
    decisions, triggers_map = _compile_bindings(cfg)
    prepared: dict[tuple[str, str], CompiledDecision] = {}
    for binding, decision in decisions.items():
        effective_card = _effective_data_card(
            data_card, binding[0], decision.adapter
        )
        decision.solver.prepare(effective_card)
        prepared[binding] = replace(
            decision,
            applicability=_applicability_report(decision.solver),
        )
    return CompiledEngine(
        decisions=prepared,
        triggers_map=triggers_map,
        data_card=data_card,
    )


# ─────────────────────────── construction ──────────────────────────────


def _construct_simulation(
    cfg: DictConfig,
    state_adapters: dict,
    triggers_map: dict,
    conditions_map: dict,
) -> SimulationEngine:
    working_dir = cfg.experiment.get("working_dir", cfg.experiment.output_dir)
    event_loggers = [
        ProgressLogger(
            every=cfg.experiment.get("progress_every", 0),
        )
    ]
    viz_cfg = cfg.get("viz") or {}
    if viz_cfg.get("record", viz_cfg.get("launch", False)):
        event_loggers.append(DashLogger(Path(cfg.experiment.output_dir) / "viz"))
    runtime = cfg.engines.get("runtime") or {}
    completion_mode = str(runtime.get("completion_mode", "drain"))
    horizon_time = runtime.get("horizon_time")
    return SimulationEngine(
        state_adapters=state_adapters,
        data_loader=build_data_loader(cfg),
        loader_kwargs=dict(cfg.input.get("load") or {}),
        triggers_map=triggers_map,
        conditions_map=conditions_map,
        event_loggers=event_loggers,
        completion_mode=completion_mode,
        horizon_time=horizon_time,
        intervention_enabled=InterventionRequest in triggers_map,
        active_batch_insertion_enabled=_is_active_insertion(
            triggers_map.get(InterventionRequest)
        ),
    )


def build_simulation(cfg: DictConfig) -> SimulationEngine:
    """Build a SimulationEngine without preparing solvers (test helper)."""
    decisions, triggers_map = _compile_bindings(cfg)
    state_adapters = {b: d.adapter for b, d in decisions.items()}
    conditions_map = {b: d.conditions for b, d in decisions.items()}
    return _construct_simulation(cfg, state_adapters, triggers_map, conditions_map)


def _is_active_insertion(binding: tuple[str, str] | None) -> bool:
    """An InterventionRequest binding is active batch insertion iff it is
    (OBRP, active): it projects the residual active tour AND buffered orders.
    """
    return binding == ("OBRP", "active")


def build_runtime(
    cfg: DictConfig,
    data_card: DataCard | None = None,
) -> tuple[SimulationEngine, DecisionEngine]:
    """Construct and prepare the complete CASIM runtime before reset.

    Routes through :func:`compile_engine` so the configuration is validated
    once and the runtime is instantiated from the compiled result instead of
    interpreting raw Hydra configuration again.
    """
    compiled = compile_engine(cfg, data_card)
    state_adapters = {b: d.adapter for b, d in compiled.decisions.items()}
    conditions_map = {b: d.conditions for b, d in compiled.decisions.items()}
    simulation = _construct_simulation(
        cfg, state_adapters, compiled.triggers_map, conditions_map
    )
    solvers = {b: d.solver for b, d in compiled.decisions.items()}
    policies = {
        b: d.policy
        for b, d in compiled.decisions.items()
        if d.policy is not None
    }
    decision_engine = DecisionEngine(solver_map=solvers, commitment_policies=policies)
    return simulation, decision_engine
