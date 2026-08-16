"""Explicit pre-runtime construction for a CASIM run.

Hydra composition ends here.  Each entry of ``engines.problems`` is a runtime
decision binding named by the pair ``(problem_class, replanning)``:
``problem_class`` selects the mathematical variables (taxonomy) and
``replanning`` (``none`` / ``unstarted`` / ``active``) selects how far into
already-committed work those variables may change.  The StateAdapter exposure
vocabulary is derived from that pair, so the user-facing YAML never names the
low-level projection terms (``available``, ``nonactive``,
``buffered_and_replannable`` ...).  Solver applicability is checked and CoSy
portfolios are synthesized before the simulation loop runs.
"""

from copy import deepcopy
from pathlib import Path

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import DataCard, load_and_flatten_data_card

from casim.decision_engine.decision_engine import (
    DecisionEngine,
    SchedulingCommitmentPolicy,
)
from casim.events import operational_events
from casim.events.operational_events import InterventionRequest
from casim.loggers import DashLogger, KPILogger
from casim.pipelines.pipeline_runner import CoSySolver
from casim.pipelines.solution_ranker import SolutionRanker
from casim.simulation_engine.simulation_engine import SimulationEngine
from casim.simulation_engine.state_adapter import StateAdapter


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


# The decision space CASIM actually supports.  Each runtime decision binding
# is the pair ``(problem_class, replanning)``.  ``problem_class`` selects the
# mathematical variables (taxonomy); ``replanning`` selects how far into
# already-committed work those variables may change.  The exposure vocabulary
# (orders / batches / resources / active_tour sources) is derived from this
# pair, so user configuration never names the low-level projection terms.
SUPPORTED_DECISIONS = {
    ("OBP", "none"),
    ("ORSP", "none"),
    ("ORSP", "unstarted"),
    ("OBRSP", "none"),
    ("OBRP", "none"),
    ("ORP", "active"),
    ("OBRP", "active"),
}

_REPLANNING_VALUES = {"none", "unstarted", "active"}


def _derive_exposure(
    problem_class: str,
    replanning: str,
    *,
    due_horizon_s,
    limit,
    congestion_penalty,
) -> dict:
    """Map a decision binding to the StateAdapter exposure vocabulary.

    The sources (``buffered`` vs ``buffered_and_replannable``, ``nonactive``
    vs ``available`` vs ``dispatchable``, ``residual``) are consequences of
    the replanning scope, not independent user choices.  Tuning parameters
    (``due_horizon_s``, ``limit``, ``congestion_penalty``) are user research
    choices and threaded into the relevant projection.
    """
    orders_limit = {"source": "buffered", "limit": limit}
    if problem_class == "OBP" and replanning == "none":
        return {
            "orders": orders_limit,
            "resources": {"source": "all"},
        }
    if problem_class == "ORSP" and replanning == "none":
        return {
            "batches": {
                "source": "buffered",
                "due_horizon_s": due_horizon_s,
                "limit": limit,
            },
            "resources": {"source": "nonactive"},
        }
    if problem_class == "ORSP" and replanning == "unstarted":
        return {
            "batches": {
                "source": "buffered_and_replannable",
                "due_horizon_s": due_horizon_s,
                "limit": limit,
            },
            "resources": {"source": "available"},
        }
    if problem_class == "OBRSP" and replanning == "none":
        return {
            "orders": orders_limit,
            "resources": {
                "source": "dispatchable",
                "scope": "trigger_if_present",
            },
        }
    if problem_class == "OBRP" and replanning == "none":
        return {
            "orders": orders_limit,
            "resources": {
                "source": "dispatchable",
                "scope": "trigger_if_present",
            },
        }
    if problem_class == "ORP" and replanning == "active":
        return {
            "active_tour": {
                "source": "residual",
                "congestion_penalty": congestion_penalty,
            },
        }
    if problem_class == "OBRP" and replanning == "active":
        return {
            "active_tour": {
                "source": "residual",
                "congestion_penalty": congestion_penalty,
            },
            "orders": orders_limit,
        }
    raise ValueError(
        f"Unsupported decision binding: problem_class={problem_class!r}, "
        f"replanning={replanning!r}. Supported bindings: "
        f"{sorted(SUPPORTED_DECISIONS)}"
    )


def _build_adapter(problem_cfg: DictConfig) -> tuple[str, str, StateAdapter]:
    problem_class = str(problem_cfg.problem_class)
    replanning = str(problem_cfg.get("replanning", "none"))
    if replanning not in _REPLANNING_VALUES:
        raise ValueError(
            f"replanning must be one of {sorted(_REPLANNING_VALUES)}, got "
            f"{replanning!r}"
        )
    if (problem_class, replanning) not in SUPPORTED_DECISIONS:
        raise ValueError(
            f"Unsupported decision binding (problem_class={problem_class!r}, "
            f"replanning={replanning!r}). Supported: "
            f"{sorted(SUPPORTED_DECISIONS)}"
        )
    exposure = _derive_exposure(
        problem_class,
        replanning,
        due_horizon_s=problem_cfg.get("due_horizon_s"),
        limit=problem_cfg.get("limit"),
        congestion_penalty=problem_cfg.get("congestion_penalty", 0.0),
    )
    adapter = StateAdapter(
        problem_class=problem_class,
        replanning=replanning,
        **exposure,
    )
    return problem_class, replanning, adapter


def _build_engine_state(cfg: DictConfig):
    """Build adapters, triggers, conditions, and solvers for every problem."""
    solvers = {}
    policies = {}
    triggers_map: dict = {}
    conditions_map: dict = {}
    state_adapters: dict = {}
    bindings: set[tuple[str, str]] = set()
    for problem_cfg in cfg.engines.problems:
        problem_class, replanning, adapter = _build_adapter(problem_cfg)
        binding = (problem_class, replanning)
        if binding in bindings:
            raise ValueError(
                f"Duplicate decision binding {binding}; each "
                f"(problem_class, replanning) pair may appear at most once"
            )
        bindings.add(binding)
        state_adapters[binding] = adapter
        conditions_map[binding] = _requirements(problem_cfg)
        for event_name in problem_cfg.get("triggers") or []:
            event_cls = _resolve_trigger(str(event_name))
            if event_cls in triggers_map:
                raise ValueError(
                    f"Event '{event_name}' is already bound to "
                    f"{triggers_map[event_cls]}, cannot also bind to {binding}"
                )
            triggers_map[event_cls] = binding
        solvers[binding] = _build_solver(cfg, problem_class, problem_cfg)
        commit_cfg = problem_cfg.get("commit")
        if commit_cfg is not None:
            policies[binding] = SchedulingCommitmentPolicy(
                n_jobs=commit_cfg.get("n_jobs"),
                max_jobs_per_picker=commit_cfg.get("max_jobs_per_picker"),
                planning_horizon_s=commit_cfg.get("planning_horizon_s"),
            )
    return solvers, policies, triggers_map, conditions_map, state_adapters


def _requirements(problem_cfg: DictConfig) -> dict:
    req = problem_cfg.get("requires")
    if req is None:
        return {}
    return OmegaConf.to_container(req, resolve=True)


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


def _construct_simulation(
    cfg: DictConfig,
    state_adapters: dict,
    triggers_map: dict,
    conditions_map: dict,
) -> SimulationEngine:
    working_dir = cfg.experiment.get("working_dir", cfg.experiment.output_dir)
    event_loggers = [
        KPILogger(
            Path(working_dir) / "kpis",
            print_every=cfg.experiment.get("progress_every", 5000),
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
    _solvers, _policies, triggers_map, conditions_map, state_adapters = (
        _build_engine_state(cfg)
    )
    return _construct_simulation(cfg, state_adapters, triggers_map, conditions_map)


def _is_active_insertion(binding: tuple[str, str] | None) -> bool:
    """An InterventionRequest binding is active batch insertion iff it is
    (OBRP, active): it projects the residual active tour AND buffered orders.
    """
    return binding == ("OBRP", "active")


def build_decision_engine(
    cfg: DictConfig,
    data_card: DataCard,
    solvers: dict,
    policies: dict,
    state_adapters: dict,
) -> DecisionEngine:
    prepared = {}
    for binding, solver in solvers.items():
        problem_class, _replanning = binding
        effective_card = deepcopy(data_card)
        effective_card.problem_class = problem_class
        features = dict((effective_card.warehouse_info or {}).get("features") or {})
        features.update(
            {feature: True for feature in state_adapters[binding].projected_features()}
        )
        warehouse_info = dict(effective_card.warehouse_info or {})
        warehouse_info["features"] = features
        effective_card.warehouse_info = warehouse_info
        solver.prepare(effective_card)
        prepared[binding] = solver
    return DecisionEngine(solver_map=prepared, commitment_policies=policies)


def build_runtime(
    cfg: DictConfig,
    data_card: DataCard | None = None,
) -> tuple[SimulationEngine, DecisionEngine]:
    """Construct and prepare the complete CASIM runtime before reset."""
    if data_card is None:
        data_card = load_and_flatten_data_card(
            OmegaConf.to_container(cfg.data_card, resolve=True)
        )
    solvers, policies, triggers_map, conditions_map, state_adapters = _build_engine_state(cfg)
    simulation = _construct_simulation(cfg, state_adapters, triggers_map, conditions_map)
    decision_engine = build_decision_engine(
        cfg, data_card, solvers, policies, state_adapters
    )
    return simulation, decision_engine
