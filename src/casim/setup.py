"""Explicit pre-runtime construction for a CASIM run.

Hydra composition ends here.  The returned simulation and decision engine
are fully configured: loaders and adapters exist, solver applicability has
been checked, CoSy portfolios have been synthesized, and event mappings
are derived from solution semantics.
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


def _exposure_dict(problem_cfg: DictConfig) -> dict | None:
    expose = problem_cfg.get("expose")
    if expose is None:
        return None
    return OmegaConf.to_container(expose, resolve=True)


def _build_adapter(problem_cfg: DictConfig) -> StateAdapter:
    expose = _exposure_dict(problem_cfg) or {}
    return StateAdapter(**expose)


def _build_engine_state(cfg: DictConfig):
    """Build adapters, triggers, conditions, and solvers for every problem."""
    solvers = {}
    policies = {}
    triggers_map: dict = {}
    conditions_map: dict = {}
    state_adapters: dict = {}
    for problem_key, problem_cfg in cfg.engines.problems.items():
        state_adapters[problem_key] = _build_adapter(problem_cfg)
        conditions_map[problem_key] = _requirements(problem_cfg)
        for event_name in problem_cfg.get("triggers") or []:
            event_cls = _resolve_trigger(str(event_name))
            if event_cls in triggers_map:
                raise ValueError(
                    f"Event '{event_name}' is already bound to problem "
                    f"'{triggers_map[event_cls]}', cannot also bind to "
                    f"'{problem_key}'"
                )
            triggers_map[event_cls] = problem_key
        solvers[problem_key] = _build_solver(cfg, problem_key, problem_cfg)
        commit_cfg = problem_cfg.get("commit")
        if commit_cfg is not None:
            policies[problem_key] = SchedulingCommitmentPolicy(
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


def _build_solver(cfg: DictConfig, problem_key: str, problem_cfg: DictConfig):
    solver_cfg = problem_cfg.solver
    solver_type = str(solver_cfg.get("type"))
    if solver_type == "cosy":
        return _build_cosy_solver(cfg, problem_key, solver_cfg)
    if solver_type == "direct":
        params = OmegaConf.to_container(solver_cfg, resolve=True)
        params.pop("type", None)
        return instantiate(params, _recursive_=False, problem_class=problem_key)
    raise ValueError(f"Unknown solver type: {solver_type!r}")


def _build_cosy_solver(cfg: DictConfig, problem_key: str, solver_cfg: DictConfig):
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
        problem_class=problem_key,
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
        active_batch_insertion_enabled=_has_active_insertion(
            cfg, triggers_map.get(InterventionRequest)
        ),
    )


def build_simulation(cfg: DictConfig) -> SimulationEngine:
    """Build a SimulationEngine without preparing solvers (test helper)."""
    _solvers, _policies, triggers_map, conditions_map, state_adapters = (
        _build_engine_state(cfg)
    )
    return _construct_simulation(cfg, state_adapters, triggers_map, conditions_map)


def _has_active_insertion(cfg: DictConfig, problem_key: str | None) -> bool:
    if problem_key is None:
        return False
    problem_cfg = cfg.engines.problems.get(problem_key)
    if problem_cfg is None:
        return False
    expose = _exposure_dict(problem_cfg) or {}
    return expose.get("active_tour") is not None and expose.get("orders") is not None


def build_decision_engine(
    cfg: DictConfig,
    data_card: DataCard,
    solvers: dict,
    policies: dict,
    state_adapters: dict,
) -> DecisionEngine:
    prepared = {}
    for problem_key, solver in solvers.items():
        effective_card = deepcopy(data_card)
        effective_card.problem_class = problem_key
        features = dict((effective_card.warehouse_info or {}).get("features") or {})
        features.update(
            {feature: True for feature in state_adapters[problem_key].projected_features()}
        )
        warehouse_info = dict(effective_card.warehouse_info or {})
        warehouse_info["features"] = features
        effective_card.warehouse_info = warehouse_info
        solver.prepare(effective_card)
        prepared[problem_key] = solver
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
