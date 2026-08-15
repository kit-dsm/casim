"""Explicit pre-runtime construction for a CASIM run.

Hydra composition ends here.  The returned simulation and decision engine are
fully configured: loaders and adapters exist, solver applicability has been
checked, CoSy portfolios have been synthesized, and event mappings are fixed.
"""

from copy import deepcopy
from pathlib import Path

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import DataCard, load_and_flatten_data_card

from casim.decision_engine.decision_engine import DecisionEngine
from casim.events.operational_events import InterventionRequest
from casim.loggers import DashLogger, KPILogger
from casim.pipelines.pipeline_runner import CoSySolver
from casim.simulation_engine.simulation_engine import SimulationEngine


def build_data_loader(cfg: DictConfig) -> DataLoader:
    loader_cfg = cfg.input.data_loader
    if not isinstance(loader_cfg, DictConfig) or not loader_cfg.get("_target_"):
        raise ValueError(
            "input.data_loader must be a Hydra object with an explicit "
            "_target_ class path"
        )
    return instantiate(loader_cfg, _recursive_=False)


def build_simulation_problems(cfg: DictConfig):
    state_adapters = {}
    conditions_map = {}
    triggers_map = {}
    for problem_key, problem_cfg in cfg.simulation_engine.problems.items():
        state_adapters[problem_key] = instantiate(problem_cfg.state_adapter)
        conditions_map[problem_key] = [
            instantiate(condition)
            for condition in (problem_cfg.get("conditions") or [])
        ]
        for event_name in (problem_cfg.get("triggers") or []):
            event_cls = get_class(str(event_name))
            if event_cls in triggers_map:
                raise ValueError(
                    f"Event '{event_name}' is already bound to problem "
                    f"'{triggers_map[event_cls]}', cannot also bind to "
                    f"'{problem_key}'"
                )
            triggers_map[event_cls] = problem_key
    return state_adapters, conditions_map, triggers_map


def build_simulation(cfg: DictConfig) -> SimulationEngine:
    state_adapters, conditions_map, triggers_map = build_simulation_problems(
        cfg.engines
    )
    working_dir = cfg.experiment.get(
        "working_dir", cfg.experiment.output_dir
    )
    event_loggers = [
        KPILogger(
            Path(working_dir) / "kpis",
            print_every=cfg.experiment.get("progress_every", 5000),
        )
    ]
    viz_cfg = cfg.get("viz") or {}
    if viz_cfg.get("record", viz_cfg.get("launch", False)):
        event_loggers.append(
            DashLogger(Path(cfg.experiment.output_dir) / "viz")
        )
    return SimulationEngine(
        state_adapters=state_adapters,
        data_loader=build_data_loader(cfg),
        loader_kwargs=dict(cfg.input.get("load") or {}),
        triggers_map=triggers_map,
        conditions_map=conditions_map,
        event_loggers=event_loggers,
        completion_mode=str(
            cfg.engines.simulation_engine.get("completion_mode", "drain")
        ),
        horizon_time=cfg.engines.simulation_engine.get("horizon_time"),
        intervention_enabled=InterventionRequest in triggers_map,
        active_batch_insertion_enabled=(
            triggers_map.get(InterventionRequest) == "OBRP"
        ),
    )


def _build_solver(cfg: DictConfig, problem_key: str, solver_cfg: DictConfig):
    """Instantiate one of CASIM's two concrete solver construction shapes."""
    solver_cls = get_class(str(solver_cfg._target_))
    kwargs = {"problem_class": problem_key}
    if issubclass(solver_cls, CoSySolver):
        working_dir = cfg.experiment.get(
            "working_dir", cfg.experiment.output_dir
        )
        kwargs.update(
            instances_dir=Path(cfg.instances_base),
            cache_dir=Path(cfg.cache_base) / cfg.data_card.name,
            output_dir=working_dir,
            instance_name=cfg.experiment.instance_name,
            verbose=False,
            luigi_cfg=cfg.luigi,
        )
    return instantiate(solver_cfg, **kwargs)


def build_decision_engine(
    cfg: DictConfig,
    data_card: DataCard,
    state_adapters: dict,
) -> DecisionEngine:
    solvers = {}
    policies = {}
    for problem_key, problem_cfg in cfg.engines.decision_engine.problems.items():
        solver = _build_solver(cfg, problem_key, problem_cfg.solver)
        effective_card = deepcopy(data_card)
        effective_card.problem_class = problem_key
        features = dict(
            (effective_card.warehouse_info or {}).get("features") or {}
        )
        features.update(
            {
                feature: True
                for feature in state_adapters[
                    problem_key
                ].projected_features()
            }
        )
        warehouse_info = dict(effective_card.warehouse_info or {})
        warehouse_info["features"] = features
        effective_card.warehouse_info = warehouse_info
        solver.prepare(effective_card)
        solvers[problem_key] = solver
        if problem_cfg.get("commitment_policy") is not None:
            policies[problem_key] = instantiate(
                problem_cfg.commitment_policy
            )

    event_map = {
        name: instantiate(event)
        for name, event in (
            cfg.engines.decision_engine.get("event_map") or {}
        ).items()
    }
    return DecisionEngine(
        solver_map=solvers,
        commitment_policies=policies,
        event_map=event_map,
    )


def build_runtime(
    cfg: DictConfig,
    data_card: DataCard | None = None,
) -> tuple[SimulationEngine, DecisionEngine]:
    """Construct and prepare the complete CASIM runtime before reset."""
    if data_card is None:
        data_card = load_and_flatten_data_card(
            OmegaConf.to_container(cfg.data_card, resolve=True)
        )
    simulation = build_simulation(cfg)
    decision_engine = build_decision_engine(
        cfg, data_card, simulation.state_adapters
    )
    return simulation, decision_engine
