from pathlib import Path
from copy import deepcopy

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import DataCard, load_and_flatten_data_card

from casim.decision_engine.decision_engine import DecisionEngine
from casim.loggers import DashLogger, KPILogger
from casim.events.operational_events import InterventionRequest
from casim.simulation_engine.simulation_engine import SimulationEngine


def build_data_loader(cfg: DictConfig) -> DataLoader:
    loader_cfg = cfg.input.data_loader
    if not isinstance(loader_cfg, DictConfig) or not loader_cfg.get(
        "_target_"
    ):
        raise ValueError(
            "input.data_loader must be a Hydra object with an "
            "explicit _target_ class path"
        )
    data_loader = instantiate(
        loader_cfg,
        _recursive_=False,
    )
    return data_loader

def build_solvers(cfg):
    solver_map = {}

    for problem_key, problem_cfg in cfg.engines.decision_engine.problems.items():
        working_dir = cfg.experiment.get(
            "working_dir",
            cfg.experiment.output_dir,
        )
        solver_map[problem_key] = instantiate(
            problem_cfg.solver,
            problem_class=problem_key,
            instances_dir=Path(cfg.instances_base),
            cache_dir=Path(cfg.cache_base) / cfg.data_card.name,
            output_dir=working_dir,
            instance_name=cfg.experiment.instance_name,
            verbose=False,
            luigi_cfg=cfg.luigi
        )

    return solver_map

def build_commitment_policies(cfg):
    return {
        problem_key: instantiate(problem_cfg.commitment_policy)
        for problem_key, problem_cfg in cfg.engines.decision_engine.problems.items()
    }

def setup_decision_engine(
    cfg: DictConfig,
    dc,
    state_adapters: dict | None = None,
) -> DecisionEngine:
    solver_map = build_solvers(cfg)
    commitment_policies = build_commitment_policies(cfg)

    for problem_key, solver in solver_map.items():
        effective_card = deepcopy(dc)
        effective_card.problem_class = problem_key
        if state_adapters is not None:
            adapter = state_adapters[problem_key]
            planning_features = adapter.projected_features()
        else:
            adapter_cfg = (
                cfg.engines.simulation_engine.problems[problem_key].state_adapter
            )
            adapter_cls = get_class(str(adapter_cfg._target_))
            planning_features = tuple(
                getattr(adapter_cls, "planning_features", ())
            )
        warehouse_info = dict(effective_card.warehouse_info or {})
        features = dict(warehouse_info.get("features") or {})
        features.update({feature: True for feature in planning_features})
        warehouse_info["features"] = features
        effective_card.warehouse_info = warehouse_info
        solver.build_pipelines(effective_card)

    return DecisionEngine(
        solver_map=solver_map,
        commitment_policies=commitment_policies,
    )

def build_simulation_problems(cfg: DictConfig):
    state_adapters = {}
    conditions_map = {}
    triggers_map = {}

    for problem_key, pcfg in cfg.simulation_engine.problems.items():
        state_adapters[problem_key] = instantiate(pcfg.state_adapter)
        conditions_map[problem_key] = [
            instantiate(c) for c in (pcfg.get("conditions") or [])
        ]
        for event_name in (pcfg.get("triggers") or []):
            event_cls = get_class(str(event_name))
            if event_cls in triggers_map:
                raise ValueError(
                    f"Event '{event_name}' is already bound to problem "
                    f"'{triggers_map[event_cls]}', cannot also bind to '{problem_key}'"
                )
            triggers_map[event_cls] = problem_key

    return state_adapters, conditions_map, triggers_map

def setup_scenario(cfg: DictConfig) -> SimulationEngine:
    state_adapters, conditions_map, triggers_map = build_simulation_problems(cfg.engines)

    loader = build_data_loader(cfg)
    loader_kwargs = dict(cfg.input.get("load") or {})

    working_dir = cfg.experiment.get(
        "working_dir",
        cfg.experiment.output_dir,
    )
    event_loggers = [
        KPILogger(
            Path(working_dir) / "kpis",
            print_every=cfg.experiment.get("progress_every", 5000),
        )
    ]
    viz_cfg = cfg.get("viz") or {}
    if viz_cfg.get("record", viz_cfg.get("launch", False)):
        event_loggers.append(DashLogger(Path(cfg.experiment.output_dir) / "viz"))

    return SimulationEngine(
        state_adapters=state_adapters,
        data_loader=loader,
        loader_kwargs=loader_kwargs,
        triggers_map=triggers_map,
        conditions_map=conditions_map,
        event_loggers=event_loggers,
        completion_mode=str(
            cfg.engines.simulation_engine.get(
                "completion_mode",
                "drain",
            )
        ),
        horizon_time=cfg.engines.simulation_engine.get("horizon_time"),
        intervention_enabled=InterventionRequest in triggers_map,
        active_batch_insertion_enabled=(
            triggers_map.get(InterventionRequest) == "OBRP"
        ),
    )
