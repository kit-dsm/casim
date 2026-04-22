from pathlib import Path
from typing import Type

from hydra.utils import instantiate
from omegaconf import DictConfig
from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import DataCard

from casim.decision_engine.decision_engine import DecisionEngine
from casim.events.base_events import Event
from casim.loggers import DashLogger, KPILogger
from casim.events.decision_events import RoutingDone, PickListDone
from casim.events.operational_events import OrderArrival, PickerArrival, PickerTourQuery, PickerIdle, TourEnd, \
    ShiftStart, FlushRemainingOrders, TruckDeparture, WMSRun
from casim.simulation_engine.simulation_engine import SimulationEngine
from scenarios.scenario_grocery_retailer.grocery_retailer_loader import GroceryRetailerLoader
from scenarios.scenario_henn_online.henn_online_loader import HennOnlineLoader

LOADER_REGISTRY = {
    "HennOnlineLoader": HennOnlineLoader,
    "GroceryRetailerLoader": GroceryRetailerLoader
}

EVENT_REGISTRY: dict[str, Type[Event]]  = {
    "OrderArrival": OrderArrival,
    "RoutingDone": RoutingDone,
    "PickerArrival": PickerArrival,
    "PickerTourQuery": PickerTourQuery,
    "PickerIdle": PickerIdle,
    "TourEnd": TourEnd,
    # "PickListSelectionDone": PickListSelectionDone,
    "ShiftStart": ShiftStart,
    "FlushRemainingOrders": FlushRemainingOrders,
    "PickListDone": PickListDone,
    "TruckDeparture": TruckDeparture,
    "WMSRun": WMSRun
}


def load_and_flatten_data_card(raw) -> DataCard:
    # with open(card_path, "r", encoding="utf-8") as f:
    #     raw = yaml.safe_load(f)
    def flatten_domain(domain: dict) -> dict:
        features = {}
        for obj in domain.get("objects", []):
            for feat in obj.get("features", []):
                name = feat["name"]
                if "value" in feat:
                    features[name] = feat["value"]
                else:
                    features[name] = True
            features.update(flatten_domain(obj))
        return features

    def section(domain: dict) -> dict:
        return {
            "type": domain.get("type"),
            "features": flatten_domain(domain),
        }

    return DataCard(
        name=raw.get("name", ""),
        problem_class=raw.get("problem_class", ""),
        objective=raw.get("objective", ""),
        layout=section(raw.get("layout", {})),
        articles=section(raw.get("articles", {})),
        orders=section(raw.get("orders", {})),
        resources=section(raw.get("resources", {})),
        storage=section(raw.get("storage", {})),
        warehouse_info=section(raw.get("warehouse_info", {})),
    )

def build_data_loader(cfg: DictConfig) -> DataLoader:
    data_loader_cls = cfg.data_card.source.data_loader
    print(data_loader_cls)
    data_loader = LOADER_REGISTRY[data_loader_cls](
        instances_dir=Path(cfg.instances_base) /
                      cfg.data_card.name,
                      cfg=cfg)
    return data_loader

def build_solvers(cfg):
    solver_map = {}

    for problem_key, problem_cfg in cfg.scenario.decision_engine.problems.items():
        solver_map[problem_key] = instantiate(
            problem_cfg.solver,
            problem_class=problem_key,
            instances_dir=Path(cfg.instances_base),
            cache_dir=Path(cfg.cache_base) / cfg.data_card.name,
            output_dir=cfg.experiment.output_dir,
            instance_name=cfg.experiment.instance_name,
            verbose=True,
            luigi_cfg=cfg.luigi
        )

    return solver_map

def build_commitment_policies(cfg):
    return {
        problem_key: instantiate(problem_cfg.commitment_policy)
        for problem_key, problem_cfg in cfg.scenario.decision_engine.problems.items()
    }

def build_state_adapters(cfg: DictConfig) -> dict:
    return {
        problem_key: instantiate(st_cfg)
        for problem_key, st_cfg in cfg.decision_engine.state_snapshot.items()
    }

def setup_decision_engine(cfg: DictConfig, dc) -> DecisionEngine:
    solver_map = build_solvers(cfg)
    commitment_policies = build_commitment_policies(cfg)

    for problem_key, solver in solver_map.items():
        dc.problem_class = problem_key
        solver.build_pipelines(dc)

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
            event_cls = EVENT_REGISTRY[event_name]
            if event_cls in triggers_map:
                raise ValueError(
                    f"Event '{event_name}' is already bound to problem "
                    f"'{triggers_map[event_cls]}', cannot also bind to '{problem_key}'"
                )
            triggers_map[event_cls] = problem_key

    return state_adapters, conditions_map, triggers_map

def setup_scenario(cfg: DictConfig) -> SimulationEngine:
    instances_dir = Path(cfg.instances_base) / cfg.data_card.name
    cache_path = Path(cfg.cache_base) / "dynamic_info.pkl"

    state_adapters, conditions_map, triggers_map = build_simulation_problems(cfg.scenario)

    loader = build_data_loader(cfg)
    loader_kwargs = {
        k: instances_dir / v
        for k, v in cfg.data_card.source.items()
        if k.endswith("path")
    }

    event_loggers = [KPILogger(Path(cfg.experiment.output_dir) / "kpis")]
    if cfg.viz.launch:
        event_loggers.append(DashLogger(Path(cfg.experiment.output_dir) / "viz"))

    return SimulationEngine(
        state_adapters=state_adapters,
        data_loader=loader,
        domain_cache_path=str(cache_path),
        loader_kwargs=loader_kwargs,
        triggers_map=triggers_map,
        conditions_map=conditions_map,
        event_loggers=event_loggers,
    )
