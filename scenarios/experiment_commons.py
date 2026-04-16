from pathlib import Path

import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig
from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import DataCard

from casim.decision_engine.decision_engine import DecisionEngine
from casim.loggers import DashLogger, KPILogger
from casim.pipelines.objective_evaluator import ObjectiveEvaluator
from casim.simulation_engine import SimulationEngine
from casim.state.conditions import build_condition_policies
from casim.state.state_snapshot import build_state_snapshots
from casim.events.decision_events import RoutingDone, PickListDone
from casim.events.operational_events import OrderArrival, PickerArrival, PickerTourQuery, PickerIdle, TourEnd, \
    ShiftStart, FlushRemainingOrders
from scenarios.scenario_henn_online.henn_online_loader import HennOnlineLoader

LOADER_REGISTRY = {
    "HennOnlineLoader": HennOnlineLoader
}

EVENT_REGISTRY = {
    "OrderArrival": OrderArrival,
    "RoutingDone": RoutingDone,
    "PickerArrival": PickerArrival,
    "PickerTourQuery": PickerTourQuery,
    "PickerIdle": PickerIdle,
    "TourEnd": TourEnd,
    # "PickListSelectionDone": PickListSelectionDone,
    "ShiftStart": ShiftStart,
    "FlushRemainingOrders": FlushRemainingOrders,
    "PickListDone": PickListDone
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


def build_trigger_map(cfg: DictConfig) -> dict:
    return {
        EVENT_REGISTRY[event_name]: policy_key
        for event_name, policy_key in cfg.decision_engine.triggers.items()
    }


def build_pipeline_runner(cfg):
    return {problem_key: instantiate(
        st_cfg,
        instance_set_name=cfg.data_card.name,
        instances_dir=Path(cfg.instances_base),
        cache_dir=Path(cfg.cache_base) / cfg.data_card.name,
        output_dir=cfg.experiment.output_dir,
        instance_name=cfg.experiment.instance_name,
        verbose=True) for problem_key, st_cfg in cfg.decision_engine.pipeline_runner.items()
    }


def build_data_loader(cfg: DictConfig) -> DataLoader:
    data_loader_cls = cfg.data_card.source.data_loader
    print(data_loader_cls)
    data_loader = LOADER_REGISTRY[data_loader_cls](
        instances_dir=Path(cfg.instances_base) /
                      cfg.data_card.name,
                      cfg=cfg)
    return data_loader


def setup_decision_engine(cfg: DictConfig, dc) -> DecisionEngine:
    pipeline_runners = build_pipeline_runner(cfg)
    for key in pipeline_runners:
        dc.problem_class = key
        pipeline_runners[key].build_pipelines(dc)

    evaluator = ObjectiveEvaluator(objective=cfg.get("objective", "makespan"))
    return DecisionEngine(execution_map=pipeline_runners, evaluator=evaluator)


def setup_scenario(cfg: DictConfig) -> SimulationEngine:
    # Configuration
    instances_dir = Path(cfg.instances_base) / cfg.data_card.name
    cache_dir = Path(cfg.cache_base) / cfg.data_card.name
    cache_path = Path(cfg.cache_base) / "dynamic_info.pkl"
    cache_dir.mkdir(parents=True, exist_ok=True)

    state_transformers = build_state_snapshots(cfg)

    loader_kwargs = {k: instances_dir / v
                     for k, v in cfg.data_card.source.items()
                     if k.endswith("path")}

    loader = build_data_loader(cfg)
    trigger_map = build_trigger_map(cfg)
    conditions_map = build_condition_policies(cfg)

    sim = SimulationEngine(
        state_transformers=state_transformers,
        data_loader=loader,
        cache_path=Path(cfg.cache_base),
        domain_cache_path=str(cache_path),
        loader_kwargs=loader_kwargs,
        triggers=trigger_map,
        conditions_map=conditions_map,
        event_loggers=[DashLogger(Path(cfg.experiment.output_dir) / "event_logs" / "dash_log.pkl"),
                       KPILogger()]
    )
    return sim
