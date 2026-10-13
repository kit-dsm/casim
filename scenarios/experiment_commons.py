"""Small scenario setup helpers: compose, resolve, reset, run, report."""

from pathlib import Path

from hydra.utils import get_class
from omegaconf import DictConfig
from ware_ops_algos.domain_models import DataCard, information_card_section

from casim.decision_engine.decision_engine import DecisionEngine
from casim.loggers import DashLogger, KPILogger
from casim.setup import resolve
from casim.simulation_engine.simulation_engine import SimulationEngine

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
        information=information_card_section(raw.get("information")),
    )


def setup_experiment(cfg: DictConfig, data_card: DataCard):
    """Build each named decision once and share it with both engines."""
    setups = resolve(cfg, data_card)
    loader_type = get_class(cfg.data_card.source.data_loader)
    loader = loader_type(
        instances_dir=Path(cfg.instances_base) / cfg.data_card.name, cfg=cfg,
    )
    loader_kwargs = {
        key: Path(cfg.instances_base) / value
        for key, value in cfg.data_card.source.items() if key.endswith("path")
    }
    loggers = [KPILogger(Path(cfg.experiment.output_dir) / "kpis")]
    if cfg.viz.launch:
        loggers.append(DashLogger(Path(cfg.experiment.output_dir) / "viz"))
    trigger_map = {trigger: name for name, setup in setups.items()
                   for trigger in setup.trigger_types}
    simulation = SimulationEngine(
        state_adapters={name: setup.adapter for name, setup in setups.items()},
        problem_classes={name: setup.problem_class for name, setup in setups.items()},
        data_loader=loader,
        domain_cache_path=str(Path(cfg.cache_base) / "dynamic_info.pkl"),
        loader_kwargs=loader_kwargs,
        triggers_map=trigger_map,
        conditions_map={name: setup.conditions for name, setup in setups.items()},
        event_loggers=loggers,
    )
    simulation.on_violation = cfg.get("on_violation", "reject")
    simulation.precedence = {name: setup.config.precedence
                             for name, setup in setups.items()}
    simulation.periods = {name: setup.period_s for name, setup in setups.items()
                          if setup.period_s is not None}
    decision = DecisionEngine(
        solver_map={name: setup.solver for name, setup in setups.items()
                    if setup.solver is not None},
        commitment={name: setup.commitment for name, setup in setups.items()},
        waiting={name: setup.waiting for name, setup in setups.items()},
        release_gates={name: setup.config.release_gates
                       for name, setup in setups.items()},
    )
    simulation.decision_engine = decision
    return simulation, decision
