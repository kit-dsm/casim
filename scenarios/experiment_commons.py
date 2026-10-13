"""Small scenario setup helpers: compose, resolve, reset, run, report."""

from pathlib import Path

from hydra.utils import get_class
from omegaconf import DictConfig
from ware_ops_algos.domain_models import DataCard

from casim.config.data_card import load_and_flatten_data_card
from casim.decision_engine.decision_engine import DecisionEngine
from casim.loggers import DashLogger, KPILogger
from casim.setup import resolve
from casim.simulation_engine.simulation_engine import SimulationEngine

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
    trigger_map = {trigger: tuple(name for name, setup in setups.items()
                                  if trigger in setup.trigger_types)
                   for setup in setups.values() for trigger in setup.trigger_types}
    simulation = SimulationEngine(
        state_adapters={name: setup.adapter for name, setup in setups.items()},
        problem_classes={name: setup.problem_class for name, setup in setups.items()},
        data_loader=loader,
        domain_cache_path=str(Path(cfg.cache_base) / "dynamic_info.pkl"),
        loader_kwargs=loader_kwargs,
        triggers_map=trigger_map,
        conditions_map={name: setup.conditions for name, setup in setups.items()},
        precedence={name: setup.config.precedence for name, setup in setups.items()},
        periods={name: setup.period_s for name, setup in setups.items()
                 if setup.period_s is not None},
        event_loggers=loggers,
    )
    decision = DecisionEngine(
        solver_map={name: setup.solver for name, setup in setups.items()
                    if setup.solver is not None},
        solution_rankers={name: setup.ranker for name, setup in setups.items()
                          if setup.ranker is not None},
        commitment_policies={name: setup.commitment_policy
                             for name, setup in setups.items()
                             if setup.commitment_policy is not None},
    )
    loggers[0].decision_tracker = decision.decision_tracker
    return simulation, decision
