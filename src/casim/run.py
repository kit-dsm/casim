"""Run a scenario directory: validate, simulate, decide, write the outputs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from casim.config.data_card import load_and_flatten_data_card
from casim.config.setup import DecisionSetup, resolve
from casim.decision_engine.decision_engine import DecisionEngine, DecisionResult
from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.outputs.audit import BaselineCapture
from casim.outputs.facts import FactsRecorder
from casim.outputs.kpis import KPILogger
from casim.outputs.reference import compare_with_reference
from casim.outputs.trace import TraceRecorder
from casim.outputs.write import write_outputs
from casim.simulation_engine.simulation_engine import SimulationEngine
from casim.sources.base import Source, SourceLoader, build_source, release

SCENARIO_FILE = "scenario.yaml"
REFERENCE_FILE = "reference.csv"


def compose_scenario(scenario_dir: Path, overrides: list[str] = (),
                     output_dir: Path | None = None) -> DictConfig:
    """Compose ``scenario.yaml`` with the run settings every scenario shares.

    ``choices`` records the file chosen in every group (for example
    ``choices.decisions``), so that ``reference.csv`` can select rows by them.
    """
    scenario_dir = scenario_dir.resolve()
    with initialize_config_dir(config_dir=str(scenario_dir), version_base="1.3"):
        cfg = compose(config_name=SCENARIO_FILE.removesuffix(".yaml"),
                      overrides=list(overrides), return_hydra_config=True)
    choices = {group: choice for group, choice in cfg.hydra.runtime.choices.items()
               if not group.startswith("hydra") and choice is not None}
    with open_dict(cfg):
        del cfg["hydra"]
    output_dir = (output_dir or scenario_dir / "outputs" / "run").resolve()
    shared = OmegaConf.create({
        "choices": choices,
        "scenario_dir": scenario_dir.as_posix(),
        "project_root": Path.cwd().resolve().as_posix(),
        "instances_base": (scenario_dir / "data").as_posix(),
        "cache_base": (output_dir / "cache").as_posix(),
        "luigi": {"log_level": "CRITICAL", "logdir": None, "background": None,
                  "logging_conf_file": None},
        "experiment": {"output_dir": output_dir.as_posix(),
                       "instance_name": cfg.get("name", scenario_dir.name),
                       "audit": False},
    })
    return OmegaConf.merge(shared, cfg)


def sweep_overrides(cfg: DictConfig) -> list[list[str]]:
    """The override lists of a ``sweep`` group file: the product of its value lists.

    A sweep file maps override keys to a value or a list of values, for
    example ``source.filepath: [a.txt, b.txt]`` and ``decisions: [x, y]``.
    """
    if not cfg.get("sweep"):
        return []
    sweep = OmegaConf.to_container(cfg.sweep, resolve=True)
    axes = [[f"{key}={value}" for value in (values if isinstance(values, list) else [values])]
            for key, values in sweep.items()]
    return [list(combination) for combination in product(*axes)] if axes else []


def check(cfg: DictConfig) -> dict[str, DecisionSetup]:
    """Validate the configuration as ``run`` uses it; raise on the first error."""
    if "source" not in cfg or "name" not in cfg.source:
        raise ValueError("scenario.yaml needs source: {name: ..., ...}")
    return resolve(cfg, load_and_flatten_data_card(cfg.data_card))


@dataclass(frozen=True)
class RunResult:
    """Where a run wrote its outputs, and its trace digest."""

    output_dir: Path
    trace_digest: str
    event_count: int


def _engines(cfg: DictConfig, setups: dict[str, DecisionSetup], source,
             recorder: TraceRecorder, facts: FactsRecorder,
             ) -> tuple[SimulationEngine, DecisionEngine]:
    output_dir = Path(cfg.experiment.output_dir)
    loggers = [KPILogger(), recorder, facts]
    if cfg.experiment.audit:
        loggers.append(BaselineCapture(output_dir / "audit.json"))
    triggers = {trigger: tuple(name for name, setup in setups.items()
                               if trigger in setup.trigger_types)
                for setup in setups.values() for trigger in setup.trigger_types}
    simulation = SimulationEngine(
        state_adapters={name: setup.adapter for name, setup in setups.items()},
        problem_classes={name: setup.problem_class for name, setup in setups.items()},
        data_loader=SourceLoader(source),
        domain_cache_path=str(Path(cfg.cache_base) / "dynamic_info.pkl"),
        loader_kwargs={},
        triggers_map=triggers,
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
        start_times={name: setup.config.start_times for name, setup in setups.items()},
    )
    return simulation, decision


def apply_decision(sim: SimulationEngine, engine: DecisionEngine,
                   result: DecisionResult, snapshot: SimWarehouseDomain) -> None:
    """Apply a decision and record its accepted choice and timing."""
    started = perf_counter()
    sim.step(result, snapshot)
    engine.on_accepted(result, snapshot)
    engine.decision_tracker.on_commands(
        sim.state.current_time, snapshot.dynamic_warehouse_info.context.decision,
        result, snapshot, perf_counter() - started)


@dataclass(frozen=True)
class Prepared:
    """Both engines of a validated scenario configuration, with its source."""

    sim: SimulationEngine
    engine: DecisionEngine
    source: Source
    recorder: TraceRecorder
    facts: FactsRecorder

    def reset(self) -> None:
        """Load the initial state and queue the source's events."""
        self.sim.reset(hooks=[release(self.source)])


def prepare(cfg: DictConfig) -> Prepared:
    """Validate ``cfg`` and build its engines and source; nothing has run yet."""
    setups = check(cfg)
    source = build_source(OmegaConf.to_container(cfg.source, resolve=True),
                          Path(cfg.scenario_dir))
    recorder, facts = TraceRecorder(), FactsRecorder(source)
    sim, engine = _engines(cfg, setups, source, recorder, facts)
    return Prepared(sim, engine, source, recorder, facts)


def run(cfg: DictConfig) -> RunResult:
    """Own the loop of ``SimulationEngine.run``, ``on_trigger`` and ``step``."""
    prepared = prepare(cfg)
    sim, engine, recorder = prepared.sim, prepared.engine, prepared.recorder
    prepared.reset()
    while True:
        done, snapshot = sim.run()
        if done:
            break
        apply_decision(sim, engine, engine.on_trigger(snapshot), snapshot)
    output_dir = Path(cfg.experiment.output_dir)
    write_outputs(sim, engine, output_dir, recorder, prepared.facts)
    reference = Path(cfg.scenario_dir) / REFERENCE_FILE
    if reference.exists():
        compare_with_reference(reference, cfg, output_dir)
    return RunResult(output_dir, recorder.digest(), len(recorder.trace))
