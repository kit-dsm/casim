"""Run a scenario directory: validate, simulate, decide, write the outputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from ware_ops_algos.planning.lifecycle import TourStates

from casim.baseline_capture import BaselineCapture
from casim.config.data_card import load_and_flatten_data_card
from casim.decision_engine.decision_engine import DecisionEngine
from casim.loggers import KPILogger
from casim.reference import compare_with_reference
from casim.runner import apply_decision
from casim.setup import DecisionSetup, resolve
from casim.simulation_engine.simulation_engine import SimulationEngine
from casim.sources.base import Source, SourceLoader, build_source, release
from casim.trace import TraceRecorder

SCENARIO_FILE = "scenario.yaml"
REFERENCE_FILE = "reference.csv"


def compose_scenario(scenario_dir: Path, overrides: list[str] = (),
                     output_dir: Path | None = None) -> DictConfig:
    """Compose ``scenario.yaml`` with the run settings every scenario shares."""
    scenario_dir = scenario_dir.resolve()
    with initialize_config_dir(config_dir=str(scenario_dir), version_base="1.3"):
        cfg = compose(config_name=SCENARIO_FILE.removesuffix(".yaml"),
                      overrides=list(overrides))
    output_dir = (output_dir or scenario_dir / "outputs" / "run").resolve()
    shared = OmegaConf.create({
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
             recorder: TraceRecorder) -> tuple[SimulationEngine, DecisionEngine]:
    output_dir = Path(cfg.experiment.output_dir)
    loggers = [KPILogger(output_dir / "kpis"), recorder]
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
    )
    loggers[0].decision_tracker = decision.decision_tracker
    return simulation, decision


def _write_outputs(sim: SimulationEngine, engine: DecisionEngine,
                   output_dir: Path, recorder: TraceRecorder) -> None:
    (output_dir / "kpis.json").write_text(
        (output_dir / "kpis" / "kpis.json").read_text(encoding="utf-8"), encoding="utf-8")
    tracker = engine.decision_tracker
    (output_dir / "decisions.json").write_text(json.dumps({
        "decisions": tracker.num_decisions,
        "pipeline_counts": dict(tracker.pipeline_counts),
    }, indent=2), encoding="utf-8")
    with open(output_dir / "tours.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("tour_id", "picker_id", "status", "start", "end", "picks",
                         "orders_at_start", "order_ids"))
        for tour in sorted(sim.state.tour_manager.get_all_tours(),
                           key=lambda tour: tour.tour_id):
            writer.writerow((tour.tour_id, tour.assigned_resource, tour.status.value,
                             tour.start_time, tour.end_time, len(tour.completed_picks),
                             len(tour.original_route.batch.orders),
                             " ".join(map(str, tour.order_numbers))))
    manager = sim.state.order_manager
    completions = {order_id: tour.end_time
                   for tour in sim.state.tour_manager.get_all_tours()
                   if tour.status is TourStates.DONE for order_id in tour.order_numbers}
    orders = {order.order_id: order for order in manager.get_order_buffer()}
    orders.update({order_id: manager.get_order_from_history(order_id)
                   for order_id in completions})
    with open(output_dir / "orders.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("order_id", "arrival", "due_date", "completion"))
        for order_id in sorted(orders):
            writer.writerow((order_id, orders[order_id].order_date, orders[order_id].due_date,
                             completions.get(order_id)))
    (output_dir / "trace.json").write_text(json.dumps({
        "trace_digest": recorder.digest(), "event_count": len(recorder.trace),
    }, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class Prepared:
    """Both engines of a validated scenario configuration, with its source."""

    sim: SimulationEngine
    engine: DecisionEngine
    source: Source
    recorder: TraceRecorder

    def reset(self) -> None:
        """Load the initial state and queue the source's events."""
        self.sim.reset(hooks=[release(self.source)])


def prepare(cfg: DictConfig) -> Prepared:
    """Validate ``cfg`` and build its engines and source; nothing has run yet."""
    setups = check(cfg)
    source = build_source(OmegaConf.to_container(cfg.source, resolve=True),
                          Path(cfg.scenario_dir))
    recorder = TraceRecorder()
    sim, engine = _engines(cfg, setups, source, recorder)
    return Prepared(sim, engine, source, recorder)


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
    _write_outputs(sim, engine, output_dir, recorder)
    reference = Path(cfg.scenario_dir) / REFERENCE_FILE
    if reference.exists():
        compare_with_reference(reference, cfg, output_dir)
    return RunResult(output_dir, recorder.digest(), len(recorder.trace))
