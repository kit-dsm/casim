"""Solve one static CASOP instance in CASIM: one decision at t = 0, every tour committed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.operational_events import OrderStreamClosed, PickerArrival
from casim.runner import apply_decision
from casim.simulation_engine.simulation_engine import SimulationEngine
from scenarios.experiment_commons import load_and_flatten_data_card, setup_experiment
from scenarios.scenario_static_benchmarks.pipelines import PIPELINES

CONFIG = Path(__file__).resolve().parent / "config"
DATA_CARDS = {
    "SPRP": "sprp", "SPRP-SS": "sprp_ss", "BahceciOencan": "bahceci_oencan",
    "HennWaescherUniform": "henn_waescher", "HennWaescherClassBased": "henn_waescher",
    "MuterOencanWG": "muter_oencan", "FoodmartData": "foodmart",
    "KrisSmallDataCorrected": "kris", "KrisLargeData": "kris",
}


@dataclass(frozen=True)
class CasimResult:
    """What the executed tours give."""

    distance: float
    total_time: float
    max_lateness: float | None
    route_distances: tuple[float, ...]
    batches: tuple[tuple[int, ...], ...]
    decisions: int


def static_release(sim: SimulationEngine, domain: SimWarehouseDomain) -> None:
    """Pickers, then orders, then the end of the stream, all at t = 0."""
    for resource in domain.resources.resources:
        sim.add_event(PickerArrival(time=0.0, picker_id=resource.id, picker_available=True))
    for order in domain.orders.orders:
        sim.add_order(order)
    sim.add_event(OrderStreamClosed(0.0))


def compose_config(instance_set: str, instance: str, label: str, output_dir: Path) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG), version_base="1.3"):
        cfg = compose(config_name="static_benchmarks_config", overrides=[
            f"data_card={DATA_CARDS[instance_set]}",
            f"benchmark.instance_set={instance_set}",
            f"benchmark.instance='{instance}'",
            f"+benchmark.label={label}",
            f"project_root={Path.cwd().as_posix()}",
            f"cache_base={(output_dir / 'cache').as_posix()}",
            f"experiment.output_dir={output_dir.as_posix()}",
        ])
    OmegaConf.update(cfg, "cosy_repo.components", PIPELINES[label].components())
    return cfg


def run(instance_set: str, instance: str, label: str, output_dir: Path,
        audit: bool = False) -> CasimResult:
    cfg = compose_config(instance_set, instance, label, output_dir)
    OmegaConf.update(cfg, "experiment.audit", audit, force_add=True)
    sim, engine = setup_experiment(cfg, load_and_flatten_data_card(cfg.data_card))
    sim.reset(hooks=[static_release])
    decisions = 0
    while True:
        done, snapshot = sim.run()
        if done:
            break
        decisions += 1
        apply_decision(sim, engine, engine.on_trigger(snapshot), snapshot)
    return _result(sim, decisions)


def _result(sim: SimulationEngine, decisions: int) -> CasimResult:
    """Total time counts each tour from the start of its setup (TourStart is after it)."""
    state = sim.state
    tours = state.tour_manager.get_all_tours()
    lateness = []
    total_time = 0.0
    for _, start, end, order_ids, picker_id, *_ in state.tracker.completed_tours:
        setup = state.resource_manager.get_resource(picker_id).tour_setup_time or 0.0
        total_time += end - start + setup
        for order_id in order_ids:
            due = state.order_manager.get_order_from_history(order_id).due_date
            if due is not None:
                lateness.append(end - due)
    return CasimResult(
        distance=sum(state.tracker.distance_by_picker.values()),
        total_time=total_time,
        max_lateness=max(lateness) if lateness else None,
        route_distances=tuple(sorted(tour.route.distance for tour in tours)),
        batches=tuple(sorted(tuple(sorted(tour.order_numbers)) for tour in tours)),
        decisions=decisions,
    )
