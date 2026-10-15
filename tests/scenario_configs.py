"""Compose the scenarios that tests run step by step."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.run import compose_scenario
from casim.sources.base import build_source

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
WALK_OR_WAIT = SCENARIOS / "walk_or_wait"
ANALYTIC_WAITING = SCENARIOS / "analytic_waiting"


def analytic_config(output_dir: Path | None = None, *overrides: str) -> DictConfig:
    """analytic_waiting with ``overrides``."""
    return compose_scenario(ANALYTIC_WAITING, list(overrides), output_dir=output_dir)


def analytic_domain(cfg: DictConfig) -> SimWarehouseDomain:
    """The initial state of analytic_waiting's source under ``cfg``."""
    source = build_source(OmegaConf.to_container(cfg.source, resolve=True), ANALYTIC_WAITING)
    return source.initial_state()


def walk_config(output_dir: Path, *overrides: str, orders: list[dict] | None = None) -> DictConfig:
    """walk_or_wait with ``overrides``; ``orders`` replaces the explicit example's orders."""
    cfg = compose_scenario(WALK_OR_WAIT, list(overrides), output_dir=output_dir)
    if orders is not None:
        cfg.source.orders = orders
    return cfg
