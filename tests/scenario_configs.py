"""Compose the scenarios that tests run step by step."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from casim.run import compose_scenario

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
WALK_OR_WAIT = SCENARIOS / "walk_or_wait"


def walk_config(output_dir: Path, *overrides: str, orders: list[dict] | None = None) -> DictConfig:
    """walk_or_wait with ``overrides``; ``orders`` replaces the explicit example's orders."""
    cfg = compose_scenario(WALK_OR_WAIT, list(overrides), output_dir=output_dir)
    if orders is not None:
        cfg.source.orders = orders
    return cfg
