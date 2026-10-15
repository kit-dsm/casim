"""The paper's measures (Section 4 of *Walking vs. Waiting*) from a run's output files.

MOCT is the mean order completion time (completion minus arrival), MLPI the
travelled distance per picked order line, TARD the mean tardiness against the
due dates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _orders(output_dir: Path) -> pd.DataFrame:
    orders = pd.read_csv(output_dir / "orders.csv")
    if orders["completion"].isna().any():
        raise ValueError("not every order was completed")
    return orders


def mean_order_completion_time(output_dir: Path) -> float:
    orders = _orders(output_dir)
    return float((orders["completion"] - orders["arrival"]).mean())


def mean_length_per_item(output_dir: Path) -> float:
    tours = pd.read_csv(output_dir / "tours.csv")
    kpis = pd.read_json(output_dir / "kpis.json", typ="series")
    return float(kpis["total_distance"] / tours["picks"].sum())


def mean_tardiness(output_dir: Path) -> float:
    orders = _orders(output_dir)
    return float((orders["completion"] - orders["due_date"]).clip(lower=0.0).mean())


def _empty_starts(output_dir: Path) -> pd.DataFrame:
    tours = pd.read_csv(output_dir / "tours.csv", dtype={"order_ids": str})
    return tours[(tours["status"] == "done") & (tours["orders_at_start"] == 0)]


def empty_tours_started(output_dir: Path) -> int:
    """Tours that left without an order (wait-0)."""
    return len(_empty_starts(output_dir))


def orders_admitted_after_empty_start(output_dir: Path) -> int:
    return int(_empty_starts(output_dir)["order_ids"].fillna("").str.split().str.len().sum())
