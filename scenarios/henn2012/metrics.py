"""Henn's objectives in minutes, from a run's ``facts/orders.csv`` and ``facts/tours.csv``.

Completion: the end of the last batch. Turnover, journal definition: completion
minus arrival of an order. Turnover, the working paper's stated definition:
completion minus the start of the order's tour, which includes the 3 min setup
(``start`` in ``facts/tours.csv`` is the start of the setup).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _orders(output_dir: Path) -> pd.DataFrame:
    orders = pd.read_csv(output_dir / "facts" / "orders.csv")
    if orders["completion"].isna().any():
        raise ValueError("not every order was completed")
    return orders


def completion_min(output_dir: Path) -> float:
    return float(_orders(output_dir)["completion"].max()) / 60


def turnover_arrival_min(output_dir: Path) -> float:
    orders = _orders(output_dir)
    return float((orders["completion"] - orders["release_time"]).mean()) / 60


def turnover_start_min(output_dir: Path) -> float:
    orders = _orders(output_dir)
    tours = pd.read_csv(output_dir / "facts" / "tours.csv", dtype={"order_ids": str})
    start = {int(order): setup_start
             for setup_start, order_ids in zip(tours["start"], tours["order_ids"].fillna(""))
             for order in order_ids.split()}
    return float((orders["completion"] - orders["order_id"].map(start)).mean()) / 60
