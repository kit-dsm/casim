"""Henn's objectives in minutes, from a run's ``orders.csv`` and ``tours.csv``.

Completion: the end of the last batch. Turnover, journal definition: completion
minus arrival of an order. Turnover, the working paper's stated definition:
completion minus the start of the order's tour, which includes the 3 min setup
(``tours.csv`` gives the start of the route, after the setup).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _orders(output_dir: Path) -> pd.DataFrame:
    orders = pd.read_csv(output_dir / "orders.csv")
    if orders["completion"].isna().any():
        raise ValueError("not every order was completed")
    return orders


def completion_min(output_dir: Path) -> float:
    return float(_orders(output_dir)["completion"].max()) / 60


def turnover_arrival_min(output_dir: Path) -> float:
    orders = _orders(output_dir)
    return float((orders["completion"] - orders["arrival"]).mean()) / 60


def turnover_start_min(output_dir: Path) -> float:
    orders = _orders(output_dir)
    tours = pd.read_csv(output_dir / "tours.csv", dtype={"order_ids": str})
    start = {int(order): route_start - setup
             for route_start, setup, order_ids in zip(tours["start"], tours["setup"],
                                                       tours["order_ids"].fillna(""))
             for order in order_ids.split()}
    return float((orders["completion"] - orders["order_id"].map(start)).mean()) / 60
