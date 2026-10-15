"""The OOBP objectives (Gil-Borrás et al. 2020, Section 2) from a run's ``orders.csv``.

Maximum completion time: the latest completion of an order. Maximum turnover
time: the largest completion minus arrival of an order. Both in seconds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _orders(output_dir: Path) -> pd.DataFrame:
    orders = pd.read_csv(output_dir / "orders.csv")
    if orders["completion"].isna().any():
        raise ValueError("not every order was completed")
    return orders


def max_completion_time(output_dir: Path) -> float:
    return float(_orders(output_dir)["completion"].max())


def max_turnover_time(output_dir: Path) -> float:
    orders = _orders(output_dir)
    return float((orders["completion"] - orders["arrival"]).max())
