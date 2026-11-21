"""P1.7: the order generator G3 checked on held-out days.

G3 (``scripts/ijpe/order_generator.py``) is fitted on Monday 13 to Wednesday
15 April only and generates Thursday 16, Friday 17 and Saturday 11 April,
which it has not seen. The held-out day gives the generator one input, its
number of store-days (the demand level a user sets); the store-day profiles
are resampled from the three calibration days, and the profile draw is held
to the calibration days' mean lines per store-day times that number, not to
the held-out day's lines. Thirty seeds per day. The seed band holds only
sampling variation; the range of the three recorded calibration days is
given next to it as the day-to-day variation the fit has seen.

Measures per day, recorded against the mean and the 90 % band of the seeds:
orders, lines, lines per order, aisles per order (aisle access), and goods
volume per order. The volume is the canonical one: packages per line are
GEWICHT_SOLL over the article's package weight, times the package volume
(``scripts/ijpe/canonical.py``). MENGE_IST counts single units, not packages;
the script also reports how far quantity times package volume would
overstate the volume. Aisles come from one fixed map of rack faces for the
recorded and the generated days.

Writes aggregates only to ``outputs/credibility/generator_holdout.csv``.

    uv run --with openpyxl python -m scripts.ijpe.generator_holdout

(The article file ``lbh_kw16.xlsx`` for the unit check needs openpyxl.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.ijpe import OUTPUTS
from scripts.ijpe.canonical import package_weights
from scripts.ijpe.order_generator import aisle_of, fit, generate, load

CALIBRATION = ("2026-04-13", "2026-04-14", "2026-04-15")
HELD_OUT = ("2026-04-16", "2026-04-17", "2026-04-11")
SEEDS = range(30)
DIRECTORY = OUTPUTS / "credibility"


def _aisle_map(lines: pd.DataFrame) -> dict[float, int]:
    faces = lines["x"].round(-2)
    return dict(zip(faces, aisle_of(lines)))


def measures(orders: pd.DataFrame, lines: pd.DataFrame, aisles: dict[float, int]) -> dict:
    """Day aggregates that matter to the model."""
    lines = lines[lines["order_id"].isin(orders["order_id"])]
    per_order = lines.assign(aisle=lines["x"].round(-2).map(aisles)).groupby("order_id").agg(
        n=("article", "size"), aisles=("aisle", "nunique"), volume=("volume", "sum"))
    return {
        "orders": len(per_order),
        "lines": int(per_order["n"].sum()),
        "lines/order mean": per_order["n"].mean(),
        "lines/order p10": per_order["n"].quantile(.1),
        "lines/order p50": per_order["n"].median(),
        "lines/order p90": per_order["n"].quantile(.9),
        "aisles/order mean": per_order["aisles"].mean(),
        "share of orders in <= 2 aisles": (per_order["aisles"] <= 2).mean(),
        "share of orders in >= 6 aisles": (per_order["aisles"] >= 6).mean(),
        "volume/order mean, m3": per_order["volume"].mean(),
        "volume/order p50, m3": per_order["volume"].median(),
        "volume/order p90, m3": per_order["volume"].quantile(.9),
        "volume, m3": per_order["volume"].sum(),
    }


def held_out_calibration(cal: dict, day: str, store_days: int) -> dict:
    """``cal`` restricted to generating ``day`` from the calibration days alone."""
    profiles = [p for d in CALIBRATION for p in cal["day_profiles"][d]]
    per_store_day = (sum(cal["day_lines"][d] for d in CALIBRATION)
                     / sum(cal["store_days_per_day"][d] for d in CALIBRATION))
    return {**cal, "days": [day], "day_profiles": {day: profiles},
            "store_days_per_day": {day: store_days},
            "day_lines": {day: store_days * per_store_day}}


def unit_check(lines: pd.DataFrame) -> dict:
    """How far quantity (single units) times package volume overstates the volume."""
    weights = package_weights()
    packages = lines["weight"] / lines["article"].astype(str).map(weights)
    package_volume = lines["volume"] / packages
    naive = lines["quantity"] * package_volume
    return {"quantity x package volume over package-based volume": naive.sum() / lines["volume"].sum(),
            "units per package, median line": (lines["quantity"] / packages).median()}


def check() -> pd.DataFrame:
    orders, lines = load()
    aisles = _aisle_map(lines)
    cal_orders = orders[orders["day"].isin(CALIBRATION)]
    cal = fit(cal_orders, lines[lines["day"].isin(CALIBRATION)])
    spread = pd.DataFrame([measures(orders[orders["day"] == d], lines, aisles) for d in CALIBRATION])
    rows = []
    for day in HELD_OUT:
        day_orders = orders[orders["day"] == day]
        recorded = measures(day_orders, lines, aisles)
        cal_day = held_out_calibration(cal, day, day_orders["customer"].nunique())
        generated = pd.DataFrame([measures(*generate(cal_day, "G3", seed), aisles)
                                  for seed in SEEDS])
        for measure, value in recorded.items():
            column = generated[measure]
            low, high = column.quantile(.05), column.quantile(.95)
            rows.append({"day": day, "measure": measure, "recorded": value,
                         "generated mean": column.mean(), "band low": low, "band high": high,
                         "inside": bool(low <= value <= high),
                         "calibration days low": spread[measure].min(),
                         "calibration days high": spread[measure].max()})
    table = pd.DataFrame(rows)
    full = pd.read_csv(OUTPUTS.parent / "data" / "canonical" / "lines.csv",
                       usecols=["order_id", "article", "quantity", "weight", "volume"])
    held = full[full["order_id"].isin(orders.loc[orders["day"].isin(HELD_OUT), "order_id"])]
    units = unit_check(held)
    table = pd.concat([table, pd.DataFrame([{"day": "held-out days", "measure": k, "recorded": v}
                                            for k, v in units.items()])], ignore_index=True)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    table.to_csv(DIRECTORY / "generator_holdout.csv", index=False)
    return table


def main() -> None:
    table = check()
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


if __name__ == "__main__":
    main()
