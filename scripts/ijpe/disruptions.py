"""Write the disruption selections of the IJPE week (plan P2.1, docs/ijpe/scenarios.md).

Each scenario advances or adds a volume V times its target: the TE the
partner's mail names for it times 0.20, an assumed share of the area in the
warehouse's volume. The experiment (plan P4.2) uses two volumes: V = 1
(realistic) and V = 5, the mail's full TE all in the area (worst case, as
submitted). A store's volume on a day is the goods volume of its orders
(``volume`` of the canonical lines) in TE of 0.445 m³.

- S1 (Monday), S2 (Thursday), S4 (Thursday): whole stores of the slot of the
  original departure on the scenario day ("13:00" is the 14:00 slot; S4's
  Thursday 10:00 is the 10:00 slot), largest first until the target is
  reached. A slot holding less than the target is taken whole and the
  shortfall is printed. A store's slot comes from the scenario's own dispatch
  rule (``source.due_dates`` of ``scenarios/ijpe_week/scenario.yaml`` with
  ``scenarios.ijpe.source.store_slot``), so another store-to-slot rule changes
  the selection with it.
- S3 (Wednesday): one order of a new store from the order generator G3
  (``scripts.ijpe.order_generator``): a zone order of a Wednesday store-day
  profile, the first of 2,000 draws within 15 % of the target volume, else
  the closest.

Selections are named lists in ``lists.csv`` (``S1_v1_fixed``,
``S1_v5_fixed``, ...) next to the warehouse's lists, S3 orders templates
(``templates/S3_v1_fixed.csv``, ``templates/S3_v5_fixed.csv``), all in the
confidential canonical folder. The variants name ``*_v1_fixed``; the full
volume overrides the list or template with ``*_v5_fixed``. Prints aggregates
only.

    python -m scripts.ijpe.disruptions
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scenarios.ijpe.source import store_slot
from scripts.ijpe.order_generator import fit, load, zone_order

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "scenarios" / "ijpe" / "data" / "canonical"
WEEK = ROOT / "scenarios" / "ijpe_week" / "scenario.yaml"
AREA_SHARE = 0.20
TE_M3 = 0.445
VOLUMES = (1, 5)
ACCEPT = 0.15
STORE_SCENARIOS = {  # day, departure hour of the selected stores' slot, TE of the mail
    "S1": ("2026-04-13", 14, 120),
    "S2": ("2026-04-16", 14, 100),
    "S4": ("2026-04-16", 10, 200),
}
S3 = ("2026-04-15", 6)
SEED = 2101


def slot_hour(stores: pd.Series) -> pd.Series:
    """Departure hour of each store under the week's dispatch rule."""
    rule = yaml.safe_load(WEEK.read_text(encoding="utf-8"))["source"]["due_dates"]
    hours = list(rule["hours"])
    return stores.map(lambda store: hours[store_slot(store, len(hours))])


def fill(te: pd.Series, target: float, order) -> list:
    chosen, total = [], 0.0
    for store in order:
        if total >= target:
            break
        chosen.append(store)
        total += te[store]
    return chosen


def store_selections(orders: pd.DataFrame, lines: pd.DataFrame):
    volume = lines.groupby("order_id")["volume"].sum()
    orders = orders.assign(te=orders["order_id"].map(volume) / TE_M3)
    rows, report = [], []
    for name, (day, hour, mail) in STORE_SCENARIOS.items():
        today = orders[orders["day"] == day]
        te = today.groupby("customer")["te"].sum()
        pool = te[slot_hour(te.index.to_series()) == hour].sort_values(ascending=False)
        for v in VOLUMES:
            target = v * mail * AREA_SHARE
            stores = fill(pool, target, pool.index)
            ids = today.loc[today["customer"].isin(stores), "order_id"]
            rows += [(f"{name}_v{v:g}_fixed", int(i)) for i in ids]
            report.append({"scenario": name, "V": v, "target TE": target, "pool TE": pool.sum(),
                           "stores": len(stores), "orders": len(ids),
                           "deviation": pool[stores].sum() / target - 1})
    return rows, pd.DataFrame(report)


def s3_templates(orders: pd.DataFrame, lines: pd.DataFrame, rng) -> pd.DataFrame:
    cal = fit(orders, lines)
    day, mail = S3
    profiles = cal["day_profiles"][day]
    report = []
    for v in VOLUMES:
        target = v * mail * AREA_SHARE * TE_M3
        best, best_error = None, np.inf
        effect = rng.normal(0.0, np.sqrt(cal["store_share"]))
        for _ in range(2000):
            zone, n_lines = profiles[rng.integers(len(profiles))][0]
            rows = zone_order(cal, rng, zone, int(n_lines), effect)
            error = abs(sum(r[4] for r in rows) - target) / target
            if error < best_error:
                best, best_error = rows, error
            if error <= ACCEPT:
                break
        frame = pd.DataFrame(best, columns=["article", "quantity", "x", "y", "volume"])
        frame.to_csv(CANONICAL / "templates" / f"S3_v{v:g}_fixed.csv", index=False)
        report.append({"scenario": "S3", "V": v, "target TE": target / TE_M3, "orders": 1,
                       "lines": len(frame), "deviation": frame["volume"].sum() / target - 1})
    return pd.DataFrame(report)


def main() -> None:
    rng = np.random.default_rng(SEED)
    orders, lines = load()
    rows, report = store_selections(orders, lines)
    for stale in (CANONICAL / "templates").glob("S3_v*.csv"):
        stale.unlink()
    report = pd.concat([report, s3_templates(orders, lines, rng)], ignore_index=True)
    lists = pd.read_csv(CANONICAL / "lists.csv")
    lists = lists[~lists["list"].str.contains("_v")]  # keep the warehouse's lists, replace ours
    lists = pd.concat([lists, pd.DataFrame(rows, columns=["list", "order_id"])], ignore_index=True)
    lists.to_csv(CANONICAL / "lists.csv", index=False)
    print(report.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
