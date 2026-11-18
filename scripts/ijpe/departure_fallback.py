"""Audit a fixed, synthetic store dispatch plan against the WMS export.

The assignment uses only the store identifier and a schedule fixed before
looking at pick completion times. It is a scenario assumption, not a recovered
truck plan. Only day-level aggregates are printed.
"""

from __future__ import annotations

import pandas as pd

from casim.sources.grid_layout import GridLayout
from casim.sources.wms_export import day_lines
from scenarios.ijpe.source import STAGING_MINUTES, fixed_departure_hour
from scripts.ijpe import CACHE, SCENARIO

DAYS = ("2026-04-11", "2026-04-13", "2026-04-14", "2026-04-15",
        "2026-04-16", "2026-04-17")
def audit() -> pd.DataFrame:
    _, locations = GridLayout().build(SCENARIO / "data" / "layout.csv")
    nodes = GridLayout.pick_nodes(locations)
    rows = []
    for day in DAYS:
        lines, _ = day_lines(day, nodes, CACHE)
        orders = lines.groupby("order").agg(customer=("customer", "first"),
                                            completion=("end", "max"))
        if (lines.groupby("order")["customer"].nunique() != 1).any():
            raise ValueError("an order belongs to several stores")
        orders["departure_hour"] = orders["customer"].map(fixed_departure_hour)
        orders["due"] = (pd.Timestamp(day) +
                         pd.to_timedelta(orders["departure_hour"], unit="h") -
                         pd.Timedelta(minutes=STAGING_MINUTES))
        late_min = ((orders["completion"] - orders["due"]).dt.total_seconds()
                    .div(60).clip(lower=0))
        rows.append({"day": day, "stores": int(orders["customer"].nunique()),
                     "orders": len(orders), "late_orders": int((late_min > 0).sum()),
                     "late_rate": float((late_min > 0).mean()),
                     "mean_tardiness_min": float(late_min.mean()),
                     "max_tardiness_min": float(late_min.max())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(audit().to_string(index=False, float_format=lambda x: f"{x:.3f}"))
