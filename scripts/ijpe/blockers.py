"""Aggregate decomposition of the anchored replay's line and order error.

Reads CASIM facts and the WMS export. Keeps order and line calculations in
memory and writes only aggregate statistics to ignored outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from casim.sources.grid_layout import GridLayout
from scripts.ijpe import OUTPUTS, SCENARIO
from scripts.ijpe.replay import RUNS, _union_s, matched_lines, recorded_day, runs


def _orders(lines: pd.DataFrame) -> pd.DataFrame:
    records = []
    tour_sizes = lines.groupby(["day", "tour_id"]).order.nunique()
    for _, group in lines.groupby(["day", "order"]):
        record_span = group["end_s"].max() - group["start_s"].min()
        model_span = group["pick_end"].max() - group["walk_start"].min()
        record_busy = _union_s(group["start_s"], group["end_s"])
        model_busy = _union_s(group["walk_start"], group["pick_end"])
        records.append({"day": group.day.iloc[0], "lines": len(group),
                        "record_idle_s": record_span - record_busy,
                        "tour_orders": int(tour_sizes.loc[(group.day.iloc[0],
                                                            group.tour_id.iloc[0])]),
                        "max_recorded_line_s": float(group.recorded_s.max()),
                        "record_overlap_s": group["recorded_s"].sum() - record_busy,
                        "has_extreme_line": bool((~group["recorded_s"].between(4, 59)).any()),
                        "has_short_line": bool((group["recorded_s"] < 4).any()),
                        "has_long_line": bool((group["recorded_s"] > 59).any()),
                        "model_idle_s": model_span - model_busy,
                        "busy_error_s": model_busy - record_busy,
                        "error_s": model_span - record_span})
    return pd.DataFrame(records)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS
    _, locations = GridLayout().build(SCENARIO / "data" / "layout.csv")
    nodes = GridLayout.pick_nodes(locations)
    matched = []
    for day, setting, directory in runs(root):
        if setting != "null":
            continue
        picks = pd.read_csv(directory / "facts" / "picks.csv")
        lines = matched_lines(recorded_day(day, nodes), picks)
        lines["day"] = day
        lines = lines.sort_values(["tour_id", "pick_start"])
        lines["first_line"] = lines.groupby("tour_id").cumcount() == 0
        matched.append(lines)
    lines = pd.concat(matched, ignore_index=True)
    orders = _orders(lines)
    line_summary = []
    for name, group in (("all", lines), ("duration <= 59s", lines[lines.recorded_s <= 59]),
                        ("duration <= 300s", lines[lines.recorded_s <= 300]),
                        ("first line", lines[lines.first_line]),
                        ("later line", lines[~lines.first_line])):
        error = group.error_s
        line_summary.append({"sample": name, "n": len(group),
                             "record_median_s": group.recorded_s.median(),
                             "model_median_s": group.model_s.median(),
                             "bias_s": error.mean(), "mae_s": error.abs().mean(),
                             "rmse_s": np.sqrt((error ** 2).mean()),
                             "squared_error_share": (error ** 2).sum() /
                             (lines.error_s ** 2).sum()})
    orders["idle_group"] = pd.cut(orders.record_idle_s, [-1, 0, 300, np.inf],
                                   labels=["no recorded idle", "idle <= 5min", "idle > 5min"])
    clean = ((orders.record_idle_s <= 300) & (orders.record_overlap_s == 0) &
             ~orders.has_extreme_line)
    order_summary = []
    screens = []
    for sample, base in (("all", orders),
                         ("train_MonWed", orders[orders.day.isin(
                             ["2026-04-13", "2026-04-14", "2026-04-15"])]),
                         ("forward_ThuFri", orders[orders.day.isin(
                             ["2026-04-16", "2026-04-17"])]),
                         ("earlier_Sat", orders[orders.day == "2026-04-11"])):
        for max_line_s in (120, 300):
            max_idle_s = 300
            mask = ((base.tour_orders == 1) &
                    (base.max_recorded_line_s <= max_line_s) &
                    (base.record_idle_s <= max_idle_s))
            group = base[mask]
            screens.append({"sample": sample, "max_line_s": max_line_s,
                            "max_order_idle_s": max_idle_s,
                            "orders": len(group), "coverage": len(group) / len(base),
                            "rmse_min": float(np.sqrt(np.mean((group.error_s / 60) ** 2))),
                            "bias_min": float(group.error_s.mean() / 60),
                            "squared_error_share": float(np.sum(group.error_s ** 2) /
                                                         np.sum(base.error_s ** 2))})
    crossed = []
    for (multi, idle, long_line), group in orders.groupby(
            [orders.tour_orders > 1, orders.record_idle_s > 300, orders.has_long_line]):
        crossed.append({"multi_order_tour": bool(multi), "recorded_idle_over_5min": bool(idle),
                        "contains_line_over_59s": bool(long_line),
                        "orders": len(group), "rmse_min": float(np.sqrt(
                            np.mean((group.error_s / 60) ** 2))),
                        "bias_min": float(group.error_s.mean() / 60),
                        "mean_lines": float(group.lines.mean()),
                        "squared_error_share": float(np.sum(group.error_s ** 2) /
                                                     np.sum(orders.error_s ** 2))})
    for name, group in (("all", orders), ("ordinary ordered logs", orders[clean]),
                        ("line over 300s", orders[orders.max_recorded_line_s > 300]),
                        ("single tour, line over 300s",
                         orders[(orders.tour_orders == 1) &
                                (orders.max_recorded_line_s > 300)]),
                        ("no long line or idle", orders[(~orders.has_long_line) &
                                                        (orders.record_idle_s <= 300)]),
                        ("with short line", orders[orders.has_short_line]),
                        ("with long line", orders[orders.has_long_line]),
                        ("with overlapping lines", orders[orders.record_overlap_s > 0]),
                        ("with extreme line", orders[orders.has_extreme_line]),
                        *orders.groupby("idle_group", observed=True)):
        error = group.error_s / 60
        busy_error = group.busy_error_s / 60
        idle_error = (group.model_idle_s - group.record_idle_s) / 60
        order_summary.append({"sample": name, "n": len(group),
                              "record_idle_mean_min": group.record_idle_s.mean() / 60,
                              "model_idle_mean_min": group.model_idle_s.mean() / 60,
                              "busy_bias_min": group.busy_error_s.mean() / 60,
                              "busy_rmse_min": np.sqrt((busy_error ** 2).mean()),
                              "idle_rmse_min": np.sqrt((idle_error ** 2).mean()),
                              "busy_idle_correlation": busy_error.corr(idle_error),
                              "bias_min": error.mean(), "mae_min": error.abs().mean(),
                              "rmse_min": np.sqrt((error ** 2).mean()),
                              "squared_error_share": (group.error_s ** 2).sum() /
                              (orders.error_s ** 2).sum()})
    target = OUTPUTS / "blockers" / root.name
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(line_summary).to_csv(target / "lines.csv", index=False)
    pd.DataFrame(order_summary).to_csv(target / "orders.csv", index=False)
    pd.DataFrame(crossed).to_csv(target / "crossed_cohorts.csv", index=False)
    pd.DataFrame(screens).to_csv(target / "quality_screens.csv", index=False)
    print("Lines:\n", pd.DataFrame(line_summary).round(3).to_string(index=False))
    print("\nOrders:\n", pd.DataFrame(order_summary).round(3).to_string(index=False))
    print("\nCrossed cohorts:\n", pd.DataFrame(crossed).round(3).to_string(index=False))
    print("\nQuality sensitivity:\n", pd.DataFrame(screens).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
