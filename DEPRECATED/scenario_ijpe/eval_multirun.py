from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_summary_files(results_root: str | Path) -> pd.DataFrame:
    results_root = Path(results_root)
    rows = []

    for summary_path in sorted(results_root.rglob("kpis/summary.json")):
        with open(summary_path) as f:
            row = json.load(f)

        rel = summary_path.relative_to(results_root)
        parts = rel.parts

        # paper_results/<cosy_repo>/<objective>/<simulation>/kpis/summary.json
        row["cosy_repo"] = parts[0]
        row["objective"] = parts[1]
        row["scenario"] = parts[2]
        row["summary_path"] = str(summary_path)

        rows.append(row)

    return pd.DataFrame(rows)


def add_baseline_deltas(df: pd.DataFrame, baseline_name: str = "baseline") -> pd.DataFrame:
    delta_cols = [
        "late_split_orders",
        "total_tardiness_h",
        "utilization",
        "idle_h",
        "max_dock_fill_pct",
        "truck_wait_events",
        "max_truck_wait_min",
        "orders_per_labor_h",
    ]

    out = []

    for _, g in df.groupby(["cosy_repo", "objective"]):
        baseline = g.loc[g["scenario"] == baseline_name].iloc[0]
        g = g.copy()

        for col in delta_cols:
            g[f"delta_{col}"] = g[col] - baseline[col]

        out.append(g)

    return pd.concat(out, ignore_index=True)


def make_paper_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "cosy_repo",
        "objective",
        "scenario",

        "tours_completed",
        "split_orders_completed",

        "on_time_rate",
        "late_split_orders",
        "total_tardiness_h",
        "max_tardiness_min",

        "affected_split_orders",
        "affected_tours",

        "utilization",
        "idle_h",
        "orders_per_labor_h",

        "max_dock_fill_pct",
        "truck_wait_events",
        "max_truck_wait_min",

        "delta_late_split_orders",
        "delta_total_tardiness_h",
        "delta_utilization",
        "delta_idle_h",
        "delta_max_dock_fill_pct",
        "delta_truck_wait_events",
        "delta_max_truck_wait_min",
        "delta_orders_per_labor_h",
    ]

    table = df[cols].copy()

    table["on_time_rate"] = 100 * table["on_time_rate"]
    table["utilization"] = 100 * table["utilization"]
    table["delta_utilization"] = 100 * table["delta_utilization"]

    number_cols = table.select_dtypes(include="number").columns
    table[number_cols] = table[number_cols].round(3)

    return table



def evaluate_multirun(
    results_root: str | Path = "../outputs/multirun/paper_results",
    baseline_name: str = "baseline",
) -> pd.DataFrame:
    results_root = Path(results_root)

    df = load_summary_files(results_root)
    df = add_baseline_deltas(df, baseline_name=baseline_name)
    table = make_paper_table(df)

    df.to_csv(results_root / "scenario_comparison_full.csv", index=False)
    table.to_csv(results_root / "scenario_comparison_paper.csv", index=False)

    print(table)

    return table


if __name__ == "__main__":
    evaluate_multirun()