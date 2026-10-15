"""Compare the L2 validation runs with Henn's published averages.

Each cell gets the mean over its successful replications, a 95 % Student-t
confidence interval, the published value of ``reference.csv`` and whether
that value lies inside the interval. Times are in minutes. The working paper
defines turnover from an order's start (correction 3 of the register), the
journal from its arrival; both are reported.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

from scenarios.scenario_henn2012.run_l2 import OUTPUT
from scenarios.scenario_henn2012.run_l2_matrix import cell_name, validation_cells

REFERENCE = Path(__file__).with_name("reference.csv")
KPIS = {"completion": "completion_s", "turnover_arrival": "mean_turnover_arrival_s",
        "turnover_start": "mean_turnover_start_s"}


def interval(values: pd.Series) -> tuple[float, float, float]:
    mean = values.mean()
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / len(values) ** 0.5
    return mean, mean - half, mean + half


def summarize() -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE)
    rows = []
    for cell in validation_cells("conditioned"):
        target, storage, orders, capacity, routing, selection, _ = cell
        path = OUTPUT / f"{cell_name(cell)}__validation.csv"
        runs = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame()
        failed = int((runs["error"] != "").sum()) if "error" in runs else 0
        ok = runs[runs["error"] == ""] if "error" in runs else runs
        published = reference[
            (reference.target == target) & (reference.storage == storage)
            & (reference.n_orders == orders) & (reference.capacity == capacity)
            & (reference.selection == selection)].set_index("kpi").published_min
        row = {"target": target, "class": f"{storage}/{orders}/{capacity}",
               "selection": selection, "runs": len(ok), "failed": failed,
               "seconds_per_run": round(ok["seconds"].astype(float).mean(), 1)
               if len(ok) else float("nan")}
        for kpi, column in KPIS.items():
            mean, low, high = (interval(ok[column].astype(float) / 60.0) if len(ok)
                               else (float("nan"),) * 3)
            row[f"{kpi}_mean"], row[f"{kpi}_low"], row[f"{kpi}_high"] = mean, low, high
        for kpi in ("completion", "turnover"):
            row[f"published_{kpi}"] = published.get(kpi, float("nan"))
        row["completion_gap_pct"] = 100 * (row["completion_mean"] / row["published_completion"] - 1)
        row["completion_in_ci"] = row["completion_low"] <= row["published_completion"] <= row["completion_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    table = summarize()
    table.to_csv(OUTPUT / "l2_validation_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
