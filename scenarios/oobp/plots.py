"""The L3 table (docs/literature_benchmarks.md, "B.5 full stage (L3)") from sweep outputs.

Run the sweeps first:

    python -m casim run scenarios/oobp +sweep=l3_henn
    python -m casim run scenarios/oobp variants=albareda +sweep=l3_albareda --output-dir scenarios/oobp/outputs/albareda

then ``python scenarios/oobp/plots.py scenarios/oobp/outputs/run scenarios/oobp/outputs/albareda``.
Each run's ``comparison.csv`` holds CASIM's value and the published one per objective; a
result below a best-known value marks the run ``protocol not reproduced``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REL_TOL = 1e-9


def runs(base: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(path for path in base.iterdir() if (path / "comparison.csv").exists()):
        overrides = dict(item.split("=", 1) for item in run_dir.name.split(","))
        comparison = pd.read_csv(run_dir / "comparison.csv")
        best = comparison[comparison.source.str.endswith("best known")].set_index("kpi")
        filepath = best["source.filepath"].iloc[0]
        row = {"filepath": filepath, "depot": overrides.get("source.depot", "layout file"),
               "stratum": str(Path(filepath).parent).replace("albareda/", "").replace("\\", "/")}
        for kpi in ("max_completion_time", "max_turnover_time"):
            row[kpi] = best.loc[kpi, "casim"]
            row[f"best_known_{kpi}"] = best.loc[kpi, "published"]
            row[f"below_{kpi}"] = row[kpi] < row[f"best_known_{kpi}"] * (1 - REL_TOL)
        rows.append(row)
    return pd.DataFrame(rows)


def table(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["stratum", "depot"])
    summary = grouped.agg(
        n=("filepath", "size"),
        completion=("max_completion_time", "mean"),
        best_known_completion=("best_known_max_completion_time", "mean"),
        turnover=("max_turnover_time", "mean"),
        best_known_turnover=("best_known_max_turnover_time", "mean"),
        below_completion=("below_max_completion_time", "sum"),
        below_turnover=("below_max_turnover_time", "sum"),
    ).reset_index()
    below = summary.below_completion + summary.below_turnover
    summary["status"] = ["protocol not reproduced" if count else "comparator only" for count in below]
    return summary


def main() -> None:
    results = pd.concat([runs(Path(base)) for base in sys.argv[1:]], ignore_index=True)
    summary = table(results)
    out = Path(__file__).parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "l3_runs.csv", index=False)
    summary.to_csv(out / "l3_summary.csv", index=False)
    print(summary.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
