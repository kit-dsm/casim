"""The L2 validation table (docs/validation.md, L2).

Run the four sweeps, then this file with their output directory:

    python -m casim run scenarios/henn2012 +sweep=validation_journal_first
    python scenarios/henn2012/plots.py scenarios/henn2012/outputs/run

Per cell (target, class, rule) it gives the mean over the successful replications
and a 95 % Student-t interval of completion and both turnover definitions, the
published values of ``reference.csv``, and the replications that failed (their
directory holds ``error.txt`` instead of outputs).
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
_spec = spec_from_file_location("henn2012_metrics", HERE / "metrics.py")
metrics = module_from_spec(_spec)
_spec.loader.exec_module(metrics)
KPIS = ("completion_min", "turnover_arrival_min", "turnover_start_min")
CELL = ("decisions", "cosy_repo", "source.storage", "source.n_orders", "source.capacity")


def runs(base: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        overrides = dict(item.split("=", 1) for item in run_dir.name.split(","))
        row = {key: overrides[key] for key in CELL}
        row["replication"] = int(overrides["source.replication"])
        row["failed"] = (run_dir / "error.txt").exists()
        if not row["failed"]:
            row.update({kpi: getattr(metrics, kpi)(run_dir) for kpi in KPIS})
        rows.append(row)
    return pd.DataFrame(rows)


def interval(values: pd.Series) -> tuple[float, float, float]:
    mean = values.mean()
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / len(values) ** 0.5
    return mean, mean - half, mean + half


def table(results: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(HERE / "reference.csv", dtype=str)
    rows = []
    for cell, group in results.groupby(list(CELL)):
        ok = group[~group.failed]
        row = dict(zip(CELL, cell), runs=len(ok), failed=int(group.failed.sum()))
        for kpi in KPIS:
            row[f"{kpi}_mean"], row[f"{kpi}_low"], row[f"{kpi}_high"] = (
                interval(ok[kpi]) if len(ok) else (float("nan"),) * 3)
        published = reference[
            (reference["choices.decisions"] == cell[0]) & (reference["choices.cosy_repo"] == cell[1])
            & (reference["source.storage"] == cell[2]) & (reference["source.n_orders"] == cell[3])
            & (reference["source.capacity"] == cell[4])].set_index("kpi").published.astype(float)
        row["published_completion_min"] = published.get("completion_min", float("nan"))
        row["published_turnover_min"] = published.get("turnover_arrival_min", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    results = pd.concat([runs(Path(base)) for base in sys.argv[1:]], ignore_index=True)
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "l2_runs.csv", index=False)
    summary = table(results)
    summary.to_csv(out / "l2_validation_summary.csv", index=False)
    print(summary.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
