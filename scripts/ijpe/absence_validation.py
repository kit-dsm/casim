"""Distribution-driven validation: the calibrated day with fitted absences.

Runs the calibrated day (``variants=calibrated``) with the recorded roster
and absences drawn from the model fitted on the other days
(``source.absence=stochastic``), for ``REPLICATIONS`` seeds per day, with
FIFO assignment (``fifo``, the base of the experiments) or each tour to its
recorded picker (``recorded``). Compares orders completed by 10:00 and
12:00, open orders and the non-picking time per picker-day with the WMS and
with the trace-driven run of the same assignment (``system_validation``).
Writes aggregates only to ``outputs/absence_validation``.

    python -m scripts.ijpe.absence_validation fifo|recorded [--run]

``--run`` (re)runs the replications first; they take about 20 s each.
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from casim.sources.grid_layout import GridLayout
from casim.sources.wms_export import day_lines
from scripts.ijpe import OUTPUTS, SCENARIO
from scripts.ijpe.system_validation import DAYS

REPLICATIONS = 30
WORKERS = 8
ASSIGNMENT = {"fifo": ([], "calibrated_fifo"),
              "recorded": (["cosy_repo=recorded_ushape"], "calibrated_recorded")}
HOURS = (10, 12)


def _run(assignment: str, day: str, seed: int) -> None:
    out = OUTPUTS / f"stochastic_{assignment}" / f"seed{seed:02d}" / day
    if (out / "service.json").exists():
        return
    subprocess.run([sys.executable, "-m", "casim", "run", str(SCENARIO), "variants=calibrated",
                    *ASSIGNMENT[assignment][0], "source.absence=stochastic",
                    f"source.absence_seed={seed}", f"source.day={day}",
                    "--output-dir", str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_all(assignment: str) -> None:
    jobs = [(assignment, day, seed) for seed in range(REPLICATIONS) for day in DAYS]
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda job: _run(*job), jobs))


def _measures(root) -> dict:
    orders = pd.read_csv(root / "facts" / "orders.csv")
    tours = pd.read_csv(root / "facts" / "tours.csv").dropna(subset=["end"])
    service = json.loads((root / "service.json").read_text())
    done = orders.completion.dropna()
    gaps = []
    for _, group in tours.groupby("picker_id"):
        group = group.sort_values("start")
        span = group.end.max() - group.setup_end.iloc[0]
        gaps.append((span - (group.end - group.start).sum()
                     + (group.setup_end - group.start).iloc[0]) / 3600)
    row = {f"by_{h}h": int((done <= h * 3600).sum()) for h in HOURS}
    row["open"] = int(service["orders"]["all"]["open_orders"])
    row["nonpicking_h"] = float(np.mean(gaps))
    return row


def _wms(day: str) -> dict:
    _, locations = GridLayout().build(SCENARIO / "data" / "layout.csv")
    lines, _ = day_lines(day, GridLayout.pick_nodes(locations), SCENARIO / "data" / "cache")
    ends = (lines.groupby("order").end.max() - pd.Timestamp(day)).dt.total_seconds()
    return {f"by_{h}h": int((ends <= h * 3600).sum()) for h in HOURS}


def main() -> None:
    assignment = sys.argv[1]
    if assignment not in ASSIGNMENT:
        raise SystemExit(f"assignment must be one of {sorted(ASSIGNMENT)}")
    if "--run" in sys.argv:
        run_all(assignment)
    rows, pooled = [], []
    for day in DAYS:
        wms = _wms(day)
        trace = _measures(OUTPUTS / ASSIGNMENT[assignment][1] / day)
        reps = pd.DataFrame([_measures(OUTPUTS / f"stochastic_{assignment}" / f"seed{s:02d}" / day)
                             for s in range(REPLICATIONS)])
        for measure in (*(f"by_{h}h" for h in HOURS), "open", "nonpicking_h"):
            low, high = reps[measure].quantile([0.05, 0.95])
            row = {"day": day, "measure": measure, "wms": wms.get(measure),
                   "trace": trace[measure], "mean": reps[measure].mean(),
                   "p05": low, "p95": high}
            if row["wms"] is not None:
                row["wms_in_band"] = bool(low <= row["wms"] <= high)
            rows.append(row)
        pooled.append(reps.assign(day=day))
    result = pd.DataFrame(rows)
    directory = OUTPUTS / "absence_validation"
    directory.mkdir(parents=True, exist_ok=True)
    result.to_csv(directory / f"days_{assignment}.csv", index=False)
    pd.set_option("display.width", 200)
    print(result.round(2).to_string(index=False))
    reps = pd.concat(pooled)
    totals = reps.groupby(reps.groupby("day").cumcount()).sum(numeric_only=True)
    print("\nSix days pooled (sum over days per replication):")
    for measure in ("by_10h", "by_12h", "open"):
        wms = int(result.loc[result.measure == measure, "wms"].sum()) if measure != "open" else 0
        trace = result.loc[result.measure == measure, "trace"].sum()
        low, high = totals[measure].quantile([0.05, 0.95])
        print(f"  {measure}: WMS {wms}, trace-driven {trace:.0f}, fitted mean "
              f"{totals[measure].mean():.0f} (90 % band {low:.0f} to {high:.0f})")


if __name__ == "__main__":
    main()
