"""Distribution-driven validation: the calibrated day with fitted non-picking time.

Runs the calibrated day (``variants=calibrated``) with the recorded roster and
breaks, tour setup and line time fitted on the other days
(``source.non_picking=fitted``, fitted by ``scripts/ijpe/fit_non_picking.py``), for
``REPLICATIONS`` seeds per day, with EDD assignment as a pull (``edd``, the
base of the experiments) or each tour to its recorded picker (``recorded``).
Compares per day orders completed by 10:00 and 12:00, open orders, the
time between tours per picker-day (not comparable across models: the
fitted model puts setup and stops inside tours) and the tour durations with the WMS, with the
run of the same assignment and recorded pauses (``system_validation``) and,
if its replications are present in ``outputs/allowance_<assignment>``, with
the previous model of breaks plus allowance per working hour. Writes
aggregates only to ``outputs/non_picking_validation``.

    python -m scripts.ijpe.non_picking_validation edd|recorded [--run]

``--run`` (re)runs the missing replications first; one takes about 15 s.
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
ASSIGNMENT = {"edd": ([], "calibrated_edd"),
              "recorded": (["decisions=recorded", "cosy_repo=recorded_ushape"],
                           "calibrated_recorded")}
HOURS = (10, 12)
MEASURES = (*(f"by_{h}h" for h in HOURS), "open", "between_tours_h")
QUANTILES = (0.1, 0.5, 0.9, 0.99)
MODELS = {"fitted": "non_picking", "previous": "allowance"}   # label -> output prefix


def _run(assignment: str, day: str, seed: int) -> None:
    out = OUTPUTS / f"non_picking_{assignment}" / f"seed{seed:02d}" / day
    if (out / "service.json").exists():
        return
    subprocess.run([sys.executable, "-m", "casim", "run", str(SCENARIO), "variants=calibrated",
                    *ASSIGNMENT[assignment][0], "source.non_picking=fitted",
                    f"source.seed={seed}", f"source.day={day}", "--output-dir", str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_all(assignment: str) -> None:
    jobs = [(assignment, day, seed) for seed in range(REPLICATIONS) for day in DAYS]
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda job: _run(*job), jobs))


def _measures(root) -> tuple[dict, np.ndarray]:
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
    row["between_tours_h"] = float(np.mean(gaps))
    # A tour's duration: from the end of its setup (the first walk) to its end,
    # as the WMS tour runs from its first line to its last.
    return row, (tours.end - tours.setup_end).to_numpy() / 60


def _wms(day: str) -> tuple[dict, np.ndarray]:
    _, locations = GridLayout().build(SCENARIO / "data" / "layout.csv")
    lines, _ = day_lines(day, GridLayout.pick_nodes(locations), SCENARIO / "data" / "cache")
    ends = (lines.groupby("order").end.max() - pd.Timestamp(day)).dt.total_seconds()
    spans = lines.groupby("order").agg(start=("start", "min"), end=("end", "max"))
    durations = (spans.end - spans.start).dt.total_seconds().to_numpy() / 60
    return {f"by_{h}h": int((ends <= h * 3600).sum()) for h in HOURS}, durations


def _replications(prefix: str, assignment: str, day: str):
    roots = [OUTPUTS / f"{prefix}_{assignment}" / f"seed{s:02d}" / day for s in range(REPLICATIONS)]
    if not all((root / "service.json").exists() for root in roots):
        return None, None
    results = [_measures(root) for root in roots]
    return pd.DataFrame([r for r, _ in results]), np.concatenate([d for _, d in results])


def _quantiles(durations: np.ndarray) -> dict:
    return {"mean": float(np.mean(durations)),
            **{f"p{int(q * 100):02d}": float(np.quantile(durations, q)) for q in QUANTILES}}


def main() -> None:
    assignment = sys.argv[1]
    if assignment not in ASSIGNMENT:
        raise SystemExit(f"assignment must be one of {sorted(ASSIGNMENT)}")
    if "--run" in sys.argv:
        run_all(assignment)
    rows, pooled, tour_rows = [], {label: [] for label in MODELS}, []
    durations = {"wms": [], "trace": [], **{label: [] for label in MODELS}}
    for day in DAYS:
        wms, wms_durations = _wms(day)
        trace, trace_durations = _measures(OUTPUTS / ASSIGNMENT[assignment][1] / day)
        durations["wms"].append(wms_durations)
        durations["trace"].append(trace_durations)
        tour_rows.append({"day": day, "model": "wms", **_quantiles(wms_durations)})
        tour_rows.append({"day": day, "model": "trace", **_quantiles(trace_durations)})
        reps = {}
        for label, prefix in MODELS.items():
            reps[label], model_durations = _replications(prefix, assignment, day)
            if reps[label] is None:
                continue
            pooled[label].append(reps[label].assign(day=day))
            durations[label].append(model_durations)
            tour_rows.append({"day": day, "model": label, **_quantiles(model_durations)})
        for measure in MEASURES:
            row = {"day": day, "measure": measure, "wms": wms.get(measure),
                   "trace": trace[measure]}
            for label, frame in reps.items():
                if frame is None:
                    continue
                low, high = frame[measure].quantile([0.05, 0.95])
                row.update({f"{label}_mean": frame[measure].mean(),
                            f"{label}_p05": low, f"{label}_p95": high})
                if row["wms"] is not None:
                    row[f"{label}_in_band"] = bool(low <= row["wms"] <= high)
            rows.append(row)
    result, tours = pd.DataFrame(rows), pd.DataFrame(tour_rows)
    directory = OUTPUTS / "non_picking_validation"
    directory.mkdir(parents=True, exist_ok=True)
    result.to_csv(directory / f"days_{assignment}.csv", index=False)
    tours.to_csv(directory / f"tour_minutes_{assignment}.csv", index=False)
    pd.set_option("display.width", 220)
    print(result.round(1).to_string(index=False))
    print("\nTour duration, minutes (first walk to end; WMS: first line to last), per day:")
    print(tours.round(1).to_string(index=False))
    print("\nSix days pooled:")
    for label, values in durations.items():
        if values:
            quantiles = _quantiles(np.concatenate(values))
            print(f"  tours {label}: " + ", ".join(f"{k} {v:.1f}" for k, v in quantiles.items()))
    for label, frames in pooled.items():
        if not frames:
            continue
        reps = pd.concat(frames)
        totals = reps.groupby(reps.groupby("day").cumcount()).sum(numeric_only=True)
        for measure in ("by_10h", "by_12h", "open"):
            wms = int(result.loc[result.measure == measure, "wms"].sum()) if measure != "open" else 0
            trace = result.loc[result.measure == measure, "trace"].sum()
            low, high = totals[measure].quantile([0.05, 0.95])
            inside = int(result.loc[result.measure == measure, f"{label}_in_band"].sum()) \
                if measure != "open" else None
            print(f"  {label} {measure}: WMS {wms}, recorded pauses {trace:.0f}, mean "
                  f"{totals[measure].mean():.0f} (90 % band {low:.0f} to {high:.0f})"
                  + ("" if inside is None else f"; WMS inside the band on {inside} of 6 days"))


if __name__ == "__main__":
    main()
