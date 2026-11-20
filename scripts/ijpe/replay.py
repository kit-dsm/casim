"""R1.4 of docs/ijpe/plan.md: CASIM's replay against the recorded day.

Reads the facts tables CASIM wrote for each run of the sweep
``sweep=days_thresholds`` (``outputs/run/<overrides>/``) and the export, and
compares at three levels; it computes no model time itself.

- Line: a recorded line's interval contains the walk to it (R0: 83 % of lines
  start when the previous one ends), so a model line runs from its walk start
  to its pick end. Model picks are matched to recorded lines of the same
  order, article and pick node in time order.
- Order: span from the first line's start to the last line's end, as in
  Figure 8 of the paper, overall and per order-size class.
- Picker-day: end of the last line, busy time (union of line intervals) and
  the completion profile of lines and orders per hour.

    python -m scripts.ijpe.replay [run directory]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from casim.sources.recorded import RecordedRow, recorded_tours

from scenarios.ijpe.source import day_lines
from scripts.ijpe import OUTPUTS as SCENARIO_OUTPUTS
from scripts.ijpe.audit import SIZE_BINS, SIZE_LABELS

RUNS = SCENARIO_OUTPUTS / "run"
OUTPUTS = SCENARIO_OUTPUTS / "replay"
CALIBRATION_DAYS = ("2026-04-13", "2026-04-14", "2026-04-15")  # Section 9 default


def _stats(error: pd.Series) -> dict:
    error = error.dropna()
    return {"n": len(error), "bias": error.mean(), "mae": error.abs().mean(),
            "rmse": float(np.sqrt((error ** 2).mean())), "q05": error.quantile(0.05),
            "median": error.median(), "q95": error.quantile(0.95)}


def _union_s(starts, ends) -> float:
    total, current = 0.0, None
    for start, end in sorted(zip(starts, ends)):
        if current is None or start > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    return total + (current[1] - current[0]) if current else 0.0


def runs(root: Path = RUNS) -> list[tuple[str, str, Path]]:
    """(day, setting, directory) of every completed run of a sweep.

    The setting is the pause threshold (sweep ``days_thresholds``) or the
    variant (sweep ``variants``).
    """
    found = []
    for directory in sorted(root.iterdir()):
        if not (directory / "facts" / "picks.csv").exists():
            continue
        parts = dict(part.split("=", 1) for part in directory.name.split(","))
        setting = parts.get("source.pause_threshold_s", parts.get("variants", "default"))
        found.append((parts["source.day"], setting, directory))
    return found


def recorded_day(day: str) -> pd.DataFrame:
    lines, _ = day_lines(day)
    midnight = pd.Timestamp(day)
    lines["start_s"] = (lines["start"] - midnight).dt.total_seconds()
    lines["end_s"] = (lines["end"] - midnight).dt.total_seconds()
    lines["node_x"] = [node[0] for node in lines["node"]]
    lines["node_y"] = [node[1] for node in lines["node"]]
    lines["order"] = lines["order"].astype(int)
    lines["article"] = lines["article"].astype(int)
    return lines


def matched_lines(recorded: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    """Recorded lines with their model pick, matched by order, article and rank.

    ``moved`` marks lines the item assignment picks at another node than the
    recorded one (articles stored at two locations).
    """
    key = ["order", "article"]
    recorded = recorded.sort_values("start_s").copy()
    recorded["rank"] = recorded.groupby(key).cumcount()
    model = picks.rename(columns={"order_id": "order", "article_id": "article",
                                  "node_x": "model_x", "node_y": "model_y"}).sort_values(
        "pick_start").copy()
    model["rank"] = model.groupby(key).cumcount()
    merged = recorded.merge(model, on=[*key, "rank"], how="left", validate="one_to_one")
    merged["moved"] = (merged["model_x"] != merged["node_x"]) | (merged["model_y"] != merged["node_y"])
    if merged["pick_end"].isna().any():
        raise ValueError(f"{int(merged['pick_end'].isna().sum())} recorded lines have no model pick")
    merged["recorded_s"] = merged["end_s"] - merged["start_s"]
    merged["model_s"] = merged["pick_end"] - merged["walk_start"]
    merged["error_s"] = merged["model_s"] - merged["recorded_s"]
    return merged


def compare_run(day: str, threshold: str, directory: Path
                ) -> dict[str, pd.DataFrame]:
    picks = pd.read_csv(directory / "facts" / "picks.csv")
    lines = matched_lines(recorded_day(day), picks)
    lines["size_class"] = pd.cut(lines.groupby("order")["row"].transform("size"),
                                 SIZE_BINS, labels=SIZE_LABELS)
    orders = lines.groupby("order").agg(
        size_class=("size_class", "first"), recorded_first=("start_s", "min"),
        recorded_last=("end_s", "max"), model_first=("walk_start", "min"),
        model_last=("pick_end", "max"))
    orders["recorded_span"] = orders["recorded_last"] - orders["recorded_first"]
    orders["model_span"] = orders["model_last"] - orders["model_first"]
    orders["error_s"] = orders["model_span"] - orders["recorded_span"]
    pickers = lines.groupby("picker").apply(lambda g: pd.Series({
        "recorded_end": g["end_s"].max(), "model_end": g["pick_end"].max(),
        "recorded_busy": _union_s(g["start_s"], g["end_s"]),
        "model_busy": _union_s(g["walk_start"], g["pick_end"]),
        "recorded_present": g["end_s"].max() - g["start_s"].min()}), include_groups=False)
    hours = range(5, 19)
    profile = pd.DataFrame({
        "recorded_lines": pd.cut(lines["end_s"] / 3600, hours).value_counts().sort_index(),
        "model_lines": pd.cut(lines["pick_end"] / 3600, hours).value_counts().sort_index(),
        "recorded_orders": pd.cut(orders["recorded_last"] / 3600, hours).value_counts().sort_index(),
        "model_orders": pd.cut(orders["model_last"] / 3600, hours).value_counts().sort_index(),
    })
    profile.index = [f"{interval.left:02d}-{interval.right:02d}" for interval in profile.index]
    tag = {"day": day, "threshold": threshold}
    rows = {
        "line": pd.DataFrame([{**tag, "level": "line", "measure": "duration [s]",
                               "picked_at_another_node": int(lines["moved"].sum()),
                               **_stats(lines["error_s"])}]),
        "order": pd.DataFrame([{**tag, "level": "order", "size_class": "all",
                                **_stats(orders["error_s"] / 60)}] +
                              [{**tag, "level": "order", "size_class": label,
                                **_stats(group["error_s"] / 60)}
                               for label, group in orders.groupby("size_class", observed=True)]),
        "picker": pd.DataFrame([
            {**tag, "measure": "end of last line [min]",
             **_stats((pickers["model_end"] - pickers["recorded_end"]) / 60)},
            {**tag, "measure": "busy time [min]",
             **_stats((pickers["model_busy"] - pickers["recorded_busy"]) / 60)},
            {**tag, "measure": "busy share of presence, recorded",
             **_stats(pickers["recorded_busy"] / pickers["recorded_present"])},
            {**tag, "measure": "busy share of presence, model",
             **_stats(pickers["model_busy"] / pickers["recorded_present"])}]),
        "profile": profile.assign(**tag),
        "tour": compare_tours(lines, directory).assign(**tag),
    }
    return rows


def compare_tours(lines: pd.DataFrame, directory: Path) -> pd.DataFrame:
    """Recorded tours against CASIM's, and the recorded idle time inside tours.

    A recorded tour's span (first line start to last line end) is the union of
    its line intervals plus its idle time inside the tour, which CASIM cannot
    model (it does not interrupt a tour). A model tour lasts setup + walk +
    picks. Tours are matched by their set of orders.
    """
    rows = [RecordedRow(picker=int(line.picker), order=int(line.order),
                        article=int(line.article), quantity=int(line.quantity),
                        node=(line.node_x, line.node_y), start_s=line.start_s, end_s=line.end_s)
            for line in lines.itertuples()]
    recorded = []
    for _, tour in recorded_tours(rows):
        span = max(r.end_s for r in tour) - min(r.start_s for r in tour)
        busy = _union_s([r.start_s for r in tour], [r.end_s for r in tour])
        recorded.append({"key": " ".join(map(str, sorted({r.order for r in tour}))),
                         "recorded_span": span, "inside_idle": span - busy,
                         "lines": len(tour)})
    recorded = pd.DataFrame(recorded).set_index("key")
    model = pd.read_csv(directory / "facts" / "tours.csv", dtype={"order_ids": str})
    model["key"] = model["order_ids"].str.split().map(lambda ids: " ".join(sorted(ids, key=int)))
    model = model.set_index("key")
    tours = recorded.join((model["end"] - model["start"]).rename("model_duration"), how="inner")
    tours["error"] = tours["model_duration"] - tours["recorded_span"]
    return pd.DataFrame([{
        "tours": len(tours), "unmatched": len(recorded) - len(tours),
        "recorded_span_h": tours["recorded_span"].sum() / 3600,
        "inside_idle_h": tours["inside_idle"].sum() / 3600,
        "inside_idle_share_of_span": tours["inside_idle"].sum() / tours["recorded_span"].sum(),
        "model_duration_h": tours["model_duration"].sum() / 3600,
        "error_h": tours["error"].sum() / 3600,
        "error_without_idle_h": (tours["error"] + tours["inside_idle"]).sum() / 3600,
        "tours_with_idle_over_5_min": int((tours["inside_idle"] > 300).sum()),
    }])


def run_all(root: Path = RUNS) -> dict[str, pd.DataFrame]:
    collected: dict[str, list] = {}
    for day, threshold, directory in runs(root):
        for name, table in compare_run(day, threshold, directory).items():
            collected.setdefault(name, []).append(table)
    tables = {name: pd.concat(parts) for name, parts in collected.items()}
    tables["order"]["days"] = np.where(tables["order"]["day"].isin(CALIBRATION_DAYS),
                                       "calibration", "validation")
    return tables


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS
    tables = run_all(root)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    for name, table in tables.items():
        table.to_csv(OUTPUTS / f"{name}.csv", index=name == "profile")
        if name != "profile":
            print(f"\n## {name}\n{table.to_string(index=False)}")
    print(f"\nwritten to {OUTPUTS}")


if __name__ == "__main__":
    main()
