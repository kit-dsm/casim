"""Summarize the six-day IJPE pause sweep without exposing picker records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from casim.sources.grid_layout import GridLayout
from casim.sources.wms_export import day_lines
from scenarios.ijpe.source import ScenarioSource
from scripts.ijpe import SCENARIO

DAYS = ("2026-04-11", "2026-04-13", "2026-04-14", "2026-04-15",
        "2026-04-16", "2026-04-17")
THRESHOLDS = (120.0, 300.0, 600.0)


def audit(root: Path = SCENARIO / "outputs" / "run") -> pd.DataFrame:
    data = SCENARIO / "data"
    _, locations = GridLayout().build(data / "layout.csv")
    nodes = GridLayout.pick_nodes(locations)
    rows = []
    for day in DAYS:
        recorded, _ = day_lines(day, nodes, data / "cache")
        recorded = recorded.sort_values(["start", "row"])
        picker_ids = {picker: index for index, picker in enumerate(recorded["picker"].unique())}
        last_recorded = recorded.assign(picker_id=recorded.picker.map(picker_ids)).groupby(
            "picker_id")["end"].max()
        midnight = pd.Timestamp(day)
        for threshold in THRESHOLDS:
            source = ScenarioSource(data, day=day, shift_start_s=19800,
                                    pause_threshold_s=threshold, speed=2306,
                                    time_per_pick=19, tour_setup_time=215)
            pauses = source.replay._pauses()
            run = root / f"source.day={day},source.pause_threshold_s={threshold}"
            picks = pd.read_csv(run / "facts" / "picks.csv")
            last_model = picks.groupby("picker_id")["pick_end"].max()
            recorded_s = (last_recorded - midnight).dt.total_seconds()
            error_min = (last_model - recorded_s).dropna() / 60
            rows.append({"day": day, "threshold_min": threshold / 60,
                         "pauses": len(pauses),
                         "pause_hours": sum(event.duration for event in pauses) / 3600,
                         "picker_days": len(error_min),
                         "picker_day_bias_min": error_min.mean(),
                         "picker_day_mae_min": error_min.abs().mean(),
                         "picker_day_rmse_min": float(np.sqrt((error_min ** 2).mean())),
                         "squared_error_sum": float((error_min ** 2).sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = audit()
    for threshold, group in table.groupby("threshold_min"):
        print(f"{threshold:g} min: {int(group.pauses.sum())} pauses, "
              f"{group.pause_hours.sum():.1f} h, "
              f"{np.sqrt(group.squared_error_sum.sum() / group.picker_days.sum()):.1f} min "
              "pooled picker-day RMSE")
    print(table.drop(columns="squared_error_sum").to_string(
        index=False, float_format=lambda x: f"{x:.1f}"))
