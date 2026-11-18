"""Compare the calibrated IJPE day with the WMS; never export private records.

The calibrated day is ``variants=calibrated`` (``docs/ijpe/validation.md``).
Three runs per day:

- ``calibrated_recorded``: each tour to its recorded picker (the validation);
- ``calibrated_fifo``: CASIM's FIFO assignment (the base of the policy
  comparisons);
- ``calibrated_recorded_arrival``: as the first, but no picker is sent home,
  so a late picker shows as a late finish instead of open work (the
  uncensored picker-level check).

Run them first, for every day::

    python -m casim run scenarios/ijpe variants=calibrated cosy_repo=recorded_ushape
        source.day=<day> --output-dir scenarios/ijpe/outputs/calibrated_recorded/<day>
    python -m casim run scenarios/ijpe variants=calibrated
        source.day=<day> --output-dir scenarios/ijpe/outputs/calibrated_fifo/<day>
    python -m casim run scenarios/ijpe variants=calibrated cosy_repo=recorded_ushape
        source.presence=arrival source.day=<day>
        --output-dir scenarios/ijpe/outputs/calibrated_recorded_arrival/<day>

With recorded presence, orders can stay open at the end of the day; picker
finish times are then censored (a late picker leaves with work open).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from casim.sources.grid_layout import GridLayout
from casim.sources.wms_export import day_lines
from scripts.ijpe import OUTPUTS, SCENARIO
from scripts.ijpe.replay import _union_s

DAYS = ("2026-04-11", "2026-04-13", "2026-04-14", "2026-04-15",
        "2026-04-16", "2026-04-17")
RUNS = ("calibrated_recorded", "calibrated_fifo", "calibrated_recorded_arrival")
HOURS = (8, 10, 12, 14, 16, 18)


def audit() -> pd.DataFrame:
    _, locations = GridLayout().build(SCENARIO / "data" / "layout.csv")
    nodes = GridLayout.pick_nodes(locations)
    rows = []
    for day in DAYS:
        lines, _ = day_lines(day, nodes, SCENARIO / "data" / "cache")
        lines = lines.sort_values(["start", "row"])
        midnight = pd.Timestamp(day)
        picker_ids = {picker: i for i, picker in enumerate(lines.picker.unique())}
        wms_picker_end = pd.Series({picker_ids[picker]:
            (group.end.max() - midnight).total_seconds()
            for picker, group in lines.groupby("picker")})
        wms_order_end = (lines.groupby("order").end.max() - midnight).dt.total_seconds()
        wms_line_end = (lines.end - midnight).dt.total_seconds()
        wms_busy = sum(_union_s((g.start - midnight).dt.total_seconds(),
                                (g.end - midnight).dt.total_seconds())
                       for _, g in lines.groupby("picker"))
        wms_span = sum((g.end.max() - g.start.min()).total_seconds()
                       for _, g in lines.groupby("picker"))
        for run in RUNS:
            root = OUTPUTS / run / day
            service = json.loads((root / "service.json").read_text())
            picks = pd.read_csv(root / "facts" / "picks.csv")
            orders = pd.read_csv(root / "facts" / "orders.csv")
            tours = pd.read_csv(root / "facts" / "tours.csv")
            complete_orders = orders.completion.dropna()
            model_picker_end = picks.groupby("picker_id").pick_end.max()
            matched = (model_picker_end - wms_picker_end).dropna() / 60
            if len(matched) != len(wms_picker_end):
                raise ValueError(f"{run} {day}: picker match failed")
            picker = service["pickers"]
            row = {"day": day, "run": run, "orders": len(wms_order_end),
                   "completed_orders": len(complete_orders),
                   "open_orders": int(service["orders"]["all"]["open_orders"]),
                   "pickers": len(wms_picker_end),
                   "wms_lines": len(wms_line_end), "model_lines": len(picks),
                   "model_tours": len(tours),
                   "completed_tours": int(tours.end.notna().sum()),
                   "wms_last_line_h": float(wms_line_end.max() / 3600),
                   "model_last_line_h": float(picks.pick_end.max() / 3600),
                   "last_line_error_min": float((picks.pick_end.max() -
                                                  wms_line_end.max()) / 60),
                   "picker_end_bias_min": float(matched.mean()),
                   "picker_end_mae_min": float(matched.abs().mean()),
                   "picker_end_rmse_min": float(np.sqrt((matched ** 2).mean())),
                   "wms_busy_share_of_first_to_last": wms_busy / wms_span,
                   "model_busy_share_of_presence": sum(p["busy_s"] for p in picker) /
                       sum(p["presence_s"] for p in picker)}
            for hour in HOURS:
                cutoff = hour * 3600
                row[f"order_cdf_error_{hour}h"] = (
                    int((complete_orders <= cutoff).sum()) -
                    int((wms_order_end <= cutoff).sum()))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    result = audit()
    directory = OUTPUTS / "system_validation"
    directory.mkdir(parents=True, exist_ok=True)
    result.to_csv(directory / "days.csv", index=False)
    pd.set_option("display.width", 250)
    print(result.round(2).to_string(index=False))
    pooled = []
    for run, group in result.groupby("run"):
        n = group.pickers.sum()
        pooled.append({"run": run, "orders_open": int(group.open_orders.sum()),
                       "model_lines": int(group.model_lines.sum()),
                       "picker_days": int(n),
                       "picker_end_bias_min": float((group.pickers *
                           group.picker_end_bias_min).sum() / n),
                       "picker_end_rmse_min": float(np.sqrt((group.pickers *
                           group.picker_end_rmse_min ** 2).sum() / n)),
                       "mean_day_last_error_min": float(group.last_line_error_min.mean()),
                       "orders_complete_by_10h_minus_wms": int(
                           group.order_cdf_error_10h.sum()),
                       "orders_complete_by_12h_minus_wms": int(
                           group.order_cdf_error_12h.sum()),
                       "largest_day_error_10h": int(group.order_cdf_error_10h.abs().max()),
                       "largest_day_error_12h": int(group.order_cdf_error_12h.abs().max())})
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(directory / "overall.csv", index=False)
    print("\nPOOLED\n", pooled.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
