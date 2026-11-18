"""Aggregate WMS versus CASIM replay clocks without exporting private rows.

The comparison distinguishes (a) a tour's first-line-to-last-line window,
(b) line-interval work within it, and (c) completion of a picker's day.
The WMS line-interval union is only a booking-time proxy for physical work.
Run after ``casim run scenarios/ijpe sweep=days_thresholds``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from casim.sources.grid_layout import GridLayout
from casim.sources.recorded import RecordedRow, recorded_tours
from casim.sources.wms_export import day_lines
from scripts.ijpe import OUTPUTS, SCENARIO
from scripts.ijpe.replay import _union_s

DAYS = ("2026-04-11", "2026-04-13", "2026-04-14", "2026-04-15",
        "2026-04-16", "2026-04-17")
SETTINGS = (("anchored", "null"), ("five_minute", "300.0"))


def _key(order_ids) -> str:
    return " ".join(map(str, sorted(map(int, order_ids))))


def _stats(error_min: pd.Series, prefix: str) -> dict:
    return {f"{prefix}_bias_min": float(error_min.mean()),
            f"{prefix}_mae_min": float(error_min.abs().mean()),
            f"{prefix}_rmse_min": float(np.sqrt((error_min ** 2).mean())),
            f"{prefix}_median_min": float(error_min.median()),
            f"{prefix}_under_count": int((error_min < 0).sum()),
            f"{prefix}_over_count": int((error_min > 0).sum()),
            f"{prefix}_q10_min": float(error_min.quantile(0.10)),
            f"{prefix}_q90_min": float(error_min.quantile(0.90))}


def audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, locations = GridLayout().build(SCENARIO / "data" / "layout.csv")
    nodes = GridLayout.pick_nodes(locations)
    days = []
    tour_errors: dict[str, list[pd.DataFrame]] = {name: [] for name, _ in SETTINGS}
    picker_errors: dict[str, list[pd.Series]] = {name: [] for name, _ in SETTINGS}
    for day in DAYS:
        lines, _ = day_lines(day, nodes, SCENARIO / "data" / "cache")
        lines = lines.sort_values(["start", "row"])
        midnight = pd.Timestamp(day)
        picker_ids = {picker: index for index, picker in enumerate(lines.picker.unique())}
        rows = [RecordedRow(picker=picker_ids[row.picker], order=int(row.order),
                            article=int(row.article), quantity=int(row.quantity),
                            node=row.node, start_s=(row.start - midnight).total_seconds(),
                            end_s=(row.end - midnight).total_seconds())
                for row in lines.itertuples()]
        recorded = []
        for picker, tour in recorded_tours(rows):
            start = min(row.start_s for row in tour)
            end = max(row.end_s for row in tour)
            busy = _union_s([row.start_s for row in tour],
                            [row.end_s for row in tour])
            recorded.append({"key": _key({row.order for row in tour}),
                             "wms_picker_id": picker, "wms_span_s": end - start,
                             "wms_busy_s": busy, "wms_inside_idle_s": end - start - busy,
                             "wms_lines": len(tour)})
        recorded = pd.DataFrame(recorded).set_index("key")
        if not recorded.index.is_unique:
            raise ValueError(f"{day}: duplicate recorded tour order sets")
        wms_picker_end = pd.Series({picker_ids[picker]:
                                    (group.end.max() - midnight).total_seconds()
                                    for picker, group in lines.groupby("picker")})
        for setting, suffix in SETTINGS:
            path = SCENARIO / "outputs" / "run" / (
                f"source.day={day},source.pause_threshold_s={suffix}")
            picks = pd.read_csv(path / "facts" / "picks.csv")
            tours = pd.read_csv(path / "facts" / "tours.csv",
                                dtype={"order_ids": str})
            model = picks.groupby("tour_id").agg(
                first_walk=("walk_start", "min"), last_pick=("pick_end", "max"),
                model_lines=("pick_end", "size"))
            active = (picks.pick_end - picks.walk_start).groupby(picks.tour_id).sum()
            model["model_active_s"] = active
            tours["key"] = tours.order_ids.str.split().map(_key)
            tours = tours.set_index("key")
            if not tours.index.is_unique:
                raise ValueError(f"{day} {setting}: duplicate model tour order sets")
            tours = tours.join(model, on="tour_id", validate="one_to_one")
            if tours.end.isna().any() or tours.first_walk.isna().any():
                raise ValueError(f"{day} {setting}: incomplete tour")
            joined = recorded.join(tours, how="outer", validate="one_to_one")
            if joined.wms_span_s.isna().any() or joined.tour_id.isna().any():
                raise ValueError(f"{day} {setting}: unmatched tour")
            if not (joined.wms_lines == joined.model_lines).all():
                raise ValueError(f"{day} {setting}: tour line counts differ")
            if not (joined.wms_picker_id == joined.picker_id).all():
                raise ValueError(f"{day} {setting}: tour picker differs")
            joined["model_window_s"] = joined.last_pick - joined.first_walk
            joined["model_full_tour_s"] = joined.end - joined.start
            joined["window_error_min"] = (joined.model_window_s - joined.wms_span_s) / 60
            joined["work_error_min"] = (joined.model_active_s - joined.wms_busy_s) / 60
            model_picker_end = picks.groupby("picker_id").pick_end.max()
            end_error = (model_picker_end - wms_picker_end).dropna() / 60
            if len(end_error) != len(wms_picker_end):
                raise ValueError(f"{day} {setting}: unmatched picker")
            picker_errors[setting].append(end_error)
            tour_errors[setting].append(joined[["window_error_min", "work_error_min"]])
            days.append({
                "day": day, "setting": setting, "lines": len(lines),
                "orders": lines.order.nunique(), "pickers": len(wms_picker_end),
                "tours": len(joined),
                "wms_last_pick_clock_h": float((lines.end.max() - midnight).total_seconds()/3600),
                "model_last_pick_clock_h": float(picks.pick_end.max()/3600),
                "last_pick_difference_min": float((picks.pick_end.max() -
                    (lines.end.max() - midnight).total_seconds())/60),
                "wms_tour_span_h": float(joined.wms_span_s.sum()/3600),
                "wms_line_busy_h": float(joined.wms_busy_s.sum()/3600),
                "wms_inside_tour_idle_h": float(joined.wms_inside_idle_s.sum()/3600),
                "model_pick_window_h": float(joined.model_window_s.sum()/3600),
                "model_active_line_h": float(joined.model_active_s.sum()/3600),
                "model_full_tour_h": float(joined.model_full_tour_s.sum()/3600),
                **_stats(joined.window_error_min, "tour_window"),
                **_stats(joined.work_error_min, "tour_work"),
                **_stats(end_error, "picker_end"),
            })
    day_table = pd.DataFrame(days)
    overall = []
    for setting, _ in SETTINGS:
        subset = day_table[day_table.setting == setting]
        tours = pd.concat(tour_errors[setting])
        pickers = pd.concat(picker_errors[setting])
        overall.append({"setting": setting, "days": len(subset),
                        "lines": int(subset.lines.sum()), "orders": int(subset.orders.sum()),
                        "pickers": int(subset.pickers.sum()), "tours": int(subset.tours.sum()),
                        "wms_tour_span_h": float(subset.wms_tour_span_h.sum()),
                        "wms_line_busy_h": float(subset.wms_line_busy_h.sum()),
                        "wms_inside_tour_idle_h": float(subset.wms_inside_tour_idle_h.sum()),
                        "model_pick_window_h": float(subset.model_pick_window_h.sum()),
                        "model_active_line_h": float(subset.model_active_line_h.sum()),
                        "model_full_tour_h": float(subset.model_full_tour_h.sum()),
                        **_stats(tours.window_error_min, "tour_window"),
                        **_stats(tours.work_error_min, "tour_work"),
                        **_stats(pickers, "picker_end")})
    return day_table, pd.DataFrame(overall)


def main() -> None:
    day_table, overall = audit()
    directory = OUTPUTS / "current_state"
    directory.mkdir(parents=True, exist_ok=True)
    day_table.to_csv(directory / "days.csv", index=False)
    overall.to_csv(directory / "overall.csv", index=False)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 260)
    print(day_table.round(2).to_string(index=False))
    print("\nOVERALL\n", overall.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
