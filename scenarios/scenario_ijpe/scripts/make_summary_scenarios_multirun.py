from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def original_order_id(order_id) -> str:
    return str(order_id).split("_")[0]


def fmt_clock(sec: float, day_sec: int = 86400) -> str:
    sec = int(sec % day_sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def collect_ids(values) -> set[str]:
    ids = set()
    for value in values:
        for x in str(value).split("|"):
            if x:
                ids.add(x)
    return ids


def work_intervals_for_day(
    day: int,
    *,
    day_sec: int,
    shift_start_hour: float,
    shift_end_hour: float,
    breaks: list[dict] | None,
) -> list[tuple[float, float]]:
    start = day * day_sec + shift_start_hour * 3600
    end = day * day_sec + shift_end_hour * 3600

    intervals = [(start, end)]

    for br in breaks or []:
        b0 = day * day_sec + br["start_hour"] * 3600
        b1 = b0 + br["duration_minutes"] * 60

        new_intervals = []
        for a, b in intervals:
            if a < b0:
                new_intervals.append((a, min(b, b0)))
            if b1 < b:
                new_intervals.append((max(a, b1), b))
        intervals = new_intervals

    return intervals


def clipped_to_work(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    return sum(overlap(start, end, a, b) for a, b in intervals)


def export_multiday_metrics(
    tracker,
    *,
    out_dir: str | Path,
    scenario_name: str,
    n_days: int,
    day_sec: int,
    pickers_per_day: list[int],
    shift_start_hour: float,
    shift_end_hour: float,
    breaks: list[dict] | None,
    due_time_by_order: dict,
    dock_capacity: int,
    picker_ids: list[int],
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # original order id -> original/effective due date
    due_time_by_order = {
        str(order_id): due_time
        for order_id, due_time in due_time_by_order.items()
    }

    # Track BOTH cases:
    # - intra-day shifts may be tracked as split ids:    O01210_P1
    # - cross-day shifts may be tracked as original ids: O01210
    shifted_due_by_split_id = {}
    shifted_due_by_original_id = {}

    shifted_from_to_by_split_id = {}
    shifted_from_to_by_original_id = {}

    # ingested_by_id = {}
    #
    # for order_id, due in tracker.ingested_orders:
    #     ingested_by_id[order_id] = due

    for order, from_time, to_time in tracker.shifted_orders:
        sid = str(order.order_id)
        oid = original_order_id(sid)

        if sid == oid:
            shifted_due_by_original_id[oid] = order.due_date
            shifted_from_to_by_original_id[oid] = (from_time, to_time)
        else:
            shifted_due_by_split_id[sid] = order.due_date
            shifted_from_to_by_split_id[sid] = (from_time, to_time)

    # ------------------------------------------------------------------
    # Tour-level execution log
    # ------------------------------------------------------------------
    tour_rows = []

    for tour_id, start, end, order_ids, picker_id, on_time, delayed, n_lines in tracker.completed_tours:
        split_ids = [str(oid) for oid in order_ids]
        original_ids = [original_order_id(oid) for oid in split_ids]

        due_times = []
        affected_split_ids = []
        affected_original_ids = []
        shifted_from = []
        shifted_to = []

        for sid, oid in zip(split_ids, original_ids):
            if sid in shifted_due_by_split_id:
                due = shifted_due_by_split_id[sid]
                from_time, to_time = shifted_from_to_by_split_id[sid]

                affected_split_ids.append(sid)
                affected_original_ids.append(oid)
                shifted_from.append(from_time)
                shifted_to.append(to_time)

            elif oid in shifted_due_by_original_id:
                due = shifted_due_by_original_id[oid]
                from_time, to_time = shifted_from_to_by_original_id[oid]

                affected_split_ids.append(sid)
                affected_original_ids.append(oid)
                shifted_from.append(from_time)
                shifted_to.append(to_time)

            else:
                due = due_time_by_order[oid]

            due_times.append(due)

        tardiness = [
            max(0.0, end - due)
            for due in due_times
        ]

        late_split_orders = sum(t > 0 for t in tardiness)
        affected_original_ids_unique = sorted(set(affected_original_ids))

        tour_rows.append({
            "scenario": scenario_name,
            "tour_id": tour_id,
            "day": int(end // day_sec),
            "picker_id": picker_id,

            "start_time_sec": start,
            "end_time_sec": end,
            "start_clock": fmt_clock(start, day_sec),
            "end_clock": fmt_clock(end, day_sec),
            "duration_sec": end - start,
            "duration_min": (end - start) / 60,

            "split_order_ids": "|".join(split_ids),
            "original_order_ids": "|".join(original_ids),
            "n_split_orders": len(split_ids),
            "n_original_orders": len(set(original_ids)),
            "n_lines": n_lines,

            "due_times_sec": "|".join(str(int(x)) for x in due_times),
            "due_clocks": "|".join(fmt_clock(x, day_sec) for x in due_times),
            "min_due_time_sec": min(due_times),
            "min_due_clock": fmt_clock(min(due_times), day_sec),

            "late_split_orders": late_split_orders,
            "all_on_time": late_split_orders == 0,
            "total_tardiness_sec": sum(tardiness),
            "total_tardiness_min": sum(tardiness) / 60,
            "max_tardiness_sec": max(tardiness),
            "max_tardiness_min": max(tardiness) / 60,

            "affected_split_orders": len(affected_split_ids),
            "affected_original_orders": len(affected_original_ids_unique),
            "affected_by_shift": len(affected_split_ids) > 0,
            "shifted_split_order_ids": "|".join(affected_split_ids),
            "shifted_original_order_ids": "|".join(affected_original_ids_unique),
            "shifted_from_sec": "|".join(str(int(x)) for x in shifted_from),
            "shifted_to_sec": "|".join(str(int(x)) for x in shifted_to),
        })

    tour_df = pd.DataFrame(tour_rows)

    # ------------------------------------------------------------------
    # Picker-day log
    # ------------------------------------------------------------------
    picker_day_rows = []

    for day in range(n_days):
        intervals = work_intervals_for_day(
            day,
            day_sec=day_sec,
            shift_start_hour=shift_start_hour,
            shift_end_hour=shift_end_hour,
            breaks=breaks,
        )

        available_sec = sum(b - a for a, b in intervals)
        available_pickers = picker_ids[:pickers_per_day[day]]

        for pid in available_pickers:
            active_sec = 0.0
            idle_sec = 0.0

            for _, start, end, _, tour_pid, _, _, _ in tracker.completed_tours:
                if tour_pid == pid:
                    active_sec += clipped_to_work(start, end, intervals)

            for idle_pid, start, end in tracker.idle_intervals:
                if idle_pid == pid:
                    idle_sec += clipped_to_work(start, end, intervals)

            picker_day_rows.append({
                "scenario": scenario_name,
                "day": day,
                "picker_id": pid,
                "available_sec": available_sec,
                "active_sec": active_sec,
                "idle_sec": idle_sec,
                "available_h": available_sec / 3600,
                "active_h": active_sec / 3600,
                "idle_h": idle_sec / 3600,
                "utilization": active_sec / available_sec,
            })

    picker_day_df = pd.DataFrame(picker_day_rows)

    # ------------------------------------------------------------------
    # Day summary
    # ------------------------------------------------------------------
    day_rows = []

    for day in range(n_days):
        tours_d = tour_df[tour_df["day"] == day]
        pickers_d = picker_day_df[picker_day_df["day"] == day]

        dock_values = [
            n_pallets
            for t, n_pallets in tracker.dock_utilization
            if int(t // day_sec) == day
        ]

        wait_times = [
            max(0.0, max(expected_finish_times) - t)
            for t, open_tours, expected_finish_times in tracker.delayed_expected_finish
            if int(t // day_sec) == day and expected_finish_times
        ]

        split_orders = int(tours_d["n_split_orders"].sum()) if len(tours_d) else 0
        late_split_orders = int(tours_d["late_split_orders"].sum()) if len(tours_d) else 0
        affected_original_ids_d = collect_ids(tours_d["shifted_original_order_ids"]) if len(tours_d) else set()

        day_rows.append({
            "scenario": scenario_name,
            "day": day,
            "n_pickers": pickers_per_day[day],

            "tours_completed": int(len(tours_d)),
            "split_orders_completed": split_orders,

            "first_tour_start_clock": fmt_clock(tours_d["start_time_sec"].min(), day_sec) if len(tours_d) else None,
            "last_tour_end_clock": fmt_clock(tours_d["end_time_sec"].max(), day_sec) if len(tours_d) else None,
            "operational_span_h": (
                (tours_d["end_time_sec"].max() - tours_d["start_time_sec"].min()) / 3600
                if len(tours_d) else 0.0
            ),

            "avg_tour_min": float(tours_d["duration_min"].mean()) if len(tours_d) else 0.0,
            "total_tour_h": float(tours_d["duration_sec"].sum() / 3600) if len(tours_d) else 0.0,

            "late_split_orders": late_split_orders,
            "on_time_rate": 1 - late_split_orders / split_orders if split_orders else 0.0,
            "total_tardiness_h": float(tours_d["total_tardiness_sec"].sum() / 3600) if len(tours_d) else 0.0,
            "max_tardiness_min": float(tours_d["max_tardiness_sec"].max() / 60) if len(tours_d) else 0.0,

            "affected_split_orders": int(tours_d["affected_split_orders"].sum()) if len(tours_d) else 0,
            "affected_original_orders": len(affected_original_ids_d),
            "affected_tours": int(tours_d["affected_by_shift"].sum()) if len(tours_d) else 0,

            "available_labor_h": float(pickers_d["available_h"].sum()),
            "active_h": float(pickers_d["active_h"].sum()),
            "idle_h": float(pickers_d["idle_h"].sum()),
            "utilization": float(pickers_d["active_sec"].sum() / pickers_d["available_sec"].sum()),

            "orders_per_labor_h": (
                split_orders / pickers_d["available_h"].sum()
                if split_orders else 0.0
            ),

            "max_dock_fill_pct": (
                100 * max(dock_values) / dock_capacity
                if dock_values else 0.0
            ),

            "truck_wait_events": int(sum(w > 0 for w in wait_times)),
            "max_truck_wait_min": max(wait_times) / 60 if wait_times else 0.0,
        })

    day_summary_df = pd.DataFrame(day_rows)

    # ------------------------------------------------------------------
    # Scenario summary
    # ------------------------------------------------------------------
    split_orders = int(tour_df["n_split_orders"].sum())
    late_split_orders = int(tour_df["late_split_orders"].sum())

    affected_original_ids = collect_ids(tour_df["shifted_original_order_ids"])

    total_available_sec = float(picker_day_df["available_sec"].sum())
    total_active_sec = float(picker_day_df["active_sec"].sum())

    summary = {
        "scenario": scenario_name,

        "tours_completed": int(len(tour_df)),
        "split_orders_completed": split_orders,

        "avg_tour_min": float(tour_df["duration_min"].mean()),
        "total_tour_h": float(tour_df["duration_sec"].sum() / 3600),

        "late_split_orders": late_split_orders,
        "on_time_rate": 1 - late_split_orders / split_orders,
        "total_tardiness_h": float(tour_df["total_tardiness_sec"].sum() / 3600),
        "max_tardiness_min": float(tour_df["max_tardiness_sec"].max() / 60),

        "affected_split_orders": int(tour_df["affected_split_orders"].sum()),
        "affected_original_orders": len(affected_original_ids),
        "affected_tours": int(tour_df["affected_by_shift"].sum()),

        "available_labor_h": total_available_sec / 3600,
        "active_h": total_active_sec / 3600,
        "idle_h": float(picker_day_df["idle_sec"].sum() / 3600),
        "utilization": total_active_sec / total_available_sec,
        "orders_per_labor_h": split_orders / (total_available_sec / 3600),

        "max_dock_fill_pct": float(day_summary_df["max_dock_fill_pct"].max()),
        "truck_wait_events": int(day_summary_df["truck_wait_events"].sum()),
        "max_truck_wait_min": float(day_summary_df["max_truck_wait_min"].max()),
    }

    tour_df.to_csv(out_dir / "tour_execution.csv", index=False)
    picker_day_df.to_csv(out_dir / "picker_day_summary.csv", index=False)
    day_summary_df.to_csv(out_dir / "day_summary.csv", index=False)

    shift_rows = []

    for order, from_time, to_time in tracker.shifted_orders:
        shift_rows.append({
            "scenario": scenario_name,
            "shift_order_id": str(order.order_id),
            "shift_original_order_id": original_order_id(order.order_id),
            "from_time_sec": from_time,
            "to_time_sec": to_time,
            "from_clock": fmt_clock(from_time, day_sec),
            "to_clock": fmt_clock(to_time, day_sec),
        })

    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv(out_dir / "shift_log.csv", index=False)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary