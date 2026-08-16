from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams["hatch.linewidth"] = 0.6
plt.rcParams["font.family"] = "Times New Roman"

DAY_SEC = 86400

def sec_to_clock(sec: float) -> str:
    sec = int(sec % DAY_SEC)
    h = sec // 3600
    m = (sec % 3600) // 60
    return f"{h:02d}:{m:02d}"


def scenario_label(scenario: str) -> str:
    labels = {
        "baseline": "Baseline",
        "scenario1": "Scenario 1",
        "scenario2": "Scenario 2",
        "scenario3": "Scenario 3",
        "scenario4": "Scenario 4",
    }
    return labels.get(scenario, scenario)


def split_cell(x) -> list[str]:
    if pd.isna(x):
        return []

    s = str(x).strip()

    if s == "":
        return []

    return s.split("|")

def fmt_num(x, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_int(x) -> str:
    if pd.isna(x):
        return "--"
    return str(int(round(float(x))))


def pick_col(df: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"None of these columns exist: {names}")

def load_kpi_files(kpi_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tour_df = pd.read_csv(kpi_dir / "tour_execution.csv")
    picker_df = pd.read_csv(kpi_dir / "picker_day_summary.csv")
    day_df = pd.read_csv(kpi_dir / "day_summary.csv")
    return tour_df, picker_df, day_df


# ---------------------------------------------------------------------
# Scenario-level and day-level KPIs
# ---------------------------------------------------------------------
def compute_summary_row(
    *,
    scenario: str,
    tour_df: pd.DataFrame,
    picker_df: pd.DataFrame,
) -> dict:
    first_start = float(tour_df["start_time_sec"].min())
    last_end = float(tour_df["end_time_sec"].max())

    makespan_sec = last_end - first_start
    overall_processing_sec = float(tour_df["duration_sec"].sum())

    orders_completed = int(tour_df["n_split_orders"].sum())
    tours_completed = int(len(tour_df))

    late_orders = int(tour_df["late_split_orders"].sum())
    total_tardiness_sec = float(tour_df["total_tardiness_sec"].sum())
    max_tardiness_sec = float(tour_df["max_tardiness_sec"].max())

    available_sec = float(picker_df["available_sec"].sum())
    active_sec = float(picker_df["active_sec"].sum())

    utilization = active_sec / available_sec if available_sec > 0 else 0.0
    on_time_rate = 1 - late_orders / orders_completed if orders_completed > 0 else 0.0

    return {
        "scenario": scenario,
        "scenario_label": scenario_label(scenario),

        "tours_completed": tours_completed,
        "orders_completed": orders_completed,

        "first_start_sec": first_start,
        "last_end_sec": last_end,
        "first_start": sec_to_clock(first_start),
        "last_end": sec_to_clock(last_end),

        "makespan_h": makespan_sec / 3600,
        "overall_processing_time_h": overall_processing_sec / 3600,
        "picker_utilization_pct": utilization * 100,

        "on_time_rate_pct": on_time_rate * 100,
        "late_orders": late_orders,
        "total_tardiness_h": total_tardiness_sec / 3600,
        "max_tardiness_min": max_tardiness_sec / 60,
    }


def compute_day_rows(
    *,
    scenario: str,
    tour_df: pd.DataFrame,
    picker_df: pd.DataFrame,
    n_days: int,
) -> list[dict]:
    rows = []

    for day in range(n_days):
        tours_d = tour_df[tour_df["day"] == day]
        pickers_d = picker_df[picker_df["day"] == day]

        orders_completed = int(tours_d["n_split_orders"].sum()) if len(tours_d) else 0
        late_orders = int(tours_d["late_split_orders"].sum()) if len(tours_d) else 0

        total_tardiness_sec = (
            float(tours_d["total_tardiness_sec"].sum())
            if len(tours_d)
            else 0.0
        )

        max_tardiness_sec = (
            float(tours_d["max_tardiness_sec"].max())
            if len(tours_d)
            else 0.0
        )

        available_sec = float(pickers_d["available_sec"].sum()) if len(pickers_d) else 0.0
        active_sec = float(pickers_d["active_sec"].sum()) if len(pickers_d) else 0.0

        utilization = active_sec / available_sec if available_sec > 0 else 0.0
        on_time_rate = 1 - late_orders / orders_completed if orders_completed > 0 else 0.0

        if len(tours_d):
            first_start = float(tours_d["start_time_sec"].min())
            last_end = float(tours_d["end_time_sec"].max())
            makespan_h = (last_end - first_start) / 3600
            overall_processing_h = float(tours_d["duration_sec"].sum()) / 3600
        else:
            first_start = np.nan
            last_end = np.nan
            makespan_h = 0.0
            overall_processing_h = 0.0

        rows.append({
            "scenario": scenario,
            "scenario_label": scenario_label(scenario),
            "day": day,
            "day_label": f"D{day + 1}",

            "orders_completed": orders_completed,
            "tours_completed": int(len(tours_d)),

            "first_start": sec_to_clock(first_start) if len(tours_d) else "",
            "last_end": sec_to_clock(last_end) if len(tours_d) else "",

            "makespan_h": makespan_h,
            "overall_processing_time_h": overall_processing_h,

            "picker_utilization_pct": utilization * 100,

            "on_time_rate_pct": on_time_rate * 100,
            "late_orders": late_orders,
            "total_tardiness_h": total_tardiness_sec / 3600,
            "max_tardiness_min": max_tardiness_sec / 60,
        })

    return rows


# ---------------------------------------------------------------------
# Order-level execution and truck waiting KPIs
# ---------------------------------------------------------------------
def explode_tour_execution_to_orders(tour_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in tour_df.iterrows():
        split_ids = split_cell(r["split_order_ids"])
        original_ids = split_cell(r["original_order_ids"])
        due_times = [float(x) for x in split_cell(r["due_times_sec"])]

        shifted_ids = set(split_cell(r.get("shifted_split_order_ids", "")))

        completion_time = float(r["end_time_sec"])

        for split_id, original_id, due_time in zip(split_ids, original_ids, due_times):
            tardiness_sec = max(0.0, completion_time - due_time)

            rows.append({
                "scenario": r["scenario"],
                "tour_day": int(r["day"]),
                "tour_id": r["tour_id"],
                "picker_id": r["picker_id"],

                "order_id": split_id,
                "original_order_id": original_id,

                "due_time_sec": due_time,
                "due_day": int(due_time // DAY_SEC),
                "due_clock": sec_to_clock(due_time),

                "completion_time_sec": completion_time,
                "completion_day": int(completion_time // DAY_SEC),
                "completion_clock": sec_to_clock(completion_time),

                "tardiness_sec": tardiness_sec,
                "tardiness_min": tardiness_sec / 60,
                "on_time": tardiness_sec <= 0,

                "shifted_order": split_id in shifted_ids,
            })

    return pd.DataFrame(rows)


def compute_truck_waiting_times(
    *,
    scenario: str,
    order_df: pd.DataFrame,
    planned_due_times: list[float],
) -> pd.DataFrame:
    rows = []

    grouped = {
        due_time: g
        for due_time, g in order_df.groupby("due_time_sec")
    }

    for due_time in planned_due_times:
        due_time = float(due_time)

        if due_time in grouped:
            g = grouped[due_time]
            latest_completion = float(g["completion_time_sec"].max())
            wait_sec = max(0.0, latest_completion - due_time)

            n_orders = int(len(g))
            n_late_orders = int((g["tardiness_sec"] > 0).sum())
            latest_completion_clock = sec_to_clock(latest_completion)
        else:
            latest_completion = np.nan
            wait_sec = 0.0

            n_orders = 0
            n_late_orders = 0
            latest_completion_clock = ""

        rows.append({
            "scenario": scenario,
            "due_time_sec": due_time,
            "due_day": int(due_time // DAY_SEC),
            "due_clock": sec_to_clock(due_time),

            "n_orders": n_orders,
            "n_late_orders": n_late_orders,
            "active_departure": n_orders > 0,

            "latest_completion_sec": latest_completion,
            "latest_completion_clock": latest_completion_clock,

            "truck_wait_sec": wait_sec,
            "truck_wait_min": wait_sec / 60,
            "truck_delayed": wait_sec > 0,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["due_time_sec"])
        .reset_index(drop=True)
    )


def compute_impact_row(
    *,
    scenario: str,
    order_df: pd.DataFrame,
    truck_df: pd.DataFrame,
) -> dict:
    shifted = order_df[order_df["shifted_order"]].copy()

    n_shifted = int(len(shifted))

    if n_shifted > 0:
        shifted_on_time = int(shifted["on_time"].sum())
        shifted_on_time_rate = 100 * shifted_on_time / n_shifted
    else:
        shifted_on_time = np.nan
        shifted_on_time_rate = np.nan

    delayed_trucks = truck_df[truck_df["truck_delayed"]].copy()

    return {
        "scenario": scenario,
        "scenario_label": scenario_label(scenario),

        "requested_due_time_slots": int(len(truck_df)),
        "active_due_time_slots": int(truck_df["active_departure"].sum()),
        "delayed_due_time_slots": int(truck_df["truck_delayed"].sum()),

        "avg_truck_wait_min": float(truck_df["truck_wait_min"].mean()) if len(truck_df) else 0.0,
        "avg_truck_wait_min_delayed": float(delayed_trucks["truck_wait_min"].mean()) if len(delayed_trucks) else 0.0,
        "max_truck_wait_min": float(truck_df["truck_wait_min"].max()) if len(truck_df) else 0.0,

        "shifted_orders": n_shifted,
        "shifted_orders_on_time": shifted_on_time,
        "shifted_on_time_rate_pct": shifted_on_time_rate,
    }


# ---------------------------------------------------------------------
# LaTeX table helpers
# ---------------------------------------------------------------------
def write_latex_table(
    df: pd.DataFrame,
    *,
    out_file: Path,
    caption: str,
    label: str,
    float_format: str = "%.2f",
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    latex = df.to_latex(
        index=False,
        escape=False,
        na_rep="--",
        float_format=lambda x: float_format % x,
        caption=caption,
        label=label,
        position="t",
    )

    out_file.write_text(latex, encoding="utf-8")


def make_paper_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scenario_label",
        "orders_completed",
        "makespan_h",
        "overall_processing_time_h",
        "picker_utilization_pct",
        "on_time_rate_pct",
        "late_orders",
        "total_tardiness_h",
        "max_tardiness_min",
    ]

    table = summary_df[cols].copy()

    table = table.rename(columns={
        "scenario_label": "Scenario",
        "orders_completed": "Orders",
        "makespan_h": "Makespan [h]",
        "overall_processing_time_h": "Overall processing [h]",
        "picker_utilization_pct": "Picker util. [\\%]",
        "on_time_rate_pct": "On-time [\\%]",
        "late_orders": "Late orders",
        "total_tardiness_h": "Tardiness [h]",
        "max_tardiness_min": "Max tard. [min]",
    })

    return table


# def make_paper_day_table(day_df: pd.DataFrame) -> pd.DataFrame:
#     cols = [
#         "scenario_label",
#         "day_label",
#         "orders_completed",
#         "picker_utilization_pct",
#         "on_time_rate_pct",
#         "late_orders",
#         "total_tardiness_h",
#     ]
#
#     table = day_df[cols].copy()
#
#     table = table.rename(columns={
#         "scenario_label": "Scenario",
#         "day_label": "Day",
#         "orders_completed": "Orders",
#         "picker_utilization_pct": "Picker util. [\\%]",
#         "on_time_rate_pct": "On-time [\\%]",
#         "late_orders": "Late orders",
#         "total_tardiness_h": "Tardiness [h]",
#     })
#
#     return table
def make_paper_day_table(
    day_df: pd.DataFrame,
    *,
    baseline: str = "baseline",
) -> pd.DataFrame:
    baseline_days = (
        day_df[day_df["scenario"] == baseline]
        .set_index("day")
    )

    rows = []

    for _, r in day_df[day_df["scenario"] != baseline].iterrows():
        day = int(r["day"])

        if day not in baseline_days.index:
            continue

        b = baseline_days.loc[day]

        changed = (
            int(r["orders_completed"]) != int(b["orders_completed"])
            or int(r["late_orders"]) > 0
            or float(r["total_tardiness_h"]) > 0
            or abs(float(r["picker_utilization_pct"]) - float(b["picker_utilization_pct"])) > 0.05
        )

        if not changed:
            continue

        rows.append({
            "Scenario": r["scenario_label"],
            "Day": r["day_label"],
            "Orders": fmt_int(r["orders_completed"]),
            "Util. [\\%]": fmt_num(r["picker_utilization_pct"], 1),
            "On-time [\\%]": fmt_num(r["on_time_rate_pct"], 1),
            "Late orders": fmt_int(r["late_orders"]),
            "Tardiness [h]": fmt_num(r["total_tardiness_h"], 2),
        })

    return pd.DataFrame(rows)


def make_paper_impact_table(impact_df: pd.DataFrame) -> pd.DataFrame:
    delayed_col = pick_col(
        impact_df,
        "delayed_due_time_slots",
        "delayed_truck_departures",
    )

    avg_wait_col = pick_col(
        impact_df,
        "avg_truck_wait_min",
        "avg_truck_wait_min_all",
    )

    rows = []

    for _, r in impact_df.iterrows():
        affected = int(r["shifted_orders"])

        affected_orders = np.nan if affected == 0 else affected
        affected_ot = np.nan if affected == 0 else r["shifted_on_time_rate_pct"]

        delayed_slots = int(r[delayed_col])
        wait_if_delayed = (
            np.nan
            if delayed_slots == 0
            else r["avg_truck_wait_min_delayed"]
        )

        rows.append({
            "Scenario": r["scenario_label"],
            "Affected orders": fmt_int(affected_orders),
            "Affected OT [\\%]": fmt_num(affected_ot, 1),
            "Delayed slots": fmt_int(delayed_slots),
            "Avg. wait [min]": fmt_num(r[avg_wait_col], 2),
            "Wait if delayed [min]": fmt_num(wait_if_delayed, 1),
            "Max wait [min]": fmt_num(r["max_truck_wait_min"], 1),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Gantt figures
# ---------------------------------------------------------------------

def tour_contains_shifted_or_affected_order(row: pd.Series) -> bool:
    """
    Returns True if a tour contains at least one shifted/affected order.

    Uses the explicit affected_by_shift flag if available.
    Falls back to shifted_split_order_ids / shifted_original_order_ids.
    """
    if "affected_by_shift" in row.index:
        value = row["affected_by_shift"]

        if pd.notna(value):
            if isinstance(value, bool):
                return value

            s = str(value).strip().lower()
            if s in {"true", "1", "yes"}:
                return True
            if s in {"false", "0", "no", ""}:
                return False

    if "shifted_split_order_ids" in row.index:
        if len(split_cell(row["shifted_split_order_ids"])) > 0:
            return True

    if "shifted_original_order_ids" in row.index:
        if len(split_cell(row["shifted_original_order_ids"])) > 0:
            return True

    return False

def plot_gantt_for_scenario(
    *,
    tour_df: pd.DataFrame,
    out_file: Path,
    title: str | None,
    n_days: int,
    x_start_hour: float = 5.0,
    x_end_hour: float | None = None,
    breaks: list[tuple[float, float]] | None = None,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if breaks is None:
        breaks = []

    day_picker_sets = {}
    for day in range(n_days):
        tours_d = tour_df[tour_df["day"] == day]
        day_picker_sets[day] = sorted(tours_d["picker_id"].unique())

    fig, axes = plt.subplots(
        n_days,
        1,
        figsize=(7.4, 4.9),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    if n_days == 1:
        axes = [axes]

    if x_end_hour is None:
        if len(tour_df):
            end_hours = (tour_df["end_time_sec"].astype(float) % DAY_SEC) / 3600
            latest_end_h = float(end_hours.max())
            x_end_hour = min(17.0, max(13.0, np.ceil(latest_end_h * 2) / 2 + 0.25))
        else:
            x_end_hour = 13.0

    color_on_time = "#8aa6c5"
    color_affected = "#c8b27a"
    color_late = "#8c4a4a"
    sep_color = "#c7c7c7"
    grid_color = "#d9d9d9"


    LABEL_FS = 8
    TICK_FS = 8
    TITLE_FS = 10
    LEGEND_FS = 7.5

    for day in range(n_days):
        ax = axes[day]
        tours_d = tour_df[tour_df["day"] == day].copy()

        pickers_d = day_picker_sets[day]
        picker_to_y = {pid: i for i, pid in enumerate(pickers_d)}

        for b_start, b_end in breaks:
            ax.axvspan(
                b_start,
                b_end,
                color=break_color,
                alpha=0.55,
                linewidth=0,
                zorder=0,
            )

        for _, r in tours_d.iterrows():
            picker_id = r["picker_id"]
            y = picker_to_y[picker_id]

            start_h = (float(r["start_time_sec"]) % DAY_SEC) / 3600
            end_h = (float(r["end_time_sec"]) % DAY_SEC) / 3600
            duration_h = max(0.001, end_h - start_h)

            is_late = int(r["late_split_orders"]) > 0

            ax.barh(
                y=y,
                width=duration_h,
                left=start_h,
                height=0.78,
                color=color_late if is_late else color_on_time,
                alpha=0.88 if is_late else 0.72,
                edgecolor="none",
                zorder=2,
            )

        n_pickers = len(pickers_d)

        ax.set_xlim(x_start_hour, x_end_hour)

        if n_pickers > 0:
            ax.set_ylim(-0.6, n_pickers - 0.4)
        else:
            ax.set_ylim(-0.5, 0.5)

        ax.set_yticks([])

        ax.set_ylabel(
            f"D{day + 1} (#P: {n_pickers})",
            rotation=0,
            ha="right",
            va="center",
            labelpad=14,
            fontsize=LABEL_FS,
        )

        ax.grid(axis="x", color=grid_color, alpha=0.35, linewidth=0.6)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(sep_color)
        ax.spines["bottom"].set_linewidth(0.6)

        ax.tick_params(axis="x", labelsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)

        if day < n_days - 1:
            ax.tick_params(axis="x", labelbottom=False)

    xticks = np.arange(
        np.ceil(x_start_hour),
        np.floor(x_end_hour) + 1,
        2,
    )

    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([f"{int(h):02d}:00" for h in xticks], fontsize=TICK_FS)
    axes[-1].set_xlabel("Time", fontsize=LABEL_FS)

    legend_handles = [
        Patch(facecolor=color_on_time, alpha=0.72, label="On time"),
        Patch(facecolor=color_late, alpha=0.88, label="Late"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        fontsize=LEGEND_FS,
    )

    if title:
        fig.suptitle(title, y=1.02, fontsize=TITLE_FS)

    fig.subplots_adjust(
        left=0.11,
        right=0.99,
        top=0.89 if title else 0.93,
        bottom=0.10,
    )

    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

def scenario_focus_days() -> dict[str, list[int]]:
    """
    Days are zero-indexed: D1 = 0, D2 = 1, ...

    Adjust these if your simulation day indexing differs.
    Based on the scenario descriptions:
    - Scenario 1: Monday intra-day shift
    - Scenario 2: Thursday intra-day shift
    - Scenario 3: urgent replenishment day
    - Scenario 5: cross-day shift from Thursday to Wednesday
    """
    return {
        "scenario1": [1],
        "scenario2": [4],
        "scenario3": [2],      # urgent replenishment; adjust if placed elsewhere
        "scenario4": [3, 4],   # Wednesday and Thursday
    }


def scenario_focus_title(scenario: str) -> str:
    titles = {
        "scenario1": "Scenario 1: driver absence on Monday",
        "scenario2": "Scenario 2: early delivery request on Thursday",
        "scenario3": "Scenario 3: emergency replenishment",
        "scenario4": "Scenario 4: cross-day workload redistribution",
    }
    return titles.get(scenario, scenario_label(scenario))


def plot_focused_gantt_for_scenario(
    *,
    baseline_tour_df: pd.DataFrame,
    scenario_tour_df: pd.DataFrame,
    scenario: str,
    focus_days: list[int],
    out_file: Path,
    x_start_hour: float | None = None,
    x_end_hour: float | None = None,
    title: str | None = None,
) -> None:
    """
    Focused paper-oriented Gantt.

    For each affected day, plot two stacked panels:
        Baseline
        Scenario

    If x_start_hour / x_end_hour are not given, the visible time window is
    inferred from the displayed tours only. This avoids empty horizontal space
    without hiding actual work.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_blocks = len(focus_days)
    n_rows = 2 * n_blocks

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(7.2, 1.00 * n_rows + 0.75),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    if n_rows == 1:
        axes = [axes]

    color_on_time = "#9fb5cc"
    color_affected = "#d8c58d"
    color_late = "#8f5a5a"
    sep_color = "#c7c7c7"
    grid_color = "#d9d9d9"

    LABEL_FS = 8
    TICK_FS = 8
    TITLE_FS = 10
    LEGEND_FS = 7.5

    # ------------------------------------------------------------
    # Determine plot window from the days actually shown.
    # ------------------------------------------------------------
    shown = pd.concat(
        [
            baseline_tour_df[baseline_tour_df["day"].isin(focus_days)],
            scenario_tour_df[scenario_tour_df["day"].isin(focus_days)],
        ],
        ignore_index=True,
    )

    if len(shown):
        start_hours = (shown["start_time_sec"].astype(float) % DAY_SEC) / 3600
        end_hours = (shown["end_time_sec"].astype(float) % DAY_SEC) / 3600

        if x_start_hour is None:
            # round down to full hour, but do not go below 05:00
            x_start_hour = max(5.0, np.floor(float(start_hours.min())))

        if x_end_hour is None:
            # round up to next half hour plus slight padding, but do not exceed 17:00
            latest = float(end_hours.max())
            x_end_hour = min(17.0, np.ceil((latest + 0.20) * 2) / 2)

            # ensure a minimally readable window
            if x_end_hour - x_start_hour < 4.0:
                x_end_hour = min(17.0, x_start_hour + 4.0)
    else:
        if x_start_hour is None:
            x_start_hour = 5.0
        if x_end_hour is None:
            x_end_hour = 13.0

    def draw_day(
        ax,
        tour_df: pd.DataFrame,
        *,
        day: int,
        row_label: str,
    ) -> None:
        tours_d = tour_df[tour_df["day"] == day].copy()

        pickers_d = sorted(tours_d["picker_id"].unique())
        picker_to_y = {pid: i for i, pid in enumerate(pickers_d)}
        n_pickers = len(pickers_d)

        for _, r in tours_d.iterrows():
            picker_id = r["picker_id"]
            y = picker_to_y[picker_id]

            start_h = (float(r["start_time_sec"]) % DAY_SEC) / 3600
            end_h = (float(r["end_time_sec"]) % DAY_SEC) / 3600

            # Clip only to the displayed, data-driven window.
            plot_start_h = max(start_h, x_start_hour)
            plot_end_h = min(end_h, x_end_hour)
            duration_h = plot_end_h - plot_start_h

            if duration_h <= 0:
                continue

            is_late = int(r["late_split_orders"]) > 0
            is_affected = tour_contains_shifted_or_affected_order(r)

            bar_height = 0.78

            if is_late and is_affected:
                ax.barh(
                    y=y,
                    width=duration_h,
                    left=plot_start_h,
                    height=bar_height,
                    facecolor=color_late,
                    edgecolor=color_affected,
                    hatch="////",
                    linewidth=0.0,
                    alpha=0.92,
                    zorder=3,
                )

            elif is_late:
                ax.barh(
                    y=y,
                    width=duration_h,
                    left=plot_start_h,
                    height=bar_height,
                    facecolor=color_late,
                    edgecolor="none",
                    alpha=0.88,
                    zorder=2,
                )

            elif is_affected:
                ax.barh(
                    y=y,
                    width=duration_h,
                    left=plot_start_h,
                    height=bar_height,
                    facecolor=color_affected,
                    edgecolor="none",
                    alpha=0.88,
                    zorder=2,
                )

            else:
                ax.barh(
                    y=y,
                    width=duration_h,
                    left=plot_start_h,
                    height=bar_height,
                    facecolor=color_on_time,
                    edgecolor="none",
                    alpha=0.72,
                    zorder=2,
                )

        ax.set_xlim(x_start_hour, x_end_hour)

        if n_pickers > 0:
            ax.set_ylim(-0.6, n_pickers - 0.4)
        else:
            ax.set_ylim(-0.5, 0.5)

        ax.set_yticks([])

        ax.set_ylabel(
            f"{row_label}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=18,
            fontsize=LABEL_FS,
        )

        ax.grid(axis="x", color=grid_color, alpha=0.35, linewidth=0.6)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(sep_color)
        ax.spines["bottom"].set_linewidth(0.6)

        ax.tick_params(axis="x", labelsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)

    row = 0
    for day in focus_days:
        ax_base = axes[row]
        ax_scen = axes[row + 1]

        # draw_day(
        #     ax_base,
        #     baseline_tour_df,
        #     day=day,
        #     row_label=f"Baseline",
        # )

        draw_day(
            ax_scen,
            scenario_tour_df,
            day=day,
            row_label=f"{scenario_label(scenario)}",
        )

        if row + 1 < n_rows - 1:
            ax_base.tick_params(axis="x", labelbottom=False)
            ax_scen.tick_params(axis="x", labelbottom=False)

        ax_scen.spines["bottom"].set_color("#a8a8a8")
        ax_scen.spines["bottom"].set_linewidth(0.8)

        row += 2

    # Tick spacing adapts to window length.
    window = x_end_hour - x_start_hour
    tick_step = 1 if window <= 6 else 2

    xticks = np.arange(
        np.ceil(x_start_hour),
        np.floor(x_end_hour) + 1,
        tick_step,
    )

    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([f"{int(h):02d}:00" for h in xticks], fontsize=TICK_FS)
    axes[-1].set_xlabel("Time", fontsize=LABEL_FS)

    legend_handles = [
        Patch(facecolor=color_on_time, edgecolor="none", alpha=0.72, label="On time"),
        Patch(facecolor=color_affected, edgecolor="none", alpha=0.88, label="Affected"),
        Patch(facecolor=color_late, edgecolor="none", alpha=0.88, label="Late"),
        Patch(
            facecolor=color_late,
            edgecolor=color_affected,
            hatch="////",
            linewidth=0.0,
            alpha=0.92,
            label="Affected + late",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=LEGEND_FS,
    )

    if title:
        fig.suptitle(title, y=1.03, fontsize=TITLE_FS)

    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        top=0.86 if title else 0.91,
        bottom=0.16,
    )

    fig.savefig(out_file, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

def plot_focused_gantt_for_scenario_day(
    *,
    scenario_tour_df: pd.DataFrame,
    focus_days: list[int],
    out_file: Path,
    x_start_hour: float | None = None,
    x_end_hour: float | None = None,
    title: str | None = None,
) -> None:
    """
    Compact focused paper-oriented Gantt.

    Plots only the selected focus days for the scenario, without baseline rows.
    If x_start_hour / x_end_hour are not given, the visible time window is
    inferred from the displayed scenario tours only.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(focus_days)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(7.2, 0.95 * n_rows + 0.75),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    if n_rows == 1:
        axes = [axes]

    color_on_time = "#9fb5cc"
    color_affected = "#d8c58d"
    color_late = "#8f5a5a"
    sep_color = "#c7c7c7"
    grid_color = "#d9d9d9"

    LABEL_FS = 8
    TICK_FS = 8
    TITLE_FS = 10
    LEGEND_FS = 7.5

    # ------------------------------------------------------------
    # Determine plot window from the scenario days actually shown.
    # ------------------------------------------------------------
    shown = scenario_tour_df[scenario_tour_df["day"].isin(focus_days)].copy()

    if len(shown):
        start_hours = (shown["start_time_sec"].astype(float) % DAY_SEC) / 3600
        end_hours = (shown["end_time_sec"].astype(float) % DAY_SEC) / 3600

        if x_start_hour is None:
            x_start_hour = max(5.0, np.floor(float(start_hours.min())))

        if x_end_hour is None:
            latest = float(end_hours.max())
            x_end_hour = min(17.0, np.ceil((latest + 0.20) * 2) / 2)

            if x_end_hour - x_start_hour < 4.0:
                x_end_hour = min(17.0, x_start_hour + 4.0)
    else:
        if x_start_hour is None:
            x_start_hour = 5.0
        if x_end_hour is None:
            x_end_hour = 13.0

    def draw_day(
        ax,
        tour_df: pd.DataFrame,
        *,
        day: int,
        row_label: str,
    ) -> None:
        tours_d = tour_df[tour_df["day"] == day].copy()

        pickers_d = sorted(tours_d["picker_id"].unique())
        picker_to_y = {pid: i for i, pid in enumerate(pickers_d)}
        n_pickers = len(pickers_d)

        for _, r in tours_d.iterrows():
            picker_id = r["picker_id"]
            y = picker_to_y[picker_id]

            start_h = (float(r["start_time_sec"]) % DAY_SEC) / 3600
            end_h = (float(r["end_time_sec"]) % DAY_SEC) / 3600

            plot_start_h = max(start_h, x_start_hour)
            plot_end_h = min(end_h, x_end_hour)
            duration_h = plot_end_h - plot_start_h

            if duration_h <= 0:
                continue

            is_late = int(r["late_split_orders"]) > 0
            is_affected = tour_contains_shifted_or_affected_order(r)

            bar_height = 0.78

            if is_late and is_affected:
                ax.barh(
                    y=y,
                    width=duration_h,
                    left=plot_start_h,
                    height=bar_height,
                    facecolor=color_late,
                    edgecolor=color_affected,
                    hatch="////",
                    linewidth=0.0,
                    alpha=0.92,
                    zorder=3,
                )

            elif is_late:
                ax.barh(
                    y=y,
                    width=duration_h,
                    left=plot_start_h,
                    height=bar_height,
                    facecolor=color_late,
                    edgecolor="none",
                    alpha=0.88,
                    zorder=2,
                )

            elif is_affected:
                ax.barh(
                    y=y,
                    width=duration_h,
                    left=plot_start_h,
                    height=bar_height,
                    facecolor=color_affected,
                    edgecolor="none",
                    alpha=0.88,
                    zorder=2,
                )

            else:
                ax.barh(
                    y=y,
                    width=duration_h,
                    left=plot_start_h,
                    height=bar_height,
                    facecolor=color_on_time,
                    edgecolor="none",
                    alpha=0.72,
                    zorder=2,
                )

        ax.set_xlim(x_start_hour, x_end_hour)

        if n_pickers > 0:
            ax.set_ylim(-0.6, n_pickers - 0.4)
        else:
            ax.set_ylim(-0.5, 0.5)

        ax.set_yticks([])

        ax.set_ylabel(
            f"{row_label}\n#P: {n_pickers}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=18,
            fontsize=LABEL_FS,
        )

        ax.grid(axis="x", color=grid_color, alpha=0.35, linewidth=0.6)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(sep_color)
        ax.spines["bottom"].set_linewidth(0.6)

        ax.tick_params(axis="x", labelsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)

    for row, day in enumerate(focus_days):
        ax = axes[row]

        draw_day(
            ax,
            scenario_tour_df,
            day=day,
            row_label=f"D{day + 1}",
        )

        if row < n_rows - 1:
            ax.tick_params(axis="x", labelbottom=False)

        ax.spines["bottom"].set_color("#a8a8a8")
        ax.spines["bottom"].set_linewidth(0.8)

    # Tick spacing adapts to window length.
    window = x_end_hour - x_start_hour
    tick_step = 1 if window <= 6 else 2

    xticks = np.arange(
        np.ceil(x_start_hour),
        np.floor(x_end_hour) + 1,
        tick_step,
    )

    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([f"{int(h):02d}:00" for h in xticks], fontsize=TICK_FS)
    axes[-1].set_xlabel("Time", fontsize=LABEL_FS)

    legend_handles = [
        Patch(facecolor=color_on_time, edgecolor="none", alpha=0.72, label="On time"),
        Patch(facecolor=color_affected, edgecolor="none", alpha=0.88, label="Affected"),
        Patch(facecolor=color_late, edgecolor="none", alpha=0.88, label="Late"),
        Patch(
            facecolor=color_late,
            edgecolor=color_affected,
            hatch="////",
            linewidth=0.0,
            alpha=0.92,
            label="Affected + late",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=LEGEND_FS,
    )

    # if title:
    #     fig.suptitle(title, y=1.03, fontsize=TITLE_FS)

    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        top=0.82 if title else 0.88,
        bottom=0.18,
    )

    fig.savefig(out_file, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


PROBLEM_ORDER = ["OBP", "ORSP", "RORSP"]


def _classify_problem(problem_class, replanning: str, pipeline: str) -> str:
    """Map a decision to its problem class.

    ``problem_class`` plus ``replanning`` is authoritative: ORSP with
    ``replanning='unstarted'`` is the former RORSP. The pipeline suffix is
    only a fallback for legacy traces that lack the replanning field.
    """
    s = str(problem_class).upper()

    if "OBP" in s:
        return "OBP"
    if "ORSP" in s or "RORSP" in s:
        if str(replanning).lower() == "unstarted" or "RORSP" in s:
            return "RORSP"
        return "ORSP"

    if pipeline.endswith("batching_sol"):
        return "OBP"
    if pipeline.endswith("scheduling_sol"):
        return "ORSP"

    return "OTHER"


def load_decision_counts(decisions_file: Path) -> dict[str, int]:
    decisions = [
        json.loads(line)
        for line in decisions_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    counts = {p: 0 for p in PROBLEM_ORDER}
    counts["OTHER"] = 0

    for decision in decisions:
        problem_class = decision["problem"]
        replanning = decision.get("replanning", "none")
        pipeline = decision["pipeline"]
        counts[_classify_problem(problem_class, replanning, pipeline)] += 1

    if counts["OTHER"] > 0:
        raw = {str(decision["problem"]) for decision in decisions}
        print(f"WARNING {decisions_file.parent.name}: "
              f"{counts['OTHER']} unclassified decisions. "
              f"Raw problem_class values seen: {sorted(raw)}")

    return counts


def build_decision_table(
    *,
    results_root: Path,
    cosy_repo: str,
    objective: str,
    scenarios: list[str],
) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        f = results_root / cosy_repo / objective / scenario / "decisions.jsonl"
        c = load_decision_counts(f)
        rows.append({
            "scenario_label": scenario_label(scenario),
            "OBP": c["OBP"],
            "ORSP": c["ORSP"],
            "RORSP": c["RORSP"],
            "Total": c["OBP"] + c["ORSP"] + c["RORSP"],
        })
    return pd.DataFrame(rows)


def make_decision_latex(table: pd.DataFrame, *, caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Setting & OBP & ORSP & RORSP & Total \\",
        r"\midrule",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"{r['scenario_label']} & {int(r['OBP'])} & {int(r['ORSP'])} & "
            f"{int(r['RORSP'])} & {int(r['Total'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
def evaluate_all(
    *,
    results_root: str | Path,
    out_dir: str | Path,
    scenarios: list[str],
    cosy_repo: str,
    objective: str,
    n_days: int,
) -> None:
    results_root = Path(results_root)
    out_dir = Path(out_dir / objective)

    table_dir = out_dir / "tables"
    figure_dir = out_dir / "figures"
    data_dir = out_dir / "data"

    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    day_rows = []
    impact_rows = []

    scenario_data: dict[str, dict[str, pd.DataFrame]] = {}

    # First pass: load all scenarios and create order-level data.
    for scenario in scenarios:
        kpi_dir = results_root / cosy_repo / objective / scenario / "kpis"

        tour_df, picker_df, _ = load_kpi_files(kpi_dir)
        order_df = explode_tour_execution_to_orders(tour_df)

        scenario_data[scenario] = {
            "tour_df": tour_df,
            "picker_df": picker_df,
            "order_df": order_df,
        }

        summary_rows.append(
            compute_summary_row(
                scenario=scenario,
                tour_df=tour_df,
                picker_df=picker_df,
            )
        )

        day_rows.extend(
            compute_day_rows(
                scenario=scenario,
                tour_df=tour_df,
                picker_df=picker_df,
                n_days=n_days,
            )
        )

        order_df.to_csv(data_dir / f"order_execution_{scenario}.csv", index=False)

        plot_gantt_for_scenario(
            tour_df=tour_df,
            out_file=figure_dir / f"gantt_{scenario}.pdf",
            title=f"{scenario_label(scenario)}",
            n_days=n_days,
        )

    focus_days_by_scenario = scenario_focus_days()

    for scenario, focus_days in focus_days_by_scenario.items():
        if scenario not in scenario_data:
            continue

        plot_focused_gantt_for_scenario(
            baseline_tour_df=scenario_data["baseline"]["tour_df"],
            scenario_tour_df=scenario_data[scenario]["tour_df"],
            scenario=scenario,
            focus_days=focus_days,
            out_file=figure_dir / f"gantt_focused_{scenario}.pdf",
            title=scenario_focus_title(scenario),
        )

        plot_focused_gantt_for_scenario_day(
            scenario_tour_df=scenario_data[scenario]["tour_df"],
            focus_days=focus_days,
            out_file=figure_dir / f"gantt_focused_{scenario}_day.pdf",
            title=scenario_focus_title(scenario),
        )

    # Fixed scheduled departure grid across all scenarios.
    # This prevents Scenario 1/5 from having fewer truck departures just
    # because one due-time slot becomes empty after shifting orders.
    planned_due_times = sorted({
        float(due_time)
        for data in scenario_data.values()
        for due_time in data["order_df"]["due_time_sec"].dropna().unique()
    })

    pd.DataFrame({
        "due_time_sec": planned_due_times,
        "due_day": [int(t // DAY_SEC) for t in planned_due_times],
        "due_clock": [sec_to_clock(t) for t in planned_due_times],
    }).to_csv(data_dir / "planned_departures.csv", index=False)

    # Second pass: compute truck waiting against fixed departure grid.
    for scenario, data in scenario_data.items():
        order_df = data["order_df"]

        truck_df = compute_truck_waiting_times(
            scenario=scenario,
            order_df=order_df,
            planned_due_times=planned_due_times,
        )

        truck_df.to_csv(data_dir / f"truck_waiting_{scenario}.csv", index=False)

        impact_rows.append(
            compute_impact_row(
                scenario=scenario,
                order_df=order_df,
                truck_df=truck_df,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    day_df = pd.DataFrame(day_rows)
    impact_df = pd.DataFrame(impact_rows)

    summary_df.to_csv(data_dir / "summary_kpis.csv", index=False)
    day_df.to_csv(data_dir / "day_kpis.csv", index=False)
    impact_df.to_csv(data_dir / "impact_kpis.csv", index=False)

    with open(data_dir / "summary_kpis.json", "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    with open(data_dir / "impact_kpis.json", "w", encoding="utf-8") as f:
        json.dump(impact_df.to_dict(orient="records"), f, indent=2)

    decision_table = build_decision_table(
        results_root=results_root,
        cosy_repo=cosy_repo,
        objective=objective,
        scenarios=scenarios,
    )
    decision_table.to_csv(data_dir / "decision_counts.csv", index=False)

    (table_dir / "decision_engine.tex").write_text(
        make_decision_latex(
            decision_table,
            caption=(
                "Number of decisions taken by the \\texttt{CoSySolver} per problem "
                "class. OBP covers the pre-shift item assignment and batching handed "
                "over by the WMS. ORSP covers routing and scheduling solved per shift. "
                "RORSP covers re-planning of unstarted tours (ORSP with "
                "replanning=unstarted) and is triggered only by disruptions, "
                "so it does not occur in the baseline. Picker routing is "
                "fixed to the U-shaped heuristic and scheduling to earliest "
                "due date under a tardiness objective."
            ),
            label="tab:decision_engine",
        ),
        encoding="utf-8",
    )

    paper_summary = make_paper_summary_table(summary_df)
    paper_day = make_paper_day_table(day_df)
    paper_impact = make_paper_impact_table(impact_df)

    write_latex_table(
        paper_summary,
        out_file=table_dir / "summary_kpis.tex",
        caption="Scenario-level KPI summary.",
        label="tab:summary-kpis",
    )

    write_latex_table(
        paper_day,
        out_file=table_dir / "day_kpis.tex",
        caption="Day-level KPI summary.",
        label="tab:day-kpis",
    )

    write_latex_table(
        paper_impact,
        out_file=table_dir / "impact_kpis.tex",
        caption=(
            "Scenario impact on scheduled truck departures and shifted orders. "
            "Truck waiting times are computed over a fixed departure grid across all scenarios."
        ),
        label="tab:impact-kpis",
    )

    print(f"Wrote KPI data to: {data_dir}")
    print(f"Wrote LaTeX tables to: {table_dir}")
    print(f"Wrote Gantt figures to: {figure_dir}")


def main() -> None:
    results_root = Path("../outputs/multirun/paper_results")
    out_dir = Path("../outputs/paper_kpi_eval")

    scenarios = [
        "baseline",
        "scenario1",
        "scenario2",
        "scenario3",
        "scenario4",
    ]

    evaluate_all(
        results_root=results_root,
        out_dir=out_dir,
        scenarios=scenarios,
        cosy_repo="ushape",
        objective="tardiness",
        n_days=6,
    )


if __name__ == "__main__":
    main()
