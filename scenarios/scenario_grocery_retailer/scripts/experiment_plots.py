from __future__ import annotations
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Patch


def _set_paper_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        palette="colorblind",
        font_scale=1.0,
    )

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.2,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.6,
        }
    )


def _tour_color_map(tour_ids: List[int]) -> Dict[int, Tuple[float, float, float]]:
    unique_tours = sorted(set(tour_ids))
    palette = sns.color_palette("colorblind", n_colors=max(len(unique_tours), 1))
    return {tour_id: palette[i % len(palette)] for i, tour_id in enumerate(unique_tours)}


def _plot_step_series(
    ax,
    x: List[float],
    y: List[float],
    ylabel: str,
    title: str,
) -> None:
    color = sns.color_palette("colorblind", 1)[0]

    ax.step(x, y, where="post", color=color)
    ax.set_title(title)
    ax.set_xlabel("Time (in Seconds)")
    ax.set_ylabel(ylabel)
    ax.margins(x=0.01)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")


def dock_fill_level_plot(
    tracker) -> Figure:
    _set_paper_style()

    dock_util_time = tracker.dock_utilization
    x_timestamps = [x for x, _ in dock_util_time]
    y_dock_capacity = [y for _, y in dock_util_time]

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    _plot_step_series(
        ax=ax,
        x=x_timestamps,
        y=y_dock_capacity,
        ylabel="Dock Fill Level",
        title=""
    )
    fig.tight_layout()
    return fig


def avg_makespan_plot(
    tracker,
) -> Figure:
    _set_paper_style()

    avg_makespan = tracker.avg_makespan
    x_tour_finish_times = [time for time, _ in avg_makespan]
    y_makespans = [makespan for _, makespan in avg_makespan]

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    _plot_step_series(
        ax=ax,
        x=x_tour_finish_times,
        y=y_makespans,
        ylabel="Average Makespan",
        title="",
    )
    fig.tight_layout()
    return fig


def picker_schedule_plot(
    tracker,
    title: str = "Picker Schedule",
    show_legend: bool = False,
) -> Figure:
    _set_paper_style()

    completed_tours = tracker.completed_tours
    idle_intervals = tracker.idle_intervals

    pickers = sorted(set(pid for _, _, _, _, pid, _, _ in completed_tours))
    picker_labels = {pid: f"P{pid}" for pid in pickers}
    y_positions = {pid: i for i, pid in enumerate(pickers)}

    tour_ids = [tour_id for tour_id, *_ in completed_tours]
    color_map = _tour_color_map(tour_ids)

    n_pickers = len(pickers)
    fig_height = min(max(2.6, 0.24 * n_pickers + 1.0), 6.2)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))

    bar_height = 0.62 if n_pickers <= 12 else 0.54

    for pid, start, end in idle_intervals:
        if pid not in y_positions:
            continue
        ax.barh(
            y=y_positions[pid],
            width=end - start,
            left=start,
            height=bar_height,
            color="0.88",
            edgecolor="0.55",
            hatch="///",
            linewidth=0.5,
            label="Idle" if pid == pickers[0] else None,
            zorder=1,
        )

    seen_tours_for_legend = set()
    for tour_id, start, end, order_ids, pid, _, _ in completed_tours:
        label = None
        if show_legend and tour_id not in seen_tours_for_legend:
            label = f"Tour {tour_id}"
            seen_tours_for_legend.add(tour_id)

        ax.barh(
            y=y_positions[pid],
            width=end - start,
            left=start,
            height=bar_height,
            color=color_map[tour_id],
            edgecolor="black",
            linewidth=0.35,
            label=label,
            zorder=2,
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Picker")
    ax.set_yticks([y_positions[pid] for pid in pickers])
    ax.set_yticklabels([picker_labels[pid] for pid in pickers])
    ax.set_ylim(-0.45, n_pickers - 0.55)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="-", alpha=0.18)
    ax.grid(axis="y", visible=False)

    if show_legend:
        ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    else:
        idle_patch = Patch(facecolor="0.88", edgecolor="0.55", hatch="///", label="Idle")
        ax.legend(handles=[idle_patch], frameon=False, loc="upper right", borderaxespad=0.2)

    fig.tight_layout(pad=0.6)
    return fig


def picker_schedule_plots(tracker) -> Dict[str, Figure]:
    return {
        "dock_fill_level": dock_fill_level_plot(tracker),
        "avg_makespan": avg_makespan_plot(tracker),
        "picker_schedule": picker_schedule_plot(tracker),
    }

