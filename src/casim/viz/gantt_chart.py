# casim/viz/plots.py
from typing import Any, Sequence, Callable

import plotly.graph_objects as go
from plotly.subplots import make_subplots

TOUR_COLORS = [
    '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
]


def gantt_chart(tracker, title="Picker Schedule") -> go.Figure:
    pickers = sorted(set(pid for _, _, _, _, pid, _, _ in tracker.completed_tours))
    picker_labels = {pid: f"Picker {pid}" for pid in pickers}
    avg_makespan = tracker.avg_makespan

    y_makespans = [makespan for _, makespan in avg_makespan]
    x_tour_finish_times = [time for time, _ in avg_makespan]
    dock_util_time = tracker.dock_utilization
    x_timestamps = [x for x, _ in dock_util_time]
    y_dock_capacity = [y for _, y in dock_util_time]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.25, 0.25, 0.5],   # top line plot, bottom gantt
    )

    # Dock-capacity line plot (top)
    fig.add_trace(
        go.Scatter(
            x=x_timestamps,
            y=y_dock_capacity,
            mode="lines+markers",
            name="Dock Fill Level",
            line=dict(color="#f97316", width=3),
            marker=dict(size=5),
            hovertemplate="Time: %{x}<br>Fill Level: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_tour_finish_times,
            y=y_makespans,
            mode="lines+markers",
            name="Avg. Makespan",
            line=dict(color="#f97316", width=3),
            marker=dict(size=5),
            hovertemplate="Time: %{x}<br>Avg. Makespan: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Idle bars
    for pid, start, end in tracker.idle_intervals:
        fig.add_trace(
            go.Bar(
                y=[picker_labels[pid]],
                x=[end - start],
                base=[start],
                orientation="h",
                marker=dict(color="#1e293b", line=dict(color="#334155", width=1)),
                hovertext=f"Idle: {end - start:.0f}",
                hoverinfo="text",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # Tour bars
    for tour_id, start, end, order_ids, pid, _, _ in tracker.completed_tours:
        color = TOUR_COLORS[tour_id % len(TOUR_COLORS)]
        fig.add_trace(
            go.Bar(
                y=[picker_labels[pid]],
                x=[end - start],
                base=[start],
                orientation="h",
                marker=dict(color=color, line=dict(color="white", width=0.5)),
                hovertext=(
                    f"Tour {tour_id}<br>"
                    f"Duration: {end - start:.0f}<br>"
                    f"Orders: {len(order_ids)} ({', '.join(str(o) for o in order_ids[:5])})"
                    f"{'...' if len(order_ids) > 5 else ''}"
                ),
                hoverinfo="text",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        title=title,
        barmode="overlay",
        plot_bgcolor="#0a0e14",
        paper_bgcolor="#0a0e14",
        font=dict(family="JetBrains Mono", color="#e2e8f0"),
        height=max(300, 100 + 80 * len(pickers)),
        margin=dict(l=100, r=20, t=50, b=40),
    )

    fig.update_yaxes(
        title_text="Dock Fill Level",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Avg. Makespan",
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="Time",
        gridcolor="#1e293b",
        zeroline=False,
        row=3,
        col=1,
    )



    fig.update_yaxes(
        autorange="reversed",
        row=3,
        col=1,
    )

    return fig



def plot_gantt_from_jobs(
        jobs: Sequence[Any],
        *,
        title: str = "Picker Schedule",
        row_key: Callable[[Any], Any] = lambda j: j.picker_id,
        color_key: Callable[[Any], str] | None = None,
        row_label: Callable[[Any], str] = lambda r: f"Picker {r}",
        hover_text: Callable[[Any], str] | None = None,
        time_scale: float = 1.0,
        xlabel: str = "Time",
) -> go.Figure:
    """Plot a Plotly Gantt chart of ScheduledJob instances.

    Each job becomes a horizontal bar from start_time to end_time on the row
    given by row_key(job). Default coloring is red for tardy, green for on-time.

    Parameters
    ----------
    jobs : sequence of ScheduledJob
        Must expose start_time, end_time, and the attribute used by row_key.
    title : str
        Figure title.
    row_key : callable
        Maps a job to its row identifier. Default: picker_id.
    color_key : callable, optional
        Maps a job to a color string. Default: red if tardiness > 0 else green.
    row_label : callable
        Formats a row identifier into a y-tick label.
    hover_text : callable, optional
        Maps a job to its hover text. Default shows start, end, duration, lateness.
    time_scale : float
        Divisor applied to start_time and end_time (e.g. 3600 for s -> h).
    xlabel : str
        X-axis title.
    """
    if color_key is None:
        def color_key(j: Any) -> str:
            return "#ef4444" if getattr(j, "tardiness", 0.0) > 0 else "#34d399"

    if hover_text is None:
        def hover_text(j: Any) -> str:
            s = j.start_time / time_scale
            e = j.end_time / time_scale
            lines = [
                f"Start: {s:.2f}",
                f"End: {e:.2f}",
                f"Duration: {e - s:.2f}",
            ]
            if hasattr(j, "tardiness"):
                lines.append(f"Tardiness: {j.tardiness:.2f}")
            if hasattr(j, "lateness"):
                lines.append(f"Lateness: {j.lateness:.2f}")
            return "<br>".join(lines)

    rows = sorted({row_key(j) for j in jobs}, key=lambda r: (r is None, r))
    labels = [row_label(r) for r in rows]

    fig = go.Figure()
    for j in jobs:
        start = j.start_time / time_scale
        end = j.end_time / time_scale
        fig.add_trace(
            go.Bar(
                y=[row_label(row_key(j))],
                x=[end - start],
                base=[start],
                orientation="h",
                marker=dict(color=color_key(j), line=dict(color="white", width=0.5)),
                hovertext=hover_text(j),
                hoverinfo="text",
                showlegend=False,
            )
        )

    n_rows = len(rows)
    row_px = 24  # px per picker
    height = 80 + row_px * n_rows  # 80 covers title + xaxis margins

    fig.update_layout(
        title=title,
        barmode="overlay",
        bargap=0.2,  # was the default; tightens bars vertically
        plot_bgcolor="#0a0e14",
        paper_bgcolor="#0a0e14",
        font=dict(family="JetBrains Mono", color="#e2e8f0"),
        height=height,
        margin=dict(l=100, r=20, t=40, b=40),
    )
    fig.update_xaxes(title_text=xlabel, gridcolor="#1e293b", zeroline=False)
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=labels,
        autorange="reversed",
    )
    return fig