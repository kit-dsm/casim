# casim/viz/plots.py
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
            row=2,
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