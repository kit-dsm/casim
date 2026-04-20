# casim/viz/plots.py
import plotly.graph_objects as go


TOUR_COLORS = [
    '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
]


def gantt_chart(tracker, title="Picker Schedule") -> go.Figure:
    pickers = sorted(set(pid for _, _, _, _, pid in tracker.completed_tours))
    picker_labels = {pid: f"Picker {pid}" for pid in pickers}

    fig = go.Figure()

    # idle bars (background, grey)
    for pid, start, end in tracker.idle_intervals:
        fig.add_trace(go.Bar(
            y=[picker_labels[pid]],
            x=[end - start],
            base=[start],
            orientation='h',
            marker=dict(color='#1e293b', line=dict(color='#334155', width=1)),
            hovertext=f"Idle: {end - start:.0f}",
            hoverinfo='text',
            showlegend=False,
        ))

    # tour bars
    for tour_id, start, end, order_ids, pid in tracker.completed_tours:
        color = TOUR_COLORS[tour_id % len(TOUR_COLORS)]
        fig.add_trace(go.Bar(
            y=[picker_labels[pid]],
            x=[end - start],
            base=[start],
            orientation='h',
            marker=dict(color=color, line=dict(color='white', width=0.5)),
            hovertext=(
                f"Tour {tour_id}<br>"
                f"Duration: {end - start:.0f}<br>"
                f"Orders: {len(order_ids)} ({', '.join(str(o) for o in order_ids[:5])})"
                f"{'...' if len(order_ids) > 5 else ''}"
            ),
            hoverinfo='text',
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        barmode='overlay',
        plot_bgcolor='#0a0e14',
        paper_bgcolor='#0a0e14',
        font=dict(family='JetBrains Mono', color='#e2e8f0'),
        xaxis=dict(title='Time', gridcolor='#1e293b', zeroline=False),
        yaxis=dict(autorange='reversed'),
        height=max(200, 80 * len(pickers)),
        margin=dict(l=100, r=20, t=50, b=40),
    )

    return fig