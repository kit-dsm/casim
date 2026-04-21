import argparse
import pickle
from pathlib import Path

import dash
import networkx as nx
import plotly.graph_objects as go
from dash import Input, Output, Patch, State, ctx, dcc, html

from casim.viz import kpis

BG, SURFACE, TEXT, MUTED, GRID, EDGE, NODE, ACCENT = (
    '#0a0e14', '#131820', '#e2e8f0', '#64748b',
    '#1e293b', '#475569', '#64748b', '#00d4ff',
)
PICKER_COLORS = [
    '#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3', '#f38181',
    '#aa96da', '#fcbad3', '#a8d8ea', '#f9ed69', '#b8de6f',
]

STATUS_COLORS = {
    "outstanding": '#f59e0b',  # amber
    "batched":     '#7c3aed',  # violet
    "assigned":    '#3b82f6',  # blue
    "in_progress": '#10b981',  # green
    "completed":   '#64748b',  # muted
    "unknown":     '#ef4444',  # red
}
STATUS_ORDER = ["in_progress", "assigned", "batched", "outstanding", "completed"]
MAX_TABLE_ROWS = 60


def load_bundle(viz_dir: Path):
    with open(viz_dir / "events.pkl", "rb") as f:
        events = pickle.load(f)
    with open(viz_dir / "static.pkl", "rb") as f:
        static = pickle.load(f)
    return events, static


def picker_pos(p):
    loc = p.current_location
    if hasattr(loc, "position"):
        loc = loc.position
    return loc[0], loc[1]


def build_figure(layout, storage, first_snapshot, colors):
    G = layout.layout_network.graph
    pos = nx.get_node_attributes(G, "pos")

    ex, ey = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]; ey += [y0, y1, None]

    sx, sy = [], []
    for slot in storage:
        if getattr(slot, "bbox", None) is None:
            continue
        b = slot.bbox
        sx += [b.x_min, b.x_max, b.x_max, b.x_min, b.x_min, None]
        sy += [b.y_min, b.y_min, b.y_max, b.y_max, b.y_min, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(  # 0: storage
        x=sx, y=sy, mode='lines', fill='toself',
        fillcolor='rgba(70,130,180,0.4)',
        line=dict(color='rgba(70,130,180,0.8)', width=1),
        hoverinfo='skip', showlegend=False,
    ))
    fig.add_trace(go.Scatter(  # 1: edges
        x=ex, y=ey, mode='lines', line=dict(width=2, color=EDGE),
        hoverinfo='skip', showlegend=False,
    ))
    nx_, ny_ = zip(*pos.values()) if pos else ([], [])
    fig.add_trace(go.Scatter(  # 2: nodes
        x=list(nx_), y=list(ny_), mode='markers',
        marker=dict(size=6, color=NODE),
        hoverinfo='skip', showlegend=False,
    ))

    picker_idx = {}
    for i, p in enumerate(first_snapshot["pickers"]):
        x, y = p["position"]
        picker_idx[p["id"]] = 3 + i
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=20, color=colors[p["id"]],
                        line=dict(width=2, color='white')),
            text=[f"P{p["id"]}"], textposition='top center',
            textfont=dict(color=colors[p["id"]], family='JetBrains Mono'),
            hoverinfo='text', hovertext=f"Picker {p["id"]}", showlegend=False,
        ))

    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family='JetBrains Mono', color=TEXT),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False,
                   showticklabels=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False,
                   showticklabels=False),
        hovermode='closest', dragmode='pan', uirevision='const',
    )
    return fig, picker_idx


def kpi_row(label, value, color=TEXT):
    return html.Div([
        html.Div(label, style={'color': MUTED, 'fontSize': '0.7rem'}),
        html.Div(value, style={'color': color, 'fontSize': '1rem',
                               'fontWeight': '600'}),
    ], style={'marginBottom': '10px'})


def create_app(events, static):
    layout_data = static["layout"]
    storage = static.get("storage_locations", [])

    app = dash.Dash(__name__, title="Warehouse Viz", update_title=None)
    n = len(events)

    pids = [p["id"] for p in events[0]["pickers"]]
    colors = {pid: PICKER_COLORS[i % len(PICKER_COLORS)]
              for i, pid in enumerate(pids)}
    fig, picker_idx = build_figure(layout_data, storage, events[0], colors)

    app.layout = html.Div([
        dcc.Store(id='frame', data=0),
        dcc.Store(id='playing', data=False),
        dcc.Interval(id='tick', interval=100, disabled=True),

        html.Div([
            html.Span(id='event-label', style={'color': ACCENT}),
            html.Span(" | ", style={'color': MUTED}),
            html.Span(id='time-label', style={'color': MUTED}),
            html.Span(" | ", style={'color': MUTED}),
            html.Span(id='frame-label', style={'color': TEXT}),
        ], style={'padding': '12px 20px', 'backgroundColor': SURFACE}),

        # Graph + KPI overlay
        html.Div([
            dcc.Graph(id='graph', figure=fig,
                      config={'scrollZoom': True, 'displaylogo': False},
                      style={'height': '55vh'}),

            html.Div([
                html.Div("Orders", style={'color': TEXT, 'fontWeight': '600',
                                          'marginBottom': '8px',
                                          'borderBottom': f'1px solid {GRID}',
                                          'paddingBottom': '6px'}),
                html.Div(id='kpi-orders'),
                html.Div("Pickers", style={'color': TEXT, 'fontWeight': '600',
                                           'marginTop': '10px',
                                           'marginBottom': '8px',
                                           'borderBottom': f'1px solid {GRID}',
                                           'paddingBottom': '6px'}),
                html.Div(id='kpi-pickers'),
            ], style={
                'position': 'absolute', 'top': '15px', 'left': '15px',
                'backgroundColor': SURFACE, 'padding': '12px',
                'borderRadius': '8px', 'border': f'1px solid {GRID}',
                'minWidth': '200px', 'maxHeight': '50vh', 'overflowY': 'auto',
                'zIndex': '1000',
            }),
        ], style={'position': 'relative'}),

        # Order table
        html.Div([
            html.Div([
                html.Span("Orders", style={'color': TEXT, 'fontWeight': '600',
                                           'marginRight': '15px'}),
                dcc.Dropdown(
                    id='status-filter',
                    options=[{'label': 'All', 'value': 'all'}] + [
                        {'label': s, 'value': s} for s in STATUS_ORDER
                    ],
                    value='all', clearable=False,
                    style={'width': '140px', 'display': 'inline-block'},
                ),
                html.Span(id='table-count', style={
                    'color': MUTED, 'marginLeft': '15px', 'fontSize': '0.8rem',
                }),
            ], style={'display': 'flex', 'alignItems': 'center',
                      'padding': '8px 15px',
                      'borderBottom': f'1px solid {GRID}'}),
            html.Div(id='order-table', style={
                'overflowY': 'auto', 'maxHeight': 'calc(30vh - 50px)',
                'padding': '8px 15px',
            }),
        ], style={
            'backgroundColor': SURFACE, 'height': '30vh',
            'borderTop': f'1px solid {GRID}',
        }),

        # Controls
        html.Div([
            html.Button('▶', id='play', n_clicks=0,
                        style={'width': '50px', 'height': '36px'}),
            dcc.Slider(id='slider', min=0, max=n - 1, step=1, value=0,
                       marks=None, tooltip={'always_visible': False},
                       updatemode='mouseup'),
            dcc.Dropdown(id='speed', value=100, clearable=False,
                         options=[{'label': lbl, 'value': v} for lbl, v in [
                             ('Max', 10), ('20 fps', 50), ('10 fps', 100),
                             ('5 fps', 200), ('1 fps', 1000)]],
                         style={'width': '110px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '15px',
                  'padding': '12px 20px', 'backgroundColor': SURFACE}),
    ], style={'backgroundColor': BG, 'color': TEXT, 'height': '100vh',
              'fontFamily': 'JetBrains Mono, monospace',
              'display': 'flex', 'flexDirection': 'column'})

    app.index_string = '''<!DOCTYPE html><html><head>{%metas%}
<title>{%title%}</title>{%favicon%}{%css%}
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>*{box-sizing:border-box;margin:0}body{overflow:hidden}</style>
</head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

    @app.callback(
        Output('graph', 'figure'),
        Output('event-label', 'children'),
        Output('time-label', 'children'),
        Output('frame-label', 'children'),
        Output('kpi-orders', 'children'),
        Output('kpi-pickers', 'children'),
        Output('order-table', 'children'),
        Output('table-count', 'children'),
        Input('frame', 'data'),
        Input('status-filter', 'value'),
    )
    def render(frame, status_filter):
        snap = events[frame]

        patched = Patch()
        for p in snap["pickers"]:
            x, y = p["position"]
            patched["data"][picker_idx[p["id"]]]["x"] = [x]
            patched["data"][picker_idx[p["id"]]]["y"] = [y]

        event_label = f"{snap['event_type']} #{snap['event_id']:05d}"
        time_label = f"t = {snap['time']}"
        frame_label = f"{frame + 1} / {n}"

        # KPIs
        s = kpis.summary(snap)
        order_panel = [
            kpi_row("Outstanding", s["outstanding"], STATUS_COLORS["outstanding"]),
            kpi_row("Batched", s["batched"], STATUS_COLORS["batched"]),
            kpi_row("In progress", s["in_progress"], STATUS_COLORS["in_progress"]),
            kpi_row("Completed", s["completed"], STATUS_COLORS["completed"]),
        ]

        picker_panel = []
        for pid in sorted(pids):
            tour = kpis.active_tour_for_picker(snap, pid)
            if tour is None:
                status_text, status_color = "idle", MUTED
            else:
                tid = snap['active_picker_tour'][pid]
                status_text = f"tour {tid} · {len(tour['order_ids'])} orders"
                status_color = TEXT
            picker_panel.append(html.Div([
                html.Span("●", style={'color': colors[pid], 'marginRight': '6px'}),
                html.Span(f"P{pid}: ", style={'color': MUTED, 'fontSize': '0.8rem'}),
                html.Span(status_text, style={'color': status_color, 'fontSize': '0.8rem'}),
            ], style={'marginBottom': '5px'}))

        # Order table
        status_map = kpis.order_status_map(snap)
        rank = {s: i for i, s in enumerate(STATUS_ORDER)}
        items = sorted(status_map.items(),
                       key=lambda kv: (rank.get(kv[1], 99), kv[0]))
        if status_filter != 'all':
            items = [kv for kv in items if kv[1] == status_filter]
        total = len(items)
        shown = items[:MAX_TABLE_ROWS]

        rows = [_order_row(oid, st) for oid, st in shown]
        count_label = (f"{total} shown" if total <= MAX_TABLE_ROWS
                       else f"{MAX_TABLE_ROWS} of {total} shown")

        return (patched, event_label, time_label, frame_label,
                order_panel, picker_panel, rows, count_label)

    @app.callback(
        Output('frame', 'data'),
        Output('playing', 'data'),
        Output('play', 'children'),
        Output('tick', 'disabled'),
        Output('slider', 'value'),
        Input('play', 'n_clicks'),
        Input('tick', 'n_intervals'),
        Input('slider', 'value'),
        State('frame', 'data'),
        State('playing', 'data'),
        prevent_initial_call=True,
    )
    def control(_, __, slider_val, frame, playing):
        trig = ctx.triggered_id
        if trig == 'slider':
            return slider_val, False, '▶', True, slider_val
        if trig == 'play':
            new_play = not playing
            f = 0 if (new_play and frame >= n - 1) else frame
            return f, new_play, '⏸' if new_play else '▶', not new_play, f
        if trig == 'tick' and playing:
            f = frame + 1
            if f >= n:
                return n - 1, False, '▶', True, n - 1
            return f, True, '⏸', False, f
        return (dash.no_update,) * 5

    @app.callback(Output('tick', 'interval'), Input('speed', 'value'))
    def set_speed(v):
        return v

    return app

def _order_row(oid, status):
    return html.Div([
        html.Span(f"#{oid}", style={
            'color': TEXT, 'fontSize': '0.8rem',
            'minWidth': '80px', 'display': 'inline-block',
        }),
        html.Span(status, style={
            'color': STATUS_COLORS.get(status, TEXT),
            'fontSize': '0.75rem', 'padding': '2px 8px',
            'borderRadius': '4px',
            'backgroundColor': f'{STATUS_COLORS.get(status, TEXT)}22',
            'border': f'1px solid {STATUS_COLORS.get(status, TEXT)}',
        }),
    ], style={
        'display': 'flex', 'alignItems': 'center', 'gap': '10px',
        'padding': '4px 0', 'borderBottom': f'1px solid {GRID}',
    })


def launch(viz_dir: Path, port: int = 8050, debug: bool = False):
    events, static = load_bundle(Path(viz_dir))
    print(f"Loaded {len(events)} frames from {viz_dir}")
    print(f"http://127.0.0.1:{port}")
    create_app(events, static).run(debug=debug, port=port)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("viz_dir", type=Path)
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    launch(args.viz_dir, args.port)