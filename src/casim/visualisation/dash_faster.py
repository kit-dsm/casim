import pickle
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, ctx, Patch
import plotly.graph_objects as go
import networkx as nx

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

COLORS = {
    'background': '#0a0e14',
    'surface': '#131820',
    'surface_light': '#1a2130',
    'primary': '#00d4ff',
    'secondary': '#ff6b35',
    'accent': '#7c3aed',
    'text': '#e2e8f0',
    'text_muted': '#64748b',
    'grid': '#1e293b',
    'edge': '#334155',
    'node': '#475569',
    'success': '#10b981',
    'warning': '#f59e0b',
}

PICKER_COLORS = [
    '#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3', '#f38181',
    '#aa96da', '#fcbad3', '#a8d8ea', '#f9ed69', '#b8de6f',
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_simulation_data(log_path, domain_path):
    with open(log_path, "rb") as f:
        log = pickle.load(f)
    with open(domain_path, "rb") as f:
        domain = pickle.load(f)
    storage_locations = []
    if hasattr(domain, 'storage') and hasattr(domain.storage, 'storage_locations'):
        storage_locations = domain.storage.storage_locations
    return log, domain.layout, storage_locations


def extract_graph_data(layout):
    G = layout.layout_network.graph
    pos = nx.get_node_attributes(G, "pos")
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_labels = [f"({n[0]:.0f}, {n[1]:.0f})" for n in G.nodes()]
    return {
        'edge_x': edge_x, 'edge_y': edge_y,
        'node_x': node_x, 'node_y': node_y,
        'node_labels': node_labels, 'graph': G, 'pos': pos,
    }


def extract_storage_data(storage_locations):
    slots_data = []
    for slot in storage_locations:
        if hasattr(slot, 'bbox') and slot.bbox is not None:
            slots_data.append({
                'id': slot.id, 'bbox': slot.bbox,
                'aisle_id': getattr(slot, 'aisle_id', None),
                'slots': getattr(slot, 'slots', []),
            })
    return slots_data


def get_picker_location(p):
    loc = p.current_location
    if hasattr(loc, 'position'):
        loc = loc.position
    return (loc[0], loc[1])


def get_picker_data(snapshot):
    pickers = []
    for p in snapshot["pickers"].resources:
        x, y = get_picker_location(p)
        pickers.append({
            'id': p.id, 'x': x, 'y': y,
            'status': getattr(p, 'tpe', getattr(p, 'occupied', 'unknown')),
        })
    return pickers


# ============================================================================
# PRE-COMPUTATION — lightweight, no dict copies per frame
# ============================================================================

def precompute_all(log, picker_ids):
    """Pre-compute cumulative scalars only. Memory: O(n*k + n)."""
    n = len(log)
    cumulative_distance = {pid: [0.0] * n for pid in picker_ids}
    times = [None] * n
    events = [None] * n
    event_type_per_frame = [None] * n
    picks_cumulative = [0] * n
    orders_cumulative = [0] * n

    prev_pos = {}
    running_picks = 0
    running_orders = 0

    for i, snapshot in enumerate(log):
        times[i] = snapshot['time']
        events[i] = str(snapshot['event'])

        event_str = events[i]
        event_type = event_str.split(' - ')[0] if ' - ' in event_str else event_str
        event_type_per_frame[i] = event_type

        event_lower = event_str.lower()
        if 'pick' in event_lower and ('complete' in event_lower or 'finish' in event_lower):
            running_picks += 1
        if 'order' in event_lower and ('complete' in event_lower or 'finish' in event_lower):
            running_orders += 1
        picks_cumulative[i] = running_picks
        orders_cumulative[i] = running_orders

        for p in snapshot["pickers"].resources:
            pid = p.id
            pos = get_picker_location(p)
            if i == 0:
                cumulative_distance[pid][0] = 0.0
            else:
                prev = prev_pos.get(pid, pos)
                dx = pos[0] - prev[0]
                dy = pos[1] - prev[1]
                cumulative_distance[pid][i] = (
                    cumulative_distance[pid][i-1] + (dx**2 + dy**2) ** 0.5
                )
            prev_pos[pid] = pos

    return {
        'cumulative_distance': cumulative_distance,
        'times': times,
        'events': events,
        'event_type_per_frame': event_type_per_frame,
        'picks_cumulative': picks_cumulative,
        'orders_cumulative': orders_cumulative,
    }


def get_kpis_for_frame(precomputed, frame, picker_ids):
    """O(k) distance lookup + O(frame) event count from flat string list."""
    times = precomputed['times']
    cum_dist = precomputed['cumulative_distance']

    start_time, end_time = times[0], times[frame]
    if hasattr(start_time, 'total_seconds'):
        elapsed = (end_time - start_time).total_seconds()
    elif isinstance(start_time, (int, float)):
        elapsed = end_time - start_time
    else:
        try:
            elapsed = float(end_time) - float(start_time)
        except Exception:
            elapsed = frame

    total_distance = {pid: cum_dist[pid][frame] for pid in picker_ids}

    # Event counts on demand from flat array
    event_counts = {}
    for i in range(frame + 1):
        et = precomputed['event_type_per_frame'][i]
        event_counts[et] = event_counts.get(et, 0) + 1

    return {
        'total_distance': total_distance,
        'total_distance_all': sum(total_distance.values()),
        'elapsed_time': elapsed,
        'picks_completed': precomputed['picks_cumulative'][frame],
        'orders_completed': precomputed['orders_cumulative'][frame],
        'events_count': event_counts,
    }


# ============================================================================
# FIGURE CREATION (built once, updated via Patch)
# ============================================================================

def build_batched_storage_trace(storage_data):
    """All storage rectangles in ONE trace with None separators."""
    if not storage_data:
        return None
    sx, sy = [], []
    for slot in storage_data:
        bbox = slot['bbox']
        sx.extend([bbox.x_min, bbox.x_max, bbox.x_max, bbox.x_min, bbox.x_min, None])
        sy.extend([bbox.y_min, bbox.y_min, bbox.y_max, bbox.y_max, bbox.y_min, None])
    return go.Scatter(
        x=sx, y=sy, mode='lines', fill='toself',
        fillcolor='rgba(70, 130, 180, 0.4)',
        line=dict(color='rgba(70, 130, 180, 0.8)', width=1),
        hoverinfo='skip', showlegend=False, name='Storage',
    )


def create_initial_figure(graph_data, pickers, picker_colors, storage_data=None,
                          show_storage=True):
    """Build once. Trace indices: 0=storage, 1=edges, 2=nodes, 3+=pickers."""
    fig = go.Figure()

    st = build_batched_storage_trace(storage_data) if show_storage else None
    fig.add_trace(st if st else go.Scatter(
        x=[], y=[], mode='markers', showlegend=False, hoverinfo='skip'))

    fig.add_trace(go.Scatter(
        x=graph_data['edge_x'], y=graph_data['edge_y'],
        mode='lines', line=dict(width=2, color=COLORS['edge']),
        hoverinfo='none', name='Aisles', showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=graph_data['node_x'], y=graph_data['node_y'],
        mode='markers',
        marker=dict(size=8, color=COLORS['node'],
                    line=dict(width=1, color=COLORS['surface_light'])),
        text=graph_data['node_labels'],
        hoverinfo='text', name='Nodes', showlegend=False,
    ))

    picker_trace_indices = {}
    for i, picker in enumerate(pickers):
        color = picker_colors.get(picker['id'], COLORS['primary'])
        trace_idx = 3 + i
        picker_trace_indices[picker['id']] = trace_idx
        fig.add_trace(go.Scatter(
            x=[picker['x']], y=[picker['y']],
            mode='markers+text',
            marker=dict(size=20, color=color,
                        line=dict(width=2, color='white'), symbol='circle'),
            text=[f"P{picker['id']}"],
            textposition='top center',
            textfont=dict(size=12, color=color, family='JetBrains Mono, monospace'),
            hoverinfo='text',
            hovertext=(f"Picker {picker['id']}<br>Status: {picker['status']}<br>"
                       f"Pos: ({picker['x']:.0f}, {picker['y']:.0f})"),
            name=f"Picker {picker['id']}", showlegend=False,
        ))

    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(family='JetBrains Mono, monospace', color=COLORS['text']),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor=COLORS['grid'], gridwidth=1,
                   zeroline=False, showticklabels=False,
                   scaleanchor='y', scaleratio=1),
        yaxis=dict(showgrid=True, gridcolor=COLORS['grid'], gridwidth=1,
                   zeroline=False, showticklabels=False),
        showlegend=False, hovermode='closest', dragmode='pan',
        uirevision='constant',
    )

    return fig, picker_trace_indices


# ============================================================================
# DASH APP
# ============================================================================

def create_app(log, layout, storage_locations=None):
    app = dash.Dash(__name__, title="Warehouse Simulation Visualizer", update_title=None)

    graph_data = extract_graph_data(layout)
    storage_data = extract_storage_data(storage_locations) if storage_locations else []
    num_frames = len(log)

    first_snapshot = log[0]
    picker_ids = [p.id for p in first_snapshot["pickers"].resources]
    picker_colors = {pid: PICKER_COLORS[i % len(PICKER_COLORS)]
                     for i, pid in enumerate(picker_ids)}

    print("Pre-computing KPIs for all frames...")
    precomputed = precompute_all(log, picker_ids)
    print("Pre-computation done.")

    initial_pickers = get_picker_data(first_snapshot)
    initial_fig, picker_trace_indices = create_initial_figure(
        graph_data, initial_pickers, picker_colors, storage_data, show_storage=True,
    )

    app._wh_log = log
    app._wh_picker_colors = picker_colors
    app._wh_picker_ids = picker_ids
    app._wh_precomputed = precomputed
    app._wh_picker_trace_indices = picker_trace_indices
    app._wh_storage_data = storage_data

    # ========================================================================
    # LAYOUT
    # ========================================================================

    app.layout = html.Div([
        dcc.Store(id='current-frame', data=0),
        dcc.Store(id='is-playing', data=False),
        dcc.Interval(id='animation-interval', interval=100, disabled=True),

        html.Div([
            # Header
            html.Div([
                html.H1("🏭 Warehouse Simulation", style={
                    'margin': '0', 'fontSize': '1.5rem', 'fontWeight': '600',
                    'color': COLORS['text'],
                    'fontFamily': 'JetBrains Mono, monospace',
                }),
                html.Div([
                    html.Span(id='event-display', style={
                        'color': COLORS['primary'],
                        'fontFamily': 'JetBrains Mono, monospace',
                        'fontSize': '0.9rem',
                    }),
                    html.Span(" | ", style={
                        'color': COLORS['text_muted'], 'margin': '0 10px'}),
                    html.Span(id='time-display', style={
                        'color': COLORS['text_muted'],
                        'fontFamily': 'JetBrains Mono, monospace',
                        'fontSize': '0.9rem',
                    }),
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ], style={
                'display': 'flex', 'justifyContent': 'space-between',
                'alignItems': 'center', 'padding': '15px 20px',
                'backgroundColor': COLORS['surface'],
                'borderBottom': f'1px solid {COLORS["grid"]}',
            }),

            # Graph
            html.Div([
                dcc.Graph(
                    id='warehouse-graph',
                    figure=initial_fig,
                    config={
                        'displayModeBar': True, 'displaylogo': False,
                        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                        'scrollZoom': True,
                    },
                    style={'height': 'calc(100vh - 180px)'},
                ),
            ], style={'flex': '1', 'overflow': 'hidden'}),

            # Controls
            html.Div([
                html.Div([
                    html.Button('⏮', id='btn-first', n_clicks=0, className='control-btn'),
                    html.Button('◀', id='btn-prev', n_clicks=0, className='control-btn'),
                    html.Button('▶', id='btn-play', n_clicks=0, className='control-btn primary'),
                    html.Button('▶', id='btn-next', n_clicks=0, className='control-btn'),
                    html.Button('⏭', id='btn-last', n_clicks=0, className='control-btn'),
                ], style={'display': 'flex', 'gap': '8px', 'alignItems': 'center'}),

                html.Div([
                    html.Span(id='frame-display', style={
                        'color': COLORS['text'],
                        'fontFamily': 'JetBrains Mono, monospace',
                        'fontSize': '0.85rem', 'minWidth': '120px',
                    }),
                    dcc.Slider(
                        id='frame-slider',
                        min=0, max=num_frames - 1, step=1, value=0,
                        marks=None, tooltip={'always_visible': False},
                        updatemode='mouseup',
                    ),
                ], style={
                    'flex': '1', 'display': 'flex', 'alignItems': 'center',
                    'gap': '15px', 'margin': '0 20px',
                }),

                html.Div([
                    html.Label("Speed:", style={
                        'color': COLORS['text_muted'], 'fontSize': '0.85rem',
                        'marginRight': '8px',
                    }),
                    dcc.Dropdown(
                        id='speed-dropdown',
                        options=[
                            {'label': 'Max', 'value': 10},
                            {'label': '20 fps', 'value': 50},
                            {'label': '10 fps', 'value': 100},
                            {'label': '5 fps', 'value': 200},
                            {'label': '2 fps', 'value': 500},
                            {'label': '1 fps', 'value': 1000},
                        ],
                        value=100, clearable=False, style={'width': '100px'},
                    ),
                ], style={'display': 'flex', 'alignItems': 'center'}),

                html.Div([
                    html.Label("Storage:", style={
                        'color': COLORS['text_muted'], 'fontSize': '0.85rem',
                        'marginRight': '8px',
                    }),
                    dcc.Checklist(
                        id='storage-toggle',
                        options=[{'label': 'Show', 'value': 'show'}],
                        value=['show'], inline=True,
                        style={'color': COLORS['text']},
                    ),
                ], style={'display': 'flex', 'alignItems': 'center',
                          'marginLeft': '20px'}),
            ], style={
                'display': 'flex', 'alignItems': 'center', 'padding': '12px 20px',
                'backgroundColor': COLORS['surface'],
                'borderTop': f'1px solid {COLORS["grid"]}',
            }),
        ], style={
            'display': 'flex', 'flexDirection': 'column', 'height': '100vh',
            'backgroundColor': COLORS['background'],
        }),

        # Picker legend (floating right)
        html.Div([
            html.Div("Pickers", style={
                'color': COLORS['text'], 'fontWeight': '600',
                'marginBottom': '10px', 'fontSize': '0.9rem',
            }),
            html.Div(id='picker-legend'),
        ], style={
            'position': 'fixed', 'top': '80px', 'right': '20px',
            'backgroundColor': COLORS['surface'], 'padding': '15px',
            'borderRadius': '8px',
            'border': f'1px solid {COLORS["grid"]}', 'zIndex': '1000',
            'fontFamily': 'JetBrains Mono, monospace', 'minWidth': '180px',
        }),

        # KPI Panel (floating left)
        html.Div([
            html.Div("📊 KPIs", style={
                'color': COLORS['text'], 'fontWeight': '600',
                'marginBottom': '12px', 'fontSize': '0.9rem',
                'borderBottom': f'1px solid {COLORS["grid"]}',
                'paddingBottom': '8px',
            }),
            html.Div([
                html.Div("Elapsed Time", style={
                    'color': COLORS['text_muted'], 'fontSize': '0.75rem'}),
                html.Div(id='kpi-elapsed-time', style={
                    'color': COLORS['primary'], 'fontSize': '1.1rem',
                    'fontWeight': '600'}),
            ], style={'marginBottom': '12px'}),
            html.Div([
                html.Div("Total Distance", style={
                    'color': COLORS['text_muted'], 'fontSize': '0.75rem'}),
                html.Div(id='kpi-total-distance', style={
                    'color': COLORS['text'], 'fontSize': '0.95rem'}),
            ], style={'marginBottom': '12px'}),
            html.Div([
                html.Div("Events", style={
                    'color': COLORS['text_muted'], 'fontSize': '0.75rem'}),
                html.Div(id='kpi-events', style={
                    'color': COLORS['text'], 'fontSize': '0.85rem'}),
            ], style={'marginBottom': '15px'}),
            html.Div([
                html.Div("Distance Traveled", style={
                    'color': COLORS['text'], 'fontWeight': '600',
                    'marginBottom': '8px', 'marginTop': '12px',
                    'fontSize': '0.85rem',
                    'borderTop': f'1px solid {COLORS["grid"]}',
                    'paddingTop': '12px',
                }),
                html.Div(id='kpi-picker-distances'),
            ]),
        ], style={
            'position': 'fixed', 'top': '80px', 'left': '20px',
            'backgroundColor': COLORS['surface'], 'padding': '15px',
            'borderRadius': '8px',
            'border': f'1px solid {COLORS["grid"]}', 'zIndex': '1000',
            'fontFamily': 'JetBrains Mono, monospace', 'minWidth': '200px',
            'maxHeight': 'calc(100vh - 200px)', 'overflowY': 'auto',
        }),
    ], style={
        'fontFamily': 'JetBrains Mono, -apple-system, BlinkMacSystemFont, sans-serif',
    })

    # ========================================================================
    # CSS
    # ========================================================================

    app.index_string = f'''
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
            <style>
                * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                body {{ overflow: hidden; }}
                .control-btn {{
                    width: 40px; height: 40px;
                    border: 1px solid {COLORS['grid']};
                    background: {COLORS['surface_light']};
                    color: {COLORS['text']};
                    border-radius: 6px; cursor: pointer; font-size: 1rem;
                    transition: all 0.15s ease;
                }}
                .control-btn:hover {{
                    background: {COLORS['grid']};
                    border-color: {COLORS['primary']};
                }}
                .control-btn.primary {{
                    background: {COLORS['primary']};
                    color: {COLORS['background']};
                    border-color: {COLORS['primary']}; width: 50px;
                }}
                .control-btn.primary:hover {{ filter: brightness(1.1); }}
                .rc-slider {{ height: 8px !important; }}
                .rc-slider-rail {{ background: {COLORS['grid']} !important; height: 4px !important; }}
                .rc-slider-track {{ background: {COLORS['primary']} !important; height: 4px !important; }}
                .rc-slider-handle {{
                    border-color: {COLORS['primary']} !important;
                    background: {COLORS['primary']} !important;
                    width: 14px !important; height: 14px !important;
                    margin-top: -5px !important;
                }}
                .Select-control {{ background: {COLORS['surface_light']} !important; border-color: {COLORS['grid']} !important; }}
                .Select-value-label, .Select-placeholder {{ color: {COLORS['text']} !important; }}
                .Select-menu-outer {{ background: {COLORS['surface']} !important; border-color: {COLORS['grid']} !important; }}
                .VirtualizedSelectOption {{ background: {COLORS['surface']} !important; color: {COLORS['text']} !important; }}
                .VirtualizedSelectFocusedOption {{ background: {COLORS['surface_light']} !important; }}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
        </body>
    </html>
    '''

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    @app.callback(
        Output('warehouse-graph', 'figure'),
        Output('event-display', 'children'),
        Output('time-display', 'children'),
        Output('frame-display', 'children'),
        Output('picker-legend', 'children'),
        Output('kpi-elapsed-time', 'children'),
        Output('kpi-total-distance', 'children'),
        Output('kpi-events', 'children'),
        Output('kpi-picker-distances', 'children'),
        Input('current-frame', 'data'),
        Input('storage-toggle', 'value'),
    )
    def update_visualization(frame, storage_toggle):
        triggered = ctx.triggered_id
        precomp = app._wh_precomputed
        pc = app._wh_picker_colors
        pids = app._wh_picker_ids
        log_data = app._wh_log

        patched = Patch()

        if triggered == 'storage-toggle':
            show_storage = 'show' in (storage_toggle or [])
            if show_storage and app._wh_storage_data:
                st = build_batched_storage_trace(app._wh_storage_data)
                patched["data"][0] = st.to_plotly_json()
            else:
                patched["data"][0] = go.Scatter(
                    x=[], y=[], mode='markers',
                    showlegend=False, hoverinfo='skip'
                ).to_plotly_json()

        snapshot = log_data[frame]
        pickers = get_picker_data(snapshot)
        picker_map = {p['id']: p for p in pickers}

        for pid, trace_idx in app._wh_picker_trace_indices.items():
            p = picker_map.get(pid)
            if p:
                patched["data"][trace_idx]["x"] = [p['x']]
                patched["data"][trace_idx]["y"] = [p['y']]
                patched["data"][trace_idx]["hovertext"] = (
                    f"Picker {pid}<br>Status: {p['status']}<br>"
                    f"Pos: ({p['x']:.0f}, {p['y']:.0f})"
                )

        event_text = f"Event: {precomp['events'][frame]}"
        time_text = f"Time: {precomp['times'][frame]}"
        frame_text = f"Frame {frame + 1} / {len(log_data)}"

        kpis = get_kpis_for_frame(precomp, frame, pids)

        elapsed = kpis['elapsed_time']
        if elapsed >= 3600:
            elapsed_str = f"{elapsed / 3600:.1f} h"
        elif elapsed >= 60:
            elapsed_str = f"{elapsed / 60:.1f} min"
        else:
            elapsed_str = f"{elapsed:.1f} s"

        dist_str = f"{kpis['total_distance_all'] / 1000:.2f} m"

        events_sorted = sorted(
            kpis['events_count'].items(), key=lambda x: -x[1])[:3]
        events_items = [
            html.Div(f"{name}: {count}", style={'marginBottom': '2px'})
            for name, count in events_sorted
        ]

        distance_items = []
        for pid in sorted(pids):
            color = pc[pid]
            dist = kpis['total_distance'].get(pid, 0)
            distance_items.append(
                html.Div([
                    html.Span("●", style={'color': color, 'marginRight': '6px'}),
                    html.Span(f"P{pid}: ", style={'color': COLORS['text_muted']}),
                    html.Span(f"{dist / 1000:.2f} m", style={'color': COLORS['text']}),
                ], style={'marginBottom': '4px', 'fontSize': '0.8rem'})
            )

        legend_items = []
        for p in pickers:
            color = pc.get(p['id'], COLORS['primary'])
            legend_items.append(
                html.Div([
                    html.Span("●", style={'color': color, 'marginRight': '8px',
                                           'fontSize': '1.2rem'}),
                    html.Span(f"P{p['id']}: {p['status']}", style={
                        'color': COLORS['text'], 'fontSize': '0.85rem'}),
                ], style={'display': 'flex', 'alignItems': 'center',
                          'marginBottom': '5px'})
            )

        return (patched, event_text, time_text, frame_text, legend_items,
                elapsed_str, dist_str, events_items, distance_items)

    @app.callback(
        Output('current-frame', 'data'),
        Output('is-playing', 'data'),
        Output('btn-play', 'children'),
        Output('animation-interval', 'disabled'),
        Output('frame-slider', 'value'),
        Input('btn-play', 'n_clicks'),
        Input('btn-prev', 'n_clicks'),
        Input('btn-next', 'n_clicks'),
        Input('btn-first', 'n_clicks'),
        Input('btn-last', 'n_clicks'),
        Input('animation-interval', 'n_intervals'),
        Input('frame-slider', 'value'),
        State('current-frame', 'data'),
        State('is-playing', 'data'),
        prevent_initial_call=True,
    )
    def handle_playback(play_clicks, prev_clicks, next_clicks, first_clicks,
                        last_clicks, n_intervals, slider_value,
                        current_frame, is_playing):
        triggered = ctx.triggered_id
        nf = len(app._wh_log)

        if triggered == 'frame-slider':
            return slider_value, False, '▶', True, slider_value

        if triggered == 'btn-play':
            new_playing = not is_playing
            btn = '⏸' if new_playing else '▶'
            f = 0 if (new_playing and current_frame >= nf - 1) else current_frame
            return f, new_playing, btn, not new_playing, f

        if triggered == 'btn-prev':
            f = max(0, current_frame - 1)
            return f, False, '▶', True, f

        if triggered == 'btn-next':
            f = min(nf - 1, current_frame + 1)
            return f, False, '▶', True, f

        if triggered == 'btn-first':
            return 0, False, '▶', True, 0

        if triggered == 'btn-last':
            return nf - 1, False, '▶', True, nf - 1

        if triggered == 'animation-interval' and is_playing:
            f = current_frame + 1
            if f >= nf:
                return nf - 1, False, '▶', True, nf - 1
            return f, True, '⏸', False, f

        return (current_frame, is_playing,
                '⏸' if is_playing else '▶', not is_playing, current_frame)

    @app.callback(
        Output('animation-interval', 'interval'),
        Input('speed-dropdown', 'value'),
    )
    def update_speed(interval):
        return interval

    return app


# ============================================================================
# MAIN
# ============================================================================

def main():
    log_path = "../../experiments/outputs/2026-02-04/14-33-17/logs.pkl"
    domain_path = "../../data/instances/caches/sim/dynamic_info.pkl"

    if not Path(log_path).exists():
        print(f"Error: Log file not found at {log_path}")
        return
    if not Path(domain_path).exists():
        print(f"Error: Domain file not found at {domain_path}")
        return

    print("Loading simulation data...")
    log, layout, storage_locations = load_simulation_data(log_path, domain_path)
    print(f"Loaded {len(log)} frames, {len(storage_locations)} storage locations")

    app = create_app(log, layout, storage_locations)
    print("\n" + "=" * 50)
    print("🏭 Warehouse Simulation Visualizer")
    print("=" * 50)
    print("Open your browser at: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop the server")
    print("=" * 50 + "\n")

    app.run(debug=True, port=8050)


if __name__ == "__main__":
    main()