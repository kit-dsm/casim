import pickle
from pathlib import Path
from collections import defaultdict

import dash
from dash import dcc, html, callback, Input, Output, State, ctx
import plotly.graph_objects as go
import networkx as nx

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

# Dark industrial theme colors
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

# Picker colors (distinct, vibrant)
PICKER_COLORS = [
    '#ff6b6b',  # coral red
    '#4ecdc4',  # teal
    '#ffe66d',  # yellow
    '#95e1d3',  # mint
    '#f38181',  # salmon
    '#aa96da',  # lavender
    '#fcbad3',  # pink
    '#a8d8ea',  # sky blue
    '#f9ed69',  # lemon
    '#b8de6f',  # lime
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_simulation_data(log_path: str, domain_path: str):
    """Load simulation log and domain data from pickle files."""
    with open(log_path, "rb") as f:
        log = pickle.load(f)

    with open(domain_path, "rb") as f:
        domain = pickle.load(f)

    # Extract storage locations from domain
    storage_locations = []
    if hasattr(domain, 'storage') and hasattr(domain.storage, 'storage_locations'):
        storage_locations = domain.storage.storage_locations

    return log, domain.layout, storage_locations


def extract_graph_data(layout):
    """Extract nodes and edges from the layout's NetworkX graph."""
    G = layout.layout_network.graph
    pos = nx.get_node_attributes(G, "pos")

    # Extract edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Extract nodes
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_labels = [f"({n[0]:.0f}, {n[1]:.0f})" for n in G.nodes()]

    return {
        'edge_x': edge_x,
        'edge_y': edge_y,
        'node_x': node_x,
        'node_y': node_y,
        'node_labels': node_labels,
        'graph': G,
        'pos': pos,
    }


def bbox_to_trace(bbox, color="steelblue", name="", opacity=0.6, line_color=None):
    """Convert a bounding box to a Plotly scatter trace (filled rectangle)."""
    x = [bbox.x_min, bbox.x_max, bbox.x_max, bbox.x_min, bbox.x_min]
    y = [bbox.y_min, bbox.y_min, bbox.y_max, bbox.y_max, bbox.y_min]

    return go.Scatter(
        x=x,
        y=y,
        mode='lines',
        fill='toself',
        fillcolor=color,
        opacity=opacity,
        line=dict(color=line_color or color, width=1),
        name=name,
        hoverinfo='text',
        hovertext=name,
        showlegend=False,
    )


def extract_storage_data(storage_locations):
    """Pre-process storage locations for visualization."""
    slots_data = []
    for slot in storage_locations:
        if hasattr(slot, 'bbox') and slot.bbox is not None:
            slots_data.append({
                'id': slot.id,
                'bbox': slot.bbox,
                'aisle_id': getattr(slot, 'aisle_id', None),
                'slots': getattr(slot, 'slots', []),
            })
    return slots_data


def get_picker_data(snapshot):
    """Extract picker positions and states from a snapshot."""
    pickers = []
    for p in snapshot["pickers"].resources:
        # Handle RouteNode or tuple location
        loc = p.current_location
        if hasattr(loc, 'position'):
            loc = loc.position

        pickers.append({
            'id': p.id,
            'x': loc[0],
            'y': loc[1],
            'status': getattr(p, 'tpe', getattr(p, 'occupied', 'unknown')),
        })
    return pickers


def calculate_kpis(log, current_frame, picker_colors):
    """Calculate KPIs from the simulation log up to the current frame."""

    if current_frame < 1:
        return {
            'total_distance': {pid: 0 for pid in picker_colors.keys()},
            'velocities': {pid: 0 for pid in picker_colors.keys()},
            'picker_states': {},
            'events_count': {},
            'avg_velocity': 0,
            'total_distance_all': 0,
            'elapsed_time': 0,
            'picks_completed': 0,
            'orders_completed': 0,
        }

    # Initialize tracking
    total_distance = {pid: 0.0 for pid in picker_colors.keys()}
    velocities = {pid: 0.0 for pid in picker_colors.keys()}
    recent_positions = {pid: [] for pid in picker_colors.keys()}  # Last N positions for velocity
    events_count = {}
    picks_completed = 0
    orders_completed = 0

    # Track positions over time
    VELOCITY_WINDOW = min(20, current_frame)  # Use last 20 frames for velocity calculation

    prev_positions = {}
    prev_time = None

    for i in range(current_frame + 1):
        snapshot = log[i]
        current_time = snapshot['time']
        event = snapshot['event']

        # Count events
        event_type = event.split(' - ')[0] if ' - ' in str(event) else str(event)
        events_count[event_type] = events_count.get(event_type, 0) + 1

        # Check for pick/order completion events
        event_lower = str(event).lower()
        if 'pick' in event_lower and ('complete' in event_lower or 'finish' in event_lower):
            picks_completed += 1
        if 'order' in event_lower and ('complete' in event_lower or 'finish' in event_lower):
            orders_completed += 1

        # Calculate distances
        for p in snapshot["pickers"].resources:
            loc = p.current_location
            if hasattr(loc, 'position'):
                loc = loc.position

            pid = p.id
            current_pos = (loc[0], loc[1])

            if pid in prev_positions:
                # Calculate distance moved
                dx = current_pos[0] - prev_positions[pid][0]
                dy = current_pos[1] - prev_positions[pid][1]
                dist = (dx**2 + dy**2) ** 0.5
                total_distance[pid] += dist

            prev_positions[pid] = current_pos

            # Track recent positions for velocity (only in velocity window)
            if i >= current_frame - VELOCITY_WINDOW:
                recent_positions[pid].append((current_pos, current_time))

    # Calculate velocities from recent movement
    for pid, positions in recent_positions.items():
        if len(positions) >= 2:
            first_pos, first_time = positions[0]
            last_pos, last_time = positions[-1]

            dx = last_pos[0] - first_pos[0]
            dy = last_pos[1] - first_pos[1]
            dist = (dx**2 + dy**2) ** 0.5

            # Handle time difference
            if hasattr(first_time, 'total_seconds'):
                # timedelta
                time_diff = (last_time - first_time).total_seconds()
            elif isinstance(first_time, (int, float)):
                time_diff = last_time - first_time
            else:
                # Try to parse or default
                try:
                    time_diff = float(last_time) - float(first_time)
                except:
                    time_diff = len(positions)  # Fallback: use frame count

            if time_diff > 0:
                velocities[pid] = dist / time_diff
            else:
                velocities[pid] = 0

    # Get current picker states
    current_snapshot = log[current_frame]
    picker_states = {}
    for p in current_snapshot["pickers"].resources:
        picker_states[p.id] = getattr(p, 'tpe', getattr(p, 'occupied', 'unknown'))

    # Calculate elapsed time
    start_time = log[0]['time']
    end_time = log[current_frame]['time']
    if hasattr(start_time, 'total_seconds'):
        elapsed_time = (end_time - start_time).total_seconds()
    elif isinstance(start_time, (int, float)):
        elapsed_time = end_time - start_time
    else:
        elapsed_time = current_frame  # Fallback

    # Average velocity
    active_velocities = [v for v in velocities.values() if v > 0]
    avg_velocity = sum(active_velocities) / len(active_velocities) if active_velocities else 0

    return {
        'total_distance': total_distance,
        'velocities': velocities,
        'picker_states': picker_states,
        'events_count': events_count,
        'avg_velocity': avg_velocity,
        'total_distance_all': sum(total_distance.values()),
        'elapsed_time': elapsed_time,
        'picks_completed': picks_completed,
        'orders_completed': orders_completed,
    }


# ============================================================================
# FIGURE CREATION
# ============================================================================

def create_warehouse_figure(graph_data, pickers, picker_colors, storage_data=None,
                            show_routes=False, show_storage=True):
    """Create the main warehouse visualization figure."""

    fig = go.Figure()

    # 0. Draw storage locations (background layer)
    if show_storage and storage_data:
        for slot in storage_data:
            bbox = slot['bbox']
            slot_id = slot['id']
            slot_info = f"Location: {slot_id}"
            if slot.get('slots'):
                slot_ids = [s.id for s in slot['slots']]
                slot_info += f"<br>Slots: {', '.join(slot_ids[:5])}"
                if len(slot_ids) > 5:
                    slot_info += f"... (+{len(slot_ids) - 5} more)"

            fig.add_trace(go.Scatter(
                x=[bbox.x_min, bbox.x_max, bbox.x_max, bbox.x_min, bbox.x_min],
                y=[bbox.y_min, bbox.y_min, bbox.y_max, bbox.y_max, bbox.y_min],
                mode='lines',
                fill='toself',
                fillcolor='rgba(70, 130, 180, 0.4)',  # steelblue with transparency
                line=dict(color='rgba(70, 130, 180, 0.8)', width=1),
                name=slot_id,
                hoverinfo='text',
                hovertext=slot_info,
                showlegend=False,
            ))

    # 1. Draw edges (warehouse aisles)
    fig.add_trace(go.Scatter(
        x=graph_data['edge_x'],
        y=graph_data['edge_y'],
        mode='lines',
        line=dict(width=2, color=COLORS['edge']),
        hoverinfo='none',
        name='Aisles',
    ))

    # 2. Draw nodes (intersections/pick points)
    fig.add_trace(go.Scatter(
        x=graph_data['node_x'],
        y=graph_data['node_y'],
        mode='markers',
        marker=dict(
            size=8,
            color=COLORS['node'],
            line=dict(width=1, color=COLORS['surface_light']),
        ),
        text=graph_data['node_labels'],
        hoverinfo='text',
        name='Nodes',
    ))

    # 3. Draw pickers
    for picker in pickers:
        color = picker_colors.get(picker['id'], COLORS['primary'])

        # Picker marker
        fig.add_trace(go.Scatter(
            x=[picker['x']],
            y=[picker['y']],
            mode='markers+text',
            marker=dict(
                size=20,
                color=color,
                line=dict(width=2, color='white'),
                symbol='circle',
            ),
            text=[f"P{picker['id']}"],
            textposition='top center',
            textfont=dict(size=12, color=color, family='JetBrains Mono, monospace'),
            hoverinfo='text',
            hovertext=f"Picker {picker['id']}<br>Status: {picker['status']}<br>Pos: ({picker['x']:.0f}, {picker['y']:.0f})",
            name=f"Picker {picker['id']}",
            showlegend=False,
        ))

    # Layout styling
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(family='JetBrains Mono, monospace', color=COLORS['text']),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=1,
            zeroline=False,
            showticklabels=False,
            scaleanchor='y',
            scaleratio=1,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=1,
            zeroline=False,
            showticklabels=False,
        ),
        showlegend=False,
        hovermode='closest',
        dragmode='pan',
    )

    return fig


# ============================================================================
# DASH APP LAYOUT
# ============================================================================

def create_app(log, layout, storage_locations=None):
    """Create and configure the Dash application."""

    app = dash.Dash(
        __name__,
        title="Warehouse Simulation Visualizer",
        update_title=None,
    )

    # Pre-process data
    graph_data = extract_graph_data(layout)
    storage_data = extract_storage_data(storage_locations) if storage_locations else []
    num_frames = len(log)

    # Assign colors to pickers
    first_snapshot = log[0]
    picker_colors = {
        p.id: PICKER_COLORS[i % len(PICKER_COLORS)]
        for i, p in enumerate(first_snapshot["pickers"].resources)
    }

    # Store data in app
    app.graph_data = graph_data
    app.storage_data = storage_data
    app.log = log
    app.picker_colors = picker_colors

    # ========================================================================
    # LAYOUT
    # ========================================================================

    app.layout = html.Div([
        # Hidden stores
        dcc.Store(id='current-frame', data=0),
        dcc.Store(id='is-playing', data=False),
        dcc.Interval(id='animation-interval', interval=100, disabled=True),

        # Main container
        html.Div([
            # Header
            html.Div([
                html.H1("🏭 Warehouse Simulation", style={
                    'margin': '0',
                    'fontSize': '1.5rem',
                    'fontWeight': '600',
                    'color': COLORS['text'],
                    'fontFamily': 'JetBrains Mono, monospace',
                }),
                html.Div([
                    html.Span(id='event-display', style={
                        'color': COLORS['primary'],
                        'fontFamily': 'JetBrains Mono, monospace',
                        'fontSize': '0.9rem',
                    }),
                    html.Span(" | ", style={'color': COLORS['text_muted'], 'margin': '0 10px'}),
                    html.Span(id='time-display', style={
                        'color': COLORS['text_muted'],
                        'fontFamily': 'JetBrains Mono, monospace',
                        'fontSize': '0.9rem',
                    }),
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'padding': '15px 20px',
                'backgroundColor': COLORS['surface'],
                'borderBottom': f'1px solid {COLORS["grid"]}',
            }),

            # Main visualization area
            html.Div([
                dcc.Graph(
                    id='warehouse-graph',
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                        'scrollZoom': True,
                    },
                    style={'height': 'calc(100vh - 180px)'},
                ),
            ], style={
                'flex': '1',
                'overflow': 'hidden',
            }),

            # Control panel
            html.Div([
                # Playback controls
                html.Div([
                    html.Button('⏮', id='btn-first', n_clicks=0, className='control-btn'),
                    html.Button('◀', id='btn-prev', n_clicks=0, className='control-btn'),
                    html.Button('▶', id='btn-play', n_clicks=0, className='control-btn primary'),
                    html.Button('▶', id='btn-next', n_clicks=0, className='control-btn'),
                    html.Button('⏭', id='btn-last', n_clicks=0, className='control-btn'),
                ], style={
                    'display': 'flex',
                    'gap': '8px',
                    'alignItems': 'center',
                }),

                # Frame slider
                html.Div([
                    html.Span(id='frame-display', style={
                        'color': COLORS['text'],
                        'fontFamily': 'JetBrains Mono, monospace',
                        'fontSize': '0.85rem',
                        'minWidth': '120px',
                    }),
                    dcc.Slider(
                        id='frame-slider',
                        min=0,
                        max=num_frames - 1,
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={'always_visible': False},
                        updatemode='drag',
                    ),
                ], style={
                    'flex': '1',
                    'display': 'flex',
                    'alignItems': 'center',
                    'gap': '15px',
                    'margin': '0 20px',
                }),

                # Speed control
                html.Div([
                    html.Label("Speed:", style={
                        'color': COLORS['text_muted'],
                        'fontSize': '0.85rem',
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
                        value=100,
                        clearable=False,
                        style={'width': '100px'},
                    ),
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                }),

                # Route toggle
                html.Div([
                    html.Label("Routes:", style={
                        'color': COLORS['text_muted'],
                        'fontSize': '0.85rem',
                        'marginRight': '8px',
                    }),
                    dcc.Checklist(
                        id='route-toggle',
                        options=[{'label': 'Show', 'value': 'show'}],
                        value=[],
                        inline=True,
                        style={'color': COLORS['text']},
                    ),
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'marginLeft': '20px',
                }),

                # Storage toggle
                html.Div([
                    html.Label("Storage:", style={
                        'color': COLORS['text_muted'],
                        'fontSize': '0.85rem',
                        'marginRight': '8px',
                    }),
                    dcc.Checklist(
                        id='storage-toggle',
                        options=[{'label': 'Show', 'value': 'show'}],
                        value=['show'],  # Enabled by default
                        inline=True,
                        style={'color': COLORS['text']},
                    ),
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'marginLeft': '20px',
                }),

            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'padding': '12px 20px',
                'backgroundColor': COLORS['surface'],
                'borderTop': f'1px solid {COLORS["grid"]}',
            }),

        ], style={
            'display': 'flex',
            'flexDirection': 'column',
            'height': '100vh',
            'backgroundColor': COLORS['background'],
        }),

        # Picker legend (floating)
        html.Div([
            html.Div("Pickers", style={
                'color': COLORS['text'],
                'fontWeight': '600',
                'marginBottom': '10px',
                'fontSize': '0.9rem',
            }),
            html.Div(id='picker-legend'),
        ], style={
            'position': 'fixed',
            'top': '80px',
            'right': '20px',
            'backgroundColor': COLORS['surface'],
            'padding': '15px',
            'borderRadius': '8px',
            'border': f'1px solid {COLORS["grid"]}',
            'zIndex': '1000',
            'fontFamily': 'JetBrains Mono, monospace',
            'minWidth': '180px',
        }),

        # KPI Panel (floating, left side)
        html.Div([
            html.Div("📊 KPIs", style={
                'color': COLORS['text'],
                'fontWeight': '600',
                'marginBottom': '12px',
                'fontSize': '0.9rem',
                'borderBottom': f'1px solid {COLORS["grid"]}',
                'paddingBottom': '8px',
            }),

            # Global metrics
            html.Div([
                html.Div("Elapsed Time", style={'color': COLORS['text_muted'], 'fontSize': '0.75rem'}),
                html.Div(id='kpi-elapsed-time', style={'color': COLORS['primary'], 'fontSize': '1.1rem', 'fontWeight': '600'}),
            ], style={'marginBottom': '12px'}),

            html.Div([
                html.Div("Total Distance", style={'color': COLORS['text_muted'], 'fontSize': '0.75rem'}),
                html.Div(id='kpi-total-distance', style={'color': COLORS['text'], 'fontSize': '0.95rem'}),
            ], style={'marginBottom': '12px'}),

            html.Div([
                html.Div("Avg Velocity", style={'color': COLORS['text_muted'], 'fontSize': '0.75rem'}),
                html.Div(id='kpi-avg-velocity', style={'color': COLORS['success'], 'fontSize': '0.95rem'}),
            ], style={'marginBottom': '12px'}),

            html.Div([
                html.Div("Events", style={'color': COLORS['text_muted'], 'fontSize': '0.75rem'}),
                html.Div(id='kpi-events', style={'color': COLORS['text'], 'fontSize': '0.85rem'}),
            ], style={'marginBottom': '15px'}),

            # Picker velocities
            # html.Div([
            #     html.Div("Picker Velocities", style={
            #         'color': COLORS['text'],
            #         'fontWeight': '600',
            #         'marginBottom': '8px',
            #         'fontSize': '0.85rem',
            #         'borderTop': f'1px solid {COLORS["grid"]}',
            #         'paddingTop': '12px',
            #     }),
            #     html.Div(id='kpi-picker-velocities'),
            # ]),

            # Picker distances
            html.Div([
                html.Div("Distance Traveled", style={
                    'color': COLORS['text'],
                    'fontWeight': '600',
                    'marginBottom': '8px',
                    'marginTop': '12px',
                    'fontSize': '0.85rem',
                    'borderTop': f'1px solid {COLORS["grid"]}',
                    'paddingTop': '12px',
                }),
                html.Div(id='kpi-picker-distances'),
            ]),

        ], style={
            'position': 'fixed',
            'top': '80px',
            'left': '20px',
            'backgroundColor': COLORS['surface'],
            'padding': '15px',
            'borderRadius': '8px',
            'border': f'1px solid {COLORS["grid"]}',
            'zIndex': '1000',
            'fontFamily': 'JetBrains Mono, monospace',
            'minWidth': '200px',
            'maxHeight': 'calc(100vh - 200px)',
            'overflowY': 'auto',
        }),

    ], style={
        'fontFamily': 'JetBrains Mono, -apple-system, BlinkMacSystemFont, sans-serif',
    })

    # ========================================================================
    # CSS STYLES
    # ========================================================================

    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
            <style>
                * { box-sizing: border-box; margin: 0; padding: 0; }
                body { overflow: hidden; }
                
                .control-btn {
                    width: 40px;
                    height: 40px;
                    border: 1px solid ''' + COLORS['grid'] + ''';
                    background: ''' + COLORS['surface_light'] + ''';
                    color: ''' + COLORS['text'] + ''';
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 1rem;
                    transition: all 0.15s ease;
                }
                .control-btn:hover {
                    background: ''' + COLORS['grid'] + ''';
                    border-color: ''' + COLORS['primary'] + ''';
                }
                .control-btn.primary {
                    background: ''' + COLORS['primary'] + ''';
                    color: ''' + COLORS['background'] + ''';
                    border-color: ''' + COLORS['primary'] + ''';
                    width: 50px;
                }
                .control-btn.primary:hover {
                    filter: brightness(1.1);
                }
                
                /* Slider styling */
                .rc-slider { height: 8px !important; }
                .rc-slider-rail { 
                    background: ''' + COLORS['grid'] + ''' !important; 
                    height: 4px !important;
                }
                .rc-slider-track { 
                    background: ''' + COLORS['primary'] + ''' !important; 
                    height: 4px !important;
                }
                .rc-slider-handle {
                    border-color: ''' + COLORS['primary'] + ''' !important;
                    background: ''' + COLORS['primary'] + ''' !important;
                    width: 14px !important;
                    height: 14px !important;
                    margin-top: -5px !important;
                }
                
                /* Dropdown styling */
                .Select-control {
                    background: ''' + COLORS['surface_light'] + ''' !important;
                    border-color: ''' + COLORS['grid'] + ''' !important;
                }
                .Select-value-label, .Select-placeholder {
                    color: ''' + COLORS['text'] + ''' !important;
                }
                .Select-menu-outer {
                    background: ''' + COLORS['surface'] + ''' !important;
                    border-color: ''' + COLORS['grid'] + ''' !important;
                }
                .VirtualizedSelectOption {
                    background: ''' + COLORS['surface'] + ''' !important;
                    color: ''' + COLORS['text'] + ''' !important;
                }
                .VirtualizedSelectFocusedOption {
                    background: ''' + COLORS['surface_light'] + ''' !important;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
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
        Output('kpi-avg-velocity', 'children'),
        Output('kpi-events', 'children'),
        Output('kpi-picker-velocities', 'children'),
        Output('kpi-picker-distances', 'children'),
        Input('frame-slider', 'value'),
        Input('route-toggle', 'value'),
        Input('storage-toggle', 'value'),
    )
    def update_visualization(frame, route_toggle, storage_toggle):
        """Update the warehouse visualization for the current frame."""
        snapshot = app.log[frame]
        pickers = get_picker_data(snapshot)
        show_routes = 'show' in (route_toggle or [])
        show_storage = 'show' in (storage_toggle or [])

        fig = create_warehouse_figure(
            app.graph_data,
            pickers,
            app.picker_colors,
            storage_data=app.storage_data,
            show_routes=show_routes,
            show_storage=show_storage,
        )

        event_text = f"Event: {snapshot['event']}"
        time_text = f"Time: {snapshot['time']}"
        frame_text = f"Frame {frame + 1} / {len(app.log)}"

        # Calculate KPIs
        kpis = calculate_kpis(app.log, frame, app.picker_colors)

        # Format elapsed time
        elapsed = kpis['elapsed_time']
        if elapsed >= 3600:
            elapsed_str = f"{elapsed/3600:.1f} h"
        elif elapsed >= 60:
            elapsed_str = f"{elapsed/60:.1f} min"
        else:
            elapsed_str = f"{elapsed:.1f} s"

        # Format total distance
        total_dist = kpis['total_distance_all']
        dist_str = f"{total_dist/1000:.2f} m"

        # Format average velocity
        avg_vel = kpis['avg_velocity']
        vel_str = f"{avg_vel/ 100:.2f} m/s"

        # Format events count (top 3)
        events_sorted = sorted(kpis['events_count'].items(), key=lambda x: -x[1])[:3]
        events_items = [
            html.Div(f"{name}: {count}", style={'marginBottom': '2px'})
            for name, count in events_sorted
        ]

        # Format picker velocities
        velocity_items = []
        for pid in sorted(app.picker_colors.keys()):
            color = app.picker_colors[pid]
            vel = kpis['velocities'].get(pid, 0)
            velocity_items.append(
                html.Div([
                    html.Span("●", style={'color': color, 'marginRight': '6px'}),
                    html.Span(f"P{pid}: ", style={'color': COLORS['text_muted']}),
                    html.Span(f"{vel/ 100:.2f} m/s", style={'color': COLORS['text']}),
                ], style={'marginBottom': '4px', 'fontSize': '0.8rem'})
            )

        # Format picker distances
        distance_items = []
        for pid in sorted(app.picker_colors.keys()):
            color = app.picker_colors[pid]
            dist = kpis['total_distance'].get(pid, 0)
            dist_fmt = f"{dist/1000:.2f} m"
            distance_items.append(
                html.Div([
                    html.Span("●", style={'color': color, 'marginRight': '6px'}),
                    html.Span(f"P{pid}: ", style={'color': COLORS['text_muted']}),
                    html.Span(dist_fmt, style={'color': COLORS['text']}),
                ], style={'marginBottom': '4px', 'fontSize': '0.8rem'})
            )

        # Build picker legend
        legend_items = []
        for picker in pickers:
            color = app.picker_colors.get(picker['id'], COLORS['primary'])
            legend_items.append(
                html.Div([
                    html.Span("●", style={
                        'color': color,
                        'marginRight': '8px',
                        'fontSize': '1.2rem',
                    }),
                    html.Span(f"P{picker['id']}: {picker['status']}", style={
                        'color': COLORS['text'],
                        'fontSize': '0.85rem',
                    }),
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'marginBottom': '5px',
                })
            )

        return (fig, event_text, time_text, frame_text, legend_items,
                elapsed_str, dist_str, vel_str, events_items,
                velocity_items, distance_items)

    @app.callback(
        Output('frame-slider', 'value'),
        Output('is-playing', 'data'),
        Output('btn-play', 'children'),
        Output('animation-interval', 'disabled'),
        Input('btn-play', 'n_clicks'),
        Input('btn-prev', 'n_clicks'),
        Input('btn-next', 'n_clicks'),
        Input('btn-first', 'n_clicks'),
        Input('btn-last', 'n_clicks'),
        Input('animation-interval', 'n_intervals'),
        State('frame-slider', 'value'),
        State('is-playing', 'data'),
        prevent_initial_call=True,
    )
    def handle_playback(play_clicks, prev_clicks, next_clicks, first_clicks,
                        last_clicks, n_intervals, current_frame, is_playing):
        """Handle all playback control interactions."""
        triggered = ctx.triggered_id
        num_frames = len(app.log)

        if triggered == 'btn-play':
            # Toggle play/pause
            new_playing = not is_playing
            btn_text = '⏸' if new_playing else '▶'
            # If at end, restart from beginning
            if new_playing and current_frame >= num_frames - 1:
                current_frame = 0
            return current_frame, new_playing, btn_text, not new_playing

        elif triggered == 'btn-prev':
            return max(0, current_frame - 1), False, '▶', True

        elif triggered == 'btn-next':
            return min(num_frames - 1, current_frame + 1), False, '▶', True

        elif triggered == 'btn-first':
            return 0, False, '▶', True

        elif triggered == 'btn-last':
            return num_frames - 1, False, '▶', True

        elif triggered == 'animation-interval' and is_playing:
            new_frame = current_frame + 1
            if new_frame >= num_frames:
                # Stop at end
                return num_frames - 1, False, '▶', True
            return new_frame, True, '⏸', False

        return current_frame, is_playing, '⏸' if is_playing else '▶', not is_playing

    @app.callback(
        Output('animation-interval', 'interval'),
        Input('speed-dropdown', 'value'),
    )
    def update_speed(interval):
        """Update animation speed."""
        return interval

    return app


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    # Update these paths to match your data location # 16-07-18
    log_path = "../../experiments/outputs/2026-02-04/14-33-17/logs.pkl"
    domain_path = "../../data/instances/caches/sim/dynamic_info.pkl"

    # Check if files exist
    if not Path(log_path).exists():
        print(f"Error: Log file not found at {log_path}")
        print("Please update the path in main() or pass as arguments.")
        return

    if not Path(domain_path).exists():
        print(f"Error: Domain file not found at {domain_path}")
        print("Please update the path in main() or pass as arguments.")
        return

    # Load data
    print("Loading simulation data...")
    log, layout, storage_locations = load_simulation_data(log_path, domain_path)
    print(f"Loaded {len(log)} frames")
    print(f"Loaded {len(storage_locations)} storage locations")

    # Create and run app
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
