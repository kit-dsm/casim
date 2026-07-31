from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
from pathlib import Path

import dash
import networkx as nx
import plotly.graph_objects as go
from dash import Input, Output, Patch, State, ctx, dcc, html

from casim.viz import kpis

BG, SURFACE, TEXT, MUTED, GRID, EDGE, NODE, ACCENT = (
    "#0a0e14",
    "#131820",
    "#e2e8f0",
    "#64748b",
    "#1e293b",
    "#475569",
    "#64748b",
    "#00d4ff",
)
PICKER_COLORS = [
    "#ff6b6b",
    "#4ecdc4",
    "#ffe66d",
    "#95e1d3",
    "#f38181",
    "#aa96da",
    "#fcbad3",
    "#a8d8ea",
    "#f9ed69",
    "#b8de6f",
]
STATUS_COLORS = {
    "outstanding": "#f59e0b",
    "batched": "#7c3aed",
    "assigned": "#3b82f6",
    "in_progress": "#10b981",
    "completed": "#64748b",
    "unknown": "#ef4444",
}
STATUS_ORDER = [
    "in_progress",
    "assigned",
    "batched",
    "outstanding",
    "completed",
]
MAX_TABLE_ROWS = 100


def _merge_state(target: dict, delta: dict) -> None:
    for key, value in delta.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_state(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _normalize_snapshot(snapshot: dict) -> dict:
    snapshot["pickers"] = [
        value
        for _, value in sorted(
            (
                (int(key), value)
                for key, value in snapshot.get("pickers", {}).items()
            )
        )
    ]
    snapshot["tours"] = {
        int(key): value for key, value in snapshot.get("tours", {}).items()
    }
    snapshot["active_picker_tour"] = {
        int(key): value
        for key, value in snapshot.get("active_picker_tour", {}).items()
    }
    return snapshot


def load_bundle(viz_dir: Path):
    """Load static data and replay compact JSON event-state deltas."""
    state = {}
    events = []
    with gzip.open(
        viz_dir / "events.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as stream:
        for line in stream:
            record = json.loads(line)
            _merge_state(state, record.pop("state"))
            snapshot = copy.deepcopy(state)
            snapshot.update(record)
            events.append(_normalize_snapshot(snapshot))
    with open(viz_dir / "static.pkl", "rb") as stream:
        static = pickle.load(stream)
    if not events:
        raise ValueError(f"No visualization frames found in {viz_dir}")
    return events, static


def build_figure(layout, storage, first_snapshot, colors):
    graph = layout.layout_network.graph
    positions = nx.get_node_attributes(graph, "pos")

    edge_x, edge_y = [], []
    for origin, destination in graph.edges():
        x0, y0 = positions[origin]
        x1, y1 = positions[destination]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    storage_x, storage_y = [], []
    for slot in storage:
        if getattr(slot, "bbox", None) is None:
            continue
        bbox = slot.bbox
        storage_x += [
            bbox.x_min,
            bbox.x_max,
            bbox.x_max,
            bbox.x_min,
            bbox.x_min,
            None,
        ]
        storage_y += [
            bbox.y_min,
            bbox.y_min,
            bbox.y_max,
            bbox.y_max,
            bbox.y_min,
            None,
        ]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=storage_x,
            y=storage_y,
            mode="lines",
            fill="toself",
            fillcolor="rgba(70,130,180,0.35)",
            line={"color": "rgba(70,130,180,0.8)", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 2, "color": EDGE},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    node_x, node_y = zip(*positions.values()) if positions else ([], [])
    figure.add_trace(
        go.Scatter(
            x=list(node_x),
            y=list(node_y),
            mode="markers",
            marker={"size": 6, "color": NODE},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    trace_indexes = {}
    for picker in first_snapshot["pickers"]:
        picker_id = picker["id"]
        color = colors[picker_id]
        route_index = len(figure.data)
        figure.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line={"width": 4, "color": color},
                opacity=0.65,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        picker_index = len(figure.data)
        x, y = picker["position"]
        figure.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker={
                    "size": 20,
                    "color": color,
                    "line": {"width": 2, "color": "white"},
                },
                text=[f"P{picker_id}"],
                textposition="top center",
                textfont={"color": color, "family": "monospace"},
                hoverinfo="text",
                hovertext=f"Picker {picker_id}",
                showlegend=False,
            )
        )
        trace_indexes[picker_id] = (route_index, picker_index)

    figure.update_layout(
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font={"family": "monospace", "color": TEXT},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={
            "showgrid": True,
            "gridcolor": GRID,
            "zeroline": False,
            "showticklabels": False,
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        yaxis={
            "showgrid": True,
            "gridcolor": GRID,
            "zeroline": False,
            "showticklabels": False,
        },
        hovermode="closest",
        dragmode="pan",
        uirevision="const",
    )
    return figure, trace_indexes


def _kpi_row(label, value, color=TEXT):
    return html.Div(
        [
            html.Div(label, style={"color": MUTED, "fontSize": "0.7rem"}),
            html.Div(
                value,
                style={
                    "color": color,
                    "fontSize": "1rem",
                    "fontWeight": "600",
                },
            ),
        ],
        style={"marginBottom": "9px"},
    )


def _order_row(order_id, status):
    return html.Div(
        [
            html.Span(
                f"#{order_id}",
                style={
                    "color": TEXT,
                    "fontSize": "0.8rem",
                    "minWidth": "80px",
                    "display": "inline-block",
                },
            ),
            html.Span(
                status,
                style={
                    "color": STATUS_COLORS.get(status, TEXT),
                    "fontSize": "0.75rem",
                    "padding": "2px 8px",
                    "borderRadius": "4px",
                    "backgroundColor": (
                        f"{STATUS_COLORS.get(status, TEXT)}22"
                    ),
                    "border": (
                        f"1px solid {STATUS_COLORS.get(status, TEXT)}"
                    ),
                },
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "10px",
            "padding": "4px 0",
            "borderBottom": f"1px solid {GRID}",
        },
    )


def create_app(events, static):
    layout_data = static["layout"]
    storage = static.get("storage_locations", [])
    picker_ids = [picker["id"] for picker in events[0]["pickers"]]
    colors = {
        picker_id: PICKER_COLORS[index % len(PICKER_COLORS)]
        for index, picker_id in enumerate(picker_ids)
    }
    figure, trace_indexes = build_figure(
        layout_data,
        storage,
        events[0],
        colors,
    )
    app = dash.Dash(__name__, title="CASIM picker replay", update_title=None)
    frame_count = len(events)

    app.layout = html.Div(
        [
            dcc.Store(id="frame", data=0),
            dcc.Store(id="playing", data=False),
            dcc.Interval(id="tick", interval=100, disabled=True),
            html.Div(
                [
                    html.Span(id="event-label", style={"color": ACCENT}),
                    html.Span(" | ", style={"color": MUTED}),
                    html.Span(id="time-label", style={"color": MUTED}),
                    html.Span(" | ", style={"color": MUTED}),
                    html.Span(id="frame-label", style={"color": TEXT}),
                ],
                style={"padding": "12px 20px", "backgroundColor": SURFACE},
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="graph",
                        figure=figure,
                        config={"scrollZoom": True, "displaylogo": False},
                        style={"height": "62vh"},
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id="route-filter",
                            options=[
                                {"label": "All routes", "value": "all"},
                                *[
                                    {
                                        "label": f"Picker {picker_id}",
                                        "value": str(picker_id),
                                    }
                                    for picker_id in picker_ids
                                ],
                            ],
                            value=[],
                            multi=True,
                            placeholder="Show picker routes",
                        ),
                        style={
                            "position": "absolute",
                            "top": "15px",
                            "right": "15px",
                            "width": "230px",
                            "zIndex": "1000",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Operational state",
                                style={
                                    "color": TEXT,
                                    "fontWeight": "600",
                                    "marginBottom": "8px",
                                    "borderBottom": f"1px solid {GRID}",
                                    "paddingBottom": "6px",
                                },
                            ),
                            html.Div(id="kpi-orders"),
                            html.Div(id="event-context"),
                            html.Div(
                                "Pickers",
                                style={
                                    "color": TEXT,
                                    "fontWeight": "600",
                                    "marginTop": "10px",
                                    "marginBottom": "8px",
                                    "borderBottom": f"1px solid {GRID}",
                                    "paddingBottom": "6px",
                                },
                            ),
                            html.Div(id="kpi-pickers"),
                        ],
                        style={
                            "position": "absolute",
                            "top": "15px",
                            "left": "15px",
                            "backgroundColor": SURFACE,
                            "padding": "12px",
                            "borderRadius": "8px",
                            "border": f"1px solid {GRID}",
                            "minWidth": "270px",
                            "maxHeight": "58vh",
                            "overflowY": "auto",
                            "zIndex": "1000",
                        },
                    ),
                ],
                style={"position": "relative"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "Orders",
                                style={
                                    "color": TEXT,
                                    "fontWeight": "600",
                                    "marginRight": "15px",
                                },
                            ),
                            dcc.Dropdown(
                                id="status-filter",
                                options=[
                                    {"label": "All", "value": "all"},
                                    *[
                                        {"label": status, "value": status}
                                        for status in STATUS_ORDER
                                    ],
                                ],
                                value="all",
                                clearable=False,
                                style={
                                    "width": "140px",
                                    "display": "inline-block",
                                },
                            ),
                            html.Span(
                                id="table-count",
                                style={
                                    "color": MUTED,
                                    "marginLeft": "15px",
                                    "fontSize": "0.8rem",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "padding": "8px 15px",
                            "borderBottom": f"1px solid {GRID}",
                        },
                    ),
                    html.Div(
                        id="order-table",
                        style={
                            "overflowY": "auto",
                            "maxHeight": "calc(23vh - 50px)",
                            "padding": "8px 15px",
                        },
                    ),
                ],
                style={
                    "backgroundColor": SURFACE,
                    "height": "23vh",
                    "borderTop": f"1px solid {GRID}",
                },
            ),
            html.Div(
                [
                    html.Button(
                        "▶",
                        id="play",
                        n_clicks=0,
                        style={"width": "50px", "height": "36px"},
                    ),
                    dcc.Slider(
                        id="slider",
                        min=0,
                        max=frame_count - 1,
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"always_visible": False},
                        updatemode="mouseup",
                    ),
                    dcc.Dropdown(
                        id="speed",
                        value=100,
                        clearable=False,
                        options=[
                            {"label": label, "value": value}
                            for label, value in [
                                ("Max", 10),
                                ("20 fps", 50),
                                ("10 fps", 100),
                                ("5 fps", 200),
                                ("1 fps", 1000),
                            ]
                        ],
                        style={"width": "110px"},
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "50px 1fr 110px",
                    "alignItems": "center",
                    "gap": "15px",
                    "padding": "12px 20px",
                    "backgroundColor": SURFACE,
                },
            ),
        ],
        style={
            "backgroundColor": BG,
            "color": TEXT,
            "height": "100vh",
            "fontFamily": "monospace",
            "display": "flex",
            "flexDirection": "column",
        },
    )
    app.index_string = (
        "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>"
        "{%favicon%}{%css%}<style>*{box-sizing:border-box;margin:0}"
        "body{overflow:hidden}</style></head><body>{%app_entry%}"
        "<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"
    )

    @app.callback(
        Output("graph", "figure"),
        Output("event-label", "children"),
        Output("time-label", "children"),
        Output("frame-label", "children"),
        Output("kpi-orders", "children"),
        Output("event-context", "children"),
        Output("kpi-pickers", "children"),
        Output("order-table", "children"),
        Output("table-count", "children"),
        Input("frame", "data"),
        Input("status-filter", "value"),
        Input("route-filter", "value"),
    )
    def render(frame, status_filter, visible_routes):
        snapshot = events[frame]
        patched = Patch()
        visible_routes = set(visible_routes or [])
        for picker in snapshot["pickers"]:
            route_index, picker_index = trace_indexes[picker["id"]]
            x, y = picker["position"]
            route = picker.get("route_suffix", [])
            patched["data"][picker_index]["x"] = [x]
            patched["data"][picker_index]["y"] = [y]
            patched["data"][picker_index]["hovertext"] = (
                f"Picker {picker['id']} · {picker['phase']} · "
                f"tour {picker['active_tour_id']} · "
                f"v{picker['route_version']}"
            )
            show_route = (
                "all" in visible_routes
                or str(picker["id"]) in visible_routes
            )
            patched["data"][route_index]["x"] = (
                [value[0] for value in route] if show_route else []
            )
            patched["data"][route_index]["y"] = (
                [value[1] for value in route] if show_route else []
            )

        event_label = (
            f"{snapshot['event_type']} #{snapshot['event_id']:05d}"
        )
        time_label = f"t = {snapshot['time']:.2f}"
        frame_label = f"{frame + 1} / {frame_count}"
        summary = kpis.summary(snapshot)
        order_panel = [
            _kpi_row(
                "Outstanding",
                summary["outstanding"],
                STATUS_COLORS["outstanding"],
            ),
            _kpi_row(
                "Batched",
                summary["batched"],
                STATUS_COLORS["batched"],
            ),
            _kpi_row(
                "In progress",
                summary["in_progress"],
                STATUS_COLORS["in_progress"],
            ),
            _kpi_row(
                "Completed",
                summary["completed"],
                STATUS_COLORS["completed"],
            ),
        ]
        context = snapshot.get("event", {})
        last_intervention = snapshot.get("last_intervention")
        context_lines = [
            html.Div(
                f"queued events: {snapshot.get('pending_event_count', 0)}",
                style={"color": MUTED, "fontSize": "0.75rem"},
            ),
            html.Div(
                "event: "
                + (
                    ", ".join(
                        f"{key}={value}"
                        for key, value in sorted(context.items())
                    )
                    or "no target"
                ),
                style={"color": MUTED, "fontSize": "0.75rem"},
            ),
        ]
        if last_intervention is not None:
            context_lines.append(
                html.Div(
                    "last intervention: "
                    f"P{last_intervention['picker_id']} / "
                    f"T{last_intervention['tour_id']} "
                    f"v{last_intervention['old_version']}→"
                    f"{last_intervention['new_version']}, "
                    f"inserted={last_intervention['inserted_order_ids']}",
                    style={"color": ACCENT, "fontSize": "0.75rem"},
                )
            )

        picker_panel = []
        by_id = {
            picker["id"]: picker for picker in snapshot["pickers"]
        }
        for picker_id in sorted(picker_ids):
            picker = by_id[picker_id]
            if picker["active_tour_id"] is None:
                status_text = "idle"
            else:
                status_text = (
                    f"T{picker['active_tour_id']} "
                    f"v{picker['route_version']} · {picker['phase']} · "
                    f"{picker['remaining_picks']} picks left · "
                    f"bins {picker['bin_owners']}"
                )
            picker_panel.append(
                html.Div(
                    [
                        html.Span(
                            "●",
                            style={
                                "color": colors[picker_id],
                                "marginRight": "6px",
                            },
                        ),
                        html.Span(
                            f"P{picker_id}: ",
                            style={"color": MUTED, "fontSize": "0.8rem"},
                        ),
                        html.Span(
                            status_text,
                            style={"color": TEXT, "fontSize": "0.8rem"},
                        ),
                    ],
                    style={"marginBottom": "5px"},
                )
            )

        status_map = kpis.order_status_map(snapshot)
        rank = {
            status: index for index, status in enumerate(STATUS_ORDER)
        }
        items = sorted(
            status_map.items(),
            key=lambda item: (rank.get(item[1], 99), item[0]),
        )
        if status_filter != "all":
            items = [
                item for item in items if item[1] == status_filter
            ]
        total = len(items)
        rows = [
            _order_row(order_id, status)
            for order_id, status in items[:MAX_TABLE_ROWS]
        ]
        count_label = (
            f"{total} shown"
            if total <= MAX_TABLE_ROWS
            else f"{MAX_TABLE_ROWS} of {total} shown"
        )
        return (
            patched,
            event_label,
            time_label,
            frame_label,
            order_panel,
            context_lines,
            picker_panel,
            rows,
            count_label,
        )

    @app.callback(
        Output("frame", "data"),
        Output("playing", "data"),
        Output("play", "children"),
        Output("tick", "disabled"),
        Input("play", "n_clicks"),
        Input("tick", "n_intervals"),
        Input("slider", "value"),
        State("frame", "data"),
        State("playing", "data"),
        prevent_initial_call=True,
    )
    def control(_, __, slider_value, frame, playing):
        triggered = ctx.triggered_id
        if triggered == "slider":
            return slider_value, False, "▶", True
        if triggered == "play":
            new_playing = not playing
            new_frame = (
                0 if new_playing and frame >= frame_count - 1 else frame
            )
            return (
                new_frame,
                new_playing,
                "⏸" if new_playing else "▶",
                not new_playing,
            )
        if triggered == "tick" and playing:
            new_frame = frame + 1
            if new_frame >= frame_count:
                return frame_count - 1, False, "▶", True
            return new_frame, True, "⏸", False
        return (dash.no_update,) * 4

    @app.callback(Output("slider", "value"), Input("frame", "data"))
    def sync_slider(frame):
        return frame

    @app.callback(Output("tick", "interval"), Input("speed", "value"))
    def set_speed(value):
        return value

    return app


def launch(viz_dir: Path, port: int = 8050, debug: bool = False):
    events, static = load_bundle(Path(viz_dir))
    print(f"Loaded {len(events)} frames from {viz_dir}")
    print(f"http://127.0.0.1:{port}")
    create_app(events, static).run(
        debug=debug,
        host="127.0.0.1",
        port=port,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("viz_dir", type=Path)
    parser.add_argument("--port", type=int, default=8050)
    arguments = parser.parse_args()
    launch(arguments.viz_dir, arguments.port)
