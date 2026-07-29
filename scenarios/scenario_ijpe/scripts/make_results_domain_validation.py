from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from omegaconf import OmegaConf

from scenarios.experiment_commons import build_data_loader
from scenarios.scenario_ijpe.grocery_retailer_loader_digraph import (
    COL_ARTICLE_ID,
    COL_DATE,
    COL_END_SEC,
    COL_ORDER_ID,
    COL_PICKER_ID,
    COL_QUANTITY,
    COL_START_SEC,
)
from ware_ops_algos.algorithms import GreedyItemAssignment, UShapeRouting


OUT = Path("../outputs/paper_kpi_eval/domain_validation")
OUT.mkdir(parents=True, exist_ok=True)

PICKER_SPEED_MM_S = 2306
TIME_PER_PICK_S = 19

COLORS = {
    "storage": "#4C78A8",
    "route": "#A15759",
    "pick": "#F58518",
    "graph": "#D6D6D6",
    "dark": "#222222",
}

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

cfg = OmegaConf.create({
    "instances_base": "data",
    "data_card": {
        "name": "grocery_retailer",
        "problem_type": "ORSP",
        "source": {
            "data_loader": "WarehousePickingLoaderDigraph",
            "orders_path": "../data/order_stream_historic.csv",
            "layout_path": "../data/layout.csv",
        },
    },
})


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def mm_to_m(x: float) -> float:
    return x / 1000


def fmt(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}"


def normalize_id(x) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def write_latex(df: pd.DataFrame, name: str, caption: str, label: str) -> None:
    tex = df.to_latex(
        index=False,
        escape=False,
        position="t",
        caption=caption,
        label=label,
    )
    (OUT / name).write_text(tex, encoding="utf-8")


def graph_pos_m(G):
    return {
        node: (x / 1000, y / 1000)
        for node, (x, y) in nx.get_node_attributes(G, "pos").items()
    }


def add_storage_bboxes(ax, domain, alpha: float, zorder: int) -> None:
    for loc in domain.storage.storage_slots:
        bbox = getattr(loc, "bbox", None)

        if bbox is None:
            continue

        ax.add_patch(
            Rectangle(
                (bbox.x_min / 1000, bbox.y_min / 1000),
                (bbox.x_max - bbox.x_min) / 1000,
                (bbox.y_max - bbox.y_min) / 1000,
                facecolor=COLORS["storage"],
                edgecolor="none",
                alpha=alpha,
                zorder=zorder,
            )
        )


def add_graph_edges(ax, G, pos, color: str, zorder: int, linewidth: float = 0.5) -> None:
    for u, v, data in G.edges(data=True):
        if data.get("edge_type") == "depot":
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=linewidth,
            zorder=zorder,
        )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def graph_quantities(domain) -> dict:
    G = domain.layout.layout_network.graph

    aisle_to_sides = defaultdict(set)
    aisle_to_side_x = defaultdict(dict)

    for node, data in G.nodes(data=True):
        aisle_id = data.get("aisle_id")
        side = data.get("side")

        if aisle_id is not None and side is not None:
            aisle_to_sides[aisle_id].add(side)
            aisle_to_side_x[aisle_id][side] = node[0]

    invalid_side_changes = 0
    invalid_vertical_edges = 0

    for u, v, data in G.edges(data=True):
        du = G.nodes[u]
        dv = G.nodes[v]

        same_aisle = (
            du.get("aisle_id") is not None
            and du.get("aisle_id") == dv.get("aisle_id")
        )
        different_side = (
            du.get("side") is not None
            and dv.get("side") is not None
            and du.get("side") != dv.get("side")
        )

        if same_aisle and different_side:
            permitted = data.get("edge_type") == "rung" and u[1] == v[1]
            invalid_side_changes += int(not permitted)

        if data.get("edge_type") == "vertical_aisle":
            valid = (
                u[0] == v[0]
                and du.get("aisle_id") == dv.get("aisle_id")
                and du.get("side") == dv.get("side")
            )
            invalid_vertical_edges += int(not valid)

    aisles = []
    for aisle_id, side_x in aisle_to_side_x.items():
        if "L" in side_x and "R" in side_x:
            left_x = side_x["L"]
            right_x = side_x["R"]
            center_x = (left_x + right_x) / 2
            aisles.append((aisle_id, left_x, right_x, center_x))

    aisles = sorted(aisles, key=lambda x: x[3])

    aisle_width = np.mean([mm_to_m(r - l) for _, l, r, _ in aisles])
    aisle_to_aisle_distance = np.mean([
        mm_to_m(b[1] - a[2])
        for a, b in zip(aisles, aisles[1:])
    ])

    aisle_lengths = []
    for aisle_id, _, _, _ in aisles:
        for side in ("L", "R"):
            ys = [
                node[1]
                for node, data in G.nodes(data=True)
                if data.get("aisle_id") == aisle_id
                and data.get("side") == side
            ]
            aisle_lengths.append(mm_to_m(max(ys) - min(ys)))

    storage_without_access = sum(
        getattr(loc, "pick_node", None) is None
        or getattr(loc, "pick_node") not in G
        for loc in domain.storage.storage_slots
    )

    vertical_lengths = [
        mm_to_m(data["weight"])
        for _, _, data in G.edges(data=True)
        if data.get("edge_type") == "vertical_aisle"
    ]

    return {
        "connected_components": nx.number_connected_components(G),
        "physical_aisles": len(aisle_to_sides),
        "invalid_side_changes": invalid_side_changes,
        "invalid_vertical_edges": invalid_vertical_edges,
        "storage_without_access": storage_without_access,
        "aisle_width_m": aisle_width,
        "aisle_length_m": np.mean(aisle_lengths),
        "storage_spacing_m": min(vertical_lengths),
        "aisle_to_aisle_distance_m": aisle_to_aisle_distance,
    }


def build_historic_order_times(orders_df: pd.DataFrame) -> pd.DataFrame:
    df = orders_df.copy()

    if df[COL_START_SEC].isna().all():
        df[COL_START_SEC] = pd.to_timedelta(df[COL_START_SEC].astype(str)).dt.total_seconds()

    if df[COL_END_SEC].isna().all():
        df[COL_END_SEC] = pd.to_timedelta(df[COL_END_SEC].astype(str)).dt.total_seconds()

    hist = (
        df.groupby(COL_ORDER_ID)
        .agg(
            historic_start_s=(COL_START_SEC, "min"),
            historic_end_s=(COL_END_SEC, "max"),
            historic_n_lines=(COL_ARTICLE_ID, "count"),
            historic_quantity=(COL_QUANTITY, "sum"),
            picker_id=(COL_PICKER_ID, "first"),
            date=(COL_DATE, "first"),
        )
        .reset_index()
    )

    hist[COL_ORDER_ID] = hist[COL_ORDER_ID].map(normalize_id)
    hist["historic_time_s"] = hist["historic_end_s"] - hist["historic_start_s"]
    hist.loc[hist["historic_time_s"] < 0, "historic_time_s"] += 86400

    return hist


def resolved_order_id(order) -> str:
    if hasattr(order, "order_id"):
        return normalize_id(order.order_id)
    if hasattr(order, "order") and hasattr(order.order, "order_id"):
        return normalize_id(order.order.order_id)
    return normalize_id(order.pick_positions[0].order_number)


def route_stats(G, route) -> dict:
    missing_edges = 0
    distance_mm = 0.0

    for u, v in zip(route, route[1:]):
        if u == v:
            continue

        if not G.has_edge(u, v):
            missing_edges += 1
            continue

        distance_mm += G.edges[u, v].get("weight", 0.0)

    return {
        "missing_edges": missing_edges,
        "distance_mm": distance_mm,
    }


def validate_processing(historic: pd.DataFrame, ia_sol, router, G) -> pd.DataFrame:
    lookup = {resolved_order_id(o): o for o in ia_sol.resolved_orders}
    rows = []

    for row in historic.itertuples(index=False):
        order_id = normalize_id(getattr(row, COL_ORDER_ID))

        if order_id not in lookup:
            rows.append({"order_id": order_id, "matched": False})
            continue

        pick_positions = lookup[order_id].pick_positions

        router.reset_parameters()
        sol = router.solve(pick_positions)

        stats = route_stats(G, sol.route.route)

        travel_time_s = stats["distance_mm"] / PICKER_SPEED_MM_S
        pick_time_s = len(pick_positions) * TIME_PER_PICK_S
        modeled_time_s = travel_time_s + pick_time_s
        historic_time_s = getattr(row, "historic_time_s")

        rows.append({
            "order_id": order_id,
            "matched": True,
            "historic_time_s": historic_time_s,
            "modeled_time_s": modeled_time_s,
            "distance_m": stats["distance_mm"] / 1000,
            "n_pick_positions": len(pick_positions),
            "missing_edges": stats["missing_edges"],
            "error_s": modeled_time_s - historic_time_s,
            "abs_error_s": abs(modeled_time_s - historic_time_s),
        })

    return pd.DataFrame(rows)


def valid_rows(validation: pd.DataFrame) -> pd.DataFrame:
    return validation[
        (validation["matched"])
        & (validation["missing_edges"] == 0)
        & (validation["historic_time_s"] > 0)
    ].copy()


def warehouse_validation_table(domain, validation: pd.DataFrame) -> pd.DataFrame:
    q = graph_quantities(domain)
    valid = valid_rows(validation)

    return pd.DataFrame([
        ["Connected graph components", q["connected_components"], "1"],
        ["Physical aisles", q["physical_aisles"], "8"],
        ["Storage locations without graph access", q["storage_without_access"], "0"],
        ["Routes with missing graph edges", int(valid["missing_edges"].sum()), "0"],
        ["Aisle width", f"{q['aisle_width_m']:.2f} m", "3.85 m"],
        ["Aisle travel length", f"{q['aisle_length_m']:.2f} m", "approx. 50 m"],
        ["Storage-location spacing", f"{q['storage_spacing_m']:.2f} m", "0.80 m"],
        ["Aisle-to-aisle travel distance", f"{q['aisle_to_aisle_distance_m']:.2f} m", "15.00 m"],
    ], columns=["Validation item", "Value", "Reference"])


def processing_validation_table(validation: pd.DataFrame) -> pd.DataFrame:
    df = valid_rows(validation)

    return pd.DataFrame([
        ["Matched orders", f"{len(df):,}", ""],
        ["Historic processing time, median", fmt(df["historic_time_s"].median(), 1), "s"],
        ["Modeled processing time, median", fmt(df["modeled_time_s"].median(), 1), "s"],
        ["Median error", fmt(df["error_s"].median(), 1), "s"],
        ["Mean absolute error", fmt(df["abs_error_s"].mean(), 1), "s"],
        ["Modeled / historic total", fmt(df["modeled_time_s"].sum() / df["historic_time_s"].sum(), 3), ""],
    ], columns=["Metric", "Value", "Unit"])


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def select_route_example(validation: pd.DataFrame, min_picks: int = 55, quantile: float = 0.80) -> str:
    df = validation[
        (validation["matched"])
        & (validation["missing_edges"] == 0)
        & (validation["n_pick_positions"] >= min_picks)
    ].copy()

    if df.empty:
        df = validation[
            (validation["matched"])
            & (validation["missing_edges"] == 0)
        ].copy()

    target = df["distance_m"].quantile(quantile)
    idx = (df["distance_m"] - target).abs().idxmin()

    return df.loc[idx, "order_id"]

def plot_graph(domain) -> None:
    G = domain.layout.layout_network.graph
    pos = graph_pos_m(G)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    add_storage_bboxes(ax, domain, alpha=0.18, zorder=1)
    add_graph_edges(ax, G, pos, color="#8c4a4a", zorder=2, linewidth=0.6)

    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS["storage"], alpha=0.18, label="Storage locations"),
    ]

    ax.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.16),
    )

    fig.tight_layout()
    fig.savefig(OUT / "fig_graph.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_route(domain, route, pick_positions) -> None:
    G = domain.layout.layout_network.graph
    pos = graph_pos_m(G)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    add_storage_bboxes(ax, domain, alpha=0.18, zorder=1)
    add_graph_edges(ax, G, pos, color=COLORS["graph"], zorder=2, linewidth=0.45)

    route_edges = [
        (u, v)
        for u, v in zip(route, route[1:])
        if u != v and u in pos and v in pos
    ]

    for i, (u, v) in enumerate(route_edges):
        if G.has_edge(u, v) and G.edges[u, v].get("edge_type") == "depot":
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=COLORS["route"],
            linewidth=1.2,
            zorder=4,
        )

        if i % 12 == 0:
            dx = x2 - x1
            dy = y2 - y1
            length = max((dx ** 2 + dy ** 2) ** 0.5, 1e-9)
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            arrow_len = min(length * 0.45, 1.8)

            ax.add_patch(
                FancyArrowPatch(
                    (mx - dx / length * arrow_len / 2, my - dy / length * arrow_len / 2),
                    (mx + dx / length * arrow_len / 2, my + dy / length * arrow_len / 2),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    linewidth=0.8,
                    color=COLORS["route"],
                    zorder=5,
                )
            )

    pick_counts = Counter(
        p.pick_node
        for p in pick_positions
        if getattr(p, "pick_node", None) in pos
    )

    for node, count in pick_counts.items():
        x, y = pos[node]
        ax.scatter(
            [x],
            [y],
            s=25 + 5 * count,
            marker="s",
            facecolor=COLORS["pick"],
            edgecolor="black",
            linewidth=0.35,
            zorder=6,
        )

    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS["storage"], alpha=0.18, label="Storage locations"),
        Line2D([0], [0], color=COLORS["route"], lw=1.5, label="Route"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["pick"],
               markeredgecolor="black", markersize=6, label="Pick node"),
    ]

    ax.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.16),
    )

    fig.tight_layout()
    fig.savefig(OUT / "fig_route_inspection.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_processing(validation: pd.DataFrame) -> None:
    df = valid_rows(validation)

    x = df["historic_time_s"].to_numpy() / 60
    y = df["modeled_time_s"].to_numpy() / 60

    # crop view to the bulk; all points and metrics stay over full data
    hi = np.percentile(np.concatenate([x, y]), 99)

    fig, ax = plt.subplots(figsize=(4.0, 4.0))

    ax.scatter(
        x, y,
        s=10,
        color=COLORS["storage"],
        alpha=0.35,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([0, hi], [0, hi], color=COLORS["dark"],
            linestyle="--", linewidth=0.9, zorder=3)

    # agreement with the 1:1 line (historic = reference)
    rmse = np.sqrt(np.mean((y - x) ** 2))
    mae = np.mean(np.abs(y - x))
    ax.text(
        0.04, 0.96,
        f"$n={len(x)}$\nRMSE = {rmse:.1f} min\nMAE = {mae:.1f} min",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
    )

    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Historic processing time [min]")
    ax.set_ylabel("Modeled processing time [min]")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_processing_validation.pdf",
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def use_case_summary_tables(orders_df: pd.DataFrame, historic: pd.DataFrame,
                            domain) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (totals, per_order_distribution) summary frames for the instance.

    A pick corresponds to one order line (one location visit), which is the
    quantity the travel-time model costs via TIME_PER_PICK_S. Item-level
    quantity is not aggregated here as it does not represent a pick action.
    """
    df = orders_df.copy()

    per_order = (
        df.groupby(COL_ORDER_ID)
        .agg(n_lines=(COL_ARTICLE_ID, "count"))
        .reset_index()
    )
    per_order[COL_ORDER_ID] = per_order[COL_ORDER_ID].map(normalize_id)

    pt_min = historic["historic_time_s"].dropna() / 60
    pt_min = pt_min[pt_min > 0]

    orders_per_day = (
        historic.groupby("date")[COL_ORDER_ID].nunique()
        if "date" in historic else pd.Series(dtype=float)
    )

    n_orders = per_order.shape[0]
    n_lines = int(per_order["n_lines"].sum())
    n_articles = df[COL_ARTICLE_ID].nunique()
    n_storage = len(domain.storage.storage_slots)
    n_days = int(historic["date"].nunique()) if "date" in historic else None

    # --- Totals -------------------------------------------------------------
    totals_rows = [
        ["Orders", f"{n_orders:,}"],
        ["Order lines", f"{n_lines:,}"],
        ["Distinct articles", f"{n_articles:,}"],
        ["Storage locations", f"{n_storage:,}"],
    ]
    if n_days is not None:
        totals_rows.append(["Picking days", f"{n_days:,}"])
        if not orders_per_day.empty:
            lo, hi = int(orders_per_day.min()), int(orders_per_day.max())
            totals_rows.append(["Orders per day (min-max)", f"{lo:,}-{hi:,}"])

    totals = pd.DataFrame(totals_rows, columns=["Quantity", "Value"])

    # --- Per-order distributions -------------------------------------------
    def dist_row(label, s, digits):
        s = pd.Series(s).dropna()
        return [
            label,
            f"{s.median():,.{digits}f}",
            f"{s.quantile(0.05):,.{digits}f}",
            f"{s.quantile(0.95):,.{digits}f}",
            f"{s.max():,.{digits}f}",
        ]

    dist = pd.DataFrame(
        [
            dist_row("Lines per order", per_order["n_lines"], 0),
            dist_row("Processing time [min]", pt_min, 1),
        ],
        columns=["Per-order distribution", "Median", "P5", "P95", "Max"],
    )

    return totals, dist

def plot_due_windows(df: pd.DataFrame) -> None:
    # hour bucket of the last pick = constructed due window
    due_hour = (df["historic_end_s"] // 3600).astype(int)
    counts = due_hour.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(4.3, 3.2))

    ax.bar(
        counts.index,
        counts.values,
        width=0.8,
        color=COLORS["storage"],
        edgecolor="none",
    )

    ax.set_xlabel("Modeled due time")
    ax.set_ylabel("Orders")
    ax.set_xticks(counts.index)
    ax.set_xticklabels([f"{h:02d}:00" for h in counts.index], rotation=45,
                       ha="right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_due_windows.pdf", bbox_inches="tight")
    plt.close(fig)


loader = build_data_loader(cfg)
domain = loader.load(
    cfg.data_card.source.orders_path,
    cfg.data_card.source.layout_path,
    use_cache=False,
)

G = domain.layout.layout_network.graph
layout_network = domain.layout.layout_network

ia_sol = GreedyItemAssignment(domain.storage).solve(domain.orders.orders)

router = UShapeRouting(
    start_node=layout_network.start_node,
    end_node=layout_network.end_node,
    closest_node_to_start=layout_network.closest_node_to_start,
    min_aisle_position=layout_network.min_aisle_position,
    max_aisle_position=layout_network.max_aisle_position,
    distance_matrix=layout_network.distance_matrix,
    predecessor_matrix=layout_network.predecessor_matrix,
    picker=domain.resources.resources,
    gen_tour=True,
    gen_item_sequence=True,
    node_list=layout_network.node_list,
    node_to_idx={node: idx for idx, node in enumerate(list(G.nodes))},
    idx_to_node={idx: node for idx, node in enumerate(list(G.nodes))},
)

orders_df = pd.read_csv(
    cfg.data_card.source.orders_path,
    dtype={COL_ARTICLE_ID: str, COL_ORDER_ID: str},
)

historic = build_historic_order_times(orders_df)

plot_due_windows(historic)

validation = validate_processing(historic, ia_sol, router, G)

write_latex(
    warehouse_validation_table(domain, validation),
    "tab_warehouse_graph_validation.tex",
    "Validation of the warehouse graph used in the simulation.",
    "tab:warehouse-graph-validation",
)

write_latex(
    processing_validation_table(validation),
    "tab_processing_validation.tex",
    "Order-level processing-time validation.",
    "tab:processing-validation",
)

write_latex(
    use_case_summary_tables(orders_df, historic, domain)[0],
    "tab_use_case_summary.tex",
    "Summary statistics of the use-case order data.",
    "tab:use-case-summary",
)

write_latex(
    use_case_summary_tables(orders_df, historic, domain)[1],
    "tab_use_case_summary_per_order.tex",
    "Summary statistics of the use-case order data.",
    "tab:use-case-summary",
)
plot_processing(validation)

resolved_lookup = {resolved_order_id(o): o for o in ia_sol.resolved_orders}
rep_order_id = select_route_example(validation, min_picks=55, quantile=0.80)
rep_positions = resolved_lookup[rep_order_id].pick_positions

router.reset_parameters()
rep_solution = router.solve(rep_positions)

plot_route(
    domain=domain,
    route=rep_solution.route.route,
    pick_positions=rep_positions,
)

plot_graph(domain=domain)

valid = valid_rows(validation)

print(f"Output written to: {OUT.resolve()}")
print(f"Representative route order: {rep_order_id}")
print(f"Matched orders: {len(valid):,}")
print(f"Missing route edges: {int(valid['missing_edges'].sum()):,}")
print(f"Median error [s]: {valid['error_s'].median():.1f}")
print(f"Modeled / historic total: {valid['modeled_time_s'].sum() / valid['historic_time_s'].sum():.3f}")