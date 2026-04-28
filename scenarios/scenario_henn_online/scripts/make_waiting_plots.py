import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCENARIO_MAP = {
    "scenario_henn_waiting": "Henn Waiting",
    "scenario_henn_order_window_waiting": "Order Window (5)",
    "scenario_henn_order_window": "Order Window",
}


def load_tracker(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def time_weighted_mean(series: list[tuple[float, float]]) -> float:
    if len(series) < 2:
        return float(series[0][1]) if series else 0.0
    ts = np.array([p[0] for p in series], dtype=float)
    vs = np.array([p[1] for p in series], dtype=float)
    dt = np.diff(ts)
    total = ts[-1] - ts[0]
    if total <= 0:
        return float(np.mean(vs))
    return float(np.sum(vs[:-1] * dt) / total)


def summarize(tracker: dict) -> dict:
    dock = tracker["dock_utilization"]
    util = tracker["avg_utilization"]
    completed = tracker["completed_tours"]

    tour_durations = [end - start for _, start, end, *_ in completed]
    final_makespan = max((end for _, _, end, *_ in completed), default=0.0)

    return {
        "final_makespan": final_makespan,
        "avg_makespan": float(np.mean(tour_durations)) if tour_durations else 0.0,
        "avg_utilization": util[-1][1] if util else 0.0,
        "avg_dock": time_weighted_mean(dock),
        "max_dock": max((v for _, v in dock), default=0),
        "n_tours": len(completed),
    }


def collect_runs(base_path: Path) -> pd.DataFrame:
    """Walks <base>/<instance>/<scenario>/ and returns one row per run."""
    rows = []
    for instance_dir in sorted(base_path.iterdir()):
        if not instance_dir.is_dir():
            continue
        instance = instance_dir.name.replace(".txt", "")

        for scenario_dir in sorted(instance_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            tracker_path = scenario_dir / "kpis" / "tracker.json"
            if not tracker_path.exists():
                print(f"[skip] {tracker_path}")
                continue

            tracker = load_tracker(tracker_path)
            row = {
                "instance": instance,
                "scenario": scenario_dir.name,
                "scenario_label": SCENARIO_MAP.get(scenario_dir.name, scenario_dir.name),
                **summarize(tracker),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def parse_instance(name: str) -> dict:
    """72s-100-75-0 → aisles=72, orders=100, items=75, seed=0."""
    parts = name.split("-")
    try:
        return {
            "aisles": int(parts[0].rstrip("s")),
            "orders": int(parts[1]),
            "items_per_order": int(parts[2]),
            "seed": int(parts[3]) if len(parts) > 3 else 0,
        }
    except (ValueError, IndexError):
        return {"aisles": np.nan, "orders": np.nan, "items_per_order": np.nan, "seed": 0}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_metric_per_instance(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
):
    """One marker per (instance, scenario). x-axis: instance, color: scenario."""
    sns.set_theme(style="whitegrid", context="paper", palette="colorblind")

    df = df.sort_values("instance").copy()
    instances = df["instance"].unique().tolist()
    scenarios = df["scenario_label"].unique().tolist()
    palette = dict(zip(scenarios, sns.color_palette("colorblind", n_colors=len(scenarios))))

    fig, ax = plt.subplots(figsize=(max(6.2, 0.45 * len(instances)), 3.0))

    x_pos = {inst: i for i, inst in enumerate(instances)}
    for scen in scenarios:
        sub = df[df["scenario_label"] == scen]
        ax.scatter(
            [x_pos[i] for i in sub["instance"]],
            sub[metric],
            label=scen,
            color=palette[scen],
            s=42,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
        )

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(instances, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Instance")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True, title="Scenario", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")

    fig.tight_layout()
    return fig


def plot_scaling(df: pd.DataFrame, metric: str, ylabel: str, x_field: str, x_label: str):
    """Metric vs. instance size (e.g. number of orders), one line per scenario."""
    sns.set_theme(style="whitegrid", context="paper", palette="colorblind")

    feats = df["instance"].apply(parse_instance).apply(pd.Series)
    plot_df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    if plot_df[x_field].isna().all():
        return None

    agg = (
        plot_df.groupby([x_field, "scenario_label"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    palette = sns.color_palette("colorblind", n_colors=agg["scenario_label"].nunique())
    for (scen, g), color in zip(agg.groupby("scenario_label"), palette):
        g = g.sort_values(x_field)
        ax.errorbar(
            g[x_field], g["mean"], yerr=g["std"].fillna(0),
            label=scen, color=color, marker="o", markersize=5,
            capsize=3, linewidth=1.4, alpha=0.9,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True, title="Scenario", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def to_latex_summary(df: pd.DataFrame, out_path: Path) -> str:
    """Per-instance × scenario summary, makespan and dock fill."""
    pivot = df.pivot_table(
        index="instance",
        columns="scenario_label",
        values=["final_makespan", "avg_dock"],
        aggfunc="first",
    )

    scenarios = df["scenario_label"].unique().tolist()
    metrics = [("final_makespan", "Makespan [s]", False),
               ("avg_dock", "Avg.\\ Dock", False)]

    header_top = (
        " & " + " & ".join(rf"\multicolumn{{{len(scenarios)}}}{{c}}{{\textbf{{{m[1]}}}}}"
                           for m in metrics) + r" \\"
    )
    header_bottom = (
        " & " + " & ".join(rf"\textbf{{{s}}}" for _ in metrics for s in scenarios) + r" \\"
    )

    body_lines = []
    for inst in pivot.index:
        cells = [inst]
        for metric, _, higher_better in metrics:
            row_vals = [pivot.loc[inst, (metric, s)] if (metric, s) in pivot.columns else np.nan
                        for s in scenarios]
            arr = np.array(row_vals, dtype=float)
            if np.all(np.isnan(arr)):
                cells.extend(["--"] * len(scenarios))
                continue
            best = np.nanmax(arr) if higher_better else np.nanmin(arr)
            for v in row_vals:
                if np.isnan(v):
                    cells.append("--")
                else:
                    s = f"{v:,.1f}" if metric == "final_makespan" else f"{v:.2f}"
                    cells.append(rf"\textbf{{{s}}}" if np.isclose(v, best) else s)
        body_lines.append(" & ".join(cells) + r" \\")

    n_cols = 1 + len(metrics) * len(scenarios)
    col_format = "l" + "r" * (n_cols - 1)

    latex = (
        "\\begin{table}[t]\n"
        "  \\centering\n"
        "  \\caption{Final makespan and time-weighted average dock fill per instance "
        "and waiting scenario. Best per instance and metric in bold.}\n"
        "  \\label{tab:waiting_scenarios}\n"
        "  \\resizebox{\\linewidth}{!}{%\n"
        f"  \\begin{{tabular}}{{{col_format}}}\n"
        "    \\toprule\n"
        f"    \\textbf{{Instance}} {header_top}\n"
        f"    {header_bottom}\n"
        "    \\midrule\n"
        + "\n".join(f"    {r}" for r in body_lines) + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "  }\n"
        "\\end{table}\n"
    )

    out_path.write_text(latex)
    return latex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

base_path = Path("../outputs/multirun/")
out_dir = Path("./plots/waiting_strategy")
out_dir.mkdir(parents=True, exist_ok=True)

df = collect_runs(base_path)
if df.empty:
    raise SystemExit(f"No runs found under {base_path}")

print(df.to_string(index=False))
df.to_csv(out_dir / "summary.csv", index=False)

# Scatter: final makespan per instance, colored by scenario
for metric, ylabel, fname in [
    ("final_makespan", "Final Makespan [s]", "scatter_final_makespan"),
    ("avg_utilization", "Avg.\\ Picker Utilization", "scatter_avg_utilization"),
]:
    fig = plot_metric_per_instance(df, metric=metric, ylabel=ylabel)
    fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)

# Scaling plot: makespan vs. number of orders
fig = plot_scaling(df, metric="final_makespan", ylabel="Final Makespan [s]",
                   x_field="orders", x_label="\\# Orders per Instance")
if fig is not None:
    fig.savefig(out_dir / "scaling_makespan_orders.pdf", bbox_inches="tight")
    plt.close(fig)

to_latex_summary(df, out_dir / "summary_table.tex")