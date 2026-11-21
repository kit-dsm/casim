"""P1.7: the credibility profile of the IJPE model, and its validity envelope.

No new fitting and no new simulation runs. The script reads the runs and
fits of P1.2 to P1.5 (``docs/ijpe/validation.md``) and writes the figures of
the profile, aggregates only:

- ``credibility_routing``: the distance of the recorded line sequence over
  CASIM's U-shape distance, per order size. The recorded sequence is walked
  on CASIM's graph (shortest paths from the depot through the lines in
  recorded start order back to the depot); the U-shape distance is
  ``route_distance`` of the trace-driven run ``calibrated_recorded`` (each
  recorded order one tour with its recorded locations). Quantiles per size
  class, no single tour;
- ``credibility_service_time``: the distribution of one-person order
  durations (first line start to last line end, orders that overlap no other
  order of their login), WMS as recorded and net of the breaks inside the
  order, against configuration C of the capacity validation (fitted line
  time, setup and breaks, fitted for each day on the other five; 30
  replications pooled) and the submitted time model A. Drawn as quantiles
  from the 2nd to the 98th percentile, so no single order is shown. It
  replaces the submitted Figure 8;
- ``credibility_completions``: cumulative completed orders per day, WMS
  against the mean and the 90 % band of the 30 replications of C;
- ``credibility_profile``: the profile table.

The validity envelope compares the six-day totals and the days at 08:00,
10:00, 12:00 and 14:00 (``envelope.csv``). The runs are those of
``scripts/ijpe/capacity_validation.py`` with FIFO as a pull
(``outputs/capacity_fifo``, recorded arrivals, no recorded departure). The
G3 check comes from ``scripts/ijpe/generator_holdout.py`` (run it first).

Figures go to ``docs/ijpe/figures/`` (committed, aggregates only), tables to
``outputs/credibility/`` (ignored).

    python -m scripts.ijpe.credibility
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from casim.sources.grid_layout import GridLayout  # noqa: E402
from scenarios.ijpe.source import CANONICAL, STUDY_DAYS, day_lines  # noqa: E402
from scripts.ijpe import OUTPUTS  # noqa: E402
from scripts.ijpe.capacity_validation import _measures, _wms, one_person_orders  # noqa: E402

FIGURES = Path(__file__).resolve().parents[2] / "docs" / "ijpe" / "figures"
DIRECTORY = OUTPUTS / "credibility"
RUNS = OUTPUTS / "capacity_fifo"
TRACE = OUTPUTS / "calibrated_recorded"
FITTED, SUBMITTED = "C", "A"
CLOCK = (8, 10, 12, 14)
ENVELOPE_END_H = 12          # last clock time inside the envelope
GRID_H = np.arange(5.5, 16.0 + 1e-9, 5 / 60)
SIZE_EDGES = (0, 10, 20, 40, 60, 80, 100, np.inf)
TABLE_EDGES = (0, 10, 60, 100, np.inf)
QUANTILE_GRID = np.linspace(0.02, 0.98, 97)
DAY_LABEL = {"2026-04-11": "Sat 11 Apr", "2026-04-13": "Mon 13 Apr", "2026-04-14": "Tue 14 Apr",
             "2026-04-15": "Wed 15 Apr", "2026-04-16": "Thu 16 Apr", "2026-04-17": "Fri 17 Apr"}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
MODEL, PAPER = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.dpi": 200})


def _save(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(FIGURES / f"{name}.{suffix}")
    plt.close(figure)


def _seeds(config: str, day: str) -> list[Path]:
    return sorted(p for p in (RUNS / config).glob(f"seed*/{day}") if (p / "service.json").exists())


# --- graph and routing ------------------------------------------------------

def routing_ratios() -> pd.DataFrame:
    """Per one-order tour: lines, recorded-sequence distance and U-shape distance (m)."""
    network = GridLayout().build(CANONICAL / "layout.csv")[0].layout_network
    distance = network.distance_matrix
    rows = []
    for day in STUDY_DAYS:
        lines, _ = day_lines(day)
        tours = pd.read_csv(TRACE / day / "facts" / "tours.csv")
        if (tours["orders_at_start"] != 1).any():
            raise ValueError("the trace-driven run must hold one order per tour")
        ushape = dict(zip(tours["order_ids"].astype(str), tours["route_distance"]))
        for order, own in lines.sort_values(["start", "row"]).groupby("order"):
            path = [network.start_node, *own["node"], network.end_node]
            walked = sum(distance.at[a, b] for a, b in zip(path, path[1:]))
            rows.append({"lines": len(own), "recorded_m": walked / 1000,
                         "ushape_m": ushape[str(order)] / 1000})
    frame = pd.DataFrame(rows)
    return frame.assign(ratio=frame["recorded_m"] / frame["ushape_m"])


def _classes(lines: pd.Series, edges) -> pd.Series:
    labels = [f"{int(lo) + 1}–{int(hi)}" if np.isfinite(hi) else f"over {int(lo)}"
              for lo, hi in zip(edges, edges[1:])]
    return pd.cut(lines, edges, labels=labels)


def routing_table(frame: pd.DataFrame, edges) -> pd.DataFrame:
    grouped = frame.groupby(_classes(frame["lines"], edges), observed=True)["ratio"]
    return pd.DataFrame({"tours": grouped.size(), "p10": grouped.quantile(.1),
                         "p25": grouped.quantile(.25), "median": grouped.median(),
                         "p75": grouped.quantile(.75), "p90": grouped.quantile(.9)})


def plot_routing(table: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 2.8))
    x = np.arange(len(table))
    axis.axhline(1.0, color=MUTED, lw=1, ls="--", zorder=1)
    axis.vlines(x, table["p10"], table["p90"], color=MODEL, lw=1.2, zorder=2)
    axis.vlines(x, table["p25"], table["p75"], color=MODEL, lw=6, zorder=2)
    axis.scatter(x, table["median"], s=34, color="white", edgecolor=INK, lw=1.2, zorder=3)
    axis.set_xticks(x, [f"{label}\nn = {n}" for label, n in zip(table.index, table["tours"])],
                    fontsize=7.5)
    axis.set_xlabel("Lines per order (one order per tour); n tours")
    axis.set_ylabel("Recorded sequence / U-shape distance")
    axis.set_ylim(min(0.4, table["p10"].min() - 0.05), table["p90"].max() + 0.05)
    axis.grid(axis="y", color=GRID, lw=0.6)
    axis.set_axisbelow(True)
    axis.text(0.99, 0.02, "dot: median; bar: 25th–75th; line: 10th–90th percentile",
              transform=axis.transAxes, ha="right", fontsize=7, color=MUTED)
    _save(figure, "credibility_routing")


# --- service time and throughput ------------------------------------------

def collect() -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    """One-person order durations (minutes) per source, hourly counts per run,
    and the cumulative completion curves per day."""
    durations: dict[str, list[np.ndarray]] = {
        "WMS": [], "WMS net of breaks": [], FITTED: [], SUBMITTED: []}
    counts, curves = [], {}
    for day in STUDY_DAYS:
        lines, _ = day_lines(day)
        clean = one_person_orders(lines)
        _, wms = _wms(day, lines, clean)
        durations["WMS"].append(wms["WMS"])
        durations["WMS net of breaks"].append(wms["WMS net of breaks"])
        midnight = pd.Timestamp(day)
        ends = (lines.groupby("order")["end"].max() - midnight).dt.total_seconds() / 3600
        counts.append({"source": "WMS", "seed": 0, "day": day,
                       **{h: int((ends <= h).sum()) for h in CLOCK}})
        model = []
        for config in (FITTED, SUBMITTED):
            for root in _seeds(config, day):
                _, minutes = _measures(root, clean)
                durations[config].append(minutes)
                done = pd.read_csv(root / "facts" / "orders.csv")["completion"].dropna() / 3600
                counts.append({"source": config, "seed": root.parent.name, "day": day,
                               **{h: int((done <= h).sum()) for h in CLOCK}})
                if config == FITTED:
                    model.append(np.searchsorted(np.sort(done), GRID_H, side="right"))
        curves[day] = {"WMS": np.searchsorted(np.sort(ends), GRID_H, side="right"),
                       "model": np.array(model), "orders": len(ends)}
    return ({k: np.concatenate(v) for k, v in durations.items() if v},
            pd.DataFrame(counts), curves)


def duration_table(durations: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({label: {"orders": len(v), "mean": v.mean(),
                                 **{f"p{q}": np.quantile(v, q / 100) for q in (10, 50, 90)}}
                         for label, v in durations.items()}).T


def plot_service_time(durations: dict[str, np.ndarray]) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 3.2))
    styles = {"WMS net of breaks": dict(color=INK, lw=2),
              "WMS": dict(color=INK, lw=1, ls=":"),
              FITTED: dict(color=MODEL, lw=2),
              SUBMITTED: dict(color=PAPER, lw=1.5, ls="--")}
    names = {"WMS net of breaks": "WMS, net of breaks inside the order",
             "WMS": "WMS, first line to last line",
             FITTED: "CASIM, fitted times (C, 30 replications)",
             SUBMITTED: "CASIM, submitted time model (A)"}
    for label, style in styles.items():
        axis.plot(np.quantile(durations[label], QUANTILE_GRID), QUANTILE_GRID,
                  label=names[label], **style)
    axis.set_xlabel("Order duration, minutes (one-person orders, six days)")
    axis.set_ylabel("Share of orders")
    axis.set_xlim(0, None)
    axis.set_ylim(0, 1)
    axis.grid(color=GRID, lw=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", fontsize=7.5)
    axis.text(0.99, 1.01, "2nd to 98th percentile", transform=axis.transAxes, ha="right",
              fontsize=7, color=MUTED)
    _save(figure, "credibility_service_time")


def envelope_table(counts: pd.DataFrame) -> pd.DataFrame:
    wms = counts[counts["source"] == "WMS"].set_index("day")
    model = counts[counts["source"] == FITTED]
    rows = []
    for h in CLOCK:
        totals = model.groupby("seed")[h].sum()
        by_day = model.groupby("day")[h]
        low, high, mean = by_day.quantile(.05), by_day.quantile(.95), by_day.mean()
        error = mean - wms[h]
        total = wms[h].sum()
        rows.append({
            "clock": f"{h:02d}:00", "WMS orders": int(total),
            "model mean": totals.mean(), "band low": totals.quantile(.05),
            "band high": totals.quantile(.95),
            "relative error": (totals.mean() - total) / total,
            "WMS inside six-day band": bool(totals.quantile(.05) <= total <= totals.quantile(.95)),
            "days inside day band": int(((wms[h] >= low) & (wms[h] <= high)).sum()),
            "mean absolute day error": error.abs().mean(),
            "largest day error": error.loc[error.abs().idxmax()],
            "WMS share of the day's orders done": total / CURVE_TOTAL[0],
        })
    return pd.DataFrame(rows)


CURVE_TOTAL = [0]


def plot_completions(curves: dict) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for axis, day in zip(axes.flat, STUDY_DAYS):
        curve = curves[day]
        low, high = np.quantile(curve["model"], [.05, .95], axis=0)
        axis.axvspan(ENVELOPE_END_H, GRID_H[-1], color=GRID, alpha=0.45, lw=0)
        axis.fill_between(GRID_H, low, high, color=MODEL, alpha=0.25, lw=0,
                          label="CASIM C, 90 % band")
        axis.plot(GRID_H, curve["model"].mean(axis=0), color=MODEL, lw=1.5, label="CASIM C, mean")
        axis.plot(GRID_H, curve["WMS"], color=INK, lw=1.5, ls="--", label="WMS")
        for h in CLOCK:
            axis.axvline(h, color=MUTED, lw=0.5, ls=":")
        axis.set_title(f"{DAY_LABEL[day]} ({curve['orders']} orders)", fontsize=8.5)
        axis.set_xticks(range(6, 17, 2), [f"{h:02d}:00" for h in range(6, 17, 2)], fontsize=7)
        axis.set_ylim(0, None)
        axis.grid(axis="y", color=GRID, lw=0.6)
        axis.set_axisbelow(True)
    for axis in axes[:, 0]:
        axis.set_ylabel("Completed orders")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, fontsize=7.5,
                  bbox_to_anchor=(0.5, 1.03))
    figure.text(0.99, -0.01, "grey: after 12:00, outside the validity envelope", ha="right",
                fontsize=7, color=MUTED)
    figure.tight_layout()
    _save(figure, "credibility_completions")


# --- the profile --------------------------------------------------------------

def _f(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def profile(routing: pd.DataFrame, service: pd.DataFrame, envelope: pd.DataFrame,
            generator: pd.DataFrame | None) -> pd.DataFrame:
    r = routing
    s = service
    e = envelope.set_index("clock")

    def clock(h: str) -> str:
        row = e.loc[h]
        return f"{h} {row['relative error']:+.0%}"

    rows = [
        ("Graph and U-shape routing",
         "median recorded-sequence / U-shape distance, by lines per order",
         "no parameter; all 1,414 one-order tours",
         f"{_f(r.loc['11–60', 'median'])} (11–60 lines), {_f(r.loc['61–100', 'median'])} (61–100), "
         f"{_f(r.loc['over 100', 'median'])} (over 100), {_f(r.loc['1–10', 'median'])} (up to 10)",
         "partly reproduced", "U-shape fixed by the warehouse; up to 10 lines it walks whole aisles"),
        ("Tour time, trace-driven",
         "bias and RMSE per one-order tour, line time fitted net of stops",
         "fit Mon–Wed, held out Thu, Fri, Sat",
         "bias +0.47 min, RMSE 6.3 min (18.74 s per line)",
         "reproduced", "replay with recorded stops only (P1.3)"),
        ("Order span, submitted time model",
         "bias, MAE, RMSE of first-to-last-line span",
         "held out Thu, Fri, Sat (586 orders)",
         "bias -5.33, MAE 8.64, RMSE 14.73 min; calibrated pick time not better",
         "not reproduced", "spans of interleaved orders; replaced by the next row (P1.2)"),
        ("Service time per order",
         "duration of one-person orders net of breaks: mean, 10th, 50th, 90th percentile",
         "each day fitted on the other five; 30 replications",
         f"CASIM {_f(s.loc[FITTED, 'mean'], 1)} / {_f(s.loc[FITTED, 'p10'], 1)} / "
         f"{_f(s.loc[FITTED, 'p50'], 1)} / {_f(s.loc[FITTED, 'p90'], 1)} min, WMS "
         f"{_f(s.loc['WMS net of breaks', 'mean'], 1)} / {_f(s.loc['WMS net of breaks', 'p10'], 1)} / "
         f"{_f(s.loc['WMS net of breaks', 'p50'], 1)} / {_f(s.loc['WMS net of breaks', 'p90'], 1)}",
         "reproduced", f"submitted model too narrow (90th percentile {_f(s.loc[SUBMITTED, 'p90'], 1)})"),
        ("Throughput up to midday",
         "orders completed by clock time, six days, model mean against WMS",
         "fitted times as above; recorded arrivals; 30 replications",
         f"{clock('08:00')}, {clock('10:00')}, {clock('12:00')}; WMS inside the day's 90 % band "
         f"on {e.loc['12:00', 'days inside day band']} of 6 days at 12:00",
         "partly reproduced", "levels low on Wed and Thu; compare policies, not levels"),
        ("Staffing per day",
         "extra pickers chosen on five days, error on the sixth",
         "leave one day out",
         "chooses +4 or +8; held-out errors by 12:00 from -36 to +49 orders",
         "not identified", "sensitivity only (recorded roster, -2, -4)"),
        ("End of the day",
         "day's last line, model minus WMS",
         "as throughput",
         "-49 and -46 min (Mon, Fri), +83 and +84 min (Wed, Thu)",
         "not reproduced", "not used"),
        ("Picker finish times",
         "bias and RMSE per picker-day",
         "trace-driven, recorded assignment, no forced departure",
         "bias +14 min, RMSE 72 min",
         "not reproduced", "not used"),
        ("Busy share",
         "busy time over presence",
         "diagnostic runs (P1.3)",
         "CASIM 76–92 %, WMS 53–73 %",
         "not reproduced", "not used; not a load target in P4.2"),
    ]
    if generator is not None:
        g = generator[generator["day"] == "2026-04-16"].set_index("measure")

        def pair(measure: str, digits: int = 2) -> str:
            return f"{_f(g.loc[measure, 'generated mean'], digits)} vs {_f(g.loc[measure, 'recorded'], digits)}"

        rows.append((
            "Order generator G3",
            "lines per order, aisles per order, m³ per order: generated mean vs recorded",
            "fit Mon–Wed, held out Thu (also Fri, Sat)",
            f"Thu: {pair('lines/order mean', 1)}, {pair('aisles/order mean')}, "
            f"{pair('volume/order mean, m3')}; outside the seed band",
            "partly reproduced", "reproduces the mix of its fitted days, not an unseen day"))
    return pd.DataFrame(rows, columns=["component", "metric", "held-out basis", "result",
                                       "verdict", "use and remark"])


def plot_profile(table: pd.DataFrame) -> None:
    widths = {"component": 16, "metric": 30, "held-out basis": 24, "result": 40, "verdict": 13,
              "use and remark": 28}
    cells = [["\n".join(textwrap.wrap(str(v), int(widths[c] * 0.8))) for c, v in row.items()]
             for _, row in table.iterrows()]
    heights = [max(cell.count("\n") + 1 for cell in row) for row in cells]
    lines = sum(heights) + 1.2 * len(heights) + 2
    figure = plt.figure(figsize=(11.5, 0.16 * lines))
    axis = figure.add_axes((0, 0, 1, 1))
    axis.axis("off")
    total = sum(widths.values())
    grid = axis.table(cellText=cells, colLabels=[c.capitalize() for c in widths],
                      colWidths=[w / total for w in widths.values()], bbox=(0, 0, 1, 1),
                      cellLoc="left", colLoc="left")
    grid.auto_set_font_size(False)
    grid.set_fontsize(7.5)
    unit = 1.0 / lines
    verdict = list(widths).index("verdict")
    for (row, column), cell in grid.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_height(2 * unit)
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_height((heights[row - 1] + 1.2) * unit)
            if column == verdict:
                cell.set_text_props(weight="bold")
    _save(figure, "credibility_profile")


def main() -> None:
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    ratios = routing_ratios()
    fine, coarse = routing_table(ratios, SIZE_EDGES), routing_table(ratios, TABLE_EDGES)
    fine.to_csv(DIRECTORY / "routing.csv")
    coarse.to_csv(DIRECTORY / "routing_coarse.csv")
    plot_routing(fine)
    print("Recorded-sequence / U-shape distance by lines per order\n",
          pd.concat([coarse, fine]).round(3).to_string())

    durations, counts, curves = collect()
    CURVE_TOTAL[0] = sum(c["orders"] for c in curves.values())
    service = duration_table(durations)
    service.to_csv(DIRECTORY / "service_time.csv")
    plot_service_time(durations)
    print("\nOne-person order durations, minutes\n", service.round(1).to_string())

    envelope = envelope_table(counts)
    envelope.to_csv(DIRECTORY / "envelope.csv", index=False)
    plot_completions(curves)
    print("\nEnvelope, six days, orders completed by clock time (C)\n",
          envelope.round(3).to_string(index=False))

    path = DIRECTORY / "generator_holdout.csv"
    generator = pd.read_csv(path) if path.exists() else None
    table = profile(coarse, service, envelope, generator)
    table.to_csv(DIRECTORY / "profile.csv", index=False)
    plot_profile(table)
    print("\nProfile\n", table[["component", "result", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
