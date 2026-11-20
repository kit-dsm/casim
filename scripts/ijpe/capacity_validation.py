"""System-level validation: does the simulated day reproduce the WMS day's capacity?

All six study days with their full workload (one tour per order), EDD
assignment as a pull on the synthetic dispatch deadlines (``variants=calibrated``), and presence from the
recorded arrivals only (``source.presence=arrival``): no picker leaves at a
recorded time, so the recorded end of the day is not an input. Configurations:

- ``A`` the paper's time model: 19 s per line, 215 s setup, breaks 08:00 and
  11:00 of 30 minutes for everyone (``variants=submitted``'s assumptions);
- ``B`` fitted service time: line time and tour setup drawn from the fit on the
  other days (``scripts/ijpe/fit_non_picking.py``), no breaks;
- ``C`` B plus the fitted breaks;
- ``D+k`` C with ``k`` pickers added to the logins (``source.extra_pickers``),
  and ``Dpaper`` with the pickers per day of the submitted configuration
  (the logins of the whole export, data_audit.md). Staffing is a
  sensitivity: a login is not always one person, and the data do not say
  how many people worked.

Measures per day: orders completed by each full hour from 07:00 to 14:00, the
day's last line, open orders, and the duration (first line start to last
line end) of the orders that overlap no other order of their login, whose
WMS duration is one person's. A break taken inside an order stretches its
WMS duration, while CASIM takes breaks between tours; so the WMS duration is
also given net of the breaks inside the order (the breaks the fit finds,
``fit_non_picking.observe``) and for the orders without one. For D, the extra pickers that fit five days
best are chosen and tested on the sixth (leave one day out). Writes
aggregates only to ``outputs/capacity_validation``.

    python -m scripts.ijpe.capacity_validation [--run]

``--run`` (re)runs the missing replications first; one takes about 15 s.
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from scenarios.ijpe.source import day_lines
from scenarios.ijpe.source import BREAK_MIN_S, BREAK_WINDOWS, STUDY_DAYS
from scripts.ijpe import OUTPUTS, SCENARIO
from scripts.ijpe.fit_non_picking import calibrated_source, foreign_records, observe

WORKERS = 8
HOURS = tuple(range(7, 15))
BASE = ["variants=calibrated", "source.presence=arrival"]
PAPER_BREAKS = "source.breaks=[[28800.0,30600.0],[39600.0,41400.0]]"
PAPER_PICKERS = dict(zip(STUDY_DAYS, (25, 17, 19, 20, 18, 7)))
EXTRA = (2, 4, 6, 8)
CONFIGS: dict[str, tuple[list[str], int]] = {     # name -> (overrides, replications)
    "A": (["source.non_picking=fixed", "source.time_per_pick=19.0", PAPER_BREAKS], 1),
    "B": (["source.non_picking=fitted", "source.breaks=[]"], 30),
    "C": (["source.non_picking=fitted"], 30),
    **{f"D+{k}": (["source.non_picking=fitted", f"source.extra_pickers={k}"], 10) for k in EXTRA},
    "Dpaper": (["source.non_picking=fitted"], 10),
}
QUANTILES = (0.1, 0.5, 0.9)


def _lines(day: str) -> pd.DataFrame:
    lines, _ = day_lines(day)
    return lines


def one_person_orders(lines: pd.DataFrame) -> set[int]:
    """Orders whose scan window (first to last line start) overlaps no other
    order of their login (the rule of ``fit_non_picking.overlapping_orders``)."""
    clean = set()
    for _, own in lines.groupby("picker"):
        windows = own.groupby("order")["start"].agg(["min", "max"])
        for order, (low, high) in windows.iterrows():
            others = windows.drop(index=order)
            if not ((others["min"] < high) & (low < others["max"])).any():
                clean.add(int(order))
    return clean


def _overrides(config: str, day: str) -> list[str]:
    overrides = list(CONFIGS[config][0])
    if config == "Dpaper":
        logins = _lines(day)["picker"].nunique()
        overrides.append(f"source.extra_pickers={max(0, PAPER_PICKERS[day] - logins)}")
    return overrides


def _root(config: str, seed: int, day: str):
    return OUTPUTS / "capacity" / config / f"seed{seed:02d}" / day


def _run(config: str, day: str, seed: int) -> None:
    out = _root(config, seed, day)
    if (out / "service.json").exists():
        return
    subprocess.run([sys.executable, "-m", "casim", "run", str(SCENARIO), *BASE,
                    *_overrides(config, day), f"source.seed={seed}", f"source.day={day}",
                    "--output-dir", str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_all() -> None:
    jobs = [(config, day, seed) for config, (_, n) in CONFIGS.items()
            for seed in range(n) for day in STUDY_DAYS]
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda job: _run(*job), jobs))


def _measures(root, clean: set[int]) -> tuple[dict, np.ndarray]:
    orders = pd.read_csv(root / "facts" / "orders.csv")
    service = json.loads((root / "service.json").read_text())
    done = orders["completion"].dropna()
    row = {f"by_{h}h": int((done <= h * 3600).sum()) for h in HOURS}
    row["last_line_h"] = float(orders["last_pick_end"].max()) / 3600
    row["open"] = int(service["orders"]["all"]["open_orders"])
    mine = orders[orders["order_id"].isin(clean)].dropna(subset=["last_pick_end"])
    return row, (mine["last_pick_end"] - mine["first_pick_start"]).to_numpy() / 60


def break_spells(day: str) -> dict[str, list[tuple[float, float]]]:
    """Per login, the (start, end) in seconds of the breaks the fit finds."""
    source = calibrated_source(day)
    obs = observe(source.replay, foreign_records(source), BREAK_WINDOWS, BREAK_MIN_S)
    login = {index: picker for picker, index in source.picker_ids.items()}
    spells: dict[str, list[tuple[float, float]]] = {}
    for picker, start, duration in obs.break_spells:
        spells.setdefault(login[picker], []).append((start, start + duration))
    return spells


def _wms(day: str, lines: pd.DataFrame, clean: set[int]) -> tuple[dict, dict[str, np.ndarray]]:
    """The WMS measures and the one-person order durations (minutes): as
    recorded, net of the breaks inside, and of the orders without a break."""
    midnight = pd.Timestamp(day)
    spans = lines.groupby("order").agg(start=("start", "min"), end=("end", "max"),
                                       picker=("picker", "first"))
    ends = (spans["end"] - midnight).dt.total_seconds()
    row = {f"by_{h}h": int((ends <= h * 3600).sum()) for h in HOURS}
    row["last_line_h"] = float(ends.max()) / 3600
    row["open"] = 0
    mine = spans[spans.index.astype(int).isin(clean)]
    spells = break_spells(day)
    low = (mine["start"] - midnight).dt.total_seconds().to_numpy()
    high = (mine["end"] - midnight).dt.total_seconds().to_numpy()
    inside = np.array([sum(max(0.0, min(h, b) - max(l, a)) for a, b in spells.get(p, ()))
                       for l, h, p in zip(low, high, mine["picker"])])
    recorded = (high - low) / 60
    return row, {"WMS": recorded, "WMS net of breaks": recorded - inside / 60,
                 "WMS orders without a break": recorded[inside == 0]}


def _quantiles(values: np.ndarray) -> dict:
    return {"orders": len(values), "mean": float(np.mean(values)),
            **{f"p{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES}}


def main() -> None:
    if "--run" in sys.argv:
        run_all()
    runs, wms_rows = [], []
    durations = {"WMS": [], "WMS net of breaks": [], "WMS orders without a break": []}
    for day in STUDY_DAYS:
        lines = _lines(day)
        clean = one_person_orders(lines)
        wms, wms_durations = _wms(day, lines, clean)
        wms_rows.append({"day": day, **wms})
        for label, values in wms_durations.items():
            durations[label].append(values)
        for config, (_, n) in CONFIGS.items():
            for seed in range(n):
                root = _root(config, seed, day)
                if not (root / "service.json").exists():
                    continue
                row, model_durations = _measures(root, clean)
                runs.append({"config": config, "seed": seed, "day": day, **row})
                durations.setdefault(config, []).append(model_durations)
    wms = pd.DataFrame(wms_rows).set_index("day")
    runs = pd.DataFrame(runs)
    directory = OUTPUTS / "capacity_validation"
    directory.mkdir(parents=True, exist_ok=True)
    runs.to_csv(directory / "runs.csv", index=False)
    wms.to_csv(directory / "wms.csv")
    pd.set_option("display.width", 220)
    configs = [c for c in CONFIGS if c in set(runs["config"])]

    print("Six days, orders completed by each hour: WMS and mean (90 % band) over replications")
    totals = runs.groupby(["config", "seed"])[[f"by_{h}h" for h in HOURS]].sum()
    table = {"WMS": {h: f"{wms[f'by_{h}h'].sum():.0f}" for h in HOURS}}
    for config in configs:
        t = totals.loc[config]
        table[config] = {h: (f"{t[f'by_{h}h'].mean():.0f}" if len(t) == 1 else
                             f"{t[f'by_{h}h'].mean():.0f} ({t[f'by_{h}h'].quantile(.05):.0f}-"
                             f"{t[f'by_{h}h'].quantile(.95):.0f})") for h in HOURS}
    print(pd.DataFrame(table).rename(index=lambda h: f"{h:02d}:00").to_string())

    print("\nPer day, orders by 10:00 and 12:00 minus WMS (mean over replications), "
          "day's last line minus WMS (minutes), open orders:")
    means = runs.groupby(["config", "day"]).mean(numeric_only=True)
    rows = []
    for config in configs:
        for day in STUDY_DAYS:
            m = means.loc[(config, day)]
            rows.append({"config": config, "day": day,
                         "by_10h": m["by_10h"] - wms.loc[day, "by_10h"],
                         "by_12h": m["by_12h"] - wms.loc[day, "by_12h"],
                         "last_line_min": 60 * (m["last_line_h"] - wms.loc[day, "last_line_h"]),
                         "open": m["open"]})
    errors = pd.DataFrame(rows)
    errors.to_csv(directory / "day_errors.csv", index=False)
    print(errors.pivot(index="day", columns="config", values="by_10h").round(0).to_string())
    print(errors.pivot(index="day", columns="config", values="by_12h").round(0).to_string())
    print(errors.pivot(index="day", columns="config", values="last_line_min").round(0).to_string())
    print("Mean absolute day error, orders by 10:00 / 12:00 and last line (min):")
    for config in configs:
        e = errors[errors.config == config]
        print(f"  {config:7s} {e.by_10h.abs().mean():5.1f} / {e.by_12h.abs().mean():5.1f}, "
              f"last line {e.last_line_min.abs().mean():5.0f}, open {e.open.sum():.0f}")

    staffing = [c for c in ("C", *(f"D+{k}" for k in EXTRA)) if c in configs]
    if len(staffing) > 1:
        print("\nLeave one day out: the staffing of C/D+k with the least error on the other five days")
        for day in STUDY_DAYS:
            others = errors[(errors.day != day) & errors.config.isin(staffing)]
            score = (others.by_10h.abs() + others.by_12h.abs()).groupby(others.config).sum()
            best = score.idxmin()
            held = errors[(errors.day == day) & (errors.config == best)].iloc[0]
            print(f"  {day}: {best:5s} held-out error by 10:00 {held.by_10h:+.0f}, "
                  f"by 12:00 {held.by_12h:+.0f}")

    print("\nDuration of one-person orders (first line start to last line end), minutes, six days:")
    quantiles = pd.DataFrame({label: _quantiles(np.concatenate(values))
                              for label, values in durations.items() if values}).T
    quantiles.to_csv(directory / "one_person_order_minutes.csv")
    print(quantiles.round(1).to_string())


if __name__ == "__main__":
    main()
