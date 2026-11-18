"""Phase R0 of docs/ijpe/plan.md: what the WMS export contains.

Every function takes the line table of ``export.lines()`` and returns
aggregate tables. ``main`` writes them as CSV to ``scenarios/ijpe/outputs/
audit/`` (ignored by git) and prints them; ``docs/ijpe/data_audit.md`` is
written from that output. Tables per picker go to ``outputs/audit/private/``.

    python -m scripts.ijpe.audit
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from casim.sources import wms_export as export
from scripts.ijpe import CACHE, OUTPUTS as SCENARIO_OUTPUTS

OUTPUTS = SCENARIO_OUTPUTS / "audit"
SIZE_BINS = [0, 20, 40, 60, 80, np.inf]
SIZE_LABELS = ["1-20", "21-40", "41-60", "61-80", ">80"]
GAP_THRESHOLDS_MIN = [1, 2, 5, 10, 15, 20, 30]
# Configured in the submitted simulation (scenario_ijpe/config/simulation).
BREAKS = [("08:00", "08:30"), ("11:00", "11:30")]
TU_PER_ORDER_WORKBOOK = 2.6  # "TE pro h": =D*2.6, D = distinct orders
TU_PER_ORDER_SIMULATION = 1.6  # palett_te_factor of the submitted hooks
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def _describe(values: pd.Series, name: str) -> dict:
    values = pd.Series(values).dropna()
    row = {"measure": name, "n": len(values), "mean": values.mean(), "sd": values.std()}
    row.update({f"q{int(q * 100):02d}": values.quantile(q) for q in QUANTILES})
    row.update({"min": values.min(), "max": values.max()})
    return row


def _minutes(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 60


def with_pick_date(lines: pd.DataFrame) -> pd.DataFrame:
    """``wms_export.with_pick_date`` with the order's size class."""
    frame = export.with_pick_date(lines)
    frame["size_class"] = pd.cut(frame.groupby("order")["row"].transform("size"),
                                 SIZE_BINS, labels=SIZE_LABELS)
    return frame


# ---------------------------------------------------------------- R0.1

def inventory_columns(lines: pd.DataFrame) -> pd.DataFrame:
    raw = export.raw(export.LINES_SHEET, CACHE)
    rows = []
    for column in raw.columns:
        values = raw[column]
        rows.append({"column": column, "missing": int(values.isna().sum()),
                     "distinct": int(values.nunique())})
    return pd.DataFrame(rows)


def inventory_days(lines: pd.DataFrame) -> pd.DataFrame:
    frame = with_pick_date(lines)
    rows = []
    for date, day in frame.groupby("pick_date"):
        rows.append({
            "date": date.date().isoformat(),
            "weekday": date.day_name(),
            "iso_week": date.isocalendar().week,
            "lines": len(day),
            "orders": day["order"].nunique(),
            "customers": day["customer"].nunique(),
            "articles": day["article"].nunique(),
            "locations": day["location"].nunique(),
            "pickers": day["picker"].nunique(),
            "pallets": day["pallet"].nunique(),
            "first_start": day["start"].min().strftime("%H:%M"),
            "last_end": day["end"].max().strftime("%H:%M"),
            "booked_lines": int((lines["date"] == date).sum()),
            "booked_orders": lines.loc[lines["date"] == date, "order"].nunique(),
        })
    total = {"date": "all", "weekday": "", "iso_week": "", "lines": len(frame),
             "orders": frame["order"].nunique(), "customers": frame["customer"].nunique(),
             "articles": frame["article"].nunique(), "locations": frame["location"].nunique(),
             "pickers": frame["picker"].nunique(), "pallets": frame["pallet"].nunique(),
             "first_start": "", "last_end": "", "booked_lines": len(lines),
             "booked_orders": lines["order"].nunique()}
    return pd.DataFrame(rows + [total])


def inventory_values(lines: pd.DataFrame) -> pd.DataFrame:
    """Distributions of the numeric line attributes, with units where derivable."""
    per_package = lines["weight"] / lines["quantity"].where(lines["quantity"] > 0)
    rows = [
        _describe(lines["quantity"], "quantity per line [packages]"),
        _describe(lines["weight"], "weight per line [kg, GEWICHT_SOLL]"),
        _describe(per_package, "weight per package [kg]"),
        _describe(lines["package_volume"], "volume per package [VOLUMEN_KOLLI]"),
        _describe(lines.groupby("order").size(), "lines per order"),
        _describe(lines.groupby("order")["pallet"].nunique(), "pallets per order"),
    ]
    return pd.DataFrame(rows)


def inventory_categories(lines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ("area", "aisle", "line_type", "carrier_type"):
        counts = lines[column].value_counts().sort_index()
        for value, count in counts.items():
            rows.append({"column": column, "value": value, "lines": int(count),
                         "orders": lines.loc[lines[column] == value, "order"].nunique()})
    return pd.DataFrame(rows)


def location_code(lines: pd.DataFrame) -> pd.DataFrame:
    """Q_PLATZ = site(3) area(2) aisle(2) house(2) place(2) level(2)."""
    code = lines["location"]
    parts = {"area": (3, 5), "aisle": (5, 7), "house": (7, 9), "place": (9, 11)}
    agree = {name: float((code.str[a:b].astype(int) == lines[name]).mean())
             for name, (a, b) in parts.items()}
    levels = code.str[11:13].value_counts().sort_index()
    rows = [{"check": f"Q_PLATZ digits match {name}", "value": value} for name, value in agree.items()]
    rows += [{"check": f"lines at level {level}", "value": int(count)} for level, count in levels.items()]
    rows.append({"check": "distinct Q_PLATZ", "value": code.nunique()})
    rows.append({"check": "distinct (aisle, house, place)",
                 "value": len(lines[["aisle", "house", "place"]].drop_duplicates())})
    return pd.DataFrame(rows)


def paper_filter(lines: pd.DataFrame) -> pd.DataFrame:
    """From 1,525 orders to the 1,422 of the model and the 1,419 of Figure 8."""
    kept = export.paper_lines(lines)
    orders_all = set(lines["order"])
    orders_kept = set(kept["order"])
    excluded = export.is_excluded_article(lines["article"])
    dropped = lines[~lines["order"].isin(orders_kept)]
    only_aisle_10 = dropped.groupby("order")["aisle"].apply(lambda a: set(a) == {10}).sum()
    only_article = excluded[dropped.index].groupby(dropped["order"]).all().sum()
    # build_historic_order_times: clock seconds, span = max end - min start
    start = pd.to_timedelta(kept["start_clock"]).dt.total_seconds()
    end = pd.to_timedelta(kept["end_clock"]).dt.total_seconds()
    span = end.groupby(kept["order"]).max() - start.groupby(kept["order"]).min()
    lines_zero = kept.groupby("order").size()[span <= 0]
    return pd.DataFrame([
        {"step": "orders in the export", "orders": len(orders_all), "lines": len(lines)},
        {"step": "lines in aisle 10 (area 10) dropped", "orders": None,
         "lines": int((lines["aisle"] == 10).sum())},
        {"step": "lines of the excluded article dropped", "orders": None,
         "lines": int((excluded & (lines["aisle"] != 10)).sum())},
        {"step": "orders with only aisle-10 lines", "orders": int(only_aisle_10), "lines": None},
        {"step": "orders with only the dropped article", "orders": int(only_article), "lines": None},
        {"step": "orders of the model (to_canonical)", "orders": len(orders_kept), "lines": len(kept)},
        {"step": "orders with historic span <= 0 (valid_rows)", "orders": int((span <= 0).sum()),
         "lines": int(lines_zero.sum())},
        {"step": "orders with historic span < 0", "orders": int((span < 0).sum()), "lines": None},
        {"step": "orders of Figure 8", "orders": int((span > 0).sum()), "lines": None},
    ])


# ---------------------------------------------------------------- R0.2

def order_structure(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per order: size, pickers, zones, aisles, and how pickers share it."""
    frame = with_pick_date(lines)
    per_picker = frame.groupby(["order", "picker"]).agg(first=("start", "min"), last=("end", "max"),
                                                       areas=("area", "nunique"))
    mode, overlap_s = {}, {}
    for order, group in per_picker.groupby(level="order"):
        if len(group) == 1:
            mode[order], overlap_s[order] = "single", 0.0
            continue
        intervals = group[["first", "last"]].sort_values("first").to_numpy()
        overlap = 0.0
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                lo = max(intervals[i][0], intervals[j][0])
                hi = min(intervals[i][1], intervals[j][1])
                overlap += max((hi - lo) / np.timedelta64(1, "s"), 0.0)
        overlap_s[order] = overlap
        mode[order] = "parallel" if overlap > 60 else "sequential"
    orders = frame.groupby("order").agg(
        pick_date=("pick_date", "first"), lines=("row", "size"), pickers=("picker", "nunique"),
        areas=("area", "nunique"), aisles=("aisle", "nunique"), houses=("house", "nunique"),
        pallets=("pallet", "nunique"), start=("start", "min"), end=("end", "max"),
        area_set=("area", lambda a: tuple(sorted(set(a)))))
    orders["mode"] = pd.Series(mode)
    orders["overlap_s"] = pd.Series(overlap_s)
    orders["size_class"] = pd.cut(orders["lines"], SIZE_BINS, labels=SIZE_LABELS)
    # Does each zone of a multi-picker order have one picker of its own?
    area_pickers = frame.groupby(["order", "area"])["picker"].nunique()
    orders["zone_split"] = (area_pickers.groupby(level="order").max() == 1) & (orders["pickers"] > 1) \
        & (per_picker["areas"].groupby(level="order").max() == 1)
    return orders


def order_structure_summary(orders: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in [("all", orders)] + list(orders.groupby("size_class", observed=True)):
        rows.append({
            "size_class": label, "orders": len(group),
            "mean_lines": group["lines"].mean(),
            "pickers_1": int((group["pickers"] == 1).sum()),
            "pickers_2": int((group["pickers"] == 2).sum()),
            "pickers_3+": int((group["pickers"] >= 3).sum()),
            "mean_pickers": group["pickers"].mean(),
            "areas_1": int((group["areas"] == 1).sum()),
            "areas_2+": int((group["areas"] >= 2).sum()),
            "mean_aisles": group["aisles"].mean(),
            "mean_houses": group["houses"].mean(),
            "parallel": int((group["mode"] == "parallel").sum()),
            "sequential": int((group["mode"] == "sequential").sum()),
            "zone_split": int(group["zone_split"].sum()),
            "median_span_min": _minutes(group["end"] - group["start"]).median(),
        })
    return pd.DataFrame(rows)


def order_areas_by_day(orders: pd.DataFrame) -> pd.DataFrame:
    """Which areas (zones) an order's lines lie in, by pick date."""
    frame = orders.copy()
    frame["area_set"] = frame["area_set"].map(lambda s: "+".join(str(a) for a in s))
    table = pd.crosstab(frame["area_set"], frame["pick_date"].dt.date, margins=True)
    return table


def store_days(lines: pd.DataFrame) -> pd.DataFrame:
    """A store's demand of one day (customer-day) and the orders it is split into."""
    frame = with_pick_date(lines)
    orders = frame.groupby("order").agg(
        customer=("customer", "first"), pick_date=("pick_date", "first"),
        main_area=("area", lambda a: a.mode().iloc[0]), picker=("picker", "first"),
        start=("start", "min"), end=("end", "max"))
    grouped = orders.groupby(["customer", "pick_date"])
    size = grouped.size()
    multi = orders[orders.set_index(["customer", "pick_date"]).index.isin(size[size > 1].index)]
    by = multi.groupby(["customer", "pick_date"])

    def overlapping(group: pd.DataFrame) -> bool:
        group = group.sort_values("start")
        return bool((group["start"].iloc[1:].to_numpy()
                     < group["end"].cummax().iloc[:-1].to_numpy()).any())

    combos = by["main_area"].apply(lambda a: "+".join(str(x) for x in sorted(a))).value_counts()
    rows = [
        {"measure": "customer-days", "value": len(size)},
        {"measure": "customer-days with 1 order", "value": int((size == 1).sum())},
        {"measure": "customer-days with 2 orders", "value": int((size == 2).sum())},
        {"measure": "customer-days with 3 orders", "value": int((size == 3).sum())},
        {"measure": "customer-days with 4-6 orders", "value": int((size >= 4).sum())},
        {"measure": "multi-order customer-days: one order per main area (share)",
         "value": by["main_area"].apply(lambda a: a.nunique() == len(a)).mean()},
        {"measure": "multi-order customer-days: one picker per order (share)",
         "value": by["picker"].apply(lambda p: p.nunique() == len(p)).mean()},
        {"measure": "multi-order customer-days: orders overlapping in time (share)",
         "value": by.apply(overlapping).mean()},
    ]
    rows += [{"measure": f"multi-order customer-days with main areas {combo}", "value": int(count)}
             for combo, count in combos.head(6).items()]
    return pd.DataFrame(rows)


def concurrency(lines: pd.DataFrame) -> pd.DataFrame:
    """Time a picker spends with k orders open (between the order's first and last line)."""
    frame = with_pick_date(lines)
    spans = frame.groupby(["picker", "pick_date", "order"]).agg(first=("start", "min"),
                                                                last=("end", "max"))
    seconds: dict[int, float] = {}
    for _, group in spans.groupby(level=["picker", "pick_date"]):
        events = sorted([(t, 1) for t in group["first"]] + [(t, -1) for t in group["last"]])
        open_orders = 0
        for (time, change), (following, _) in zip(events, events[1:]):
            open_orders += change
            seconds[open_orders] = seconds.get(open_orders, 0.0) + (following - time).total_seconds()
    table = pd.Series(seconds).sort_index().rename("hours") / 3600
    table.index.name = "open orders"
    table = table.to_frame()
    working = table.loc[table.index > 0, "hours"].sum()
    table["share_of_time_with_an_order"] = (table["hours"] / working).where(table.index > 0)
    return table


def picker_zones(lines: pd.DataFrame) -> pd.DataFrame:
    """Are pickers bound to a zone? Areas and aisles per picker-day."""
    frame = with_pick_date(lines)
    per_day = frame.groupby(["picker", "pick_date"]).agg(
        areas=("area", "nunique"), aisles=("aisle", "nunique"), lines=("row", "size"))
    main_share = (frame.groupby(["picker", "pick_date", "area"]).size()
                  .groupby(level=["picker", "pick_date"]).apply(lambda s: s.max() / s.sum()))
    return pd.DataFrame([
        {"measure": "picker-days", "value": len(per_day)},
        {"measure": "picker-days in one area", "value": int((per_day["areas"] == 1).sum())},
        {"measure": "picker-days in two areas", "value": int((per_day["areas"] == 2).sum())},
        {"measure": "picker-days in three areas", "value": int((per_day["areas"] == 3).sum())},
        {"measure": "median share of a picker-day's lines in its main area",
         "value": main_share.median()},
        {"measure": "picker-days with >= 95 % of lines in one area",
         "value": int((main_share >= 0.95).sum())},
        {"measure": "pickers working in more than one area over the week",
         "value": int((frame.groupby("picker")["area"].nunique() > 1).sum())},
        {"measure": "pickers", "value": frame["picker"].nunique()},
    ])


def interleaving(lines: pd.DataFrame) -> pd.DataFrame:
    """Does a picker alternate between orders? Runs of one order in time order."""
    frame = with_pick_date(lines).sort_values(["picker", "pick_date", "start", "row"])
    key = ["picker", "pick_date"]
    change = frame["order"] != frame.groupby(key)["order"].shift()
    frame["run"] = change.groupby([frame["picker"], frame["pick_date"]]).cumsum()
    runs = frame.groupby(key)["run"].max()
    distinct = frame.groupby(key)["order"].nunique()
    runs_per_order = frame.groupby(key + ["order"])["run"].nunique()
    # Intervals of one picker's orders that overlap another order of the same picker.
    spans = frame.groupby(key + ["order"]).agg(first=("start", "min"), last=("end", "max"))
    overlapping = 0
    for _, group in spans.groupby(level=key):
        group = group.sort_values("first")
        latest = group["last"].cummax().shift()
        overlapping += int((group["first"] < latest).sum())
    return pd.DataFrame([
        {"measure": "picker-days", "value": len(runs)},
        {"measure": "picker-days with interleaved orders (runs > orders)",
         "value": int((runs > distinct).sum())},
        {"measure": "picker-order pairs", "value": len(runs_per_order)},
        {"measure": "picker-order pairs picked in one run", "value": int((runs_per_order == 1).sum())},
        {"measure": "picker-order pairs picked in 2 runs", "value": int((runs_per_order == 2).sum())},
        {"measure": "picker-order pairs picked in 3+ runs", "value": int((runs_per_order >= 3).sum())},
        {"measure": "picker-order intervals starting inside an earlier order of the same picker",
         "value": overlapping},
    ])


def zone_affinity(lines: pd.DataFrame) -> pd.DataFrame:
    """R1.8: does a picker stay in one zone (area) over the day, and how often do they change?

    A picker-day's orders, in the order of their first line, each with its main
    area; a change is a consecutive pair of orders with different main areas.
    """
    frame = with_pick_date(lines)
    orders = frame.groupby(["picker", "pick_date", "order"]).agg(
        first=("start", "min"), area=("area", lambda a: a.mode().iloc[0])).reset_index()
    orders = orders.sort_values(["picker", "pick_date", "first"])
    orders["change"] = orders.groupby(["picker", "pick_date"])["area"].transform(
        lambda a: a.ne(a.shift()) & a.shift().notna())
    days = orders.groupby(["picker", "pick_date"]).agg(orders=("order", "size"),
                                                       changes=("change", "sum"),
                                                       areas=("area", "nunique"))
    share = (frame.groupby(["picker", "pick_date", "area"]).size()
             .groupby(level=["picker", "pick_date"]).apply(lambda s: s.max() / s.sum()))
    return pd.DataFrame([
        {"measure": "picker-days", "value": len(days)},
        {"measure": "picker-days whose orders all have one main area", "value": int((days["areas"] == 1).sum())},
        {"measure": "picker-days with at least one zone change", "value": int((days["changes"] > 0).sum())},
        {"measure": "zone changes per picker-day, mean", "value": days["changes"].mean()},
        {"measure": "zone changes per picker-day, max", "value": int(days["changes"].max())},
        {"measure": "zone changes per order change, all picker-days",
         "value": days["changes"].sum() / (days["orders"] - 1).clip(lower=0).sum()},
        {"measure": "share of a picker-day's lines in its main area, median", "value": share.median()},
        {"measure": "share of a picker-day's lines in its main area, 10 % quantile",
         "value": share.quantile(0.1)},
    ])


# ---------------------------------------------------------------- R0.3

def timelines(lines: pd.DataFrame) -> pd.DataFrame:
    """Lines of each picker-day in time order with the gap before each line."""
    frame = with_pick_date(lines).sort_values(["picker", "pick_date", "start", "row"])
    key = [frame["picker"], frame["pick_date"]]
    frame["previous_order"] = frame.groupby(key)["order"].shift()
    frame["gap"] = frame["start"] - frame.groupby(key)["end"].shift()
    frame["order_change"] = frame["previous_order"].notna() & (frame["order"] != frame["previous_order"])
    frame["duration"] = frame["end"] - frame["start"]
    return frame


def gap_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    gap = timeline["gap"].dropna()
    within = timeline.loc[timeline["previous_order"] == timeline["order"], "gap"]
    between = timeline.loc[timeline["order_change"], "gap"]
    rows = [
        _describe(timeline["duration"].dt.total_seconds(), "line duration [s]"),
        _describe(gap.dt.total_seconds(), "gap before line, all [s]"),
        _describe(within.dt.total_seconds(), "gap within an order [s]"),
        _describe(between.dt.total_seconds(), "gap at an order change [s]"),
    ]
    return pd.DataFrame(rows)


def gap_thresholds(timeline: pd.DataFrame) -> pd.DataFrame:
    """How many gaps exceed a threshold, within orders and at order changes."""
    within = timeline.loc[timeline["previous_order"] == timeline["order"], "gap"]
    between = timeline.loc[timeline["order_change"], "gap"]
    rows = []
    for minutes in GAP_THRESHOLDS_MIN:
        limit = pd.Timedelta(minutes=minutes)
        rows.append({
            "threshold_min": minutes,
            "gaps_within_order_above": int((within > limit).sum()),
            "share_within": float((within > limit).mean()),
            "gaps_at_order_change_above": int((between > limit).sum()),
            "share_between": float((between > limit).mean()),
            "orders_split_by_within_gaps": timeline.loc[
                (timeline["previous_order"] == timeline["order"]) & (timeline["gap"] > limit),
                "order"].nunique(),
            "time_in_gaps_above_h": float(timeline.loc[timeline["gap"] > limit, "gap"]
                                          .dt.total_seconds().sum() / 3600),
        })
    return pd.DataFrame(rows)


def break_gaps(timeline: pd.DataFrame, minimum_min: int = 15) -> pd.DataFrame:
    """Long gaps by the half hour in which they start, and their fit to the configured breaks."""
    long = timeline[timeline["gap"] > pd.Timedelta(minutes=minimum_min)].copy()
    long["gap_start"] = long["start"] - long["gap"]
    clock = long["gap_start"] - long["pick_date"]
    long["half_hour"] = (clock.dt.total_seconds() // 1800 * 1800).map(
        lambda s: f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}")
    table = long.groupby("half_hour").agg(gaps=("gap", "size"),
                                         median_min=("gap", lambda g: _minutes(g).median()))
    picker_days = timeline.groupby(["picker", "pick_date"]).ngroups
    rows = []
    for start, end in BREAKS:
        lo, hi = pd.to_timedelta(start + ":00"), pd.to_timedelta(end + ":00")
        gap_lo = long["gap_start"] - long["pick_date"]
        gap_hi = long["start"] - long["pick_date"]
        hits = long[(gap_lo < hi) & (gap_hi > lo)]
        rows.append({"half_hour": f"configured break {start}-{end}",
                     "gaps": f"{hits.groupby(['picker', 'pick_date']).ngroups} of {picker_days} "
                             "picker-days have a long gap overlapping it",
                     "median_min": _minutes(hits["gap"]).median()})
    return pd.concat([table.reset_index(), pd.DataFrame(rows)], ignore_index=True)


def _union(intervals: pd.DataFrame) -> pd.Timedelta:
    total, current = pd.Timedelta(0), None
    for start, end in intervals.sort_values("start").itertuples(index=False):
        if current is None or start > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    return total + (current[1] - current[0])


def presence(timeline: pd.DataFrame) -> pd.DataFrame:
    """Per picker-day: first and last activity, present time, busy time by definition."""
    key = ["picker", "pick_date"]
    days = timeline.groupby(key).agg(first=("start", "min"), last=("end", "max"),
                                     lines=("row", "size"), orders=("order", "nunique"),
                                     line_time=("duration", "sum"))
    days["present"] = days["last"] - days["first"]
    # Lines of concurrently open orders overlap; the union counts that time once.
    days["line_union"] = timeline.groupby(key)[["start", "end"]].apply(_union)
    for minutes in (2, 5, 10, 15):
        idle = timeline["gap"].where(timeline["gap"] > pd.Timedelta(minutes=minutes))
        days[f"active_{minutes}"] = days["present"] - idle.groupby(
            [timeline["picker"], timeline["pick_date"]]).sum()
    return days


def presence_summary(days: pd.DataFrame) -> pd.DataFrame:
    clock_first = _minutes(days["first"] - days.index.get_level_values("pick_date").to_series(
        index=days.index)) / 60
    clock_last = _minutes(days["last"] - days.index.get_level_values("pick_date").to_series(
        index=days.index)) / 60
    present_h = _minutes(days["present"]) / 60
    rows = [
        _describe(clock_first, "first activity [clock hour]"),
        _describe(clock_last, "last activity [clock hour]"),
        _describe(present_h, "present (last - first) [h]"),
        _describe(days["lines"], "lines per picker-day"),
        _describe(days["orders"], "orders per picker-day"),
        _describe(days["line_time"] / days["present"], "line time (sum) / present"),
        _describe(days["line_union"] / days["present"], "line time (union) / present"),
    ]
    for minutes in (2, 5, 10, 15):
        rows.append(_describe(days[f"active_{minutes}"] / days["present"],
                              f"active (gaps > {minutes} min removed) / present"))
    total_present = days["present"].sum()
    totals = pd.DataFrame([{"measure": "pooled line time (sum) / present",
                            "mean": days["line_time"].sum() / total_present},
                           {"measure": "pooled line time (union) / present",
                            "mean": days["line_union"].sum() / total_present}] +
                          [{"measure": f"pooled active_{m} / present",
                            "mean": days[f"active_{m}"].sum() / total_present} for m in (2, 5, 10, 15)])
    return pd.concat([pd.DataFrame(rows), totals], ignore_index=True)


def presence_by_day(days: pd.DataFrame) -> pd.DataFrame:
    grouped = days.groupby(level="pick_date")
    table = pd.DataFrame({
        "pickers": grouped.size(),
        "present_h": grouped["present"].sum().dt.total_seconds() / 3600,
        "line_union_h": grouped["line_union"].sum().dt.total_seconds() / 3600,
        "active_5_h": grouped["active_5"].sum().dt.total_seconds() / 3600,
    })
    table["line_union_share"] = table["line_union_h"] / table["present_h"]
    table["active_5_share"] = table["active_5_h"] / table["present_h"]
    table.index = table.index.date
    return table


# ---------------------------------------------------------------- R0.4

def timestamps(lines: pd.DataFrame) -> pd.DataFrame:
    frame = with_pick_date(lines)
    clocks = pd.concat([lines["start_clock"], lines["end_clock"]])
    duration = lines["end"] - lines["start"]
    content = [c for c in lines.columns if c not in ("row", "seq", "order_gap")]
    split = lines.groupby(["order", "position"]).agg(n=("row", "size"), pallets=("pallet", "nunique"))
    rebooked = frame[frame["rebooked"]]
    majority = frame[~frame["rebooked"]].groupby("order").agg(lo=("start", "min"), hi=("end", "max"),
                                                              pickers=("picker", lambda p: set(p)))
    inside = rebooked.join(majority, on="order")
    same_position = rebooked.set_index(["order", "position"]).index.isin(
        frame[~frame["rebooked"]].set_index(["order", "position"]).index)
    # Rebooked lines that collide with the same picker's other lines on their booking date.
    booked = lines.set_index(["picker", "date"]).sort_index()
    collide = 0
    for line in rebooked.itertuples():
        others = booked.loc[(line.picker, line.date)]
        others = others[others["order"] != line.order]
        # rebooked start/end were shifted to the pick date; compare on the booking date
        lo = line.date + pd.to_timedelta(line.start_clock)
        hi = line.date + pd.to_timedelta(line.end_clock)
        collide += bool(((others["start"] <= hi) & (others["end"] >= lo)).any())
    # The workbook's gap column recomputed in sheet row order
    ordered = lines.sort_values("row")
    nxt = ordered.shift(-1)
    recomputed = (nxt["start"] - ordered["end"]).where(
        (nxt["picker"] == ordered["picker"]) & (nxt["order"] == ordered["order"])
        & (nxt["start"] - ordered["end"] > pd.Timedelta(seconds=10)))
    gap_match = ((recomputed == ordered["order_gap"]) | (recomputed.isna() & ordered["order_gap"].isna()))
    timeline = timelines(lines)
    negative = timeline["gap"] < pd.Timedelta(0)
    within = timeline.loc[timeline["previous_order"] == timeline["order"], "gap"]
    return pd.DataFrame([
        {"check": "clock format HH:MM:SS (seconds resolution)",
         "value": float(clocks.str.fullmatch(r"\d{2}:\d{2}:\d{2}").mean())},
        {"check": "missing start or end", "value": int(lines[["start_clock", "end_clock"]].isna().sum().sum())},
        {"check": "earliest clock start", "value": lines["start_clock"].min()},
        {"check": "latest clock end", "value": lines["end_clock"].max()},
        {"check": "lines across midnight or end < start", "value": int((duration < pd.Timedelta(0)).sum())},
        {"check": "zero-length lines (end = start)", "value": int((duration == pd.Timedelta(0)).sum())},
        {"check": "lines of 1 s", "value": int((duration == pd.Timedelta(seconds=1)).sum())},
        {"check": "exact duplicates (all fields but LFDNR)",
         "value": int(lines.duplicated(subset=content).sum())},
        {"check": "duplicate LFDNR", "value": int(lines["seq"].duplicated().sum())},
        {"check": "order positions with more than one line", "value": int((split["n"] > 1).sum())},
        {"check": "lines in such positions", "value": int(split.loc[split["n"] > 1, "n"].sum())},
        {"check": "of these positions, on more than one pallet",
         "value": int(((split["n"] > 1) & (split["pallets"] > 1)).sum())},
        {"check": "orders with lines on two booking dates",
         "value": int((lines.groupby("order")["date"].nunique() > 1).sum())},
        {"check": "rebooked lines (booking date != pick date)", "value": len(rebooked)},
        {"check": "rebooked lines splitting a position also picked on the pick date",
         "value": float(same_position.mean())},
        {"check": "rebooked lines by a picker of the order on the pick date",
         "value": float(inside.apply(lambda r: r.picker in r.pickers, axis=1).mean())},
        {"check": "rebooked lines inside the order's interval on the pick date (+-5 min)",
         "value": float(((inside["start"] >= inside["lo"] - pd.Timedelta(minutes=5))
                         & (inside["end"] <= inside["hi"] + pd.Timedelta(minutes=5))).mean())},
        {"check": "rebooked lines colliding with the picker's other lines on the booking date",
         "value": collide / max(len(rebooked), 1)},
        {"check": "rebooked lines booked one day after the pick date",
         "value": int((rebooked["date"] - rebooked["pick_date"] == pd.Timedelta(days=1)).sum())},
        {"check": "workbook gap column reproduced in sheet row order", "value": float(gap_match.mean())},
        {"check": "lines starting before the same picker's previous line ended",
         "value": int(negative.sum())},
        {"check": "of these, overlap > 10 s",
         "value": int((timeline["gap"] < pd.Timedelta(seconds=-10)).sum())},
        {"check": "of the overlaps > 10 s, at an order change (share)",
         "value": float(timeline.loc[timeline["gap"] < pd.Timedelta(seconds=-10), "order_change"].mean())},
        {"check": "within an order, line starts exactly at the previous line's end (share)",
         "value": float((within == pd.Timedelta(0)).mean())},
        {"check": "within an order, gap of 0 or 1 s (share)",
         "value": float(within.between(pd.Timedelta(0), pd.Timedelta(seconds=1)).mean())},
    ])


# ---------------------------------------------------------------- R0.5

def stores_and_units(lines: pd.DataFrame) -> pd.DataFrame:
    frame = with_pick_date(lines)
    per_day = frame.groupby(["pick_date", "customer"])["order"].nunique()
    per_pallet = lines.groupby("pallet")["order"].nunique()
    return pd.DataFrame([
        {"measure": "orders with more than one customer",
         "value": int((lines.groupby("order")["customer"].nunique() > 1).sum())},
        {"measure": "customers", "value": lines["customer"].nunique()},
        {"measure": "customer-days", "value": len(per_day)},
        {"measure": "customer-days with 1 order", "value": int((per_day == 1).sum())},
        {"measure": "customer-days with 2 orders", "value": int((per_day == 2).sum())},
        {"measure": "customer-days with 3+ orders", "value": int((per_day >= 3).sum())},
        {"measure": "max orders per customer-day", "value": int(per_day.max())},
        {"measure": "pallets", "value": lines["pallet"].nunique()},
        {"measure": "pallets with more than one order", "value": int((per_pallet > 1).sum())},
        {"measure": "pallets with lines on two booking dates",
         "value": int((lines.groupby("pallet")["date"].nunique() > 1).sum())},
        {"measure": "TU per order in the workbook (formula)", "value": TU_PER_ORDER_WORKBOOK},
        {"measure": "TU per order in the submitted simulation", "value": TU_PER_ORDER_SIMULATION},
        {"measure": "columns naming a truck, route, tour or departure", "value": 0},
    ])


def pallets_per_day(lines: pd.DataFrame) -> pd.DataFrame:
    frame = with_pick_date(lines)
    grouped = frame.groupby("pick_date")
    table = pd.DataFrame({"orders": grouped["order"].nunique(), "pallets": grouped["pallet"].nunique(),
                          "customers": grouped["customer"].nunique()})
    table["tu_workbook"] = table["orders"] * TU_PER_ORDER_WORKBOOK
    table["pallets_per_order"] = table["pallets"] / table["orders"]
    table["tu_per_pallet"] = table["tu_workbook"] / table["pallets"]
    table.index = table.index.date
    return table


def disruption_sheets(lines: pd.DataFrame) -> pd.DataFrame:
    frame = with_pick_date(lines)
    orders = frame.groupby("order").agg(customer=("customer", "first"), pick_date=("pick_date", "first"),
                                        lines=("row", "size"), pallets=("pallet", "nunique"),
                                        start=("start", "min"), end=("end", "max"))
    tu = {"S1": 120, "S2": 100, "S3": 6, "S4": 80, "S5": 200}
    rows = []
    for sheet, group in export.disruption_orders(CACHE).groupby("sheet"):
        sheet_date = group["date"].iloc[0]
        selected = orders.reindex(group["order"])
        same_customer = orders[(orders["customer"].isin(selected["customer"]))
                               & (orders["pick_date"] == sheet_date)]
        clock_end = selected["end"] - selected["pick_date"]
        rows.append({
            "sheet": sheet, "sheet_date": sheet_date.date(), "tu": tu[sheet],
            "orders": len(group), "tu_over_2.6": tu[sheet] / TU_PER_ORDER_WORKBOOK,
            "tu_over_1.6": tu[sheet] / TU_PER_ORDER_SIMULATION,
            "orders_found": int(selected["customer"].notna().sum()),
            "orders_picked_previous_day": int((selected["pick_date"] < sheet_date).sum()),
            "customers": selected["customer"].nunique(),
            "all_orders_of_these_customers_that_day": len(same_customer),
            "lines": int(selected["lines"].sum()), "pallets": int(selected["pallets"].sum()),
            "end_clock_min": str(clock_end.min()).split()[-1][:5],
            "end_clock_max": str(clock_end.max()).split()[-1][:5],
        })
    template = export.disruption_lines(CACHE)
    rows.append({"sheet": "S3", "sheet_date": template["date"].min().date(), "tu": tu["S3"],
                 "orders": 0, "tu_over_2.6": tu["S3"] / TU_PER_ORDER_WORKBOOK,
                 "tu_over_1.6": tu["S3"] / TU_PER_ORDER_SIMULATION,
                 "customers": template["customer"].nunique(), "lines": len(template),
                 "orders_found": None, "orders_picked_previous_day": None,
                 "all_orders_of_these_customers_that_day": None, "pallets": None,
                 "end_clock_min": "", "end_clock_max": ""})
    return pd.DataFrame(rows).sort_values("sheet")


def completion_by_customer_hour(lines: pd.DataFrame) -> pd.DataFrame:
    """Completion clock of orders: the only trace of departures the export holds."""
    orders = with_pick_date(lines).groupby("order").agg(pick_date=("pick_date", "first"),
                                                         end=("end", "max"))
    hour = ((orders["end"] - orders["pick_date"]).dt.total_seconds() // 3600).astype(int)
    table = pd.crosstab(hour, orders["pick_date"].dt.date)
    table.index.name = "completion hour"
    return table


# ----------------------------------------------------------------

def run() -> dict[str, pd.DataFrame]:
    lines = export.lines(CACHE)
    timeline = timelines(lines)
    days = presence(timeline)
    orders = order_structure(lines)
    return {
        "r01_columns": inventory_columns(lines),
        "r01_days": inventory_days(lines),
        "r01_values": inventory_values(lines),
        "r01_categories": inventory_categories(lines),
        "r01_location_code": location_code(lines),
        "r01_paper_filter": paper_filter(lines),
        "r02_orders_summary": order_structure_summary(orders),
        "r02_order_areas_by_day": order_areas_by_day(orders),
        "r02_store_days": store_days(lines),
        "r02_concurrency": concurrency(lines),
        "r02_picker_zones": picker_zones(lines),
        "r02_interleaving": interleaving(lines),
        "r18_zone_affinity": zone_affinity(lines),
        "r03_gaps": gap_summary(timeline),
        "r03_gap_thresholds": gap_thresholds(timeline),
        "r03_break_gaps": break_gaps(timeline),
        "r03_presence": presence_summary(days),
        "r03_presence_by_day": presence_by_day(days),
        "r04_timestamps": timestamps(lines),
        "r05_stores_units": stores_and_units(lines),
        "r05_pallets_per_day": pallets_per_day(lines),
        "r05_disruption_sheets": disruption_sheets(lines),
        "r05_completion_hours": completion_by_customer_hour(lines),
        "private/r02_orders": orders,
        "private/r03_presence": days,
    }


def main() -> None:
    tables = run()
    (OUTPUTS / "private").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    for name, table in tables.items():
        table.to_csv(OUTPUTS / f"{name}.csv")
        if not name.startswith("private/"):
            print(f"\n## {name}\n{table.to_string()}")
    print(f"\nwritten to {OUTPUTS}")


if __name__ == "__main__":
    main()
