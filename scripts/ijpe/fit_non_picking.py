"""Fit the IJPE non-picking time: breaks, tour setup and line time (calibration).

For each study day, fits on the other five days (leave one day out) what the
recorded days show of three parts, each measured on a picker's time without
any WMS record, and writes the fit to ``scenarios/ijpe/data/non_picking/<day>.json``
(confidential, not committed). ``source.non_picking=fitted`` reads it; the
source realizes the times of a run from it (``casim.domain_objects.process_truth``).

- breaks, tied to the clock: per break window (``BREAK_WINDOWS`` of the
  source), a picker's longest spell without a record that starts in the
  window and lasts at least ``BREAK_MIN_S`` is its break: a gap between tours,
  a gap inside a tour, or a line that long. Written: per window the share of
  present picker-days that took it and the (start, duration) of each break;
- tour setup: the time without a record between the end of a picker's tour and
  the first line of its next tour, net of a break in it and of the walk from
  the last line to the depot and from the depot to the first line, which the
  simulation walks itself; a tour that starts before the previous one ended
  gives zero. Written: the samples;
- line time: a line's recorded duration plus the time without a record before
  it inside its tour, net of the walk from the previous line of the tour, per
  class of quantity (``QUANTITY_BOUNDS``). A line that holds another record of
  the picker inside it, or overlaps a record outside the replayed lines, is
  bookkeeping (its time is counted twice) and left out; so is a line that is
  a break. A line inside another one, which a picker working on two orders at
  once leaves, is kept. Written: the samples per class.

Only what one person did is a sample. The system lets a login work on one
order at a time (the use case partner), yet the scan windows (first to last
line start) of some orders of one login overlap: two people worked under it,
or the records are wrong, and the scans do not tell how many people. So an
order that overlaps another order of its login gives no line sample, a gap
next to such an order no setup sample, and a picker-day with such an order
neither a break nor a presence in the break windows; breaks are still found
on every picker-day, so that no sample holds one. Time covered by a picker's
records outside the replayed lines (lines without a pick node, orders of
several pickers) is work and is left out of every part.

The check (``Observations.own_lines``): every line's time measured on its
own order's scans alone (its duration plus the time since the order's
previous line ended, net of the walk), for orders with and without overlap.
If the overlapping orders were picked by a person of their own, their lines
look like the others. The source settings (speed, setup, presence) are those of
``variants=calibrated``.

    python -m scripts.ijpe.fit_non_picking

Prints aggregates only.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from casim.run import compose_scenario
from casim.sources.recorded import RecordedReplay, _uncovered
from casim.sources.wms_export import lines as export_lines, with_pick_date
from scenarios.ijpe.source import BREAK_MIN_S, BREAK_WINDOWS, STUDY_DAYS, ScenarioSource
from scripts.ijpe import CACHE, DATA, SCENARIO

FIT_DIR = DATA / "non_picking"

Interval = tuple[float, float]               # (start, end) in seconds
QUANTITY_BOUNDS = (6, 12, 24)                # classes: <= 6, 7-12, 13-24, > 24 packages


def quantity_class(quantity: float, bounds: Sequence[int] = QUANTITY_BOUNDS) -> int:
    return bisect_left(bounds, quantity)


def _merged(intervals: Sequence[Interval]) -> list[Interval]:
    merged: list[list[float]] = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return [(low, high) for low, high in merged]


def _overlap(a: Interval, b: Interval) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


@dataclass
class Observations:
    """What one recorded day shows of the three parts, and what was left out."""

    windows: tuple[Interval, ...]
    breaks: list[list[tuple[float, float]]]   # per window: (start, duration)
    present: list[int]                        # per window: pickers present
    setups: list[float] = field(default_factory=list)
    lines: list[tuple[int, float]] = field(default_factory=list)   # (quantity, seconds)
    lines_bookkeeping: int = 0                # left out: hold another record, or overlap one outside
    lines_as_breaks: int = 0
    clipped_lines: int = 0                    # net of walking below zero
    clipped_setups: int = 0
    covered_s: float = 0.0                    # gap time covered by records outside the lines
    orders: int = 0
    orders_overlapping: int = 0               # overlap another order of the picker: no sample
    setups_left_out: int = 0                  # next to an overlapping order
    picker_days: int = 0
    picker_days_left_out: int = 0             # with an overlapping order: no break sample
    # The check: (quantity, seconds, overlapping) per line, on its order's scans alone.
    own_lines: list[tuple[int, float, bool]] = field(default_factory=list)

    def non_picking_s(self) -> dict[str, float]:
        """Seconds of each part on this day (lines: net of walking, all lines kept)."""
        return {"breaks": sum(d for window in self.breaks for _, d in window),
                "setup": sum(self.setups), "line": sum(s for _, s in self.lines)}


def observe(replay: RecordedReplay, foreign: Mapping[int, Sequence[Interval]],
            break_windows: Sequence[Interval], break_min_s: float) -> Observations:
    """Breaks, setups and line times of one recorded day.

    ``foreign`` holds, per picker, the intervals of its records outside the
    replayed lines; they count as work.
    """
    network = replay.layout.layout_network
    matrix = network.distance_matrix
    index = {node: i for i, node in enumerate(matrix.index)}
    distance = matrix.to_numpy()
    speed = replay.time_model.speed

    def walk(a, b) -> float:
        return float(distance[index[a], index[b]]) / speed

    roster = replay.roster()
    windows = tuple(tuple(w) for w in break_windows)
    obs = Observations(windows=windows, breaks=[[] for _ in windows], present=[0 for _ in windows])
    by_picker: dict[int, list[list]] = defaultdict(list)
    for picker, lines in replay.tours:
        by_picker[picker].append(sorted(lines, key=lambda r: r.start_s))
    for picker, tours in by_picker.items():
        overlapping = overlapping_orders(tours)
        one_person = not any(overlapping)
        obs.orders += len(tours)
        obs.orders_overlapping += sum(overlapping)
        obs.picker_days += 1
        obs.picker_days_left_out += not one_person
        if one_person:
            for w, window in enumerate(windows):
                obs.present[w] += _overlap(roster[picker], window) > 0
        for lines, overlaps in zip(tours, overlapping):
            reach = lines[0].end_s
            for k, row in enumerate(lines):
                gap = 0.0 if k == 0 else max(0.0, row.start_s - reach) - walk(lines[k - 1].node, row.node)
                obs.own_lines.append((row.quantity, max(0.0, row.end_s - row.start_s + gap), overlaps))
                reach = max(reach, row.end_s)
        own = [(r.start_s, r.end_s) for lines in tours for r in lines]
        other = [tuple(map(float, i)) for i in foreign.get(picker, ())]
        busy_own, busy = _merged(own), _merged(own + other)
        records = sorted([(s, e, id(r)) for lines in tours for r in lines
                          for s, e in [(r.start_s, r.end_s)]]
                         + [(s, e, -1 - k) for k, (s, e) in enumerate(other)])
        starts = [s for s, _, _ in records]
        longest = max((e - s for s, e, _ in records), default=0.0)

        def bookkeeping(row) -> bool:
            # The line holds another record of the picker inside it, or
            # overlaps a record outside the lines: its time is not its own.
            low = bisect_left(starts, row.start_s - longest)
            high = bisect_left(starts, row.end_s)
            span = (row.start_s, row.end_s)
            return any(key != id(row) and _overlap((s, e), span) > 0
                       and (key < 0 or (row.start_s <= s and e <= row.end_s))
                       for s, e, key in records[low:high])

        # Items whose time is a sample: ("setup", pieces, walk) or
        # ("line", pieces, walk, row). Pieces are spells without a record.
        items: list[list] = []
        previous_end, previous_last = None, None
        for t, lines in enumerate(tours):
            first = lines[0]
            if previous_end is not None:
                gap = (previous_end, first.start_s)
                pieces = _uncovered(*gap, busy) if gap[1] > gap[0] else []
                if gap[1] > gap[0]:
                    obs.covered_s += (sum(h - l for l, h in _uncovered(*gap, busy_own))
                                      - sum(h - l for l, h in pieces))
                items.append(["setup", pieces, walk(previous_last.node, network.end_node)
                              + walk(network.start_node, first.node),
                              overlapping[t] or overlapping[t - 1]])
            reach = first.end_s
            for k, row in enumerate(lines):
                if k == 0:
                    pieces, step = [], 0.0
                else:
                    pieces = _uncovered(reach, row.start_s, busy) if row.start_s > reach else []
                    step = walk(lines[k - 1].node, row.node)
                    reach = max(reach, row.end_s)
                items.append(["line", pieces, step, row, bookkeeping(row), overlapping[t]])
            end = max(r.end_s for r in lines)
            previous_end = end if previous_end is None else max(previous_end, end)
            previous_last = lines[-1]
        # Break candidates: (duration, start, item, piece index or None for a line).
        candidates = []
        for n, item in enumerate(items):
            for p, (low, high) in enumerate(item[1]):
                if high - low >= break_min_s:
                    candidates.append((high - low, low, n, p))
            if item[0] == "line" and not item[4]:
                row = item[3]
                if row.end_s - row.start_s >= break_min_s:
                    candidates.append((row.end_s - row.start_s, row.start_s, n, None))
        taken: set[tuple[int, int | None]] = set()
        for w, window in enumerate(windows):
            if _overlap(roster[picker], window) <= 0:
                continue
            inside = [c for c in candidates if window[0] <= c[1] < window[1]
                      and (c[2], c[3]) not in taken
                      and (c[2], None) not in taken]
            if inside:
                duration, start, n, p = max(inside)
                taken.add((n, p))
                if one_person:
                    obs.breaks[w].append((start, duration))
        for n, item in enumerate(items):
            free = sum(high - low for p, (low, high) in enumerate(item[1]) if (n, p) not in taken)
            if item[0] == "setup":
                if item[3]:
                    obs.setups_left_out += 1
                    continue
                value = free - item[2]
                obs.clipped_setups += value < 0
                obs.setups.append(max(0.0, value))
                continue
            _, _, step, row, bookkeeping, overlaps = item
            if overlaps:
                continue
            if (n, None) in taken:
                obs.lines_as_breaks += 1
            elif bookkeeping:
                obs.lines_bookkeeping += 1
            else:
                value = row.end_s - row.start_s + free - step
                obs.clipped_lines += value < 0
                obs.lines.append((row.quantity, max(0.0, value)))
    return obs


def overlapping_orders(tours: Sequence[Sequence]) -> list[bool]:
    """Per tour (one order, lines by start) of one picker: does its scan window
    (first to last line start) overlap another one's? Touching windows do not."""
    windows = [(lines[0].start_s, lines[-1].start_s) for lines in tours]
    return [any(i != j and a[0] < b[1] and b[0] < a[1] for j, b in enumerate(windows))
            for i, a in enumerate(windows)]


def foreign_records(source: ScenarioSource) -> dict[int, list[Interval]]:
    """Per picker of the day, the intervals of its records outside the replayed lines.

    Records of the day (by pick date or by start) of the same pickers that
    the source left out: lines without a pick node and orders of several pickers.
    """
    midnight = pd.Timestamp(source.day)
    frame = with_pick_date(export_lines(CACHE))
    kept = set(source.rows_kept)
    frame = frame[((frame["pick_date"] == midnight) | (frame["start"].dt.normalize() == midnight))
                  & ~frame["row"].isin(kept) & frame["picker"].isin(source.picker_ids)]
    foreign: dict[int, list[Interval]] = {}
    for record in frame.itertuples():
        foreign.setdefault(source.picker_ids[record.picker], []).append(
            ((record.start - midnight).total_seconds(), (record.end - midnight).total_seconds()))
    return foreign


def fit(days: Sequence[Observations], bounds: Sequence[int] = QUANTITY_BOUNDS) -> dict:
    """The fit written for one held-out day: pooled over ``days``."""
    breaks = []
    for w, window in enumerate(days[0].windows):
        samples = sorted(s for day in days for s in day.breaks[w])
        present = sum(day.present[w] for day in days)
        breaks.append({"window": list(window), "share": len(samples) / present if present else 0.0,
                       "samples": [list(s) for s in samples]})
    by_class: list[list[float]] = [[] for _ in range(len(bounds) + 1)]
    for day in days:
        for quantity, seconds in day.lines:
            by_class[quantity_class(quantity, bounds)].append(seconds)
    pooled = sorted(s for c in by_class for s in c)
    return {"breaks": breaks,
            "setup_s": sorted(s for day in days for s in day.setups),
            "quantity_bounds": list(bounds),
            "line_s": [sorted(c) if c else pooled for c in by_class]}


def main() -> None:
    cfg = compose_scenario(SCENARIO, ["variants=calibrated"])
    settings = OmegaConf.to_container(cfg.source, resolve=True)
    settings = {key: settings[key] for key in (
        "shift_start_s", "pause_threshold_s", "speed", "time_per_pick", "tour_setup_time",
        "per_unit_s", "setup_in_gap", "presence")}
    observed = {}
    for day in STUDY_DAYS:
        source = ScenarioSource(DATA, day=day, **settings)
        observed[day] = observe(source.replay, foreign_records(source), BREAK_WINDOWS, BREAK_MIN_S)
        o = observed[day]
        hours = {k: round(v / 3600, 1) for k, v in o.non_picking_s().items()}
        print(f"{day}: orders overlapping {o.orders_overlapping} of {o.orders}, picker-days left out "
              f"{o.picker_days_left_out} of {o.picker_days}, setups left out {o.setups_left_out}; "
              f"breaks {[len(b) for b in o.breaks]} of {o.present} present, hours {hours}, "
              f"bookkeeping lines {o.lines_bookkeeping}, lines as breaks {o.lines_as_breaks}, "
              f"covered by outside records {o.covered_s / 3600:.1f} h")
    own = pd.DataFrame([row for o in observed.values() for row in o.own_lines],
                       columns=["quantity", "seconds", "overlapping"])
    own["class"] = own["quantity"].map(quantity_class)
    check = own.groupby(["class", "overlapping"])["seconds"].agg(["size", "median", "mean"]).unstack()
    print("Check, line time on its own order's scans (s), by quantity class and overlap:")
    print(check.round(1).to_string())
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    for day in STUDY_DAYS:
        result = fit([observed[other] for other in STUDY_DAYS if other != day])
        (FIT_DIR / f"{day}.json").write_text(json.dumps(result))
        means = [round(float(np.mean(c)), 1) for c in result["line_s"]]
        print(f"fit for {day}: setup mean {np.mean(result['setup_s']):.0f} s, line means {means}, "
              f"break shares {[round(b['share'], 2) for b in result['breaks']]}")


if __name__ == "__main__":
    main()
