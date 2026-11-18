"""A fitted model of picker absences during the shift (distribution-driven days).

A recorded replay takes every absence from the record: pauses between tours
and stops inside tours over the threshold. A fitted day draws them instead,
from a model fitted on other recorded days (in the IJPE scenario,
``source.absence=stochastic``). The model has two parts:

- scheduled breaks: for each break window (for example around 08:00 to 09:30),
  a picker's longest recorded absence starting in the window and lasting at
  least ``break_min_s`` is its break. A picker whose roster overlaps the
  window takes the break with the share of such picker-days that took it on
  the fit days; its request time and duration are drawn together from the
  recorded breaks of that window;
- allowance (personal needs, fatigue, delays): every other recorded absence.
  Requests follow a Poisson process per picker over its rostered window. The
  window has three segments: start-up (the first ``edge_s``), wind-down (the
  last ``edge_s``) and the regular time between; each has its own rate
  (allowance absences per present picker-hour in that segment) and draws
  durations from the recorded allowance absences requested in it.

As with recorded absences, an absence begins when the picker's running tour
ends. The draws depend only on the seed, the day and the roster, not on the
decisions, so every policy of a replication sees the same absences.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from casim.sources.recorded import RecordedReplay

HOUR = 3600.0
Window = tuple[float, float]
SEGMENTS = ("start", "regular", "end")
EDGE_S = 3600.0
Absence = tuple[int, float, float]          # picker, request time, duration


def windows(replay: RecordedReplay) -> dict[int, Window]:
    """Each picker's rostered window: setup start of its first tour to its last line."""
    result: dict[int, Window] = {}
    for picker, lines in replay.tours:
        start = replay._setup_start(lines)
        end = max(row.end_s for row in lines)
        if picker in result:
            start, end = min(start, result[picker][0]), max(end, result[picker][1])
        result[picker] = (start, end)
    return result


def recorded_absences(replay: RecordedReplay) -> list[Absence]:
    """(picker, request time, duration) of every recorded absence of a chained replay."""
    return [(d.picker_id, d.time, d.duration) for d in replay._pauses()]


def _overlaps(roster: Window, window: Window) -> bool:
    return min(roster[1], window[1]) > max(roster[0], window[0])


def split_breaks(roster: dict[int, Window], absences: Sequence[Absence],
                 break_windows: Sequence[Window], break_min_s: float
                 ) -> tuple[list[list[tuple[float, float]]], list[int], list[Absence]]:
    """Per break window, the breaks taken and the pickers present; the allowance."""
    breaks: list[list[tuple[float, float]]] = []
    present: list[int] = []
    used: set[int] = set()
    for window in break_windows:
        taken = []
        for picker in roster:
            candidates = [(duration, index) for index, (p, time, duration) in enumerate(absences)
                          if p == picker and index not in used
                          and window[0] <= time < window[1] and duration >= break_min_s]
            if candidates:
                _, index = max(candidates)
                used.add(index)
                taken.append(absences[index][1:])
        breaks.append(taken)
        present.append(sum(_overlaps(r, window) for r in roster.values()))
    allowance = [a for index, a in enumerate(absences) if index not in used]
    return breaks, present, allowance


@dataclass(frozen=True)
class Break:
    window: Window
    share: float                              # of present pickers that take it
    samples: tuple[tuple[float, float], ...]  # (request time, duration)


def segments(roster: Window, edge_s: float = EDGE_S) -> dict[str, Window]:
    """Start-up, regular and wind-down part of a rostered window."""
    start, end = roster
    first = min(end, start + edge_s)
    last = max(first, end - edge_s)
    return {"start": (start, first), "regular": (first, last), "end": (last, end)}


def segment_of(roster: Window, time: float, edge_s: float = EDGE_S) -> str:
    return next(name for name, (low, high) in segments(roster, edge_s).items()
                if time < high or name == "end")


@dataclass(frozen=True)
class Allowance:
    rate: float                               # per present picker-hour
    durations: tuple[float, ...]


@dataclass(frozen=True)
class AbsenceModel:
    breaks: tuple[Break, ...]
    allowance: dict[str, Allowance]           # by segment of the roster

    @classmethod
    def fit(cls, days: Sequence[tuple[dict[int, Window], Sequence[Absence]]],
            break_windows: Sequence[Window], break_min_s: float) -> AbsenceModel:
        """Fit on (roster windows, recorded absences) of each fit day."""
        taken = [[] for _ in break_windows]
        present = [0] * len(break_windows)
        durations: dict[str, list[float]] = {name: [] for name in SEGMENTS}
        exposure = dict.fromkeys(SEGMENTS, 0.0)
        for roster, absences in days:
            day_breaks, day_present, day_allowance = split_breaks(
                roster, absences, break_windows, break_min_s)
            for k in range(len(break_windows)):
                taken[k].extend(day_breaks[k])
                present[k] += day_present[k]
            for picker, time, duration in day_allowance:
                durations[segment_of(roster[picker], time)].append(duration)
            for window in roster.values():
                for name, (low, high) in segments(window).items():
                    exposure[name] += (high - low) / HOUR
        breaks = tuple(Break(window=tuple(window), share=len(taken[k]) / present[k] if present[k] else 0.0,
                             samples=tuple(sorted(taken[k])))
                       for k, window in enumerate(break_windows))
        allowance = {name: Allowance(
            rate=len(durations[name]) / exposure[name] if exposure[name] else 0.0,
            durations=tuple(sorted(durations[name]))) for name in SEGMENTS}
        return cls(breaks=breaks, allowance=allowance)

    def draw(self, roster: dict[int, Window], rng: np.random.Generator) -> list[Absence]:
        """(picker, request time, duration) for each picker of the day."""
        absences: list[Absence] = []
        for picker in sorted(roster):
            start, end = roster[picker]
            for brk in self.breaks:
                if not brk.samples or not _overlaps((start, end), brk.window):
                    continue
                if rng.random() < brk.share:
                    time, duration = brk.samples[rng.integers(len(brk.samples))]
                    if start <= time <= end:
                        absences.append((picker, float(time), float(duration)))
            for name, (low, high) in segments((start, end)).items():
                part = self.allowance.get(name)
                if part is None or not part.durations:
                    continue
                count = rng.poisson(part.rate * (high - low) / HOUR)
                for time in np.sort(rng.uniform(low, high, count)):
                    duration = part.durations[rng.integers(len(part.durations))]
                    absences.append((picker, float(time), float(duration)))
        return absences
