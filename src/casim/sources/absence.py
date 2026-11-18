"""A fitted model of picker absences during the shift (distribution-driven days).

A recorded replay takes every absence from the record: pauses between tours
and stops inside tours over the threshold. A fitted day draws them instead,
from a model fitted on other recorded days (in the IJPE scenario,
``source.absences=fitted``). A picker's presence runs from its arrival to its
departure; the roster is the presence of every picker of a day. The model has
two parts:

- scheduled breaks: for each break window (for example 07:00 to 09:30), a
  picker's longest recorded absence starting in the window and lasting at
  least ``break_min_s`` is its break. A picker present in the window takes
  the break with the share of such picker-days that took it on the fit days;
  its request time and duration are drawn together from the recorded breaks
  of that window;
- allowance (personal needs, fatigue, delays): every other recorded absence.
  Requests follow a Poisson process per picker over its presence, which has
  three parts: start-up (the first hour), wind-down (the last hour) and the
  regular time between. Each part has its own rate (allowance absences per
  present picker-hour in that part) and draws durations from the recorded
  allowance absences requested in it.

As with recorded absences, an absence begins when the picker's running tour
ends. The draws depend only on the seed and the roster, not on the
decisions, so every policy of a replication sees the same absences.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

HOUR = 3600.0
EDGE_S = HOUR                               # length of start-up and of wind-down
Interval = tuple[float, float]              # (start, end) in seconds
Absence = tuple[int, float, float]          # picker, request time, duration
Roster = dict[int, Interval]                # picker -> presence
PARTS = ("start-up", "regular", "wind-down")


def _overlaps(a: Interval, b: Interval) -> bool:
    return min(a[1], b[1]) > max(a[0], b[0])


def parts(presence: Interval) -> dict[str, Interval]:
    """Start-up, regular time and wind-down of one picker's presence."""
    start, end = presence
    regular_from = min(end, start + EDGE_S)
    regular_to = max(regular_from, end - EDGE_S)
    return {"start-up": (start, regular_from), "regular": (regular_from, regular_to),
            "wind-down": (regular_to, end)}


def part_of(presence: Interval, time: float) -> str:
    """The part of ``presence`` in which ``time`` lies."""
    start_up, regular, _ = parts(presence).values()
    if time < start_up[1]:
        return "start-up"
    if time < regular[1]:
        return "regular"
    return "wind-down"


def split_breaks(roster: Roster, absences: Sequence[Absence],
                 break_windows: Sequence[Interval], break_min_s: float
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
        present.append(sum(_overlaps(presence, window) for presence in roster.values()))
    allowance = [a for index, a in enumerate(absences) if index not in used]
    return breaks, present, allowance


@dataclass(frozen=True)
class Break:
    window: Interval
    share: float                              # of present pickers that take it
    samples: tuple[tuple[float, float], ...]  # (request time, duration)


@dataclass(frozen=True)
class Allowance:
    rate: float                               # per present picker-hour
    durations: tuple[float, ...]


@dataclass(frozen=True)
class AbsenceModel:
    breaks: tuple[Break, ...]
    allowance: dict[str, Allowance]           # by part of the presence (PARTS)

    @classmethod
    def fit(cls, days: Sequence[tuple[Roster, Sequence[Absence]]],
            break_windows: Sequence[Interval], break_min_s: float) -> AbsenceModel:
        """Fit on the roster and the recorded absences of each fit day."""
        taken = [[] for _ in break_windows]
        present = [0] * len(break_windows)
        durations: dict[str, list[float]] = {name: [] for name in PARTS}
        hours = dict.fromkeys(PARTS, 0.0)
        for roster, absences in days:
            day_breaks, day_present, day_allowance = split_breaks(
                roster, absences, break_windows, break_min_s)
            for k in range(len(break_windows)):
                taken[k].extend(day_breaks[k])
                present[k] += day_present[k]
            for picker, time, duration in day_allowance:
                durations[part_of(roster[picker], time)].append(duration)
            for presence in roster.values():
                for name, (low, high) in parts(presence).items():
                    hours[name] += (high - low) / HOUR
        breaks = tuple(Break(window=tuple(window),
                             share=len(taken[k]) / present[k] if present[k] else 0.0,
                             samples=tuple(sorted(taken[k])))
                       for k, window in enumerate(break_windows))
        allowance = {name: Allowance(rate=len(durations[name]) / hours[name] if hours[name] else 0.0,
                                     durations=tuple(sorted(durations[name])))
                     for name in PARTS}
        return cls(breaks=breaks, allowance=allowance)

    def draw(self, roster: Roster, rng: np.random.Generator) -> list[Absence]:
        """(picker, request time, duration) for each picker of the day.

        A drawn break whose time lies outside the picker's presence is not taken.
        """
        absences: list[Absence] = []
        for picker in sorted(roster):
            presence = roster[picker]
            for brk in self.breaks:
                if brk.samples and _overlaps(presence, brk.window) and rng.random() < brk.share:
                    time, duration = brk.samples[rng.integers(len(brk.samples))]
                    if presence[0] <= time <= presence[1]:
                        absences.append((picker, float(time), float(duration)))
            for name, (low, high) in parts(presence).items():
                allowance = self.allowance[name]
                if not allowance.durations:
                    continue
                count = rng.poisson(allowance.rate * (high - low) / HOUR)
                for time in np.sort(rng.uniform(low, high, count)):
                    duration = allowance.durations[rng.integers(len(allowance.durations))]
                    absences.append((picker, float(time), float(duration)))
        return absences
