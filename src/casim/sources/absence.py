"""A fitted model of picker absences during the shift (distribution-driven days).

A recorded replay takes every absence from the record: pauses between tours
and stops inside tours over the threshold. A fitted day draws them instead,
from a model fitted on other recorded days (in the IJPE scenario,
``source.absence=stochastic``):

- requests: per picker, a Poisson process over the picker's rostered window
  (setup start of its first tour to its last line) whose rate is constant
  within each clock hour: recorded absences starting in that hour per
  present picker-hour;
- durations: drawn from the recorded absences that started within 30 minutes
  of the request's clock time (widened to 60 and 120 minutes when fewer
  than ``MIN_SAMPLE`` remain).

As with recorded absences, an absence begins when the picker's running tour
ends. The draws depend only on the seed, the day and the roster, not on the
decisions, so every policy of a replication sees the same absences.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from casim.sources.recorded import RecordedReplay

MIN_SAMPLE = 20
HOUR = 3600.0


def windows(replay: RecordedReplay) -> dict[int, tuple[float, float]]:
    """Each picker's rostered window: setup start of its first tour to its last line."""
    result: dict[int, tuple[float, float]] = {}
    for picker, lines in replay.tours:
        start = replay._setup_start(lines)
        end = max(row.end_s for row in lines)
        if picker in result:
            start, end = min(start, result[picker][0]), max(end, result[picker][1])
        result[picker] = (start, end)
    return result


def recorded_absences(replay: RecordedReplay) -> list[tuple[float, float]]:
    """(request time, duration) of every recorded absence of a chained replay."""
    return [(d.time, d.duration) for d in replay._pauses()]


@dataclass(frozen=True)
class AbsenceModel:
    rates: dict[int, float]                 # clock hour -> requests per present picker-hour
    samples: tuple[tuple[float, float], ...]  # (request time, duration) of the fit days

    @classmethod
    def fit(cls, days: Sequence[tuple[dict[int, tuple[float, float]],
                                      list[tuple[float, float]]]]) -> AbsenceModel:
        """Fit on (windows, recorded absences) of each fit day."""
        exposure: dict[int, float] = {}
        counts: dict[int, int] = {}
        samples = []
        for day_windows, absences in days:
            for start, end in day_windows.values():
                for hour in range(int(start // HOUR), int(end // HOUR) + 1):
                    overlap = min(end, (hour + 1) * HOUR) - max(start, hour * HOUR)
                    if overlap > 0:
                        exposure[hour] = exposure.get(hour, 0.0) + overlap / HOUR
            for time, duration in absences:
                hour = int(time // HOUR)
                counts[hour] = counts.get(hour, 0) + 1
                samples.append((time, duration))
        rates = {hour: counts.get(hour, 0) / hours for hour, hours in exposure.items()
                 if hours > 0}
        return cls(rates=rates, samples=tuple(sorted(samples)))

    def _duration(self, time: float, rng: np.random.Generator) -> float:
        times = np.array([t for t, _ in self.samples])
        durations = np.array([d for _, d in self.samples])
        for width in (1800.0, 3600.0, 7200.0, np.inf):
            near = durations[np.abs(times - time) <= width]
            if len(near) >= MIN_SAMPLE or width == np.inf:
                return float(rng.choice(near))
        raise AssertionError("unreachable")

    def draw(self, day_windows: dict[int, tuple[float, float]], rng: np.random.Generator
             ) -> list[tuple[int, float, float]]:
        """(picker, request time, duration) for each picker of the day."""
        absences = []
        for picker in sorted(day_windows):
            start, end = day_windows[picker]
            for hour in range(int(start // HOUR), int(end // HOUR) + 1):
                low, high = max(start, hour * HOUR), min(end, (hour + 1) * HOUR)
                if high <= low:
                    continue
                count = rng.poisson(self.rates.get(hour, 0.0) * (high - low) / HOUR)
                for time in np.sort(rng.uniform(low, high, count)):
                    absences.append((picker, float(time), self._duration(time, rng)))
        return absences
