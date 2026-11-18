"""Seconds per line of the calibrated day: tour time net of stops, minus walking.

A recorded tour's time is its first line start to its last line end. Stops
over the pause threshold inside it (a gap between lines, or a line's excess
over the submitted 19 s) are absences in the calibrated day
(``source.stops_in_tours``), so they are taken out. CASIM's walking time
(its route distance at the scenario speed) is taken out too. What is left,
divided by the lines, is the time per line. It is fitted on Monday 13 to
Wednesday 15 April and checked per tour on the other three days.

Reads the export and the tours of ``calibrated_recorded_arrival`` (every
tour done; routing does not depend on the line time). Prints aggregates only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scenarios.ijpe.source import ScenarioSource
from scripts.ijpe import DATA, OUTPUTS

DAYS = ("2026-04-11", "2026-04-13", "2026-04-14", "2026-04-15",
        "2026-04-16", "2026-04-17")
TRAIN = {"2026-04-13", "2026-04-14", "2026-04-15"}
THRESHOLD_S, SUBMITTED_PICK_S, SPEED = 300.0, 19.0, 2306.0
RUN = "calibrated_recorded_arrival"


def tours() -> pd.DataFrame:
    rows = []
    for day in DAYS:
        source = ScenarioSource(DATA, day=day, shift_start_s=19800.0,
                                pause_threshold_s=THRESHOLD_S, speed=SPEED,
                                time_per_pick=SUBMITTED_PICK_S, tour_setup_time=215.0)
        model = pd.read_csv(OUTPUTS / RUN / day / "facts" / "tours.csv")
        by_picker: dict[int, list] = {}
        for picker, lines in source.replay.tours:
            by_picker.setdefault(picker, []).append(sorted(lines, key=lambda r: r.start_s))
        for picker, recorded in by_picker.items():
            recorded.sort(key=lambda lines: lines[0].start_s)
            simulated = model[model.picker_id == picker].sort_values("start")
            if len(simulated) != len(recorded) or simulated.end.isna().any():
                raise ValueError(f"{day} picker {picker}: tours do not match")
            for lines, distance in zip(recorded, simulated.distance):
                window = max(r.end_s for r in lines) - lines[0].start_s
                stops, reach = 0.0, lines[0].end_s
                for row in lines[1:]:
                    if row.start_s - reach > THRESHOLD_S:
                        stops += row.start_s - reach
                    reach = max(reach, row.end_s)
                for row in lines:
                    excess = row.end_s - row.start_s - SUBMITTED_PICK_S
                    if excess > THRESHOLD_S:
                        stops += excess
                rows.append({"day": day, "lines": len(lines), "net_s": window - stops,
                             "walk_s": distance / SPEED})
    return pd.DataFrame(rows)


def main() -> None:
    data = tours()
    train = data[data.day.isin(TRAIN)]
    per_line = (train.net_s.sum() - train.walk_s.sum()) / train.lines.sum()
    print(f"seconds per line (Monday to Wednesday): {per_line:.2f}")
    for name, part in (("fit days", train), ("held-out days", data[~data.day.isin(TRAIN)])):
        error = part.walk_s + per_line * part.lines - part.net_s
        print(f"{name}: {len(part)} tours, bias {error.mean() / 60:+.2f} min, "
              f"RMSE {np.sqrt((error ** 2).mean()) / 60:.1f} min per tour")
    for day, part in data.groupby("day"):
        print(f"  {day}: {(part.net_s.sum() - part.walk_s.sum()) / part.lines.sum():.1f} s per line")


if __name__ == "__main__":
    main()
