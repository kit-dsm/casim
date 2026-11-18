"""R1.7 of docs/ijpe/plan.md: calibrate CASIM's time model on recorded lines.

CASIM's line takes walk / speed + pick, the pick being ``LinePickTime``
(base + per package), and a tour starts with a setup. The recorded line
contains the walk to it, and a tour's first line also the setup, so the
recorded duration is regressed on the walk distance (m), the quantity and a
first-line indicator:

    recorded_s = base + per_unit * quantity + walk_m / speed_m_s + setup * first_line

on lines of at most ``CAP_S`` s (idle time inside a tour is not part of the
model, R1.4), estimated on Monday to Wednesday and leaving out each day.
The original unconstrained fit is kept as a diagnostic. A second fit fixes
speed at 2,306 mm/s and uses only interior lines lasting 4-59 s, avoiding
unobserved first-tour legs and long WMS waiting intervals. These fit choices
do not remove any lines from CASIM's validation replay. Writes aggregate
``parameters.csv`` and ``interior_parameters.csv``.

    python -m scripts.ijpe.calibrate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from casim.sources.grid_layout import GridLayout
from scripts.ijpe import OUTPUTS as SCENARIO_OUTPUTS, SCENARIO
from scripts.ijpe.causes import CAP_S, line_features
from scripts.ijpe.replay import CALIBRATION_DAYS, RUNS, runs

OUTPUTS = SCENARIO_OUTPUTS / "calibration"


def fit(frame: pd.DataFrame) -> dict:
    x = np.column_stack([np.ones(len(frame)), frame["quantity"], frame["walk_m"],
                         frame["first_line"]])
    base, per_unit, seconds_per_m, setup = np.linalg.lstsq(x, frame["recorded_s"], rcond=None)[0]
    return {"time_per_pick": base, "per_unit_s": per_unit,
            "speed": 1000.0 / seconds_per_m,  # mm/s, the layout's unit
            "tour_setup_time": max(setup, 0.0), "lines": len(frame)}


def fit_fixed_speed(train: pd.DataFrame, held_out: pd.DataFrame,
                    speed_m_s: float) -> dict:
    """Quantify the loss of fit at a physically specified walking speed."""
    columns = ["quantity", "first_line"]
    x = np.column_stack([np.ones(len(train)), train[columns].to_numpy(float)])
    adjusted = train["recorded_s"].to_numpy() - train["walk_m"].to_numpy() / speed_m_s
    base, per_unit, first_line = np.linalg.lstsq(x, adjusted, rcond=None)[0]

    def rmse(frame: pd.DataFrame) -> float:
        prediction = (base + per_unit * frame["quantity"] +
                      first_line * frame["first_line"] + frame["walk_m"] / speed_m_s)
        return float(np.sqrt(np.mean((prediction - frame["recorded_s"]) ** 2)))

    return {"speed_m_s": speed_m_s, "base_s": base, "per_unit_s": per_unit,
            "first_line_s": first_line, "training_lines": len(train),
            "held_out_lines": len(held_out), "training_rmse_s": rmse(train),
            "held_out_rmse_s": rmse(held_out)}


def fit_interior(frame: pd.DataFrame, speed_mm_s: float = 2306.0) -> dict:
    """Estimate pick and quantity terms without unobserved tour-start travel.

    The Feb 6 correspondence gives 8.3 km/h as an operational estimate of
    truck speed. Restrict to non-first lines with recorded durations 4-59 s,
    the range used in that correspondence for cleaned pick-time analysis.
    The restriction is for estimation only; validation keeps the full export.
    """
    sample = frame[(frame["first_line"] == 0) & frame["recorded_s"].between(4, 59)]
    x = np.column_stack([np.ones(len(sample)), sample["quantity"]])
    response = sample["recorded_s"] - sample["walk_m"] * 1000 / speed_mm_s
    base, per_unit = np.linalg.lstsq(x, response, rcond=None)[0]
    return {"time_per_pick": base, "per_unit_s": per_unit,
            "speed": speed_mm_s, "tour_setup_time": 215.0,
            "lines": len(sample)}


def main() -> None:
    _, locations = GridLayout().build(SCENARIO / "data" / "layout.csv")
    nodes = GridLayout.pick_nodes(locations)
    anchored = [(day, directory) for day, setting, directory in runs(RUNS) if setting == "null"]
    data = pd.concat([line_features(day, directory, nodes) for day, directory in anchored])
    data = data[data["recorded_s"] <= CAP_S]
    training = data[data["day"].isin(CALIBRATION_DAYS)]
    held_out = data[~data["day"].isin(CALIBRATION_DAYS)]
    fixed = pd.DataFrame([fit_fixed_speed(training, held_out, speed)
                          for speed in (1.0, 1.5, 2.306)])
    correlation = data[["walk_m", "quantity", "first_line"]].corr()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(OUTPUTS / "fixed_speed_parameters.csv", index=False)
    correlation.to_csv(OUTPUTS / "collinearity.csv")
    print("Fixed-speed fits (Mon-Wed -> held-out):\n", fixed.round(3).to_string(index=False))
    print("\nFeature correlations:\n", correlation.round(3).to_string())
    rows = [{"estimation": "mon_wed", "for_days": "2026-04-16 2026-04-17 2026-04-11",
             **fit(data[data["day"].isin(CALIBRATION_DAYS)])}]
    rows += [{"estimation": f"without_{day}", "for_days": day, **fit(data[data["day"] != day])}
             for day in sorted(data["day"].unique())]
    table = pd.DataFrame(rows)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUTS / "parameters.csv", index=False)
    interior = pd.DataFrame([
        {"estimation": "mon_wed", "for_days": "2026-04-11 2026-04-16 2026-04-17",
         **fit_interior(data[data["day"].isin(CALIBRATION_DAYS)])},
        *({"estimation": f"without_{day}", "for_days": day,
           **fit_interior(data[data["day"] != day])}
          for day in sorted(data["day"].unique())),
    ])
    interior.to_csv(OUTPUTS / "interior_parameters.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print("\nFixed-speed interior fit:\n", interior.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
