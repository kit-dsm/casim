"""R1.5 of docs/ijpe/plan.md: what explains a recorded line's duration.

A recorded line's interval contains the walk to it, so walk and pick are
estimated jointly: the recorded duration (s) is regressed by least squares on
the walk distance of the line's leg in the recorded sequence (m, read from
CASIM's pick table of the anchored replay), an aisle change from the
previous line of the tour, quantity, weight, volume, the first line of a tour,
an order change inside the tour (interleaving) and the hour of the day.
Intervals are heteroscedasticity-robust (HC1). Estimated

- leave-one-day-out: coefficients on five days, prediction error on the sixth;
- calibration Monday to Wednesday, validation Thursday, Friday, Saturday.

Two samples: all lines, and lines of at most ``CAP_S`` seconds, since idle
time inside a tour (R1.4) sits in a few long lines.

    python -m scripts.ijpe.causes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from casim.sources.grid_layout import GridLayout
from scripts.ijpe import OUTPUTS as SCENARIO_OUTPUTS, SCENARIO
from scripts.ijpe.replay import CALIBRATION_DAYS, RUNS, matched_lines, recorded_day, runs

OUTPUTS = SCENARIO_OUTPUTS / "causes"
CAP_S = 300.0
FEATURES = ["walk_m", "aisle_change", "quantity", "weight_kg", "volume", "first_line",
            "order_change", "hour"]


def line_features(day: str, directory, nodes: dict) -> pd.DataFrame:
    picks = pd.read_csv(directory / "facts" / "picks.csv")
    lines = matched_lines(recorded_day(day, nodes), picks).sort_values(["tour_id", "pick_start"])
    previous = lines.groupby("tour_id")
    lines["first_line"] = (previous.cumcount() == 0).astype(float)
    lines["aisle_change"] = ((lines["aisle"] != previous["aisle"].shift())
                             & (lines["first_line"] == 0)).astype(float)
    lines["order_change"] = ((lines["order"] != previous["order"].shift())
                             & (lines["first_line"] == 0)).astype(float)
    lines["walk_m"] = lines["walk_distance"] / 1000
    lines["weight_kg"] = lines["weight"]
    lines["volume"] = lines["package_volume"] * lines["quantity"]
    lines["hour"] = lines["start_s"] / 3600
    lines["day"] = day
    return lines[["day", "recorded_s", "model_s", *FEATURES]]


def ols(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x = np.column_stack([np.ones(len(frame)), frame[FEATURES].to_numpy(float)])
    y = frame["recorded_s"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    n, k = x.shape
    bread = np.linalg.inv(x.T @ x)
    meat = x.T @ (x * residual[:, None] ** 2)
    se = np.sqrt(np.diag(bread @ meat @ bread) * n / (n - k))
    table = pd.DataFrame({"term": ["intercept", *FEATURES], "coefficient": beta, "se": se})
    table["ci_low"], table["ci_high"] = beta - 1.96 * se, beta + 1.96 * se
    return table, beta


def predict(frame: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(frame)), frame[FEATURES].to_numpy(float)]) @ beta


def errors(frame: pd.DataFrame, prediction: np.ndarray, label: str) -> dict:
    error = prediction - frame["recorded_s"].to_numpy(float)
    submitted = frame["model_s"].to_numpy(float) - frame["recorded_s"].to_numpy(float)
    return {"evaluation": label, "lines": len(frame), "bias_s": error.mean(),
            "mae_s": np.abs(error).mean(), "rmse_s": float(np.sqrt((error ** 2).mean())),
            "submitted_bias_s": submitted.mean(), "submitted_mae_s": np.abs(submitted).mean(),
            "submitted_rmse_s": float(np.sqrt((submitted ** 2).mean()))}


def main() -> None:
    _, locations = GridLayout().build(SCENARIO / "data" / "layout.csv")
    nodes = GridLayout.pick_nodes(locations)
    anchored = [(day, directory) for day, setting, directory in runs(RUNS) if setting == "null"]
    data = pd.concat([line_features(day, directory, nodes) for day, directory in anchored])
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    for sample, frame in (("all", data), (f"<= {CAP_S:.0f} s", data[data["recorded_s"] <= CAP_S])):
        pooled, _ = ols(frame)
        lodo, scores = [], []
        for day in sorted(frame["day"].unique()):
            table, beta = ols(frame[frame["day"] != day])
            lodo.append(table.assign(held_out=day))
            held = frame[frame["day"] == day]
            scores.append(errors(held, predict(held, beta), f"leave out {day}"))
        calibration = frame[frame["day"].isin(CALIBRATION_DAYS)]
        validation = frame[~frame["day"].isin(CALIBRATION_DAYS)]
        cal_table, beta = ols(calibration)
        scores.append(errors(validation, predict(validation, beta), "Mon-Wed -> Thu, Fri, Sat"))
        spread = pd.concat(lodo).groupby("term")["coefficient"].agg(["min", "max"])
        summary = pooled.set_index("term").join(spread.rename(columns={"min": "lodo_min",
                                                                       "max": "lodo_max"}))
        summary = summary.join(cal_table.set_index("term")["coefficient"].rename("mon_wed"))
        tag = "all" if sample == "all" else "capped"
        summary.to_csv(OUTPUTS / f"coefficients_{tag}.csv")
        pd.DataFrame(scores).to_csv(OUTPUTS / f"prediction_{tag}.csv", index=False)
        print(f"\n## coefficients, {sample} lines (n = {len(frame)})\n{summary.to_string()}")
        print(f"\n## prediction error, {sample}\n{pd.DataFrame(scores).to_string(index=False)}")
    print(f"\nwritten to {OUTPUTS}")


if __name__ == "__main__":
    main()
