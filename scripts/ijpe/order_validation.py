"""P1.2: compare CASIM order spans with out-of-day WMS orders.

The fit is the fixed-speed interior-line calibration. Each held-out day is
simulated with parameters estimated without that day. A second parameter set
trained on Monday to Wednesday is evaluated on Saturday, Thursday and Friday.
The submitted constant-time model is the reference. Only aggregate size-class
errors are written to the ignored output directory.
"""

from __future__ import annotations

import pandas as pd

from casim.run import compose_scenario, run
from casim.sources.grid_layout import GridLayout
from scripts.ijpe import OUTPUTS, SCENARIO
from scripts.ijpe.replay import compare_run

CALIBRATION = OUTPUTS / "calibration" / "interior_parameters.csv"
DESTINATION = OUTPUTS / "order_validation"
DAYS = ("2026-04-11", "2026-04-13", "2026-04-14", "2026-04-15",
        "2026-04-16", "2026-04-17")


def main() -> None:
    parameters = pd.read_csv(CALIBRATION).set_index("estimation")
    _, locations = GridLayout().build(SCENARIO / "data" / "layout.csv")
    nodes = GridLayout.pick_nodes(locations)
    collected = []
    for day in DAYS:
        for label in ("submitted", "leave_one_day_out", "mon_wed"):
            if label == "mon_wed" and day in ("2026-04-13", "2026-04-14", "2026-04-15"):
                continue
            if label == "submitted":
                directory = OUTPUTS / "run" / f"source.day={day},source.pause_threshold_s=null"
            else:
                fit = parameters.loc["mon_wed" if label == "mon_wed" else f"without_{day}"]
                directory = DESTINATION / label / day
                overrides = [f"source.day={day}",
                             f"source.time_per_pick={fit.time_per_pick}",
                             f"source.per_unit_s={fit.per_unit_s}",
                             "source.pause_threshold_s=null"]
                run(compose_scenario(SCENARIO, overrides, output_dir=directory))
            table = compare_run(day, label, directory, nodes)["order"]
            table.insert(1, "model", label)
            collected.append(table)
    result = pd.concat(collected, ignore_index=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    result.to_csv(DESTINATION / "order_size_errors.csv", index=False)
    print(result[["day", "model", "size_class", "n", "bias", "mae", "rmse"]].to_string(
        index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
