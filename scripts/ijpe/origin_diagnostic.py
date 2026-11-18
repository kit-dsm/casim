"""Compare CASIM replay outputs under explicit, unobserved tour-origin hypotheses.

The variants differ only in where CASIM begins a recorded tour. All durations
and distances in this report are read from CASIM's facts tables and the WMS
export. No line, order or picker identifiers are written to the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from casim.sources.grid_layout import GridLayout
from scripts.ijpe import OUTPUTS, SCENARIO
from scripts.ijpe.causes import line_features
from scripts.ijpe.replay import RUNS, compare_run, runs

DAYS = ("2026-04-11", "2026-04-13", "2026-04-14", "2026-04-15",
        "2026-04-16", "2026-04-17")
MODES = ("depot", "aisle_head", "first_pick")


def main() -> None:
    _, locations = GridLayout().build(SCENARIO / "data" / "layout.csv")
    nodes = GridLayout.pick_nodes(locations)
    anchored = {day: path for day, setting, path in runs(RUNS) if setting == "null"}
    rows = []
    for day in DAYS:
        for mode in MODES:
            directory = (anchored[day] if mode == "depot" else
                         OUTPUTS / "origin_checks" /
                         f"{day}_{mode}{'_fixed' if mode == 'first_pick' else ''}")
            print(f"reading {day} {mode}", flush=True)
            lines = line_features(day, directory, nodes)
            errors = lines["model_s"] - lines["recorded_s"]
            later = lines["first_line"] == 0
            comparison = compare_run(day, mode, directory, nodes)
            order = comparison["order"].query("size_class == 'all'").iloc[0]
            picker = comparison["picker"].query("measure == 'end of last line [min]'").iloc[0]
            rows.append({
                "day": day, "origin": mode, "lines": len(lines),
                "first_lines": int((~later).sum()),
                "first_line_bias_s": float(errors[~later].mean()),
                "first_line_mae_s": float(errors[~later].abs().mean()),
                "later_line_mae_s": float(errors[later].abs().mean()),
                "line_rmse_s": float(np.sqrt(np.mean(errors ** 2))),
                "orders": int(order["n"]), "order_rmse_min": float(order["rmse"]),
                "order_mae_min": float(order["mae"]),
                "picker_end_bias_min": float(picker["bias"]),
            })
    result = pd.DataFrame(rows)
    target = OUTPUTS / "origin_checks" / "summary.csv"
    result.to_csv(target, index=False)
    print(result.round(2).to_string(index=False))
    print(f"\naggregate report: {target}")


if __name__ == "__main__":
    main()
