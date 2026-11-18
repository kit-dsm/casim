"""Build the approved L1 full run list from l1_subset.csv.

Keep all three smoke rows of each pipeline, then select four more distinct
instances spread over the remaining strata. Both pipelines of a distance set
run on the same ten instances. Kris keeps its entire 295-instance subset.
"""

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
source = pd.read_csv(HERE / "l1_subset.csv", dtype={"instance_name": str, "stratum": str})
selected = []
for instance_set, rows in source.groupby("instance_set", sort=False):
    if instance_set == "Kris":
        selected.append(rows)
        continue
    smoke = list(dict.fromkeys(rows.loc[rows.smoke, "instance_name"]))
    if len(smoke) > 10:
        raise ValueError(f"More than ten smoke instances for {instance_set}")
    remaining = rows.loc[~rows.instance_name.isin(smoke)].drop_duplicates("instance_name")
    remaining = remaining.sort_values(
        ["reference_is_opt", "stratum", "instance_name"],
        ascending=[False, True, True],
    ).drop_duplicates("stratum")
    count = 10 - len(smoke)
    indices = np.linspace(0, len(remaining) - 1, count, dtype=int)
    names = set(smoke) | set(remaining.iloc[indices].instance_name)
    if len(names) != 10:
        raise ValueError(f"Could not select ten distinct instances for {instance_set}")
    chosen = rows[rows.instance_name.isin(names)]
    if set(rows.loc[rows.smoke].index) - set(chosen.index):
        raise ValueError(f"Smoke row missing for {instance_set}")
    selected.append(chosen)

runlist = pd.concat(selected, ignore_index=True)
assert len(runlist) == 415 and runlist.smoke.sum() == 39
runlist.to_csv(HERE / "l1_runlist.csv", index=False)
