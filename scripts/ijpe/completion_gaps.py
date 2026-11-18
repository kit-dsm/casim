"""Compare picker completion gaps without treating WMS line starts as walks.

Run on a six-day anchored CASIM output directory. Prints aggregates only.
The comparison pairs the kth WMS completion and kth model completion per
picker-day; signature mismatches are reported so ordering remains auditable.
It never writes confidential line or picker records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from casim.sources.grid_layout import GridLayout
from scripts.ijpe import SCENARIO
from scripts.ijpe.replay import recorded_day, runs


def analyze(root: Path) -> dict:
    _, locations = GridLayout().build(SCENARIO / "data" / "layout.csv")
    nodes = GridLayout.pick_nodes(locations)
    records = []
    for day, setting, directory in runs(root):
        if setting not in {"null", "default"}:
            continue
        observed = recorded_day(day, nodes).sort_values(["start_s", "row"])
        picker_ids = {picker: index for index, picker in
                      enumerate(observed.picker.unique())}
        picks = pd.read_csv(directory / "facts" / "picks.csv")
        for picker, group in observed.groupby("picker"):
            wms = group.sort_values(["end_s", "row"])
            model = picks[picks.picker_id == picker_ids[picker]].sort_values(
                ["pick_end", "tour_id"])
            if len(wms) != len(model):
                raise ValueError(f"different line counts on {day}")
            observed_keys = list(zip(wms.order, wms.article, wms.node_x, wms.node_y))
            model_keys = list(zip(model.order_id, model.article_id,
                                  model.node_x, model.node_y))
            mismatches = sum(a != b for a, b in zip(observed_keys, model_keys))
            matched = np.asarray([a == b for a, b in zip(observed_keys, model_keys)])
            same_tour = (model.tour_id.to_numpy()[1:] ==
                         model.tour_id.to_numpy()[:-1])
            wms_gaps = np.diff(wms.end_s.to_numpy(float))
            model_gaps = np.diff(model.pick_end.to_numpy(float))
            records.append((day, len(wms), mismatches, wms_gaps, model_gaps,
                            same_tour & matched[1:] & matched[:-1]))
    if not records:
        raise ValueError(f"no anchored facts in {root}")
    result = {}
    for label, subset in (("all", records),
                          ("forward_ThuFri", [r for r in records if r[0] in
                                              {"2026-04-16", "2026-04-17"}])):
        raw_wms = np.concatenate([r[3] for r in subset])
        raw_model = np.concatenate([r[4] for r in subset])
        within = np.concatenate([r[5] for r in subset])
        for scope, mask in (("all_gaps", np.ones(len(raw_wms), bool)),
                            ("within_tour_matched", within),
                            ("within_tour_gap_at_most_120s", within & (raw_wms <= 120))):
            wms = raw_wms[mask]
            model = raw_model[mask]
            result[f"{label}_{scope}"] = {
                "picker_days": len(subset),
                "gaps": len(wms),
                "completion_signature_mismatches": sum(r[2] for r in subset),
                "wms_gap_median_s": float(np.median(wms)),
                "model_gap_median_s": float(np.median(model)),
                "gap_mae_s": float(np.mean(abs(model - wms))),
                "gap_rmse_s": float(np.sqrt(np.mean((model - wms) ** 2))),
                "wms_gap_over_300_s": int((wms > 300).sum()),
                "model_gap_over_300_s": int((model > 300).sum()),
            }
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(Path(sys.argv[1])), indent=2))
