"""Task B.4: run the L1 subset and write the three comparisons per instance and pipeline.

1. Simulator: CASIM (one decision at t = 0) against the direct algorithm calls.
2. Algorithm drift: the direct result against CASOP's published v0.1.0 result.
3. Literature: the direct result against the reference value.

Usage: python -m scenarios.scenario_static_benchmarks.run_benchmark {smoke,full} [--sets SET ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from math import isclose
from pathlib import Path
import re
import tempfile

import pandas as pd

from scenarios.scenario_static_benchmarks.direct import TIME_LIMIT_S, solve
from scenarios.scenario_static_benchmarks.experiment import run
from scenarios.scenario_static_benchmarks.loader import load_casop_domain, to_static_sim_domain
from scenarios.scenario_static_benchmarks.pipelines import PIPELINES

ROOT = Path(__file__).resolve().parents[2]
SUBSET = ROOT / "docs" / "literature_benchmarks" / "l1_subset.csv"
RESULTS = Path(__file__).resolve().parent / "data" / "casop-results-v0.1.0.parquet"
INSTANCES = ROOT.parent / "ware_ops_pipes" / "data" / "instances"
OUTPUT = Path(__file__).resolve().parent / "outputs"
REL_TOL = 1e-9
NOT_EXECUTABLE = {"RatliffRosenthal": "RatliffRosenthalRouting builds no node sequence (annotated_route)"}

logger = logging.getLogger(__name__)


def raw_instance(instance_set: str, name: str) -> tuple[str, str]:
    """CASOP's comparison names -> the instance set directory and file stem."""
    if instance_set == "HennWaescher":
        stem, suffix = re.fullmatch(r"(.+)_(u|cb)", name).groups()
        return ("HennWaescherUniform" if suffix == "u" else "HennWaescherClassBased"), stem
    if instance_set == "Foodmart":
        return "FoodmartData", f"instances_{name}_MAL"
    if instance_set == "Kris":
        stem, size = re.fullmatch(r"(.+)_(small|large)", name).groups()
        return ("KrisSmallDataCorrected" if size == "small" else "KrisLargeData"), stem
    return instance_set, name


def equal(a: float | None, b: float | None) -> bool | None:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    return isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-9)


def published(results: pd.DataFrame, instance_set: str, stem: str, label: str) -> pd.Series | None:
    rows = results[(results.instance_set == instance_set) & (results.instance_name == stem)
                   & (results.strategy == label)]
    return None if rows.empty else rows.iloc[0]


def violations(audit: Path) -> int:
    data = json.loads(audit.read_text(encoding="utf-8"))
    return sum(sum(counts.values()) for counts in data["violations"].values())


def evaluate(row: pd.Series, results: pd.DataFrame) -> dict:
    instance_set, stem = raw_instance(row.instance_set, row.instance_name)
    pipeline = PIPELINES[row.pipeline]
    objective = "total_time" if pipeline.scheduling == "EDDScheduler" else "distance"
    domain = to_static_sim_domain(load_casop_domain(INSTANCES, instance_set, stem))
    direct = solve(pipeline, domain)
    direct_value = direct.total_time if objective == "total_time" else direct.distance
    out = {
        "set": row.instance_set, "instance": row.instance_name, "pipeline": row.pipeline,
        "objective": objective, "direct": direct_value, "direct_distance": direct.distance,
        "direct_walked_distance": direct.walked_distance, "direct_max_lateness": direct.max_lateness,
        "batching_s": round(direct.batching_seconds, 3),
        "time_limit_reached": direct.batching_seconds >= TIME_LIMIT_S,
    }
    reason = NOT_EXECUTABLE.get(pipeline.routing)
    if reason is None:
        with tempfile.TemporaryDirectory() as folder:
            casim = run(instance_set, stem, row.pipeline, Path(folder), audit=True)
            out["casim_violations"] = violations(Path(folder) / "audit.json")
        casim_value = casim.total_time if objective == "total_time" else casim.distance
        out.update(casim=casim_value, casim_distance=casim.distance, casim_decisions=casim.decisions,
                   same_batches=casim.batches == direct.batches,
                   c1_equal=equal(casim_value, direct_value))
    else:
        out.update(casim=None, casim_not_run=reason)
    v010 = published(results, instance_set, stem, row.pipeline)
    v010_value = None if v010 is None else v010["total_time" if objective == "total_time" else "total_distance"]
    out.update(casop_v010=v010_value, c2_equal=equal(direct_value, v010_value))
    out.update(reference=row.reference_value, reference_is_opt=row.reference_is_opt,
               c3_gap_pct=100 * (direct_value - row.reference_value) / row.reference_value,
               below_proven_optimum=bool(row.reference_is_opt)
               and direct_value < row.reference_value * (1 - REL_TOL))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["smoke", "full"])
    parser.add_argument("--sets", nargs="*")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    subset = pd.read_csv(SUBSET, dtype={"instance_name": str, "stratum": str})
    if args.stage == "smoke":
        subset = subset[subset.smoke]
    if args.sets:
        subset = subset[subset.instance_set.isin(args.sets)]
    results = pd.read_parquet(RESULTS, columns=["instance_set", "instance_name", "strategy",
                                                "total_distance", "total_time"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"l1_{args.stage}.csv"
    rows = []
    for _, row in subset.iterrows():
        rows.append(evaluate(row, results))
        logger.info("%s %s %s: direct %s, CASIM %s, v0.1.0 %s", row.instance_set,
                    row.instance_name, row.pipeline, rows[-1]["direct"], rows[-1]["casim"],
                    rows[-1]["casop_v010"])
    fields = sorted({key for row in rows for key in row}, key=lambda key: list(rows[0]).index(key)
                    if key in rows[0] else len(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
