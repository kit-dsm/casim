"""Task B.4: run the L1 subset and write the three comparisons per instance and pipeline.

1. Simulator: CASIM (one decision at t = 0) against the direct algorithm calls.
2. Algorithm drift: the direct result against CASOP's published v0.1.0 result.
3. Literature: the direct result against the reference value.

Usage: python -m scenarios.scenario_static_benchmarks.run_benchmark {smoke,full} [--sets SET ...]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import replace
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
from ware_ops_algos.domain_models import Resources

ROOT = Path(__file__).resolve().parents[2]
RUNLIST = ROOT / "docs" / "literature_benchmarks" / "l1_runlist.csv"
RESULTS = Path(__file__).resolve().parent / "data" / "casop-results-v0.1.0.parquet"
INSTANCES = ROOT / "scenarios" / "scenario_static_benchmarks" / "data" / "instances"
OUTPUT = Path(__file__).resolve().parent / "outputs"
REL_TOL = 1e-9
NOT_EXECUTABLE = {}

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


def evaluate(row: pd.Series, results: pd.DataFrame, smoke: bool = False) -> dict:
    instance_set, stem = raw_instance(row.instance_set, row.instance_name)
    pipeline = PIPELINES[row.pipeline]
    objective = "total_time" if pipeline.scheduling == "EDDScheduler" else "distance"
    domain = to_static_sim_domain(load_casop_domain(INSTANCES, instance_set, stem), instance_set)
    direct = solve(pipeline, domain)
    direct_value = direct.total_time if objective == "total_time" else direct.distance
    out = {
        "set": row.instance_set, "instance": row.instance_name, "pipeline": row.pipeline,
        "objective": objective, "direct": direct_value, "direct_distance": direct.distance,
        "direct_walked_distance": direct.walked_distance, "direct_max_lateness": direct.max_lateness,
        "batching_s": round(direct.batching_seconds, 3),
        "time_limit_reached": direct.batching_seconds >= TIME_LIMIT_S,
    }
    if smoke and row.instance_set != "Kris":
        second = to_static_sim_domain(load_casop_domain(INSTANCES, instance_set, stem), instance_set)
        second.resources = Resources(second.resources.tpe, [
            replace(resource, speed=2.0, time_per_pick=7.0, tour_setup_time=11.0)
            for resource in second.resources.resources
        ])
        out["second_times_distance_equal"] = equal(
            direct.distance, solve(pipeline, second).distance)
        if not out["second_times_distance_equal"]:
            raise RuntimeError(f"Distance changed with time parameters: {row.instance_set} "
                               f"{row.instance_name} {row.pipeline}")
    if pipeline.routing in ("NearestNeighbourhood", "RatliffRosenthal"):
        if pipeline.routing == "RatliffRosenthal" and any(
                route.walked_distance is None
                or not equal(route.reported_distance, route.walked_distance)
                for route in direct.routes):
            raise RuntimeError(f"RatliffRosenthalRouting defect: a route's walked node sequence "
                               f"does not equal its reported distance on {row.instance_set} "
                               f"{row.instance_name}")
        changed = [route for route in direct.routes if route.walked_distance is not None
                   and not equal(route.reported_distance, route.walked_distance)]
        walked_value = (direct_value + (direct.walked_distance - direct.distance)
                        / domain.resources.resources[0].speed
                        if objective == "total_time" else direct.walked_distance)
        out.update(
            routes_json=json.dumps([asdict(route) for route in direct.routes]),
            routes_with_distance_gap=len(changed),
            distance_shortfall=sum(route.walked_distance - route.reported_distance
                                   for route in changed),
            walked_reference_gap_pct=(100 * (walked_value - row.reference_value)
                                      / row.reference_value),
        )
    reason = NOT_EXECUTABLE.get(pipeline.routing)
    casim_value = None
    out["c1_f64_explained"] = False
    if reason is None:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUTPUT) as folder:
            casim = run(instance_set, stem, row.pipeline, Path(folder), audit=True)
            out["casim_violations"] = violations(Path(folder) / "audit.json")
        casim_value = casim.total_time if objective == "total_time" else casim.distance
        out.update(casim=casim_value, casim_distance=casim.distance, casim_decisions=casim.decisions,
                   same_batches=casim.batches == direct.batches,
                   c1_equal=equal(casim_value, direct_value))
        speed = domain.resources.resources[0].speed
        predicted_gap = (direct.walked_distance - direct.distance) / speed if objective == "total_time" \
            else direct.walked_distance - direct.distance
        out["c1_f64_explained"] = (pipeline.routing == "NearestNeighbourhood"
                                   and not equal(direct.walked_distance, direct.distance)
                                   and out["same_batches"]
                                   and equal(casim.distance, direct.walked_distance)
                                   and equal(casim_value - direct_value, predicted_gap))
    else:
        out.update(casim=None, casim_not_run=reason, c1_equal=None)
    if out["c1_equal"] is False and not out["c1_f64_explained"]:
        raise RuntimeError(f"Unexplained CASIM/direct difference: {row.instance_set} "
                           f"{row.instance_name} {row.pipeline}: {casim_value} != {direct_value}")
    v010 = published(results, instance_set, stem, row.pipeline)
    v010_value = None if v010 is None else v010["total_time" if objective == "total_time" else "total_distance"]
    out.update(casop_v010=v010_value, c2_equal=equal(direct_value, v010_value))
    out.update(reference=row.reference_value, reference_is_opt=row.reference_is_opt,
               c3_gap_pct=100 * (direct_value - row.reference_value) / row.reference_value,
               below_proven_optimum=bool(row.reference_is_opt)
               and direct_value < row.reference_value * (1 - REL_TOL))
    out["below_proven_optimum_f64_explained"] = (
        out["below_proven_optimum"] and out["c1_f64_explained"]
        and casim_value is not None
        and casim_value >= row.reference_value * (1 - REL_TOL))
    if out["below_proven_optimum"] and not out["below_proven_optimum_f64_explained"]:
        raise RuntimeError(f"Below proven optimum: {row.instance_set} {row.instance_name} "
                           f"{row.pipeline}: {direct_value} < {row.reference_value}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["smoke", "full"])
    parser.add_argument("--sets", nargs="*")
    parser.add_argument("--pipelines", nargs="*")
    parser.add_argument("--exclude-runlist", type=Path,
                        help="run only rows absent from this prior run list")
    parser.add_argument("--resume", action="store_true", help="continue an existing output CSV")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    subset = pd.read_csv(RUNLIST, dtype={"instance_name": str, "stratum": str})
    if args.stage == "smoke":
        subset = subset[subset.smoke]
    if args.sets:
        subset = subset[subset.instance_set.isin(args.sets)]
    if args.pipelines:
        subset = subset[subset.pipeline.isin(args.pipelines)]
    if args.exclude_runlist:
        prior = pd.read_csv(args.exclude_runlist)
        keys = set(zip(prior.instance_set, prior.instance_name, prior.pipeline))
        subset = subset[[key not in keys for key in zip(
            subset.instance_set, subset.instance_name, subset.pipeline)]]
    results = pd.read_parquet(RESULTS, columns=["instance_set", "instance_name", "strategy",
                                                "total_distance", "total_time"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suffix = "_missing" if args.exclude_runlist else (
        "_filtered" if args.pipelines or args.sets else "")
    path = OUTPUT / f"l1_{args.stage}{suffix}.csv"
    rows = pd.read_csv(path).to_dict("records") if args.resume and path.exists() else []
    for index, (_, row) in enumerate(subset.iterrows()):
        if index < len(rows):
            prior = rows[index]
            if (prior["set"], prior["instance"], prior["pipeline"]) != (
                    row.instance_set, row.instance_name, row.pipeline):
                raise RuntimeError(f"Resume CSV row {index} does not match the run list")
            continue
        rows.append(evaluate(row, results, smoke=args.stage == "smoke"))
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("%s %s %s: direct %s, CASIM %s, v0.1.0 %s", row.instance_set,
                    row.instance_name, row.pipeline, rows[-1]["direct"], rows[-1]["casim"],
                    rows[-1]["casop_v010"])
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
