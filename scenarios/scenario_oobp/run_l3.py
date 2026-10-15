"""Run the L3 benchmark (Gil-Borrás et al. 2020) and compare with the published values.

Usage: python -m scenarios.scenario_oobp.run_l3 {smoke,full} [--groups ...] [--resume]

The rule is FCFS batching, the oldest-order batch selection, S-shape routing and
no waiting, under the protocol of Table 4. The 64 Henn instances run under both
depot readings (corner and centre), the 64 Albareda instances under their layout
file's depot. A result below a published best-known value is an error.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

import pandas as pd
from hydra import compose, initialize_config_dir

from casim.runner import apply_decision
from tests.verification.kpi_recomputation import recompute_kpis
from scenarios.experiment_commons import load_and_flatten_data_card, setup_experiment
from scenarios.scenario_oobp.experiment_oobp import add_orders_hook, picker_arrival_hook

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "config"
INSTANCES = ROOT / "scenarios" / "scenario_henn_online" / "data" / "oobp"
RUNLIST = ROOT / "docs" / "literature_benchmarks" / "l3_instances.csv"
REFERENCE = ROOT / "scenarios" / "scenario_henn_online" / "data" / "oobp_reference.csv"
OUTPUT = Path(__file__).resolve().parent / "outputs"
BASELINES = ROOT / "tests" / "baseline" / "benchmarks" / "l3"
REL_TOL = 1e-9

HENN_GROUP = "Henn"


@dataclass(frozen=True)
class OobpResult:
    distance: float
    max_completion_s: float
    max_turnover_s: float
    tours: int
    decisions: int
    violations: int


def instance_key(group: str, stratum: str, path: str) -> str:
    """The instance's name in ``data/oobp_reference.csv``."""
    if group == HENN_GROUP:
        setting, orders, _, _ = Path(path).stem.split("-")
        return f"H_{stratum}_{orders}_{setting.rstrip('sl')}"
    stem = Path(path)
    warehouse = stem.name.split("_")[-2]
    return f"A_{int(warehouse)}_{stem.parent.name}_{stem.name.split('_')[-1].split('.')[0]}"


def albareda_depot(pedido_path: str) -> int:
    """The layout file's depot: 0 bottom left, 1 bottom centre."""
    pedido = INSTANCES / pedido_path
    match = re.fullmatch(r"wsrp_input_pedido_(\d{2})_(\d{3})\.txt", pedido.name)
    if match is None:
        raise ValueError(f"unexpected pedido path {pedido_path!r}")
    layout = pedido.parent / f"wsrp_input_layout_{match.group(1)}_{match.group(2)}.txt"
    return int(layout.read_text(encoding="windows-1252").splitlines()[3].strip())


def depot_of(row: pd.Series) -> list[str]:
    if row.group == "Henn":
        return ["corner", "centre"]
    return [str(albareda_depot(row.path_in_data_oobp))]


def violations(audit: Path) -> int:
    data = json.loads(audit.read_text(encoding="utf-8"))
    return sum(sum(counts.values()) for counts in data["violations"].values())


def run_case(row: pd.Series, depot: str, output: Path) -> OobpResult:
    card = "henn_w5" if row.group == HENN_GROUP else "albareda"
    with tempfile.TemporaryDirectory(dir=output) as folder:
        out = Path(folder)
        with initialize_config_dir(config_dir=str(CONFIG), version_base="1.3"):
            cfg = compose(config_name="oobp_config", overrides=[
                f"data_card={card}",
                f"data_card.source.filepath='{row.path_in_data_oobp}'",
                *( [f"data_card.source.depot={depot}"] if card == "henn_w5" else [] ),
                f"project_root={ROOT.as_posix()}",
                f"cache_base={(out / 'cache').as_posix()}",
                f"experiment.output_dir={out.as_posix()}",
                "+experiment.audit=true",
            ])
        sim, engine = setup_experiment(cfg, load_and_flatten_data_card(cfg.data_card))
        sim.reset(hooks=[add_orders_hook, picker_arrival_hook])
        decisions = 0
        while True:
            done, snapshot = sim.run()
            if done:
                break
            decisions += 1
            apply_decision(sim, engine, engine.on_trigger(snapshot), snapshot)
        audit_path = out / "audit.json"
        audit_violations = violations(audit_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        kpis = json.loads((out / "kpis" / "kpis.json").read_text(encoding="utf-8"))
        record = {
            "instance": row.instance, "stratum": row.stratum, "depot": depot,
            "command": ("python -m scenarios.scenario_oobp.run_l3 full "
                        f"# {row.stratum}/{row.instance} depot={depot}"),
            "kpis": kpis, "decisions": audit["decisions"],
            "casim_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "ware_ops_algos_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT.parent / "ware_ops_algos",
                text=True).strip(),
            "committed_tours": audit["committed_tours"],
            "trace_digest": audit["trace_digest"],
            "event_count": audit["event_count"],
            "violations": audit["violations"],
            "first_violations": audit["first_violations"],
            "kpi_differences": recompute_kpis(audit, kpis),
        }
        record["status"] = ("inconsistent" if record["kpi_differences"] or
                            audit_violations else "consistent")
        baseline_path = BASELINES / f"{row.stratum.replace('/', '_')}__{row.instance}__{depot}.json"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return _result(sim, decisions, audit_violations)


def _result(sim, decisions: int, audit_violations: int) -> OobpResult:
    """Max completion time and max turnover time from the executed tours (7.3)."""
    completion = 0.0
    turnover = 0.0
    for _, _, end, order_ids, _picker_id, *_ in sim.state.tracker.completed_tours:
        completion = max(completion, end)
        for order_id in order_ids:
            arrival = sim.state.order_manager.get_order_from_history(order_id).order_date
            turnover = max(turnover, end - arrival)
    return OobpResult(
        distance=sum(sim.state.tracker.distance_by_picker.values()),
        max_completion_s=completion,
        max_turnover_s=turnover,
        tours=len(sim.state.tracker.completed_tours),
        decisions=decisions,
        violations=audit_violations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["smoke", "full"])
    parser.add_argument("--groups", nargs="*")
    parser.add_argument("--strata", nargs="*")
    parser.add_argument("--exclude-results", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = pd.read_csv(RUNLIST, dtype={"instance": str, "stratum": str})
    for column in ("group", "stratum", "instance", "path_in_data_oobp"):
        rows[column] = rows[column].str.strip()
    if args.stage == "smoke":
        smoke = {"abc1": ["29s-40-30-0", "30s-40-45-0"], "abc2": ["10l-40-45-0", "11l-40-60-0"],
                 "ran1": ["29s-40-30-0", "30s-40-45-0"], "ran2": ["10l-40-45-0", "11l-40-60-0"],
                 "W1/100": ["wsrp_input_pedido_01_000", "wsrp_input_pedido_01_030"],
                 "W2/100": ["wsrp_input_pedido_02_000", "wsrp_input_pedido_02_030"],
                 "W3/100": ["wsrp_input_pedido_03_000", "wsrp_input_pedido_03_030"],
                 "W4/100": ["wsrp_input_pedido_04_000", "wsrp_input_pedido_04_030"]}
        rows = rows[[stem in smoke.get(stratum, []) for stratum, stem
                     in zip(rows.stratum, rows.instance)]]
    if args.groups:
        rows = rows[rows.group.isin(args.groups)]
    if args.strata:
        rows = rows[rows.stratum.isin(args.strata)]
    reference = pd.read_csv(REFERENCE).set_index("instance")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if not args.strata else "_" + "_".join(
        stratum.replace("/", "-") for stratum in args.strata)
    path = OUTPUT / f"l3_{args.stage}{suffix}.csv"
    records = pd.read_csv(path, dtype={"depot": str}).to_dict("records") \
        if args.resume and path.exists() else []
    done_keys = {(record["stratum"], record["instance"], record["depot"])
                 for record in records}
    if args.exclude_results:
        prior = pd.read_csv(args.exclude_results, dtype={"depot": str})
        done_keys.update((item.stratum, item.instance, item.depot)
                         for item in prior.itertuples())
    for _, row in rows.iterrows():
        for depot in depot_of(row):
            key = (row.stratum, row.instance, depot)
            if key in done_keys:
                continue
            started = time.perf_counter()
            result = run_case(row, depot, OUTPUT)
            ref = reference.loc[instance_key(row.group, row.stratum, row.path_in_data_oobp)]
            record = {
                "group": row.group, "stratum": row.stratum,
                "instance": row.instance, "depot": depot,
                "reference": instance_key(row.group, row.stratum, row.path_in_data_oobp),
                "distance": result.distance,
                "max_completion_s": result.max_completion_s,
                "max_turnover_s": result.max_turnover_s,
                "tours": result.tours, "decisions": result.decisions,
                "violations": result.violations,
                "gil_completion_s": ref.gil_completion_s,
                "gil_turnover_s": ref.gil_turnover_s,
                "best_known_completion_s": ref.best_known_completion_s,
                "best_known_turnover_s": ref.best_known_turnover_s,
                "seconds": round(time.perf_counter() - started, 1),
            }
            for objective in ("completion", "turnover"):
                record[f"below_best_known_{objective}"] = (
                    record[f"max_{objective}_s"]
                    < record[f"best_known_{objective}_s"] * (1 - REL_TOL))
            record["protocol_status"] = (
                "protocol not reproduced" if
                record["below_best_known_completion"] or
                record["below_best_known_turnover"] else "comparator only")
            records.append(record)
            pd.DataFrame(records).to_csv(path, index=False)
            print(f"{row.group} {row.instance} {depot}: completion "
                  f"{record['max_completion_s']:.1f} (bk {record['best_known_completion_s']:.1f}), "
                  f"turnover {record['max_turnover_s']:.1f} "
                  f"(bk {record['best_known_turnover_s']:.1f}), "
                  f"violations {record['violations']}", flush=True)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
