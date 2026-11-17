"""Record and compare Phase B scenario baselines at fixed configuration inputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "tests" / "baseline"
RUNS = ROOT / ".phaseb_runs"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)
sys.path.insert(0, str(ROOT))
from tests.verification.kpi_recomputation import recompute_kpis  # noqa: E402
from scripts.l1.run_benchmark import raw_instance, scenario_overrides  # noqa: E402


def cases() -> dict[str, tuple[str, str | None, tuple[str, ...]]]:
    """Inventory defaults, sweeps and documented configuration selections.

    A case without a module is a scenario of the Phase S structure and runs
    with ``casim run``; the others still run their own entry point.
    """
    result = {}

    def add(name, scenario, module, overrides=()):
        result[name] = (scenario, module, tuple(overrides))

    henn = None
    add("henn__default", "henn_waiting", henn)
    add("henn__repair", "henn_waiting", henn,
        ("cosy_repo=repair_waiting_henn", "+decisions.OBRSP.change_penalty=10"))
    add("henn__release_only", "henn_waiting", henn,
        ("decisions=release_only_example",))
    # One small and one large instance cover both strategies' code paths; the
    # sweep over all 15 instances is a study, not a regression test.
    instances = ("29s-40-30-0.txt", "63s-80-60-0.txt")
    for instance in instances:
        for strategy in ("henn_waiting", "order_window"):
            add(f"henn__{instance[:-4]}__{strategy}", "henn_waiting", henn,
                (f"source.filepath={instance}", f"decisions={strategy}"))

    walk = None
    add("walk__default", "walk_or_wait", walk)
    for minimum in (0, 2, 3, 4):
        add(f"walk__wait_{minimum}", "walk_or_wait", walk, (f"cosy_repo=wait_{minimum}",))
    add("walk__radar_wait_1", "walk_or_wait", walk, ("variants=radar_standard_case",))
    add("walk__patrol", "walk_or_wait", walk,
        ("cosy_repo=paper_patrol", "variants=explicit_wait0"))
    add("walk__center", "walk_or_wait", walk,
        ("cosy_repo=paper_center", "decisions=paper_intervention_center",
         "variants=explicit_wait0"))
    add("walk__no_intervention_return", "walk_or_wait", walk,
        ("decisions=paper_no_intervention", "cosy_repo=wait_4_return"))
    add("walk__no_intervention_nearest", "walk_or_wait", walk,
        ("decisions=paper_no_intervention", "cosy_repo=wait_2_nearest_neighbour"))
    add("walk__no_intervention_largest_gap", "walk_or_wait", walk,
        ("decisions=paper_no_intervention", "cosy_repo=wait_1_largest_gap"))

    stochastic = None
    for repo in ("analytic_waiting", "henn_waiting", "no_waiting"):
        add(f"stochastic__{repo}", "analytic_waiting", stochastic,
            (f"cosy_repo={repo}",))

    ijpe = "scenarios.scenario_ijpe.scenario_grocery_retailer"
    for simulation in ("baseline", "scenario1", "scenario2", "scenario3",
                       "scenario4"):
        add(f"ijpe__{simulation}", "scenario_ijpe", ijpe,
            (f"simulation={simulation}",))
    return result


def _config_name(scenario: str) -> str:
    return {
        "scenario_ijpe": "grocery_retailer_config",
    }[scenario]


def _run(command: list[str], env: dict | None = None,
         timeout: int = 900) -> tuple[int, str]:
    try:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", timeout=timeout)
        return result.returncode, result.stdout[-2000:]
    except subprocess.TimeoutExpired as error:
        return 124, f"Timed out after {timeout}s: {error}"


def _relative_command(command: list[str]) -> str:
    """Store reproducible commands without checkout-specific absolute paths."""
    root = str(ROOT)
    return " ".join(part.replace(root + os.sep, "").replace(root, ".")
                    for part in command)


_CHECKS: dict[tuple, tuple[int, str]] = {}
_CHECK_LOCKS: dict[tuple, threading.Lock] = {}
_CHECK_GUARD = threading.Lock()


def sweep_key(definition: tuple[str, str | None, tuple[str, ...]]) -> tuple:
    """The sweep a case belongs to: its configuration without the instance data.

    Runs of one sweep differ only in ``source.*`` overrides, which select data
    and do not change what ``casim check`` resolves, so the sweep is checked once.
    """
    scenario, module, overrides = definition
    return scenario, module, tuple(item for item in overrides
                                   if not item.lstrip("+").startswith("source."))


def _check_once(definition, check: list[str]) -> tuple[int, str]:
    key = sweep_key(definition)
    with _CHECK_GUARD:
        lock = _CHECK_LOCKS.setdefault(key, threading.Lock())
    with lock:
        if key not in _CHECKS:
            _CHECKS[key] = _run(check)
        return _CHECKS[key]


def run_case(name: str, definition: tuple[str, str, tuple[str, ...]]) -> dict:
    """Check the case's sweep once and run the configuration; only a failed run keeps its outputs."""
    scenario, module, overrides = definition
    out_dir = RUNS / f"{name}-{uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = out_dir / "audit.json"
    if module is None:
        directory = str(ROOT / "scenarios" / scenario)
        check = [str(PYTHON), "-m", "casim", "check", directory, *overrides, "--json"]
        command = [str(PYTHON), "-m", "casim", "run", directory, *overrides, "--audit",
                   "--output-dir", str(out_dir)]
        recorded_command = command[:-2]
        kpi_path = out_dir / "kpis.json"
    else:
        check = [str(PYTHON), "-m", "casim", "check",
                 str(ROOT / "scenarios" / scenario / "config"),
                 _config_name(scenario), *overrides, "--json"]
        command = [str(PYTHON), "-m", module, *overrides,
                   "viz.launch=false", "+experiment.audit=true",
                   f"hydra.run.dir={out_dir}"]
        recorded_command = command[:-1]
        kpi_path = out_dir / "kpis" / "kpis.json"
    check_code, check_tail = _check_once(definition, check)
    run_code, run_tail = _run(command)
    result = {
        "name": name, "scenario": scenario,
        "command": _relative_command(recorded_command),
        "check_command": _relative_command(check),
        "check_exit": check_code, "check_output": check_tail if check_code else "ok",
        "run_exit": run_code, "run_output": run_tail if run_code else "ok",
    }
    if run_code != 0 or not capture.exists() or not kpi_path.exists():
        result["status"] = "not_runnable"
        return result
    audit = json.loads(capture.read_text(encoding="utf-8"))
    kpis = json.loads(kpi_path.read_text(encoding="utf-8"))
    # The record holds everything the comparison needs; a failed run keeps its
    # folder for diagnosis, a successful one does not.
    shutil.rmtree(out_dir, ignore_errors=True)
    differences = recompute_kpis(audit, kpis)
    result.update({
        "casim_commit": _run(["git", "rev-parse", "HEAD"])[1].strip(),
        "ware_ops_algos_commit": _run([
            "git", "-C", str(ROOT.parent / "ware_ops_algos"),
            "rev-parse", "HEAD"])[1].strip(),
        "kpis": kpis,
        # Counts on_decision callbacks by decision name, including callbacks
        # that commit no tour; this is not a count of committed tours.
        "decisions": audit["decisions"],
        "committed_tours": audit["committed_tours"],
        "trace_digest": audit["trace_digest"],
        "event_count": audit["event_count"],
        "violations": audit["violations"],
        "first_violations": audit["first_violations"],
        "open_orders": audit["open_orders"],
        "kpi_differences": differences,
    })
    result["status"] = ("inconsistent" if differences or any(
        audit["violations"][kind] for kind in audit["violations"])
        else "consistent")
    return result


def compare_recorded(recorded: dict, current: dict) -> list[str]:
    """Describe every difference from one recorded configuration."""
    findings = []
    if recorded["status"] != current["status"]:
        findings.append(f"status {recorded['status']} -> {current['status']}")
    if current["status"] == "not_runnable":
        return findings or ["run became unavailable"]
    old_kpis, new_kpis = recorded["kpis"], current["kpis"]
    for key in sorted(set(old_kpis) | set(new_kpis)):
        if old_kpis.get(key) != new_kpis.get(key):
            findings.append(f"KPI {key}: {old_kpis.get(key)} -> {new_kpis.get(key)}")
    for key in ("decisions", "violations", "first_violations",
                "kpi_differences", "event_count", "open_orders"):
        if recorded.get(key) != current.get(key):
            findings.append(f"{key}: {recorded[key]} -> {current[key]}")
    old_tours, new_tours = recorded["committed_tours"], current["committed_tours"]
    for index in range(max(len(old_tours), len(new_tours))):
        old = old_tours[index] if index < len(old_tours) else None
        new = new_tours[index] if index < len(new_tours) else None
        if old != new:
            findings.append(f"first committed tour {index}: {old} -> {new}")
            break
    if recorded["trace_digest"] != current["trace_digest"]:
        findings.append("trace digest changed")
    return findings


def record(selected: set[str] | None = None) -> list[dict]:
    """Write one immutable baseline file for every runnable case."""
    BASELINES.mkdir(parents=True, exist_ok=True)
    inventory = []
    for name, definition in cases().items():
        if selected is not None and name not in selected:
            continue
        print(f"record {name}", flush=True)
        result = run_case(name, definition)
        inventory.append({key: result[key] for key in (
            "name", "scenario", "command", "check_command", "check_exit",
            "check_output", "run_exit", "run_output", "status")})
        if result["status"] != "not_runnable":
            (BASELINES / f"{name}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8")
        print(f"  {result['status']} check={result['check_exit']} "
              f"run={result['run_exit']}", flush=True)
    inventory_path = BASELINES / "inventory.json"
    if selected is not None and inventory_path.exists():
        existing = {item["name"]: item for item in json.loads(
            inventory_path.read_text(encoding="utf-8"))}
        existing.update({item["name"]: item for item in inventory})
        inventory_to_write = [existing[name] for name in cases() if name in existing]
    else:
        inventory_to_write = inventory
    inventory_path.write_text(json.dumps(inventory_to_write, indent=2),
                              encoding="utf-8")
    return inventory


def compare(selected: set[str] | None = None) -> list[str]:
    """Re-run and compare every saved case."""
    findings = []
    for name, definition in cases().items():
        if selected is not None and name not in selected:
            continue
        path = BASELINES / f"{name}.json"
        if not path.exists():
            continue
        print(f"compare {name}", flush=True)
        old = json.loads(path.read_text(encoding="utf-8"))
        new = run_case(name, definition)
        differences = compare_recorded(old, new)
        for difference in differences:
            findings.append(f"{name}: {difference}")
            print(f"  {difference}", flush=True)
    return findings


def _l3_overrides(record: dict) -> tuple[str, ...]:
    if record["stratum"].startswith("W"):
        return ("variants=albareda",
                f"source.filepath=albareda/{record['stratum']}/{record['instance']}.txt")
    return (f"source.filepath={record['stratum']}/{record['instance']}.txt",
            f"source.depot={record['depot']}")


def _l2_overrides(record: dict) -> tuple[str, ...]:
    target, storage, orders, capacity, routing, rule, replication, mode = \
        record["name"].split("__")
    if routing != "SShape":
        raise ValueError(f"{record['name']}: only S-shape runs are recorded")
    return (f"decisions=l2_{'journal' if target == 'journal' else 'working'}",
            f"cosy_repo={target}_sshape_{rule.lower()}", f"source.storage={storage}",
            f"source.n_orders={orders}", f"source.capacity={capacity}",
            f"source.replication={replication}", f"source.arrival_mode={mode}")


def _l1_overrides(record: dict) -> tuple[str, ...]:
    instance_set, instance = raw_instance(record["instance_set"], record["instance"])
    return tuple(scenario_overrides(instance_set, instance, record["pipeline"]))


BENCHMARKS = {"l1": ("casop_static", _l1_overrides), "l2": ("henn2012", _l2_overrides),
              "l3": ("oobp", _l3_overrides)}


def benchmark_cases(kind: str) -> dict[str, tuple[Path, tuple[str, str | None, tuple[str, ...]]]]:
    """Every recorded run of a benchmark, as a casim-run case of its scenario."""
    scenario, overrides = BENCHMARKS[kind]
    return {path.stem: (path, (scenario, None, overrides(json.loads(
                path.read_text(encoding="utf-8")))))
            for path in sorted((BASELINES / "benchmarks" / kind).glob("*.json"))}


def compare_benchmark(kind: str, workers: int,
                      patterns: list[str] | None = None) -> list[str]:
    """Re-run the recorded runs of a benchmark in parallel and compare.

    ``patterns`` (shell globs on the run name) restrict the rerun to the cells a
    change can affect; without them every recorded run is rerun.
    """
    selected = {name: case for name, case in benchmark_cases(kind).items()
                if not patterns or any(fnmatch(name, pattern) for pattern in patterns)}
    return _compare_parallel({name: (kind, path, definition)
                              for name, (path, definition) in selected.items()}, workers)


def _compare_parallel(selected: dict[str, tuple[str, Path, tuple]], workers: int) -> list[str]:
    def one(item):
        name, (_, path, definition) = item
        return name, compare_recorded(json.loads(path.read_text(encoding="utf-8")),
                                      run_case(name, definition))

    findings = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, differences in pool.map(one, selected.items()):
            print(f"compare {selected[name][0]} {name}: "
                  f"{'; '.join(differences) or 'equal'}", flush=True)
            findings.extend(f"{name}: {difference}" for difference in differences)
    return findings


RECORDED_FIELDS = ("command", "casim_commit", "ware_ops_algos_commit", "kpis", "decisions",
                   "committed_tours", "trace_digest", "event_count", "violations",
                   "first_violations", "kpi_differences", "open_orders", "status")


def record_benchmark(kind: str, workers: int, patterns: list[str]) -> list[str]:
    """Re-run the recorded runs matching ``patterns`` and record them again.

    Fields a run does not produce (its name, the stream closure of L2) are kept
    from the old record. A run that is no longer runnable keeps its old record
    and is reported.
    """
    selected = {name: case for name, case in benchmark_cases(kind).items()
                if any(fnmatch(name, pattern) for pattern in patterns)}

    def one(item):
        name, (path, definition) = item
        old = json.loads(path.read_text(encoding="utf-8"))
        new = run_case(name, definition)
        if new["status"] == "not_runnable":
            return f"{name}: not runnable, old record kept: {new['run_output'][-300:]}"
        old.update({key: new[key] for key in RECORDED_FIELDS})
        path.write_text(json.dumps(old, indent=2), encoding="utf-8")
        return f"{name}: recorded, {new['status']}"

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(one, selected.items()):
            print(f"record {kind} {line}", flush=True)
            if "not runnable" in line:
                failures.append(line)
    print(f"{len(selected) - len(failures)} of {len(selected)} runs recorded", flush=True)
    return failures


# The fast tier: a fixed subset of one or two recorded cases per scenario and
# benchmark set, under five minutes in total on four workers, run after every
# commit. The full baseline (``compare`` and ``compare-benchmark`` for l1, l2
# and l3) runs at the end of a task. Timings are in docs/revision_scopes_status.md.
FAST = {
    "scenarios": ("henn__default", "henn__29s-40-30-0__order_window", "walk__default",
                  "walk__patrol", "stochastic__henn_waiting"),
    "l1": ("HennWaescher__21s-20-30-1_u__GIA+SavingsSShape+SShape",
           "BahceciOencan__Pr_20_1_20_Store1_13__GIA+LSBatchingSShapeFiFoOrderNr+SShape"),
    "l2": ("journal__C__30__45__SShape__FIRST__0__conditioned",
           "working__C__30__45__SShape__SAV__0__conditioned"),
    "l3": ("W1_100__wsrp_input_pedido_01_000__0", "abc1__29s-40-30-0__centre"),
}


def compare_fast(workers: int) -> list[str]:
    """Re-run and compare the fast tier."""
    selected = {name: ("scenario", BASELINES / f"{name}.json", cases()[name])
                for name in FAST["scenarios"]}
    for kind in BENCHMARKS:
        recorded = benchmark_cases(kind)
        selected.update({name: (kind, *recorded[name]) for name in FAST[kind]})
    return _compare_parallel(selected, workers)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("record", "compare", "compare-benchmark", "compare-fast",
                                           "record-benchmark"))
    parser.add_argument("--case", action="append", dest="selected")
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--match", action="append", dest="patterns",
                        help="compare-benchmark, record-benchmark: only runs whose name "
                             "matches this glob (required for record-benchmark)")
    args = parser.parse_args()
    selected = set(args.selected) if args.selected else None
    if args.action == "record":
        record(selected)
        return 0
    if args.action == "compare-benchmark":
        return 1 if compare_benchmark(args.benchmark, args.workers, args.patterns) else 0
    if args.action == "record-benchmark":
        if not args.patterns:
            parser.error("record-benchmark needs --match")
        return 1 if record_benchmark(args.benchmark, args.workers, args.patterns) else 0
    if args.action == "compare-fast":
        return 1 if compare_fast(args.workers) else 0
    return 1 if compare(selected) else 0


if __name__ == "__main__":
    raise SystemExit(main())
