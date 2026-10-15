"""Record and compare Phase B scenario baselines at fixed configuration inputs."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "tests" / "baseline"
RUNS = ROOT / ".phaseb_runs"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)
sys.path.insert(0, str(ROOT))
from tests.verification.kpi_recomputation import recompute_kpis  # noqa: E402


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
    instances = (
        "29s-40-30-0.txt", "30s-40-45-0.txt", "31s-40-60-0.txt",
        "32s-40-75-0.txt", "37s-60-30-0.txt", "38s-60-45-0.txt",
        "39s-60-60-0.txt", "40s-60-75-0.txt", "61s-80-30-0.txt",
        "62s-80-45-0.txt", "63s-80-60-0.txt", "64s-80-75-0.txt",
        "69s-100-30-0.txt", "70s-100-45-0.txt", "72s-100-75-0.txt",
    )
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

    grocery = "scenarios.scenario_grocery_retailer.experiment_grocery_retailer"
    for repo in ("return", "midpoint", "largest_gap", "nearest_neighbour",
            "sshape", "adaptive_routing", "full_portfolio"):
        add(f"grocery__{repo}", "scenario_grocery_retailer", grocery,
            (f"cosy_repo={repo}",))

    ijpe = "scenarios.scenario_ijpe.scenario_grocery_retailer"
    for simulation in ("baseline", "scenario1", "scenario2", "scenario3",
                       "scenario4"):
        add(f"ijpe__{simulation}", "scenario_ijpe", ijpe,
            (f"simulation={simulation}",))
    return result


def _config_name(scenario: str) -> str:
    return {
        "scenario_grocery_retailer": "grocery_retailer_config",
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


def run_case(name: str, definition: tuple[str, str, tuple[str, ...]]) -> dict:
    """Check and run one configuration, retaining only diagnostic outputs."""
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
    check_code, check_tail = _run(check)
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
                "kpi_differences", "event_count"):
        if recorded[key] != current[key]:
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


BENCHMARKS = {"l3": ("oobp", _l3_overrides)}


def benchmark_cases(kind: str) -> dict[str, tuple[Path, tuple[str, str | None, tuple[str, ...]]]]:
    """Every recorded run of a benchmark, as a casim-run case of its scenario."""
    scenario, overrides = BENCHMARKS[kind]
    return {path.stem: (path, (scenario, None, overrides(json.loads(
                path.read_text(encoding="utf-8")))))
            for path in sorted((BASELINES / "benchmarks" / kind).glob("*.json"))}


def compare_benchmark(kind: str, workers: int) -> list[str]:
    """Re-run every recorded run of a benchmark in parallel and compare."""
    def one(item):
        name, (path, definition) = item
        return name, compare_recorded(json.loads(path.read_text(encoding="utf-8")),
                                      run_case(name, definition))

    findings = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, differences in pool.map(one, benchmark_cases(kind).items()):
            print(f"compare {kind} {name}: {'; '.join(differences) or 'equal'}", flush=True)
            findings.extend(f"{name}: {difference}" for difference in differences)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("record", "compare", "compare-benchmark"))
    parser.add_argument("--case", action="append", dest="selected")
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    selected = set(args.selected) if args.selected else None
    if args.action == "record":
        record(selected)
        return 0
    if args.action == "compare-benchmark":
        return 1 if compare_benchmark(args.benchmark, args.workers) else 0
    return 1 if compare(selected) else 0



if __name__ == "__main__":
    raise SystemExit(main())
