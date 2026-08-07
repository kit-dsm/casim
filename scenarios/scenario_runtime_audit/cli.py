from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import math
import os
import platform
import pstats
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(__file__).parents[2].resolve()
AUDIT_ROOT = ROOT / "outputs" / "runtime_audit"
ONLINE_MODES = (
    "no_wait",
    "reopt",
    "periodic_rolling",
    "unstarted_reopt",
    "routing_intervention",
)
DYNAMIC_MODES = ONLINE_MODES[2:]


class _NullProgress:
    n = 0

    def update(self, amount=1):
        self.n += amount

    def set_postfix_str(self, value):
        return None

    def close(self):
        return None


class _CountingLogger:
    def __init__(self):
        self.events = 0

    def on_reset(self, sim, domain):
        return None

    def on_event(self, event, sim):
        self.events += 1

    def on_done(self, sim):
        return None


class _PhaseRecorder:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = Counter()

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - started
            self.calls[name] += 1

    def wrap(self, obj, method_name: str, phase: str) -> None:
        original = getattr(obj, method_name)

        def measured(*args, **kwargs):
            with self.measure(phase):
                return original(*args, **kwargs)

        setattr(obj, method_name, measured)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _powershell_value(command: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def machine_metadata() -> dict:
    lock = ROOT / "uv.lock"
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "windows_version": platform.win32_ver(),
        "python": sys.version,
        "python_executable": sys.executable,
        "processor": platform.processor() or _powershell_value(
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 "
            "-ExpandProperty Name)"
        ),
        "logical_cpu_count": os.cpu_count(),
        "active_power_plan": _powershell_value("powercfg /getactivescheme"),
        "uv_lock_sha256": (
            hashlib.sha256(lock.read_bytes()).hexdigest()
            if lock.is_file()
            else None
        ),
    }


def _compose_config(mode: str, profile: str, output_dir: Path):
    if mode in {"no_wait", "reopt"}:
        config_dir = ROOT / "scenarios" / "scenario_reopt" / "config"
        with initialize_config_dir(
            version_base="1.3", config_dir=str(config_dir)
        ):
            cfg = compose(
                config_name="reopt_config",
                overrides=[f"variant={mode}"],
            )
    else:
        config_dir = (
            ROOT / "scenarios" / "scenario_dynamic_operations" / "config"
        )
        simulation = "three_day" if profile == "three_day" else "smoke"
        with initialize_config_dir(
            version_base="1.3", config_dir=str(config_dir)
        ):
            cfg = compose(
                config_name="dynamic_operations_config",
                overrides=[f"engines={mode}", f"simulation={simulation}"],
            )
        OmegaConf.update(cfg, "viz.record", False, merge=False)
        OmegaConf.update(cfg, "viz.launch", False, merge=False)

    OmegaConf.update(cfg, "project_root", str(ROOT), merge=False)
    OmegaConf.update(
        cfg, "instances_base", str(ROOT / "scenarios"), merge=False
    )
    OmegaConf.update(
        cfg, "cache_base", str(output_dir / "cache"), merge=False
    )
    OmegaConf.update(
        cfg, "experiment.output_dir", str(output_dir), merge=False
    )
    OmegaConf.update(
        cfg, "experiment.working_dir", str(output_dir / "work"), merge=False
    )
    return cfg


def _instrument(simulation, decision_engine, recorder: _PhaseRecorder):
    for adapter in simulation.state_adapters.values():
        recorder.wrap(adapter, "transform_state", "snapshot")
    for conditions in simulation.conditions_map.values():
        for condition in conditions:
            if condition is not None:
                recorder.wrap(condition, "get_decision", "conditions")
    for solver in decision_engine.solver_map.values():
        recorder.wrap(solver, "solve", "solver")
    recorder.wrap(decision_engine, "solution_to_events", "event_conversion")
    for logger in simulation.event_loggers:
        recorder.wrap(logger, "on_event", "logging")
        recorder.wrap(logger, "on_done", "logging")


def _setup(mode: str, profile: str, output_dir: Path, observed: bool, recorder):
    from scenarios.experiment_commons import (
        load_and_flatten_data_card,
        setup_decision_engine,
        setup_scenario,
    )

    with recorder.measure("configuration"):
        cfg = _compose_config(mode, profile, output_dir)
        data_card = load_and_flatten_data_card(
            OmegaConf.to_container(cfg.data_card, resolve=True)
        )
    with recorder.measure("simulation_setup"):
        simulation = setup_scenario(cfg)
    if not observed:
        simulation.event_loggers = []
        import casim.simulation_engine.simulation_engine as simulation_module

        simulation_module.tqdm = lambda *args, **kwargs: _NullProgress()
    counter = _CountingLogger()
    simulation.event_loggers.append(counter)
    with recorder.measure("decision_setup"):
        decision_engine = setup_decision_engine(
            cfg, data_card, simulation.state_adapters
        )
        decision_engine.event_map = {
            name: instantiate(event)
            for name, event in cfg.engines.decision_engine.event_map.items()
        }
    _instrument(simulation, decision_engine, recorder)
    return cfg, simulation, decision_engine, counter


def _run_online(mode, cfg, simulation, decision_engine, recorder):
    if mode in {"no_wait", "reopt"}:
        from scenarios.scenario_reopt.scenario_specific_hooks import (
            build_sim_hooks,
        )

        hooks = build_sim_hooks()
    else:
        from scenarios.scenario_dynamic_operations.scenario_specific_hooks import (
            build_hooks,
        )

        hooks = build_hooks(cfg)

    with recorder.measure("reset"):
        initial_domain = simulation.reset(hooks=hooks)

    decisions = 0
    while True:
        with recorder.measure("simulation_run"):
            done, snapshot = simulation.run()
        if done:
            break
        if snapshot is None:
            raise RuntimeError("Simulation paused without a decision snapshot")
        with recorder.measure("decision_total"):
            selected = decision_engine.on_trigger(snapshot)
        if selected is None:
            raise RuntimeError(
                f"No decision for {snapshot.problem_class} at "
                f"t={snapshot.dynamic_warehouse_info.time}"
            )
        events, solution = selected
        with recorder.measure("commitment_step"):
            simulation.step(events, snapshot.problem_class, solution)
        decisions += 1

    tracker = simulation.state.tracker
    return {
        "completion_reason": simulation.state.completion_reason,
        "received_orders": len(initial_domain.orders.orders),
        "completed_orders": len(
            simulation.state.order_manager.completed_orders
        ),
        "completed_tours": len(tracker.completed_tours),
        "decisions": decisions,
        "decision_problem_counts": dict(
            Counter(row[0] for row in decision_engine.decision_tracker.decisions)
        ),
        "pipeline_counts": dict(
            decision_engine.decision_tracker.pipeline_counts
        ),
        "algorithm_runtime_s": sum(
            float(row[5])
            for row in decision_engine.decision_tracker.decisions
        ),
        "decision_elapsed_s": sum(
            float(row[6])
            for row in decision_engine.decision_tracker.decisions
        ),
        "interventions": len(tracker.interventions),
        "route_replacements": sum(
            bool(value.get("replaced")) for value in tracker.interventions
        ),
    }


def execute_worker(
    mode: str,
    profile: str,
    output_dir: Path,
    *,
    observed: bool = False,
) -> dict:
    if mode not in ONLINE_MODES:
        raise ValueError(f"Unknown audit mode: {mode}")
    if profile == "three_day" and mode not in DYNAMIC_MODES:
        raise ValueError("Three-day scaling is only defined for dynamic modes")

    recorder = _PhaseRecorder()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    cfg, simulation, decision_engine, counter = _setup(
        mode, profile, output_dir, observed, recorder
    )
    outcomes = _run_online(
        mode, cfg, simulation, decision_engine, recorder
    )
    cpu_time = time.process_time() - cpu_started
    wall_time = time.perf_counter() - wall_started

    phase = dict(recorder.seconds)
    phase["decision_overhead"] = max(
        0.0,
        phase.get("decision_total", 0.0)
        - phase.get("solver", 0.0)
        - phase.get("event_conversion", 0.0),
    )
    phase["event_loop_exclusive"] = max(
        0.0,
        phase.get("simulation_run", 0.0)
        - phase.get("snapshot", 0.0)
        - phase.get("conditions", 0.0)
        - phase.get("logging", 0.0),
    )
    denominator = wall_time or math.nan
    return {
        "schema_version": 1,
        "status": "success",
        "mode": mode,
        "profile": (
            "paper_example" if mode in {"no_wait", "reopt"} else profile
        ),
        "observed": observed,
        "wall_time_s": wall_time,
        "process_cpu_time_s": cpu_time,
        "cpu_to_wall_ratio": cpu_time / denominator,
        "phase_times_s": phase,
        "phase_calls": dict(recorder.calls),
        "counts": {
            "events": counter.events,
            "decisions": outcomes["decisions"],
            "completed_orders": outcomes["completed_orders"],
        },
        "rates": {
            "events_per_s": counter.events / denominator,
            "decisions_per_s": outcomes["decisions"] / denominator,
            "completed_orders_per_s": (
                outcomes["completed_orders"] / denominator
            ),
        },
        "existing_timing": {
            "algorithm_runtime_s": outcomes.pop("algorithm_runtime_s"),
            "decision_elapsed_s": outcomes.pop("decision_elapsed_s"),
        },
        "outcomes": outcomes,
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_repetitions(rows: list[dict]) -> dict:
    successful = [row for row in rows if row.get("status") == "success"]
    if not successful:
        return {"successful": 0, "failed": len(rows)}
    values = [float(row["subprocess_wall_time_s"]) for row in successful]
    in_process = [float(row["wall_time_s"]) for row in successful]
    mean = statistics.fmean(values)
    cv = (
        statistics.stdev(values) / mean
        if len(values) > 1 and mean
        else 0.0
    )
    return {
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "wall_time_s": {
            "minimum": min(values),
            "median": statistics.median(values),
            "q1": _quantile(values, 0.25),
            "q3": _quantile(values, 0.75),
            "mean": mean,
            "coefficient_of_variation": cv,
            "high_variability": cv > 0.10,
        },
        "in_process_wall_time_s": {
            "minimum": min(in_process),
            "median": statistics.median(in_process),
            "q1": _quantile(in_process, 0.25),
            "q3": _quantile(in_process, 0.75),
        },
        "median_rates": {
            name: statistics.median(
                float(row["rates"][name]) for row in successful
            )
            for name in (
                "events_per_s",
                "decisions_per_s",
                "completed_orders_per_s",
            )
        },
        "median_phase_times_s": {
            name: statistics.median(
                float(row["phase_times_s"].get(name, 0.0))
                for row in successful
            )
            for name in sorted(
                {
                    key
                    for row in successful
                    for key in row["phase_times_s"]
                }
            )
        },
    }


def _worker_command(
    mode: str,
    profile: str,
    output_file: Path,
    observed: bool,
    profile_file: Path | None = None,
) -> list[str]:
    command = [sys.executable]
    command.extend(
        [
            "-m",
            "scenarios.scenario_runtime_audit.cli",
            "_worker",
            "--mode",
            mode,
            "--profile",
            profile,
            "--output-file",
            str(output_file),
        ]
    )
    if observed:
        command.append("--observed")
    if profile_file is not None:
        command.extend(["--profile-file", str(profile_file)])
    return command


def _run_worker_process(
    mode: str,
    profile: str,
    run_dir: Path,
    timeout_s: float,
    *,
    observed: bool = False,
    profile_file: Path | None = None,
) -> dict:
    output_file = run_dir / "worker.json"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            _worker_command(
                mode, profile, output_file, observed, profile_file
            ),
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "status": "timeout",
            "mode": mode,
            "profile": profile,
            "timeout_s": timeout_s,
            "subprocess_wall_time_s": time.perf_counter() - started,
            "stdout_tail": (error.stdout or "")[-2000:],
            "stderr_tail": (error.stderr or "")[-2000:],
        }
    subprocess_wall = time.perf_counter() - started
    if completed.returncode != 0 or not output_file.is_file():
        return {
            "status": "failed",
            "mode": mode,
            "profile": profile,
            "returncode": completed.returncode,
            "subprocess_wall_time_s": subprocess_wall,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    row = json.loads(output_file.read_text(encoding="utf-8"))
    row["subprocess_wall_time_s"] = subprocess_wall
    row["stdout_tail"] = completed.stdout[-1000:]
    row["stderr_tail"] = completed.stderr[-1000:]
    return row


def run_benchmark(args) -> dict:
    output_dir = Path(args.output_dir or AUDIT_ROOT / _timestamp()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = [("small", ONLINE_MODES, args.smoke_repetitions)]
    if not args.skip_scaling:
        groups.append(("three_day", DYNAMIC_MODES, args.scale_repetitions))

    result = {
        "schema_version": 1,
        "command": "benchmark",
        "machine": machine_metadata(),
        "protocol": {
            "warmups_per_mode": args.warmups,
            "small_repetitions": args.smoke_repetitions,
            "three_day_repetitions": (
                0 if args.skip_scaling else args.scale_repetitions
            ),
            "timeout_s": args.timeout_s,
            "interleaved_mode_order": True,
            "profiled": False,
        },
        "groups": {},
    }
    for profile, modes, repetitions in groups:
        for warmup in range(args.warmups):
            for mode in modes:
                _run_worker_process(
                    mode,
                    profile,
                    output_dir / "warmups" / profile / mode / str(warmup),
                    args.timeout_s,
                )
        rows_by_mode = {mode: [] for mode in modes}
        for repetition in range(repetitions):
            ordered_modes = (
                modes if repetition % 2 == 0 else tuple(reversed(modes))
            )
            for mode in ordered_modes:
                row = _run_worker_process(
                    mode,
                    profile,
                    output_dir / "runs" / profile / mode / str(repetition),
                    args.timeout_s,
                )
                row["repetition"] = repetition
                rows_by_mode[mode].append(row)
        result["groups"][profile] = {
            mode: {
                "repetitions": rows,
                "summary": summarize_repetitions(rows),
            }
            for mode, rows in rows_by_mode.items()
        }

    if not args.skip_observed:
        observed = {}
        for mode in ONLINE_MODES:
            row = _run_worker_process(
                mode,
                "small",
                output_dir / "observed" / mode,
                args.timeout_s,
                observed=True,
            )
            baseline = result["groups"]["small"][mode]["summary"]
            if row.get("status") == "success":
                row["overhead_ratio"] = {
                    "subprocess": row["subprocess_wall_time_s"]
                    / baseline["wall_time_s"]["median"],
                    "in_process": row["wall_time_s"]
                    / baseline["in_process_wall_time_s"]["median"],
                }
            observed[mode] = row
        result["observability_smoke"] = observed
    _write_json(output_dir / "benchmark.json", result)
    return result


def _profile_rows(
    profile_file: Path,
    limit: int = 30,
    *,
    sort_by: str = "cumulative",
) -> list[dict]:
    stats = pstats.Stats(str(profile_file))
    value_index = {"self": 2, "cumulative": 3}[sort_by]
    ordered = sorted(
        stats.stats.items(),
        key=lambda item: item[1][value_index],
        reverse=True,
    )[:limit]
    return [
        {
            "file": key[0],
            "line": key[1],
            "function": key[2],
            "primitive_calls": values[0],
            "total_calls": values[1],
            "self_time_s": values[2],
            "cumulative_time_s": values[3],
        }
        for key, values in ordered
    ]


def run_profile(args) -> dict:
    output_dir = Path(args.output_dir or AUDIT_ROOT / _timestamp()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = ONLINE_MODES if args.mode == "all" else (args.mode,)
    result = {
        "schema_version": 1,
        "command": "profile",
        "machine": machine_metadata(),
        "warning": (
            "cProfile adds overhead; profiled durations are not baseline "
            "performance measurements."
        ),
        "modes": {},
    }
    for mode in modes:
        baseline = _run_worker_process(
            mode,
            "small",
            output_dir / "baseline" / mode,
            args.timeout_s,
        )
        profile_file = output_dir / "profiles" / f"{mode}.prof"
        profile_file.parent.mkdir(parents=True, exist_ok=True)
        profiled = _run_worker_process(
            mode,
            "small",
            output_dir / "profiled" / mode,
            args.timeout_s,
            profile_file=profile_file,
        )
        slowdown = None
        if baseline.get("status") == profiled.get("status") == "success":
            slowdown = (
                profiled["wall_time_s"] / baseline["wall_time_s"]
            )
        result["modes"][mode] = {
            "baseline": baseline,
            "profiled": profiled,
            "profile_slowdown": slowdown,
            "top_cumulative": (
                _profile_rows(profile_file) if profile_file.is_file() else []
            ),
            "top_self": (
                _profile_rows(profile_file, sort_by="self")
                if profile_file.is_file()
                else []
            ),
        }
    _write_json(output_dir / "profile.json", result)
    return result


def _add_output_argument(parser):
    parser.add_argument("--output-dir", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark")
    _add_output_argument(benchmark)
    benchmark.add_argument("--warmups", type=int, default=1)
    benchmark.add_argument("--smoke-repetitions", type=int, default=7)
    benchmark.add_argument("--scale-repetitions", type=int, default=3)
    benchmark.add_argument("--timeout-s", type=float, default=900.0)
    benchmark.add_argument("--skip-scaling", action="store_true")
    benchmark.add_argument("--skip-observed", action="store_true")

    profile_parser = subparsers.add_parser("profile")
    _add_output_argument(profile_parser)
    profile_parser.add_argument(
        "--mode", choices=("all",) + ONLINE_MODES, default="all"
    )
    profile_parser.add_argument("--timeout-s", type=float, default=900.0)

    all_parser = subparsers.add_parser("all")
    _add_output_argument(all_parser)
    all_parser.add_argument("--timeout-s", type=float, default=900.0)
    all_parser.add_argument("--skip-scaling", action="store_true")

    worker = subparsers.add_parser("_worker")
    worker.add_argument("--mode", choices=ONLINE_MODES, required=True)
    worker.add_argument("--profile", choices=("small", "three_day"), required=True)
    worker.add_argument("--output-file", type=Path, required=True)
    worker.add_argument("--observed", action="store_true")
    worker.add_argument("--profile-file", type=Path)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "_worker":
        if args.profile_file is None:
            result = execute_worker(
                args.mode,
                args.profile,
                args.output_file.parent,
                observed=args.observed,
            )
        else:
            profiler = cProfile.Profile()
            profiler.enable()
            result = execute_worker(
                args.mode,
                args.profile,
                args.output_file.parent,
                observed=args.observed,
            )
            profiler.disable()
            args.profile_file.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(args.profile_file))
        serialization_started = time.perf_counter()
        json.dumps(result, sort_keys=True)
        result["phase_times_s"]["output_serialization"] = (
            time.perf_counter() - serialization_started
        )
        _write_json(args.output_file, result)
        return
    if args.command == "benchmark":
        result = run_benchmark(args)
    elif args.command == "profile":
        result = run_profile(args)
    else:
        base = Path(args.output_dir or AUDIT_ROOT / _timestamp()).resolve()
        benchmark_args = argparse.Namespace(
            output_dir=base,
            warmups=1,
            smoke_repetitions=7,
            scale_repetitions=3,
            timeout_s=args.timeout_s,
            skip_scaling=args.skip_scaling,
            skip_observed=False,
        )
        profile_args = argparse.Namespace(
            output_dir=base,
            mode="all",
            timeout_s=args.timeout_s,
        )
        result = {
            "benchmark": run_benchmark(benchmark_args),
            "profile": run_profile(profile_args),
        }
        _write_json(base / "all.json", result)
    print(json.dumps({"status": "complete", "command": args.command}))


if __name__ == "__main__":
    main()
