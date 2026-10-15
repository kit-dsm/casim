"""Run seeded L2 pilots for Henn's working-paper and journal targets.

The original individual instances and random seeds were not published, so
comparison with the papers is statistical. Both turnover definitions are
recorded for every run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pandas as pd

from casim.runner import apply_decision
from casim.events.operational_events import OrderStreamClosed
from scenarios.experiment_commons import load_and_flatten_data_card, setup_experiment
from scenarios.scenario_henn2012.generator import generate_instance
from scenarios.scenario_oobp.experiment_oobp import picker_arrival_hook
from tests.verification.kpi_recomputation import recompute_kpis


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "scenarios" / "scenario_oobp" / "config"
OUTPUT = ROOT / "scenarios" / "scenario_henn2012" / "outputs"
BASELINES = ROOT / "tests" / "baseline" / "benchmarks" / "l2"
ROUTERS = {
    "SShape": "casim.pipelines.subproblems.picker_routing.SShape",
    "LargestGap": "casim.pipelines.subproblems.picker_routing.LargestGap",
}
SCHEDULERS = {
    "FIRST": "casim.pipelines.subproblems.scheduling.FIFOScheduler",
    "SHORT": "casim.pipelines.subproblems.scheduling.SPTScheduler",
    "LONG": "casim.pipelines.subproblems.scheduling.HennLongScheduler",
    "SAV": "casim.pipelines.subproblems.scheduling.SAVScheduler",
    "IGNORE": "casim.pipelines.subproblems.scheduling.FIFOScheduler",
}
WAITING = {
    "journal": "casim.pipelines.subproblems.waiting.NoWaitingNode",
    "working": "casim.pipelines.subproblems.waiting.HennWaitingNode",
}


def run_case(target: str, storage: str, n_orders: int, capacity: int,
             routing: str, selection: str, replication: int,
             arrival_mode: str) -> dict:
    if target == "working" and storage != "C":
        raise ValueError("working paper has only class-based storage")
    if target == "working" and selection == "IGNORE":
        raise ValueError("IGNORE is a journal benchmark only")
    data_root = OUTPUT / "generated"
    relative = generate_instance(
        data_root, storage=storage, n_orders=n_orders, capacity=capacity,
        replication=replication, arrival_mode=arrival_mode)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUTPUT) as folder:
        out = Path(folder)
        with initialize_config_dir(config_dir=str(CONFIG), version_base="1.3"):
            cfg = compose(config_name="oobp_config", overrides=[
                f"data_card.source.filepath='HennOOBP/{relative}'",
                f"instances_base={data_root.as_posix()}",
                f"project_root={ROOT.as_posix()}",
                f"cache_base={(out / 'cache').as_posix()}",
                f"experiment.output_dir={out.as_posix()}",
                "stream_closure=announced",
                "+experiment.audit=true",
            ])
        cfg.cosy_repo.components[4] = ROUTERS[routing]
        cfg.cosy_repo.components[5] = SCHEDULERS[selection]
        cfg.cosy_repo.components[6] = (WAITING["journal"] if selection == "IGNORE"
                                       else WAITING[target])
        if selection == "LONG":
            cfg.decisions.OBRSP.solver._target_ = (
                "scenarios.scenario_henn2012.solver.HennLongSolver")
        if selection == "IGNORE":
            cfg.decisions.OBRSP.state_information.resources = ["idle"]
            cfg.decisions.OBRSP.commitment_policy = OmegaConf.create({
                "_target_": "casim.decision_engine.commitment_policies.CommitAllPolicy",
            })
        sim, engine = setup_experiment(cfg, load_and_flatten_data_card(cfg.data_card))
        horizon_s = 480 * 60

        def orders_hook(simulation, domain):
            for order in domain.orders.orders:
                simulation.add_order(order)
            simulation.add_event(OrderStreamClosed(horizon_s))
        sim.reset(hooks=[orders_hook, picker_arrival_hook])
        while True:
            done, snapshot = sim.run()
            if done:
                break
            apply_decision(sim, engine, engine.on_trigger(snapshot), snapshot)

        audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
        kpis = json.loads((out / "kpis" / "kpis.json").read_text(encoding="utf-8"))
        arrivals = {int(key): value for key, value in audit["order_arrivals"].items()}
        arrival_turnovers = []
        start_turnovers = []
        completion = 0.0
        for _, start, end, order_ids, picker_id, *_ in sim.state.tracker.completed_tours:
            picker = sim.state.resource_manager.get_resource(picker_id)
            setup_start = start - (picker.tour_setup_time or 0.0)
            completion = max(completion, end)
            for order_id in order_ids:
                arrival_turnovers.append(end - arrivals[order_id])
                start_turnovers.append(end - setup_start)
        realized_orders = len(arrivals)
        if len(arrival_turnovers) != realized_orders:
            raise RuntimeError(f"completed {len(arrival_turnovers)} of {realized_orders} orders")

        label = (f"{target}__{storage}__{n_orders}__{capacity}__{routing}__"
                 f"{selection}__{replication}__{arrival_mode}")
        baseline = {
            "name": label,
            "stream_closure": "announced_at_480",
            "command": "python -m scenarios.scenario_henn2012.run_l2 pilot",
            "casim_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "ware_ops_algos_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT.parent / "ware_ops_algos",
                text=True).strip(),
            "kpis": kpis, "decisions": audit["decisions"],
            "committed_tours": audit["committed_tours"],
            "trace_digest": audit["trace_digest"],
            "event_count": audit["event_count"],
            "violations": audit["violations"],
            "first_violations": audit["first_violations"],
            "kpi_differences": recompute_kpis(audit, kpis),
        }
        violations = sum(sum(counts.values()) for counts in audit["violations"].values())
        baseline["status"] = ("inconsistent" if baseline["kpi_differences"] or
                              violations else "consistent")
        BASELINES.mkdir(parents=True, exist_ok=True)
        (BASELINES / f"{label}.json").write_text(json.dumps(baseline, indent=2),
                                                  encoding="utf-8")
        return {
            "target": target, "storage": storage, "n_orders": n_orders,
            "capacity": capacity, "routing": routing, "selection": selection,
            "replication": replication,
            "arrival_mode": arrival_mode,
            "realized_orders": realized_orders,
            "stream_closure": "announced_at_480",
            "completion_s": completion,
            "mean_turnover_arrival_s": sum(arrival_turnovers) / realized_orders,
            "mean_turnover_start_s": sum(start_turnovers) / realized_orders,
            "tours": len(sim.state.tracker.completed_tours),
            "decisions": sum(audit["decisions"].values()),
            "violations": violations,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "full"))
    parser.add_argument("--target", choices=tuple(WAITING), required=True)
    parser.add_argument("--storage", choices=("C", "U"), default="C")
    parser.add_argument("--orders", type=int, choices=(30, 60, 90, 120), required=True)
    parser.add_argument("--capacity", type=int, choices=(45, 75), required=True)
    parser.add_argument("--routing", choices=tuple(ROUTERS), default="SShape")
    parser.add_argument("--selection", choices=tuple(SCHEDULERS), default="FIRST")
    parser.add_argument("--arrival-mode", choices=("truncated", "conditioned"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    label = (f"{args.target}__{args.storage}__{args.orders}__{args.capacity}__"
             f"{args.routing}__{args.selection}__{args.arrival_mode}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{label}__{args.stage}.csv"
    rows = pd.read_csv(path).to_dict("records") if args.resume and path.exists() else []
    count = 10 if args.stage == "pilot" else 50
    for replication in range(len(rows), count):
        started = time.perf_counter()
        row = run_case(args.target, args.storage, args.orders, args.capacity,
                       args.routing, args.selection, replication,
                       arrival_mode=args.arrival_mode)
        row["seconds"] = round(time.perf_counter() - started, 2)
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"{label} {replication}: completion={row['completion_s']:.1f}s "
              f"turnover={row['mean_turnover_arrival_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
