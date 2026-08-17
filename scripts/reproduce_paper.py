#!/usr/bin/env python3
"""Reproduce the WSC26 paper experiment results and plot.

Usage:
    python scripts/reproduce_paper.py            # uses anonymized data
    python scripts/reproduce_paper.py --real      # uses real data (must be in data/)

NOTE: The repository includes an anonymized version of the grocery retailer
data (data/anonymized/) that preserves the structure and statistical
properties of the original but replaces all identifying article, customer,
order, and picker IDs with sequential integers. Results from the anonymized
data will differ slightly from the paper due to changes in FiFo batching order.

For the original data used in the paper, please contact the authors.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO = "scenarios/scenario_grocery_retailer"
OUTPUT_DIR = REPO_ROOT / SCENARIO / "outputs" / "multirun" / "paper_results"
PLOT_OUTPUT = REPO_ROOT / SCENARIO / "outputs" / "combined_operational_decision_makespan.pdf"

VARIANTS = ["return", "midpoint", "largest_gap", "nearest_neighbour", "sshape", "adaptive_routing"]


def run_sweep(data_dir: str = "data/anonymized"):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scenarios" / "scenario_grocery_retailer" / "experiment_grocery_retailer.py"),
        "--multirun",
        "cosy_repo=return,midpoint,largest_gap,nearest_neighbour,sshape,adaptive_routing",
        "data_card.objective=makespan",
        "scenario.simulation_engine.problems.ORSP.conditions.2.threshold=98",
        f"hydra.sweep.dir={OUTPUT_DIR}",
        "hydra.job.chdir=false",
    ]
    if data_dir:
        cmd.append(f"data_dir={data_dir}")
    env = {"PYTHONPATH": str(REPO_ROOT), "PYTHONIOENCODING": "utf-8"}
    print(f"Running sweep: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, env={**env}, check=True)


def generate_plot():
    script = REPO_ROOT / "scenarios" / "scenario_grocery_retailer" / "scripts" / "reproduce_combined_plot.py"
    cmd = [sys.executable, str(script)]
    env = {"PYTHONPATH": str(REPO_ROOT), "PYTHONIOENCODING": "utf-8"}
    print(f"Generating plot: {cmd}")
    subprocess.run(cmd, cwd=REPO_ROOT, env={**env}, check=True)
    print(f"Plot saved to: {PLOT_OUTPUT}")


def print_kpis():
    import numpy as np
    labels = {
        "sshape": "SShape", "return": "Return", "largest_gap": "Largest Gap",
        "midpoint": "Midpoint", "nearest_neighbour": "Nearest Neighbour",
        "adaptive_routing": "Portfolio",
    }
    print("\n=== Table 2: Routing Strategy KPIs ===")
    print(f"{'Strategy':<20} {'AvgUtil':>7} {'FinalComp':>10} {'AvgMake':>9} {'AvgDock':>8} {'MaxDock':>8}")
    for v in VARIANTS:
        tracker_path = OUTPUT_DIR / v / "makespan" / "kpis" / "tracker.json"
        if not tracker_path.exists():
            print(f"  MISSING: {tracker_path}")
            continue
        t = json.load(open(tracker_path))
        util = t["avg_utilization"]
        dock = t["dock_utilization"]
        completed = t["completed_tours"]
        durations = [e - s for _, s, e, *_ in completed]
        final = max(e for _, _, e, *_ in completed)
        ts = np.array([p[0] for p in dock], dtype=float)
        vs = np.array([p[1] for p in dock], dtype=float)
        dt = np.diff(ts)
        total = ts[-1] - ts[0]
        tw_dock = float(np.sum(vs[:-1] * dt) / total) if total > 0 else 0.0
        print(f"{labels[v]:<20} {util[-1][1]:>7.3f} {final:>10.1f} {np.mean(durations):>9.1f} {tw_dock:>8.2f} {max(d for _, d in dock):>8d}")


if __name__ == "__main__":
    if "--real" in sys.argv:
        run_sweep(data_dir=None)
    else:
        run_sweep(data_dir="scenarios/scenario_grocery_retailer/data/anonymized")
    generate_plot()
    print_kpis()
