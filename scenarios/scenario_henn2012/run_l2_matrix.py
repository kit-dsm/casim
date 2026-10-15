"""Run the L2 matrix under the pilot-selected arrival reading."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from itertools import product
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

from scenarios.scenario_henn2012.run_l2 import OUTPUT, ROOT, run_case


def cells(arrival_mode):
    for target in ("journal", "working"):
        storages = ("C", "U") if target == "journal" else ("C",)
        selections = ("FIRST", "SHORT", "LONG", "SAV", "IGNORE") if target == "journal" else ("FIRST", "SHORT", "LONG", "SAV")
        for storage, orders, capacity, routing, selection in product(
            storages, (30, 60, 90, 120), (45, 75), ("SShape", "LargestGap"), selections
        ):
            yield target, storage, orders, capacity, routing, selection, arrival_mode


def cell_name(cell):
    target, storage, orders, capacity, routing, selection, mode = cell
    return f"{target}__{storage}__{orders}__{capacity}__{routing}__{selection}__{mode}"


def run_cell(cell, stage):
    label = cell_name(cell)
    path = OUTPUT / f"{label}__{stage}.csv"
    prior = pd.read_csv(path).to_dict("records") if path.exists() else []
    count = 10 if stage == "pilot" else 50
    if stage == "pilot" and not prior:
        full_path = OUTPUT / f"{label}__full.csv"
        if full_path.exists():
            prior = pd.read_csv(full_path).head(10).to_dict("records")
    if stage == "full" and not prior:
        pilot_path = OUTPUT / f"{label}__pilot.csv"
        if pilot_path.exists():
            prior = pd.read_csv(pilot_path).to_dict("records")
    if len(prior) > count:
        prior = prior[:count]
    if len(prior) == count:
        if not path.exists():
            pd.DataFrame(prior).to_csv(path, index=False)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(Path(__file__).with_name("l2_run_stdout.log"), "a", encoding="utf-8") as detail:
        for replication in range(len(prior), count):
            started = time.perf_counter()
            with redirect_stdout(detail):
                row = run_case(*cell[:6], replication, arrival_mode=cell[6])
            row["seconds"] = round(time.perf_counter() - started, 2)
            prior.append(row)
            pd.DataFrame(prior).to_csv(path, index=False)


def worker(stage, shard, shards, arrival_mode):
    selected = [cell for index, cell in enumerate(cells(arrival_mode))
                if index % shards == shard]
    for index, cell in enumerate(selected, 1):
        run_cell(cell, stage)
        print(f"{stage} shard {shard}: {index}/{len(selected)} {cell_name(cell)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--stage", choices=("pilot", "full", "both"), default="both")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--arrival-mode", choices=("truncated", "conditioned"), required=True)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.stage, args.worker, args.workers, args.arrival_mode)
        return
    if args.workers < 1 or args.workers > 8:
        parser.error("workers must be between 1 and 8")
    if len(list(cells(args.arrival_mode))) != 224:
        raise AssertionError("L2 matrix must have 224 cells")
    for stage in (("pilot", "full") if args.stage == "both" else (args.stage,)):
        processes = [subprocess.Popen(
            [sys.executable, "-m", "scenarios.scenario_henn2012.run_l2_matrix",
             "--stage", stage, "--workers", str(args.workers),
             "--worker", str(shard), "--arrival-mode", args.arrival_mode],
            cwd=ROOT, stdout=(OUTPUT / f"l2_{stage}_shard{shard}.log").open("a", encoding="utf-8"),
            stderr=(OUTPUT / f"l2_{stage}_shard{shard}.err").open("a", encoding="utf-8"),
        ) for shard in range(args.workers)]
        failures = [p.wait() for p in processes]
        if any(failures):
            raise RuntimeError(f"{stage} workers failed: {failures}")


if __name__ == "__main__":
    main()
