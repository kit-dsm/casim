"""Henn's FIRST rule in the recorded L2 runs: each tour is the first FCFS batch.

Henn's FCFS (working paper, Section 4.1; journal, Algorithm 1) assigns the
unprocessed orders in arrival order to batches and opens a new batch when the
next order does not fit; FIRST starts the first batch so formed. In CASIM,
``FifoBatching`` sorts by ``order_date`` and fills batches next-fit, routing
keeps the batch order, ``build_jobs`` numbers jobs by that order,
``FIFOScheduling`` sorts by job id, and with one picker
``SchedulingCommitmentPolicy(n_jobs=1)`` commits the earliest job, job 0.

So, with orders sorted by arrival and tours sorted by start (setup begin):

1. the tours partition the order sequence into consecutive runs (compared as
   sets, since a tour lists its orders in pick sequence), and
2. each tour is maximal: the next order either exceeds the capacity or had
   not arrived when the tour started.
"""

from pathlib import Path
import json

import pytest

from scenarios.scenario_henn2012.generator import generate_instance

BASELINES = Path(__file__).resolve().parents[1] / "baseline" / "benchmarks" / "l2"
FIRST_RUNS = sorted(BASELINES.glob("*__SShape__FIRST__*__conditioned.json"))


def _instance(tmp_path: Path, name: str) -> tuple[list[float], list[int], int]:
    _, storage, orders, capacity, *_rest = name.split("__")
    replication = int(name.split("__")[6])
    relative = generate_instance(tmp_path, storage=storage, n_orders=int(orders),
                                 capacity=int(capacity), replication=replication,
                                 arrival_mode="conditioned")
    path = tmp_path / "HennOOBP" / relative
    arrival_file = path.parent / f"Arrival_Time_{path.stem}.csv"
    arrivals = [float(line.split(";")[1]) / 1000.0
                for line in arrival_file.read_text().split()]
    sizes = [int(line.split()[-1]) for line in path.read_text().splitlines()
             if line.startswith("Order")]
    return arrivals, sizes, int(capacity)


def test_first_runs_are_recorded():
    assert FIRST_RUNS


@pytest.mark.parametrize("path", FIRST_RUNS, ids=lambda path: path.stem)
def test_each_tour_is_the_first_fcfs_batch(path: Path, tmp_path: Path):
    record = json.loads(path.read_text(encoding="utf-8"))
    arrivals, sizes, capacity = _instance(tmp_path, record["name"])
    sequence = sorted(range(len(arrivals)), key=lambda order: (arrivals[order], order))
    tours = sorted(record["committed_tours"], key=lambda tour: tour["start"])

    position = 0
    for tour in tours:
        run = sequence[position:position + len(tour["order_ids"])]
        assert sorted(tour["order_ids"]) == sorted(run), (
            f"tour {tour['tour_id']} is not the next FCFS run")
        position += len(run)
        if position == len(sequence):
            continue
        following = sequence[position]
        load = sum(sizes[order] for order in run)
        assert (load + sizes[following] > capacity
                or arrivals[following] >= tour["start"]), (
            f"tour {tour['tour_id']} leaves out order {following}, which fits and had arrived")
    assert position == len(sequence)
