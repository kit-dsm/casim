# Walking vs. Waiting

## Question

How do the wait-k rules, wait-0 with an empty walk, and intervention (admitting new orders to a
running tour) of "Walking vs. Waiting: Performance Impact of Waiting Strategies in Manual
Warehouses" (Barlang, Lehmann and Furmans) perform in CASIM, on a small example and on the RADAR
standard case?

## Data source

- Default: a small deterministic example written out in `scenario.yaml` (six orders, an 8 × 16
  layout, speed 1, 1 s per pick); it is an integration case, not a published instance.
- `variants=radar_standard_case`: the published standard-case orders, storage assignment and
  layout of the RADAR dataset (Barlang, Lehmann and Furmans 2026, doi 10.35097/mwsv59v8sk9sqaan),
  read by the library source `radar`. `python scripts/fetch_data.py radar` restores the data;
  `data/SOURCE.md` lists the files.

## Decisions

| Paper rule | CASIM |
|---|---|
| FCFS batching, four order bins | `FifoBatching` |
| FCFS release, next available picker | `FIFOScheduler`; with one picker the assignment is unambiguous |
| Wait until at least k orders, k = 1 to 4 | one `OrderCountWaiting` card with four configurations; `cosy_repo=wait_1` to `wait_4` |
| Wait-0: leave at once with an empty route and take reachable arrivals | `AislePatrolPositioningNode` adds an empty tour that traverses every aisle; `cosy_repo=paper_patrol variants=explicit_wait0` makes the empty walk observable |
| Centre relocation with an empty cart | `CenterRelocationPositioningNode` moves to a centre node and holds; an unused cart returns at stream closure. This is CASIM's choice for the cart, not a reproduction of Lorenz et al. |
| S-shape, Return, Largest Gap, Nearest Neighbour routing | configured routing algorithms; the `wait_k` repositories use `WalkOrWaitSShape`, the original simulator's S-shape traversal |
| Intervention on arrival: a free bin and every new pick on the remaining route; reroute after admission | a second decision (`decisions/paper_intervention.yaml`, default) with `RemainingRouteAdmission`, triggered by `ActiveTourOpportunity`; it sees only started tours and may insert orders into them |
| Mean order completion time, distance per item, tardiness | `metrics.py`, from the run's output files |

`decisions/paper_no_intervention.yaml` runs the wait-k rules without the admission decision;
`decisions/paper_intervention_center.yaml` combines centre relocation with admission from the
observed position.

Supported limits:

- Wait-0 and intervention are supported for one block, S-shape routing and one picker. The other
  routers can start tours but cannot reroute a running tour from an arbitrary position;
  selecting one with intervention raises an error.
- An arrival during a pick is considered after the pick completes. The original simulator updates
  the batch at arrival and estimates the picker's position on a reversed tour, which can admit an
  order whose location was already passed; CASIM keeps the actual forward position. Exact
  numerical parity with the original simulator is therefore not established.
- The paper's full experiment grid (arrival rates, order sizes, layouts, storage, due dates,
  workload, picker counts) is not configured; only the standard case is.

## How to run

```bash
python -m casim run scenarios/walk_or_wait
python -m casim run scenarios/walk_or_wait variants=radar_standard_case
python -m casim run scenarios/walk_or_wait cosy_repo=wait_3
python -m casim run scenarios/walk_or_wait cosy_repo=paper_patrol variants=explicit_wait0
python -m casim run scenarios/walk_or_wait cosy_repo=paper_center decisions=paper_intervention_center variants=explicit_wait0
python -m casim run scenarios/walk_or_wait decisions=paper_no_intervention cosy_repo=wait_4_return
```

Another standard-case instance:
`variants=radar_standard_case source.instance=radar/selected/Instances/Orders/OFAT/Standard_case/instance_25.json`.

## Expected outputs

The standard outputs. `metrics.py` also gives `empty_tours_started` and
`orders_admitted_after_empty_start`, so the wait-0 behaviour can be checked directly: the patrol
run starts tour 1 at t = 0 with no orders.

## Reference result

The published standard-case values (RADAR `Results_Standard_Case.csv`, 100 instances) for
S-shape routing are, as mean order completion time in seconds with 95 % interval: SI (start
immediately) 144.20 [141.39, 147.00], WOC_1 125.62 [122.62, 128.62], WOC_2 127.79, WOC_3 150.73,
W (full batch) 180.34. The comparison with CASIM is not run yet: it needs the commit of the
original simulator that produced the published values and its speed and pick time, which the
paper does not state (benchmark L5 in `docs/validation.md`).

Recorded regression values (`tests/baseline/walk__*.json`) on the small example:

| Configuration | Makespan (s) | Mean order completion (s) | Tours | Distance |
|---|---:|---:|---:|---:|
| default (wait-1 with intervention) | 167 | 107.2 | 3 | 160 |
| wait-2 | 152 | 98.3 | 2 | 144 |
| wait-3 | 137 | 92.5 | 2 | 128 |
| wait-4 | 147 | 102.5 | 2 | 128 |
| patrol (wait-0, empty walk) | 87 | 54.3 | 2 | 84 |
| centre relocation | 91 | 58.3 | 2 | 88 |

On RADAR standard-case instance 1 with wait-1, CASIM records 1,021 orders in 414 tours, a mean
order completion time of 153.1 s and a distance of 26,024. The variant assumes a speed of 1
length unit per second and 1 s per pick; both are assumptions, not values from the paper.
