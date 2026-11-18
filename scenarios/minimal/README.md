# Minimal scenario

## Question

Does CASIM give the values derived by hand for the smallest possible case? This scenario is also
the template for new scenarios (`docs/scenarios.md`).

## Data source

None. The scenario's own `source.py` (`source: {name: local}`) builds the case from the values in
`scenario.yaml`: a conventional warehouse with 8 aisles of 16 pick locations, the depot in front
of aisle 1, walking speed 1 length unit per second, 1 s per pick, no setup. One order arrives at
t = 0 with one pick at aisle 2, location 3. The picker arrives at t = 0.

## Decisions

One decision (`decisions/obrsp.yaml`) batches first come first served, routes S-shape
(`cosy_repo/fifo_sshape.yaml`) and commits the tour at once (`CommitAllPolicy`).

## How to run

```bash
python -m casim run scenarios/minimal
python -m casim run scenarios/minimal --audit
```

## Expected outputs

`scenarios/minimal/outputs/run/` with `kpis.json`, `tracker.json`, `decisions.json`,
`service.json`, `trace.json`, `facts/` and `comparison.csv`; with `--audit` also `audit.json`. The run has 89 events and one
tour.

## Reference result

The S-shape tour leaves the depot node (-1, -1) for the front cross aisle at (1, 0), moves to
aisle 2 at (2, 0), traverses the aisle to its rear end (2, 17), picking at (2, 3) on the way,
comes back down to (2, 0) and returns over (1, 0) to the end node (-2, -1). Every edge has length
1: 1 + 1 + 17 + 17 + 1 + 1 = 38 edges, so 38 s of walking, plus one pick of 1 s.

| KPI | Value | Derivation | Kind |
|---|---:|---|---|
| `total_distance` | 38 | 38 unit edges | exact |
| `makespan` | 39 | start 0, 38 s walking, 1 s picking | exact |
| `mean_order_completion_time` | 39 | completion 39 minus arrival 0 | exact |

`reference.csv` holds these values; the run reproduces them (`comparison.csv`).
