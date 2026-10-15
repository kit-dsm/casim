# Minimal scenario

The smallest case CASIM runs, and the template for new scenarios: one picker,
one order, every expected value derived by hand. It is verification scenario 1
of Section 7.3 of `docs/revision_scopes_corrections.md`.

## The case

The source is the scenario's own `source.py` (`source: {name: local}`): the
orders are written out in `scenario.yaml`.

A conventional warehouse with 8 aisles of 16 pick locations, the depot in
front of aisle 1, walking speed 1 length unit per second, 1 s per pick, no
setup. One order arrives at t = 0 with one pick at aisle 2, location 3. The
picker arrives at t = 0. One decision (`decisions/obrsp.yaml`) batches FCFS,
routes S-shape and commits the tour at once.

## Expected values (derived by hand)

The walk-or-wait S-shape tour leaves the depot node (-1, -1) for the front
cross aisle at (1, 0), moves to aisle 2 at (2, 0), traverses the aisle to its
rear end (2, 17), picking at (2, 3) on the way, comes back down the same aisle
to (2, 0) and returns over (1, 0) to the end node (-2, -1). Every edge has
length 1: 1 + 1 + 17 + 17 + 1 + 1 = 38 unit edges, so 38 s of walking. With
one pick of 1 s:

| KPI | Value | Derivation |
|---|---:|---|
| `total_distance` | 38 | 38 unit edges |
| `makespan` | 39 | start 0, 38 s walking, 1 s picking |
| `mean_order_completion_time` | 39 | completion 39 minus arrival 0 |

`reference.csv` holds these values; `casim run scenarios/minimal` writes
`comparison.csv` next to the other outputs.

## Running

```bash
python -m casim run scenarios/minimal
```

Outputs go to `scenarios/minimal/outputs/run/`: `kpis.json`, `decisions.json`,
`tours.csv`, `trace.json` (digest of the event trace) and `comparison.csv`.
Add `--audit` to check the plan invariants after every event (`audit.json`).

## Making a new scenario from this one

Copy this directory, then change, in this order: the `source` in
`scenario.yaml` (a library source by name, or `name: local` with your own
`source.py`), the data card, the decisions and the CoSy repository. Put
published values into `reference.csv` with their kind (exact, statistical,
bound, reproduction). Data files never go into git: `data/SOURCE.md` says
where they come from and `data/SHA256SUMS.txt` lists their checksums.
`casim check --all` must pass.
