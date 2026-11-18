# OOBP: online order batching benchmark (L3)

## Question

Does CASIM, running a simple FCFS rule on the instances of the online order batching benchmark of
Gil-Borrás et al. (2020), stay above the published best-known values, as a simpler method must?

## Data source

The 128 instances of the OOBP archive (grafo.etsii.urjc.es/optsicom/oobp.html): 64 Henn instances
of warehouse W5 and 64 Albareda-Sambola instances of warehouses W1 to W4, read by the library
source `oobp`, one file named by `source.filepath`. `python scripts/fetch_data.py oobp` restores
them. For the Henn instances, `source.depot` is `corner` (bottom left, as on the benchmark page)
or `centre` (as in Table 3 of the paper); the sources conflict, so both are run. The Albareda
instances take the depot from their layout file. The archive numbers storage locations from 0;
CASIM places location l in cell l + 1, as the protocol's cell distances require. The stream
closes at the last arrival.

## Decisions

One OBRSP decision (`decisions/oobp.yaml`, repository `cosy_repo/oobp_baseline.yaml`): FCFS
batching, selection of the batch with the oldest order, S-shape routing, no waiting, under the
protocol of Table 4 (48 length units per minute, 6 items per minute, 3 min setup, capacity by
weight).

## How to run

```bash
python scripts/fetch_data.py oobp
python -m casim run scenarios/oobp
python -m casim run scenarios/oobp +sweep=l3_henn
python -m casim run scenarios/oobp variants=albareda +sweep=l3_albareda --output-dir scenarios/oobp/outputs/albareda
python scenarios/oobp/plots.py scenarios/oobp/outputs/run scenarios/oobp/outputs/albareda
```

## Expected outputs

The standard outputs per run. `metrics.py` gives the maximum completion and turnover times;
`plots.py` builds the table against `reference.csv`, which holds the published best-known and
GRASP+VND values per instance (from the benchmark's `experiments.xlsx`, converted by
`scripts/oobp_reference.py`; the run list is `benchmarks/l3_instances.csv`).

## Reference result

The comparison has kind bound. Of 192 runs, 40 lie below at least one best-known value (30 in
completion time, 11 in turnover time, one in both), so the protocol is not reproduced for them;
no documented reading of the protocol removes these gaps. Details and the per-group table in
`docs/validation.md` (L3); per-instance results in `benchmarks/l3_full_results.csv`.

## Source

S. Gil-Borrás, E. G. Pardo, A. Alonso-Ayuso, A. Duarte (2020), GRASP with variable neighborhood
descent for the online order batching problem, Journal of Global Optimization 78, 295-325.
