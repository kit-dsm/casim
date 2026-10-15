# OOBP: the online order batching benchmark of Gil-Borrás et al. (2020)

Literature benchmark L3 of `docs/literature_benchmarks.md`: the 128 instances
of the OOBP archive (64 Henn instances of warehouse W5, 64 Albareda-Sambola
instances of W1 to W4), run with FCFS batching, oldest-order batch selection,
S-shape routing and no waiting under the protocol of Table 4 (48 length units
per minute, 6 items per minute, 3 min setup). The comparison has kind
**bound**: CASIM runs a rule, the paper a GRASP+VND search.

## Source, decisions, runs

- Source: `oobp` (`casim.sources.oobp`), one archive file named by
  `source.filepath`; `source.depot` is `corner` (bottom left, the benchmark
  page) or `centre` (Table 3 of the paper) for the Henn instances, the layout
  file's depot for Albareda. The stream closes at the last arrival.
- Decisions: `decisions/oobp.yaml`, repository `cosy_repo/oobp_baseline.yaml`.
- Runs: `sweep/l3_henn.yaml` (64 instances × 2 depots) and
  `sweep/l3_albareda.yaml` (64 instances, with `variants=albareda`).
  `reference.csv` holds the published best-known and GRASP+VND values per
  instance (from `experiments.xlsx`, `scripts/oobp_reference.py`);
  `metrics.py` gives the maximum completion and turnover times;
  `plots.py` builds the L3 table from the sweeps' outputs.

## Running

```bash
python scripts/fetch_data.py oobp
python -m casim run scenarios/oobp
python -m casim run scenarios/oobp +sweep=l3_henn
python -m casim run scenarios/oobp variants=albareda +sweep=l3_albareda --output-dir scenarios/oobp/outputs/albareda
python scenarios/oobp/plots.py scenarios/oobp/outputs/run scenarios/oobp/outputs/albareda
```

The recorded runs are `tests/baseline/benchmarks/l3/`;
`python scripts/baseline.py compare-benchmark --benchmark l3` reruns and
compares them.

## Source

S. Gil-Borrás, E. G. Pardo, A. Alonso-Ayuso, A. Duarte (2020), GRASP with
variable neighborhood descent for the online order batching problem, Journal of
Global Optimization 78, 295-325. Instances: grafo.etsii.urjc.es/optsicom/oobp.html.
