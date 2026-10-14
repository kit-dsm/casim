# Static benchmarks (L1, Task B.4)

A CASOP benchmark instance as a static CASIM instance: every order and every
picker arrives at t = 0, the order stream closes at t = 0, one OBRSP decision
with `CommitAllPolicy` solves the whole instance, and the pickers execute the
committed tours. The run answers three questions, each with one possible cause
of a difference:

1. Simulator: CASIM against the same ware_ops_algos algorithms called directly
   (`direct.py`, built with the arguments of casim's components), objective
   from the routes and the time model. A difference is CASIM's.
2. Algorithm drift: the direct result against CASOP's published v0.1.0 result
   (`casop-results-v0.1.0.parquet`, ware_ops_pipes `f04527ba` with
   ware_ops_algos `042dcff`). A difference is a change on casim's
   ware_ops_algos branch.
3. Literature: the direct result against the reference value of
   `docs/literature_benchmarks/l1_subset.csv`.

## Data

- Instances: `python scripts/fetch_data.py casop_instances` restores the selected
  public files into `data/instances/` and verifies their committed checksums.
  Loaded with ware_ops_pipes' own loaders (`loader.py`), ware_ops_pipes
  `554a153` as a development dependency of casim.
- CASOP v0.1.0 results: `python scripts/fetch_data.py casop` downloads the
  release asset into `data/` and checks it against `data/SHA256SUMS.txt`.

## Pipelines

`pipelines.py` maps each CASOP v0.1.0 label (the names
`experiments/evaluation/01_prepare_pipeline_results.py` gives in that release)
to today's classes and casim's components:

| v0.1.0 label | Item assignment | Batching | Routing | Scheduling |
|---|---|---|---|---|
| `GIA+SingleOrderBatching+RR` | `GreedyIA` | `FiFo` (one order per SPRP instance) | `RatliffRosenthal` | FIFO |
| `GIA+SingleOrderBatching+SShape` | `GreedyIA` | `FiFo` | `SShape` | FIFO |
| `MinMinIA+SingleOrderBatching+RR` | `MinMinIA` | `FiFo` | `RatliffRosenthal` | FIFO |
| `MinMinIA+SingleOrderBatching+SShape` | `MinMinIA` | `FiFo` | `SShape` | FIFO |
| `GIA+LSBatchingRROrderNrFiFo+RR` | `GreedyIA` | `LSBatchingRROrderNrFiFo` | `RatliffRosenthal` | FIFO |
| `GIA+LSBatchingSShapeFiFoOrderNr+SShape` | `GreedyIA` | `LSBatchingSShapeFiFoOrderNr` | `SShape` | FIFO |
| `GIA+SavingsSShape+SShape` | `GreedyIA` | `ClarkAndWrightSShape` | `SShape` | FIFO |
| `GIA+LSOrdNrNN+NN` | `GreedyIA` | `LSBatchingNNFiFoOrderNr` | `NearestNeighbourhood` | FIFO |
| `GIA+DueDate+NN+EDD` | `GreedyIA` | `DueDate` | `NearestNeighbourhood` | `EDDScheduler` |

`SingleOrderBatching` was a ware_ops_pipes template stage, not an algorithm
card; every SPRP and SPRP-SS instance has one order, which the loader gives one
box, so `FiFo` returns the same single batch. The FIFO scheduler and no waiting
only let the picker execute the tours; they change no distance.
`RatliffRosenthalNF` and `CombinedBatchingRoutingAssigning` need integrated
stages that CASIM does not have. `RatliffRosenthalRouting` has no node
sequence yet, so its pipelines are solved directly but not executed in CASIM.

## Running

    python -m scenarios.scenario_static_benchmarks.run_benchmark smoke
    python -m scenarios.scenario_static_benchmarks.run_benchmark full [--sets SET ...]

The full run reads the committed `docs/literature_benchmarks/l1_runlist.csv`,
reproducible from `docs/literature_benchmarks/select_l1_runlist.py`.
Pass `--resume` after an interrupted run to continue from its existing CSV;
the prefix of instance and pipeline keys is checked against the run list.

The result table is written to `outputs/l1_<stage>.csv` (not committed).
