# Static CASOP benchmarks (L1)

## Question

Does CASIM, running a CASOP benchmark instance as a static case, give exactly the result of the
same ware_ops_algos algorithms called directly, and do those still give the published CASOP
v0.1.0 results?

## Data source

The seven CASOP benchmark sets (Bahçeci and Öncan, Foodmart, Henn and Wäscher with class-based
and uniform storage, Muter and Öncan, SPRP, SPRP-SS, Kris small and large), read by the library
source `casop` with ware_ops_pipes' own loaders; `source.instance_set` and `source.instance`
select one instance, and a variant per set sets the matching data card. Every order and picker
arrives at t = 0 and the order stream closes at t = 0.

- Instances: `python scripts/fetch_data.py casop_instances` restores the selected public files
  into `data/instances/` and checks their checksums.
- Published CASOP v0.1.0 results: `python scripts/fetch_data.py casop` downloads the release's
  results file into `data/`.
- Run list and reference values: `benchmarks/l1_runlist.csv` and `benchmarks/l1_subset.csv`,
  reproducible with `benchmarks/select_l1_runlist.py` and `select_l1_subset.py`.

The Kris loader of ware_ops_pipes leaves wrong distance-matrix entries in instances with aisles
without products (`docs/validation.md`); Kris results are affected until it is fixed.

## Decisions

One OBRSP decision (`decisions/static.yaml`) solves the whole instance and commits every tour
(`CommitAllPolicy`). One CoSy repository per CASOP pipeline (`cosy_repo/`, named by the v0.1.0
label):

| v0.1.0 label | Item assignment | Batching | Routing | Scheduling |
|---|---|---|---|---|
| `GIA+SingleOrderBatching+RR` | `GreedyIA` | `FiFo` | `RatliffRosenthal` | FIFO |
| `GIA+SingleOrderBatching+SShape` | `GreedyIA` | `FiFo` | `SShape` | FIFO |
| `MinMinIA+SingleOrderBatching+RR` | `MinMinIA` | `FiFo` | `RatliffRosenthal` | FIFO |
| `MinMinIA+SingleOrderBatching+SShape` | `MinMinIA` | `FiFo` | `SShape` | FIFO |
| `GIA+LSBatchingRROrderNrFiFo+RR` | `GreedyIA` | `LSBatchingRROrderNrFiFo` | `RatliffRosenthal` | FIFO |
| `GIA+LSBatchingSShapeFiFoOrderNr+SShape` | `GreedyIA` | `LSBatchingSShapeFiFoOrderNr` | `SShape` | FIFO |
| `GIA+SavingsSShape+SShape` | `GreedyIA` | `ClarkAndWrightSShape` | `SShape` | FIFO |
| `GIA+LSOrdNrNN+NN` | `GreedyIA` | `LSBatchingNNFiFoOrderNr` | `NearestNeighbourhood` | FIFO |
| `GIA+DueDate+NN+EDD` | `GreedyIA` | `DueDate` | `NearestNeighbourhood` | `EDDScheduler` |

`SingleOrderBatching` was a template stage of ware_ops_pipes, not an algorithm; every SPRP and
SPRP-SS instance has one order, so `FiFo` returns the same single batch. The FIFO scheduler and
no waiting only let the picker execute the tours and change no distance. CASOP's integrated
stages (`RatliffRosenthalNF`, `CombinedBatchingRoutingAssigning`) have no CASIM counterpart.

## How to run

```bash
python -m casim run scenarios/casop_static
python -m casim run scenarios/casop_static data_card=kris source.instance_set=KrisSmallDataCorrected source.instance=instances_100_1 "cosy_repo='GIA+DueDate+NN+EDD'"
python -m scripts.l1.run_benchmark smoke
python -m scripts.l1.run_benchmark full [--sets SET ...] [--resume]
```

`scripts/l1/run_benchmark.py` runs the run list and compares, per row, CASIM with the direct
algorithm calls (`scripts/l1/direct.py`), the direct calls with CASOP v0.1.0, and the direct
result with the literature value.

## Expected outputs

The standard outputs per run; the benchmark script writes `outputs/l1_<stage>.csv` with the three
comparisons per instance and pipeline.

## Reference result

On all 415 rows of the run list, the direct calls equal CASOP v0.1.0. CASIM equals the direct
calls on 310 rows; the other 105 are Kris rows that the loader defect explains. No result lies
below a proven optimum. Details in `docs/validation.md` (L1).
