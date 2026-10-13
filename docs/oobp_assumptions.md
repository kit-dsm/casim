# OOBP reference feasibility

Sources: [Gil-Borrás et al. (2020), reproduced in the author's thesis, pp. 304-318](https://grafo.etsii.urjc.es/phd/PhD_Thesis_SergioGil.pdf), [official OOBP benchmark page](https://grafo.etsii.urjc.es/optsicom/oobp.html), its `experiments.xlsx`, and its `oobp_instances.zip`. The downloaded workbook contains 128 rows: 64 Albareda and 64 Henn. Its only result groups are “Gil et al. 2020” and “Best Known”, each with maximum completion and maximum turnover time in seconds. The paper gives ILS only as aggregate deviations and best counts in Tables 11 and 12. The CSV therefore has no ILS field.

The analysis below refers to the present Henn configuration and loader. “Mapped” means the relevant input or operation exists; it does not assert that the present simulator reproduces the paper's numerical results. Phase 3b runs no benchmark simulation.

| Paper assumption | Current mapping | Verdict |
| --- | --- | --- |
| Orders arrive over 4 h (Table 4, §4.2) | `data/HennOnline/Arrival_Time_abc1_*.csv` records arrivals in milliseconds over `[0, 14400000]`; `henn_online_loader.py` converts them to seconds. | Mapped |
| Exponential interarrival distribution (Table 4, §4.2) | The supplied arrival CSVs fix the realized arrival times. The Henn loader reads those times instead of resampling. | Mapped for replay; distribution generation is not reproducible from this configuration |
| Initial orders in queue: 0 (Table 4) | The chosen arrival CSV begins after time zero; orders enter through `OrderArrival`. | Mapped |
| Random seed: 50 (Table 4) | A fixed arrival CSV is replayed; the Henn configuration does not reconstruct the paper's stochastic generator from seed 50. | Not reproducible as a generator |
| Picker speed: 48 length units/min = 0.8 length units/s (Table 4) | `henn_online_loader.py` sets `Resource.speed=1` length unit/s. This is hardcoded, not a configuration value. | Needs code |
| Pick rate: 6 items/min = 10 s/item (Table 4) | `henn_online_loader.py` sets `time_per_pick=10`. | Mapped |
| Tour setup: 3 min = 180 s (Table 4) | The loader sets `tour_setup_time=180`. The current execution path counts setup twice; Task 4.8 fixes that. | Needs code |
| Time-window fit: 400 s (Table 4) | No matching 400 s GRASP fit parameter appears in the configured FiFo/SPT/HennWaiting pipeline. | Needs code |
| S-shape routing (Table 4, §3.3) | `config/cosy_repo/baseline_waiting_henn.yaml` selects `SShape`, backed by `ware_ops_algos.algorithms.routing.SShapeRouting`. | Mapped |
| Picker starts when available (Table 4) | Scheduling uses picker availability, but `HennWaiting` may delay release and the current setup timing differs. A `NoWaiting` component can be selected in configuration. | Mappable with a configuration change, subject to Task 4.8 |
| Select batch containing oldest order (Table 4, §3.1) | The baseline selects `SPTScheduler` by processing time. `FIFOScheduler` sorts by job id, not the minimum `order_date` in each batch. | Needs code |
| Maximum completion time (paper objective) | The Henn data card requests `makespan`; `KPILogger` records the latest completed tour time. Current setup timing still affects its value. | Mapped, with Task 4.8 outstanding |
| Maximum order turnover time (paper objective) | `KPILogger` computes mean completion delay, not the maximum of `completion_time - order_date`; the Henn data card uses `makespan`. | Needs code |
| Real-time solving between picker departures and returns (§4) | CASIM measures wall-clock solve time but advances no simulated time for it. Paper GRASP+VND and ILS use real-time budgets. | Needs code |
| Depot position (Table 3 and benchmark page) | Paper Table 3 says “Center” for W5; the benchmark page says Henn's depot is at the bottom left; the local instance header says `DEPOT_AISLE : 0`, `DEPOT_LOCATION : bottom`, which the loader maps to aisle 1. These sources conflict. | Not reproducible as a single agreed protocol until the source conflict is resolved |
| Capacity unit (Table 3) | Paper W5 has item weight 1 and capacity 30/45/60/75 by weight; local headers have `PICKER_CAPACITY`, and the loader uses `DimensionType.WEIGHT`. | Mapped |

## Instance provenance

`oobp_reference.py` compared all 32 local Henn `abc1` files against the official archive. All **16 arrival CSVs are byte-identical**. All **16 instance text files differ**. For example, the archive's `29s-40-30-0.txt` begins with `Order 0` and is 15,010 bytes, while the local file begins with a `NAME`/layout/article header and is 30,881 bytes. The local files appear to be a converted representation; this byte check does not establish semantic equivalence of their orders or geometry. The archive also has inconsistent punctuation in several arrival filenames, which the comparison script normalizes for matching, without altering the bytes.

SHA-256 of the inspected downloads: `experiments.xlsx` `E68995050E245BDAF8C0A733D370597CE8484FED3ED24BC5F3C4566BD1D64435`; `oobp_instances.zip` `C8C7E1096EF515D373C770A705AD8AEDA975D5874CAFCEFAB2FDB4DCEA29E20D`.

## What can be compared

The committed CSV holds the paper's Gil et al. 2020 and best-known values for both objectives. A valid instance-level comparison first needs the same instance semantics, timing, objective, depot interpretation, and online compute-time protocol. For a minimization objective, a deterministic rule under that protocol should not score below a verified best-known value; a lower score would warrant investigation of the model or source data. Exact equality with Gil et al. 2020 is not expected from a different algorithm or a solver that consumes zero simulated time. ILS has no per-instance column in the workbook; the paper's Tables 11 and 12 supply only aggregate comparisons.

Available ware_ops_algos building blocks include `FiFoBatching`, `LocalSearchBatching` and `SShapeRouting`; the baseline pipeline uses `FiFoBatching`, `SPTScheduler`, and `SShapeRouting`. `FIFOScheduler` sorts jobs by id, so it is not the paper's oldest-order batch selector. There is no algorithm explicitly implementing the paper's GRASP+VND batching or Henn's ILS in ware_ops_algos; `LocalSearchBatching` is only a related search framework. A like-for-like comparison would need those methods or a clearly specified deterministic substitute, the oldest-order selector, and a real-time solution budget. Task 6.0 performs benchmark runs after the verification gate.
