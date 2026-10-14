# Literature benchmarks

Register of Section 7.4 of `docs/revision_scopes_corrections.md`, written in Task B.3 on 14 October 2026 at casim `fbc7223` and ware_ops_algos `dd4ffc9`. Every number below was checked against the source named next to it; where a source could not be read, the entry says so and the number is marked unverified. The register's summary in Section 7.4 was second-hand; the differences found are listed first.

Kinds, as defined in Section 7.4: *exact* (same instances, deterministic method, per-instance values; CASIM must reproduce them), *statistical* (published generator and averages; CASIM must reproduce them within sampling error), *bound* (values of a method CASIM does not have; CASIM must not be better than a proven optimum and should not be better than a best-known value), *reproduction* (L5: rerun of the original simulator, then a decision-by-decision comparison).

## Summary

| # | Benchmark | Kind | Instances here | Published values here | CASIM needs | Status |
|---|---|---|---|---|---|---|
| L1 | Seven CASOP sets, static | exact | all, in ware_ops_pipes `data/instances/` (not redistributable) | ware_ops_pipes `data/reference/`; CASOP results `experiments/evaluation/df_results.parquet` | `scenario_static_benchmarks` with the ware_ops_pipes loader (B.4) | ready for B.4; subset proposed |
| L2 | Henn (2012), online FCFS with four selection rules | statistical | generator in the paper; no files | Tables 7.1 to 7.4 of the working paper, transcribed below | seeded generator, Henn layout and storage, selection rules as scheduling cards, Algorithm 4.1 waiting rule, turnover KPIs (B.5) | sources verified; subset proposed |
| L3 | OOBP, Gil-Borrás et al. (2020) | bound | 64 Henn and 64 Albareda, `scripts/fetch_data.py oobp` | `experiments.xlsx` (128 rows), `data/oobp_reference.csv` | archive loader for both formats, weight capacity, depot per warehouse, maximum completion and turnover time (B.5) | sources verified; depot conflict open; all 128 proposed |
| L4 | D'Haen, Braekers and Ramaekers (2023) | bound | OSF online instances (960), `scripts/fetch_data.py dhaen` | OSF results workbook (5,760 rows) | loader, two-block layout (unverified), trucks and due dates as departures, tardiness and pick time | paper not reachable; protocol unverified; subset provisional |
| L5 | Walking vs. Waiting, RADAR standard case | reproduction | RADAR dataset, `scripts/fetch_data.py radar` | RADAR `Results/OFAT/Standard_Case/Results_Standard_Case.csv` | exists for S-shape with intervention; original simulator at the paper's commit | simulator commit unknown; subset proposed |
| L6 | Lorenz, Otto and Gendreau (2025) | bound | none: electronic companion not reachable | Tables 3, 4 in the paper; S5, S6 in the companion | a reoptimization policy and the Pcart/Robot systems | blocked: companion |
| L7 | Gil-Borrás, Pardo, Jiménez and Sörensen (2024), time-window strategies | bound (expected) | probably the OOBP archive (unverified) | paper not read | time-window waiting rules | candidate; paper needed |
| L8 | Gil-Borrás et al. (2021), OOBP with multiple pickers | bound (expected) | unknown | paper not read | multi-picker OOBP protocol | candidate; paper needed |
| L9 | van Gils et al. (2019), static integrated batching, routing and picker scheduling | bound | OSF static `Instances.zip` (not downloaded, 55.7 MB) | OSF `Results ILS.zip`, `Results LNS.zip` (not downloaded) | the van Gils warehouse and due dates; ware_ops_pipes has an IBRSP runner (`experiments/run_ibrsp.py`, not inspected) | candidate; download on approval |

## Corrections to the register of Section 7.4

Each was checked against the source in brackets.

1. L2: the second row labelled `120/45` in the S-shape block of Table 7.1 is `120/75`. The working paper's text says so: "82 min (120/45) and 25 min (120/75)" for FCFS/LONG against C&W(ii)/LONG (p. 21), which is 840 − 758 = 82 for the first row and 627 − 602 = 25 for the second. Table 7.2 and the Largest Gap tables label the row `120/75` correctly. [Henn 2009, pp. 20 to 23]
2. L2: the arrival rate for n = 30 is printed as λ = 0.08625; the stated rule E[X(8 h)] = n gives 30 / 480 min = 0.0625 per minute, and the other three values (0.125, 0.1875, 0.25) follow that rule. 0.08625 is a misprint or a different choice; the generator must state which it uses. [Henn 2009, p. 17]
3. L2: Henn defines the turnover time as "the time period between the completion and the start time of a customer order" (p. 3), while the same sentence calls it the time an order "stays in the system". The plan's KPI `max(completion_time - order_date)` follows the second reading. Which one Tables 7.2 and 7.4 use must be settled before comparison; the arrival-based reading is the usual one and gives the larger values. [Henn 2009, p. 3]
4. L2: the published values come from the 2009 working paper (FEMM WP No. 34, 09034; the OVGU page `article/view/204` is this paper). The journal version, Computers & Operations Research 39(11), 2549-2563, doi 10.1016/j.cor.2011.12.019, was not available here; its tables may differ.
5. L2: Algorithm 4.1 is not a pure batching rule. With a single batch it waits until 2r<sub>i'</sub> + st<sub>i'</sub> − st<sub>j'</sub> unless "the last customer order is known" (p. 8); after the last arrival all batches start one after another. The benchmark therefore needs `HennWaiting` and a stream-closure rule; Henn assumes the arrival of the last order is recognised (closure at the last arrival, CASIM's current `OrderStreamClosed` behaviour). [Henn 2009, pp. 2, 8-9]
6. L3: the OOBP set has 128 instances, 64 Henn and 64 Albareda-Sambola, not 144 with 80 Albareda. The benchmark page says 144 and 80; the archive contains 64 Albareda order files (W1 to W4 × 100, 150, 200, 250 orders × instances 000, 030, 060, 090; the README mentions five order-count folders, the archive has four); the paper uses "64 instances" of each set (Section 4.1, pp. 310-311); `experiments.xlsx` has 128 rows. The paper's Table 2 gives Albareda order counts from 50 to 250; the archive starts at 100.
7. L3: the paper gives the depot of the Henn warehouse (W5) as "Center" (Table 3, p. 311); the benchmark page says bottom left, and the instance header says `DEPOT_AISLE : 0`, `DEPOT_LOCATION : bottom`. This conflict was already recorded in `docs/oobp_assumptions.md`; the protocol in Section 7.4 states bottom left as if settled.
8. L3: the archive's terms: the benchmark page states no licence, only a citation request. Its 64 Henn files had been committed in `6eaf5ee`; they were removed from the tree again, and all benchmark data is now restored by `scripts/fetch_data.py` (Section 3.1).
9. L4: OSF holds 96 instance-level factor combinations × 10 replications = 960 instances, not 72 × 10. The factors in the results workbook are the number of due dates (10, 15, 20), the truck distribution (2 levels), the release-time distribution (4 levels), the uncertainty (0.1, 0.3) and the number of pickers (3, 5): 3 × 2 × 4 × 2 × 2 = 96. Each instance is solved under six settings (staging lanes 20, 40 or 60 %, forecasting off or on), giving 5,760 rows. Batch capacity is 10 in every row; orders per instance range from 546 to about 1,022. The other protocol values of the register (1 m/s, 180 s setup, 10 s per pick, two-block layout with 12 aisles) could not be checked: the paper was not reachable. The distance matrix file is named `DistanceMatrix_12SubAisles`, which fits 6 aisles in two blocks as well as 12 aisles in one; the paper must decide.
10. L5: the paper in `docs/research/walk_or_wait/` does not report standard-case values per wait-k policy; Section 5.2.1 describes them in words. Its numbers are Figure 3 (means over all LHS runs) and a mean absolute percentage error of 2.3 % on 24 Albareda instances (Section 4, p. 8). The per-policy values are in the RADAR dataset's `Results_Standard_Case.csv` (doi 10.35097/mwsv59v8sk9sqaan, CC BY 4.0, Barlang, Lehmann and Furmans 2026), transcribed below. The paper is in `docs/research/`, not `docs/literature/`.
11. L5: neither the paper nor the RADAR readme states the walking speed or the pick time; distances are in abstract length units and times in seconds. CASIM's `radar_standard_case` uses speed 1 and 1 s per pick, taken from the original simulator. The value of MLPI does not depend on them; MOCT and TARD do.
12. L6: arXiv 2409.12619 is the Networks paper (2025, doi 10.1002/net.22281), with 2,600 instances = 2 systems (Pcart, Robot) × 10 settings × 13 order counts (3 to 15) × 10 (Section 4.1). The paper says the instances "can be found in the Electronic Companion"; its data availability statement says the data are "available from the corresponding author upon reasonable request". arXiv 2410.14316 (complete-information counterpart) uses 120 instances, 103 solved to proven optimality for the makespan (Section 5, Table 4).

## Where I searched

Local: `docs/research/` (CASIM, CASOP, Gil-Borrás, Henn, Lorenz, Walk-or-Wait, IJPE, CoAML), ware_ops_pipes `data/` and `experiments/evaluation/`, `project_4D4L/2_Stochastic_Waiting`, `~/Downloads/oobp_instances.zip`. Web: the OOBP benchmark page grafo.etsii.urjc.es/optsicom/oobp.html (lists 13 papers), the OSF API for project 7r4gn, the arXiv API for both Lorenz papers (no ancillary files), the OVGU working-paper page, the RADAR dataset page, three web searches ("online order batching benchmark instances published results", "dynamic order picking batching routing benchmark instances", "van Gils integrated batching routing picker scheduling instances"). Unreachable: the UHasselt document server (HTTP 503, three attempts), Wiley's supporting information (HTTP 403), Taylor & Francis.

Found and not added as benchmarks, because no published values exist for them: a reimplementation of Henn (2012) on GitHub (PhilippXXY/on-line-order-batching), useful as a cross-check of L2 only after review; deep reinforcement learning studies such as arXiv 2408.01656, which generate their own instances and publish none; the Walk-or-Wait paper's 24-instance Albareda reproduction, which reports only an aggregate MAPE. The OOBP page also cites Gil-Borrás et al. (2019, 2020 LNCS; 2020 Progress in Artificial Intelligence) on the same instances; whether those report per-instance values is unknown.

## L1: the seven CASOP sets as static instances

| Field | Value |
|---|---|
| Sources | Heßler and Irnich, SPRP and SPRP-SS, [doi:10.1287/ijoc.2023.0075](https://doi.org/10.1287/ijoc.2023.0075), results at [github.com/katrinhessler/SPRP-SS](https://github.com/katrinhessler/SPRP-SS); Heßler and Irnich, Technical Report LM-2022-03 (BahceciOencan); Wahlen and Gschwind (2023), [doi:10.1287/trsc.2023.1198](https://doi.org/10.1287/trsc.2023.1198) (HennWaescher, MuterOencan); Wahlen (2026), [doi:10.1016/j.ejor.2025.08.055](https://doi.org/10.1016/j.ejor.2025.08.055), Table B.15 (Foodmart); Briant et al., [arXiv:2303.17834](https://arxiv.org/abs/2303.17834) (Kris) |
| Instances | ware_ops_pipes `data/instances/` (all present: SPRP 2,400, SPRP-SS 14,300, BahceciOencan 1,350, HennWaescher 2 × 2,880, MuterOencanWG 270, Foodmart 144, Kris 243 + 237 files) |
| Licence | SPRP-SS: MIT. The others state no redistribution licence (ware_ops_pipes `data/instances/README.md`); they stay in ware_ops_pipes and are not copied into casim |
| Published values | ware_ops_pipes `data/reference/processed/reference_values.csv` (`reference_value`, `reference_is_opt`, `reference_policy`) and Kris tables under `data/reference/raw/briant_et_al/`; with references: SPRP 2,400 (all optimal), SPRP-SS 14,300 (all optimal), BahceciOencan 1,350 (all optimal), HennWaescher 5,759 (5,457 optimal), MuterOencanWG 270 (234), Foodmart 42 (42), Kris 295 (242) |
| CASOP values | `experiments/evaluation/df_results.parquet` (1,114,430 rows; objective and `total_cpu_time` per instance and pipeline) |
| Protocol | as CASOP: one decision at t = 0 with all orders, `CommitAllPolicy`; distance for the routing and batching sets, total picking time (setup, picking, travel) for Kris (Section 7.4) |
| CASIM needs | `scenario_static_benchmarks` with the ware_ops_pipes loader and ware_ops_pipes as development dependency (B.4) |
| Status | sources as recorded by CASOP; the original papers of Wahlen and Gschwind, Wahlen and Heßler and Irnich were not read here, only their machine-readable result files through CASOP |

Pipelines per set, chosen from CASOP's results over all instances with a reference value (`docs/literature_benchmarks/select_l1_subset.py`, run in ware_ops_pipes): the pipeline with the smallest mean gap to the reference, and the best S-shape pipeline.

| Set | Closest pipeline (mean gap, CASOP CPU per instance) | S-shape pipeline (mean gap, CPU) | CASOP below a proven optimum |
|---|---|---|---|
| SPRP | `GIA+SingleOrderBatching+RR` (0.000 %, 0.007 s) | `GIA+SingleOrderBatching+SShape` (33.85 %, 0.0001 s) | 0 |
| SPRP-SS | `RatliffRosenthalNF` (0.000 %, 2.68 s) | `MinMinIA+SingleOrderBatching+SShape` (98.39 %, 0.008 s) | 0 |
| BahceciOencan | `GIA+CombinedBatchingRoutingAssigning` (0.108 %, 151.1 s) | `GIA+LSBatchingSShapeFiFoOrderNr+SShape` (28.83 %, 0.001 s) | 0 |
| HennWaescher | `GIA+LSBatchingRROrderNrFiFo+RR` (4.543 %, 7.66 s) | `GIA+SavingsSShape+SShape` (17.49 %, 0.17 s) | 0 |
| MuterOencanWG | `GIA+LSBatchingRROrderNrFiFo+RR` (6.580 %, 11.77 s) | `GIA+SavingsSShape+SShape` (18.15 %, 0.11 s) | 0 |
| Foodmart | `GIA+LSOrdNrNN+NN` (8.128 %, 0.17 s) | `GIA+LSBatchingSShapeFiFoOrderNr+SShape` (14.48 %, 0.03 s) | 0 |
| Kris | `GIA+DueDate+NN+EDD` (19.76 % on the 264 deadline-feasible instances, 0.001 s) | none in the full-coverage pipelines | 0 |

SPRP and SPRP-SS are the sharpest test: CASOP's exact routers equal the published optima on every instance, so CASIM must equal the published optimum, not only CASOP.

## L2: Henn (2012), online order batching

| Field | Value |
|---|---|
| Citation | Henn, S. (2012), Algorithms for on-line order batching in an order picking warehouse, Computers & Operations Research 39(11), 2549-2563, doi [10.1016/j.cor.2011.12.019](https://doi.org/10.1016/j.cor.2011.12.019). Values from the working paper: Henn, S. (2009), FEMM Working Paper No. 34 (09034), Otto-von-Guericke-Universität Magdeburg, [journals.ub.ovgu.de/index.php/FEMM-WPS/article/view/204](https://journals.ub.ovgu.de/index.php/FEMM-WPS/article/view/204), local copy `docs/research/henn/2009_34.pdf` |
| Instances | none published; generated from the protocol below |
| Licence | not applicable (generator) |
| Status | protocol and values verified against the working paper; journal version not read |

| Parameter | Value | Source (working paper) |
|---|---|---|
| Layout | one block, front and rear cross aisle, depot in front of the leftmost aisle | p. 11, Fig. 5.1; p. 16 |
| Aisles U, cells per side C | 10, 45 (900 locations, one article each) | p. 16, Table 6.1 |
| Cell width l<sub>cw</sub>, aisle centre distance l<sub>c</sub> | 1 LU, 5 LU | Table 6.1 |
| Depot | 1.5 LU from the first location of the leftmost aisle, 0.5 LU from the front cross aisle | p. 16, Table 6.1 |
| Leaving an aisle | one cell width l<sub>cw</sub> between the first or last location and the cross aisle | p. 11 |
| Travel velocity, pick velocity, setup | 48 LU/min, 6 items/min (10 s per article), 3 min per tour | p. 17, Table 6.1 |
| Storage | class-based: A in aisle 1, B in aisles 2 to 4, C in aisles 5 to 10; 52 %, 36 %, 12 % of demand; random within a class | p. 17 |
| Order size | uniform on {5, ..., 25} items | p. 17 |
| Capacity W | 45 or 75 items | p. 17, Table 6.2 |
| Orders n, horizon, arrivals | 30, 60, 90, 120 over 8 h, exponential interarrival times, λ = 0.08625 (see correction 2), 0.125, 0.1875, 0.25 per minute | p. 17 |
| Routing | S-shape (with the last-aisle exception) and Largest Gap | p. 2, Table 6.2 |
| Instances per class | 50; 16 classes, 800 instances | p. 18 |
| Algorithm | Algorithm 4.1: batching heuristic, single-batch waiting rule, selection rule; one picker | pp. 8-11 |
| Selection rules | FIRST (first batch of B(t)), SHORT (smallest service time), LONG (longest), SAV (largest savings, ties by lowest index) | pp. 10-11 |
| Objectives | completion time of the last batch; average turnover time per order (definition: correction 3) | pp. 3, 19 |

Published values that CASIM can run (FCFS batching), in minutes, averages over 50 instances. The C&W(ii) and ILS columns are not transcribed: CASIM has no ILS, and whether CASOP's savings cards implement C&W(ii) is not established.

| n/W | 7.1 S-shape completion FIRST / SHORT / LONG / SAV | 7.2 S-shape turnover | 7.3 Largest Gap completion | 7.4 Largest Gap turnover |
|---|---|---|---|---|
| 30/45 | 467 / 488 / 467 / 467 | 44 / 55 / 44 / 44 | 466 / 485 / 466 / 465 | 43 / 53 / 44 / 43 |
| 30/75 | 457 / 482 / 457 / 457 | 57 / 72 / 57 / 57 | 458 / 482 / 458 / 458 | 58 / 72 / 58 / 58 |
| 60/45 | 519 / 551 / 518 / 518 | 45 / 66 / 45 / 44 | 514 / 539 / 513 / 514 | 43 / 59 / 43 / 41 |
| 60/75 | 499 / 526 / 499 / 499 | 46 / 63 / 46 / 46 | 500 / 523 / 500 / 500 | 47 / 61 / 47 / 47 |
| 90/45 | 651 / 722 / 638 / 643 | 113 / 161 / 106 / 96 | 630 / 690 / 618 / 622 | 102 / 141 / 97 / 88 |
| 90/75 | 520 / 593 / 519 / 518 | 55 / 102 / 54 / 53 | 525 / 586 / 524 / 523 | 58 / 95 / 57 / 56 |
| 120/45 | 856 / 930 / 840 / 847 | 207 / 260 / 195 / 175 | 827 / 893 / 813 / 821 | 192 / 239 / 183 / 163 |
| 120/75 | 634 / 733 / 627 / 630 (printed 120/45) | 101 / 170 / 98 / 93 | 646 / 731 / 641 / 643 | 107 / 163 / 104 / 98 |

The register's example (FCFS, FIRST, S-shape, 30/45: 467 min completion, 44 min turnover) is correct.

CASIM needs: a seeded generator of these instances (layout with Henn's depot and aisle-exit geometry, class-based storage), the four selection rules as scheduling cards, `HennWaiting` combined with them as in Algorithm 4.1, picker speed as an instance value, the turnover KPIs, and a Largest Gap router with Henn's definition. No published standard deviations exist, so the confidence interval comes from CASIM's own replications.

## L3: OOBP, Gil-Borrás et al. (2020)

| Field | Value |
|---|---|
| Citation | Gil-Borrás, S., Pardo, E. G., Alonso-Ayuso, A., Duarte, A. (2020), GRASP with Variable Neighborhood Descent for the online order batching problem, Journal of Global Optimization 78, 295-325, doi [10.1007/s10898-020-00910-2](https://doi.org/10.1007/s10898-020-00910-2); local copy in `docs/research/gil/` |
| Instances | [oobp_instances.zip](https://grafo.etsii.urjc.es/optsicom/oobp/oobp-files/oobp_instances.zip) (SHA-256 `c8c7e109...e20d`), restored by `scripts/fetch_data.py oobp` into `scenarios/scenario_henn_online/data/oobp/` (Henn groups at the top level, Albareda below `albareda/`) |
| Published values | [experiments.xlsx](https://grafo.etsii.urjc.es/optsicom/oobp/oobp-files/experiments.xlsx), converted to `data/oobp_reference.csv` in Phase 3b: per instance GR+VND and best known, maximum completion time and maximum turnover time in seconds; ILS only as aggregates (Tables 11 and 12) |
| Licence | none stated; citation requested |
| Status | protocol verified; depot conflict open (correction 7) |

| Parameter | Value | Source |
|---|---|---|
| Instances | 64 Albareda (W1 to W4), 64 Henn (W5) | Section 4.1, pp. 310-311 |
| Warehouses W1 to W4 | Table 2: random or ABC storage, depot centre or corner, order sizes U(1,7), U(2,10), U(5,25), U(1,36), capacity by weight 12, 24, 150, 80, aisles 4, 10, 25, 12, items per aisle 2×30, 2×20, 2×25, 2×16, aisle length 50, 10, 50, 80 m, aisle centre distance 4.3, 2.4, 5, 15 m | Table 2, p. 311 |
| Warehouse W5 (Henn) | random or ABC, depot "Center", order size U(5,25), item weight 1, capacity 30/45/60/75, 10 aisles of 2×45 items, aisle length 45, centre distance 5 | Table 3, p. 311 |
| Arrival period, distribution | 4 h, exponential | Table 4, p. 312 |
| λ per order count | 40: 0.167, 60: 0.250, 80: 0.333, 100: 0.417, 150: 0.625, 200: 0.833, 250: 1.042 | Table 5, p. 312 |
| Picker | 48 LU/min, 6 items/min, setup 3 min | Table 4 |
| Time window fit, initial orders, random seed | 400 s, 0, 50 | Table 4 |
| Routing, start, batch selection | S-shape; as soon as the picker is available; batch with the oldest order | Table 4 |
| Objectives | maximum completion time; maximum turnover time (arrival to completion) | Section 4.4, p. 318 |

The rule CASIM runs (FCFS batching, oldest-order selection, S-shape, no waiting) is not the published method, so the comparison is a bound: CASIM must not beat a best-known value. CASIM needs a loader for both archive formats (Henn `settNN.txt` with order files and arrival files; Albareda layout, order and arrival files), capacity by weight for W1 to W4, the depot of each warehouse, and the maximum KPIs.

## L4: D'Haen, Braekers and Ramaekers (2023)

| Field | Value |
|---|---|
| Citation | D'Haen, R., Braekers, K., Ramaekers, K. (2023), Integrated scheduling of order picking operations under dynamic order arrivals, International Journal of Production Research 61(10), 3205-3226, doi [10.1080/00207543.2022.2078747](https://doi.org/10.1080/00207543.2022.2078747); author version at documentserver.uhasselt.be (handle 1942/37567), unreachable from here |
| Instances | OSF doi [10.17605/OSF.IO/7R4GN](https://doi.org/10.17605/OSF.IO/7R4GN), online instances restored by `scripts/fetch_data.py dhaen` into `scenarios/scenario_dhaen_online/data/osf/` (see its `SOURCE.md`) |
| Published values | `Results online problem instances.xlsx`: per instance and setting `OrderPickTime`, `Tardiness`, `Makespan`, `WaitingTime`, `Earliness`, `TotalSolutionTime` and the factor levels |
| Licence | CC BY 4.0 |
| Status | protocol unverified (correction 9); paper needed |

Units of the workbook's times are not stated in the OSF README; the distance matrix is in decimetres. The LNS used a real-time budget (mean `TotalSolutionTime` 1,763 s per run), which CASIM does not spend; for a rule without search this does not matter.

## L5: Walking vs. Waiting, RADAR standard case

| Field | Value |
|---|---|
| Citation | Barlang, M., Lehmann, T., Furmans, K., Walking vs. Waiting: Performance Impact of Waiting Strategies in Manual Warehouses (manuscript, `docs/research/walk_or_wait/Walk_or_Wait_orig_with_els_tex.pdf`); dataset: Barlang, Lehmann, Furmans (2026), Benchmark Instances for Manual Warehouse Order Picking, KIT RADAR, doi [10.35097/mwsv59v8sk9sqaan](https://doi.org/10.35097/mwsv59v8sk9sqaan) |
| Instances | restored by `scripts/fetch_data.py radar` into `scenarios/scenario_walk_or_wait/data/radar/selected/` from the RADAR archive (see its `SOURCE.md`) |
| Published values | `selected/Instances/Results/OFAT/Standard_Case/Results_Standard_Case.csv` (SHA-256 `2d6c085867b72f8a9c86416716d268acfb175c693c4e2e487f85a3b7118e6031`): per waiting rule and routing, mean, standard deviation and 95 % interval of MOCT, TARD, MLPI and utilisation, order and batch counts over 100 instances; further OFAT factors and LHS results in the same tree |
| Licence | CC BY 4.0 |
| Original simulator | `project_4D4L/2_Stochastic_Waiting` (gitlab.kit.edu/kit/ifl/forschung/project_4D4L), imported into that repository on 2026-03-03 (`031d6685`); the head `404c2d33` and `bf81a290` are configured for analytic S-shape without waiting and read orders from `U:\Diss\2_Stochastic_Waiting\Orders`. The commit that produced the published CSV is not recorded |
| Status | values verified; simulator commit and speed and pick-time values needed |

| Parameter | Value | Source |
|---|---|---|
| Layout | 8 aisles × 16 locations, cross aisles at bottom and top, depot at (0, 0) | paper Table 2; RADAR readme |
| Batch capacity | 4 orders | paper Table 2 |
| Orders per shift | 1,000 expected; 100,597 over the 100 instances | paper Table 2; CSV |
| Order lines | P(1) = 0.80, P(2) = 0.15, P(3) = 0.05 | paper Table 2 |
| Pickers, storage, arrivals, due date | 1, random uniform, Poisson over 8 h, end of shift | paper Table 2 |
| Waiting rules | SI (immediate start with a dummy route), WOC_1 to WOC_3 (wait for k orders), W (full batch) | paper Section 3.2 |
| Intervention | orders that fit on the remaining route are accepted, the remaining tour is re-optimised | paper Section 3.2 |
| Speed, pick time | not stated (correction 11) | |

Published standard-case values (MOCT and TARD in seconds, MLPI in length units per item; 95 % interval of MOCT in brackets):

| Waiting | S-shape MOCT | NNH MOCT | Return MOCT | Largest Gap MOCT | S-shape MLPI | S-shape TARD |
|---|---|---|---|---|---|---|
| SI | 144.20 [141.39, 147.00] | 98.30 | 194.16 | 187.91 | 22.10 | 0.48 |
| WOC_1 | 125.62 [122.62, 128.62] | 63.91 | 109.98 | 131.49 | 20.58 | 0.41 |
| WOC_2 | 127.79 [125.00, 130.57] | 76.88 | 124.87 | 136.83 | 18.44 | 0.39 |
| WOC_3 | 150.73 [148.12, 153.35] | 94.67 | 146.42 | 152.61 | 17.75 | 0.47 |
| W | 180.34 [177.76, 182.93] | 118.53 | 174.76 | 178.75 | 17.65 | 0.62 |

CASIM supports intervention with S-shape and one picker only (`scenario_walk_or_wait/README.md`), so the CASIM side covers the S-shape column; the original simulator covers all four.

## L6: Lorenz, Otto and Gendreau (2025)

| Field | Value |
|---|---|
| Citation | Lorenz, C., Otto, A., Gendreau, M. (2025), Picking operations in warehouses with dynamically arriving orders: How good is reoptimization?, Networks (2025, issue 2), doi [10.1002/net.22281](https://doi.org/10.1002/net.22281), arXiv 2409.12619; Lorenz, Otto (2024), On picking operations in e-commerce warehouses: insights from the complete-information counterpart, arXiv 2410.14316 |
| Instances | electronic companion (Wiley supporting information, HTTP 403 from here); otherwise "upon reasonable request" |
| Published values | worst observed optimality ratios, Tables 3 and 4; averages, Tables S5 and S6 of the companion |
| Status | blocked until the companion is available |

## Subset proposal and expected run time

Strategy for every benchmark: stratify by the factors that define the set, represent every stratum, fix the seed (20261014), take every instance with a proven optimum or best-known value where the set marks them and the budget allows. Run times were measured on this machine on 14 October 2026; a CASIM run costs a fixed 2.6 s (measured: the explicit walk-or-wait example, three runs, 2.56 to 2.74 s; imports alone 2.0 s), and the estimate uses 4 s per static run to be conservative.

| # | Subset | Size | Basis of the run time | Expected time |
|---|---|---|---|---|
| L1 | one instance per stratum (stratum = instance name without its replication index), drawn among proven optima; all of Foodmart and Kris; two per stratum for MuterOencanWG; two pipelines per set (one for Kris). List: `docs/literature_benchmarks/l1_subset.csv` | SPRP 48, SPRP-SS 286, BahceciOencan 45, HennWaescher 144, MuterOencanWG 54, Foodmart 42, Kris 295 instances; 1,533 runs | CASOP's measured CPU of the same pipelines and instances (2.75 h, of which BahceciOencan's integrated pipeline 2.05 h) plus 4 s per run | CASIM 4.5 h; the CASOP rerun of B.4 another 2.75 h |
| L2 | all 8 classes × 2 routings × 4 selection rules (FCFS), 50 generated instances per class as in the paper, so CASIM's sampling error matches Henn's; a pilot of 10 per class first, whose standard deviations give the interval widths | 3,200 runs | Henn online scenario, 100 orders over 8 h: 8.7 s per run | 7.7 h sequential |
| L3 | all 128 instances, one rule | 128 runs | Henn online, 40 orders: 5.1 s, 100 orders: 8.7 s; Albareda up to 250 orders estimated at 15 s until a loader exists | 0.4 h |
| L4 | provisional: replication 21 of each of the 96 combinations, forecasting off, staging 20 %, earliest-due-date rule; only after the paper confirms the layout | 96 runs | not measurable without a loader; estimated 30 s for about 800 orders and 3 to 5 pickers | 0.8 h |
| L5 | KPI reproduction: all 100 standard-case instances (the published intervals are over 100) × 5 waiting rules, CASIM with S-shape; original simulator × 4 routings. Decision-by-decision comparison: instances 12, 17, 45, 56, 60, 61, 62, 72, 86, 94. List: `docs/literature_benchmarks/l5_instances.csv` | CASIM 500 runs; original 2,000 runs | CASIM `instance_1`, wait-1, S-shape: 27.9 s (1,021 orders, 414 tours); the original simulator could not be timed (commit unknown) | CASIM 3.9 h; original unknown |
| L6 | none until the companion is available | | | |

L3's list is `docs/literature_benchmarks/l3_instances.csv`. Without L6 and the original simulator, the total is about 17 h of CASIM time plus 2.75 h of CASOP time, sequentially; the runs are independent and can run in parallel.
