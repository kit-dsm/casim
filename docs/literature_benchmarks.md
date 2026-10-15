# Literature benchmarks

Register of Section 7.4 of `docs/revision_scopes_corrections.md`, written in Task B.3 on 14 October 2026 at casim `fbc7223` and ware_ops_algos `dd4ffc9`. Every number below was checked against the source named next to it; where a source could not be read, the entry says so and the number is marked unverified. The register's summary in Section 7.4 was second-hand; the differences found are listed first.

Kinds, as defined in Section 7.4: *exact* (same instances, deterministic method, per-instance values; CASIM must reproduce them), *statistical* (published generator and averages; CASIM must reproduce them within sampling error), *bound* (values of a method CASIM does not have; CASIM must not be better than a proven optimum and should not be better than a best-known value), *reproduction* (L5: rerun of the original simulator, then a decision-by-decision comparison).

## Summary

| # | Benchmark | Kind | Instances here | Published values here | CASIM needs | Status |
|---|---|---|---|---|---|---|
| L1 | Seven CASOP sets, static | exact | all, in ware_ops_pipes `data/instances/` (not redistributable) | ware_ops_pipes `data/reference/`; CASOP results `experiments/evaluation/df_results.parquet` | `scenario_static_benchmarks` with the ware_ops_pipes loader (B.4) | ready for B.4; subset proposed |
| L2 | Henn (2012), online FCFS with four selection rules | statistical | generator in the papers; no files | journal Tables 2 to 7 (α = 0, class-based and uniform storage) and working-paper Tables 7.1 to 7.4 (α = 1, class-based), transcribed below | seeded generator, Henn layout and both storage policies, selection rules as scheduling cards, the waiting rule with α, IGNORE, `announced` stream closure, both turnover definitions (B.5) | sources verified; subset approved |
| L3 | OOBP, Gil-Borrás et al. (2020) | bound | 64 Henn and 64 Albareda, `scripts/fetch_data.py oobp` | `experiments.xlsx` (128 rows), `data/oobp_reference.csv` | archive loader for both formats, weight capacity, depot per warehouse, maximum completion and turnover time (B.5) | sources verified; Henn instances run with both depots; subset approved |
| L4 | D'Haen, Braekers and Ramaekers (2023) | bound | OSF online instances (960), `scripts/fetch_data.py dhaen` | OSF results workbook (5,760 rows) | loader, two-block layout, trucks and due dates as departures, tardiness and pick time | protocol verified (correction 9); subset approved |
| L5 | Walking vs. Waiting, RADAR standard case | reproduction | RADAR dataset, `scripts/fetch_data.py radar` | RADAR `Results/OFAT/Standard_Case/Results_Standard_Case.csv` | exists for S-shape with intervention; the 4D4L commit and parameters that reproduce the CSV | simulator commit to be established; subset approved; run last |
| L6 | Lorenz, Otto and Gendreau (2025) | bound | none: electronic companion not reachable | Tables 3, 4 in the paper; S5, S6 in the companion | a reoptimization policy and the Pcart/Robot systems | not pursued for now: companion not available |
| L7 | Gil-Borrás, Pardo, Jiménez and Sörensen (2024), time-window strategies | bound (expected) | probably the OOBP archive (unverified) | paper not read | time-window waiting rules | not pursued for now: paper not available |
| L8 | Gil-Borrás et al. (2021), OOBP with multiple pickers | bound (expected) | unknown | paper not read | multi-picker OOBP protocol | not pursued for now: paper not available |
| L9 | van Gils et al. (2019), static integrated batching, routing and picker scheduling | bound | OSF 7r4gn static `Instances.zip` (55.7 MB, not fetched) | OSF `Results ILS.zip`, `Results LNS.zip` (not fetched); aggregates in D'Haen et al. (2023), Table 3 | the van Gils warehouse and due dates; ware_ops_pipes has an IBRSP runner (`experiments/run_ibrsp.py`, not inspected) | not pursued for now |
| L10 | D'Haen, Ramaekers, Archetti and Braekers (2024), integrating order picking and vehicle routing in a dynamic e-commerce setting, Computers & Operations Research 170, 106762, doi 10.1016/j.cor.2024.106762 | bound | OSF [dkgjq](https://osf.io/dkgjq/) (not fetched) | in the paper (not read in detail) | vehicle routing after picking | not pursued: it combines picking with vehicle routing, which CASIM does not model |

## Corrections to the register of Section 7.4

Each was checked against the source in brackets.

1. L2: the second row labelled `120/45` in the S-shape block of Table 7.1 is `120/75`. The working paper's text says so: "82 min (120/45) and 25 min (120/75)" for FCFS/LONG against C&W(ii)/LONG (p. 21), which is 840 − 758 = 82 for the first row and 627 − 602 = 25 for the second. Table 7.2 and the Largest Gap tables label the row `120/75` correctly. [Henn 2009, pp. 20 to 23]
2. L2: the arrival rate for n = 30 is printed as λ = 0.08625; the stated rule E[X(8 h)] = n gives 30 / 480 min = 0.0625 per minute, and the other three values (0.125, 0.1875, 0.25) follow that rule. 0.08625 is a misprint or a different choice; the generator must state which it uses. [Henn 2009, p. 17]
3. L2: the working paper defines the turnover time as "the time period between the completion and the start time of a customer order" (p. 3) while calling it the time an order "stays in the system"; the journal defines it as "the time period between the arrival and the completion time of a customer order" (p. 2550). Both definitions are computed, and the report says which one each table matches. [Henn 2009, p. 3; Henn 2012, p. 2550]
4. L2: the journal version (Computers & Operations Research 39(11), 2549-2563, doi 10.1016/j.cor.2011.12.019, in `docs/research/henn/`) is not the working paper in a new layout. Its algorithm has a waiting parameter α ∈ [0, 1] (p. 2553): a single batch j starts at max{t, (1 + α) r<sub>i'</sub> + α st<sub>i'</sub> − st<sub>j</sub>}; α = 1 is the working paper's rule 2r<sub>i'</sub> + st<sub>i'</sub> − st<sub>j</sub>, and α = 0 starts the batch at once. The configuration test (Table 1, p. 2558) finds α = 0 best, and all of Tables 2 to 7 use α = 0. The journal adds a uniform storage policy (classes C and U: 32 classes, 1,600 instances, p. 2557), the benchmark strategy IGNORE (with FCFS, C&W(ii) or ILS) and a seed algorithm, and reports turnover only for FCFS/LONG, C&W(ii)/LONG, ILS/SAV, IGNORE and Seed (Tables 4 and 7). Its values differ from the working paper's (C/30/45, FCFS, FIRST, S-shape: 473 min against 467; FCFS/LONG turnover 16 min against 44). The two sources are therefore two targets: the journal with α = 0, the working paper with α = 1. The journal also prints λ = 0.08625 for n = 30 (p. 2557). The working paper is FEMM WP No. 34 (09034), the OVGU page `article/view/204`.
5. L2: Algorithm 4.1 (working paper) and Algorithm 1 (journal) are not pure batching rules. With a single batch they wait as in correction 4 unless "the last customer order is known" (working paper, p. 8; journal decision point of type C, p. 2553); when the last order arrives, all batches are started one after another. The benchmark therefore needs the waiting rule with α and an `announced` stream closure: the algorithm knows at the last arrival that it is the last. [Henn 2009, pp. 2, 8-9; Henn 2012, pp. 2552-2553]
6. L3: the OOBP set has 128 instances, 64 Henn and 64 Albareda-Sambola, not 144 with 80 Albareda. The benchmark page says 144 and 80; the archive contains 64 Albareda order files (W1 to W4 × 100, 150, 200, 250 orders × instances 000, 030, 060, 090; the README mentions five order-count folders, the archive has four); the paper uses "64 instances" of each set (Section 4.1, pp. 310-311); `experiments.xlsx` has 128 rows. The paper's Table 2 gives Albareda order counts from 50 to 250; the archive starts at 100.
7. L3: the paper gives the depot of the Henn warehouse (W5) as "Center" (Table 3, p. 311); the benchmark page says bottom left, and the instance header says `DEPOT_AISLE : 0`, `DEPOT_LOCATION : bottom`. This conflict was already recorded in `docs/oobp_assumptions.md`; the protocol in Section 7.4 states bottom left as if settled.
8. L3: the archive's terms: the benchmark page states no licence, only a citation request. Its 64 Henn files had been committed in `6eaf5ee`; they were removed from the tree again, and all benchmark data is now restored by `scripts/fetch_data.py` (Section 3.1).
9. L4: the paper's design has 2 × 3 × 2 × 3 × 2 = 72 factor combinations (average orders 600 or 1,000; 10, 15 or 20 trucks; uniform or peak truck distribution; uniform, progressive or degressive order arrivals; variation 10 % or 30 %), ten instances each (Section 5.2, Table 5). OSF holds 96 combinations × 10 = 960 instances because the workbook's release-time factor has a fourth level, the "uniform+" pattern of Section 5.4.1; 72 of the 96 are the main design. Each instance is solved six times (20, 40 or 60 % of the trucks active, with and without anticipation; 5,760 rows). Table 4 confirms the register's other values: two blocks, 12 aisles (24 sub-aisles), 240 locations per aisle, across-aisle storage with classes of 1/6, 1/3 and 1/2 of the SKUs and 60, 30 and 10 % of the picks, aisle width 3.0 m, cross aisle 6.0 m, 1 m/s, 180 s setup, 10 s per pick, 10 orders per batch, one picker per 200 orders, 8 h; order lines exponential with mean 3.5, rounded up. Times are in seconds (Table 7). [D'Haen et al. 2023, pp. 3213-3215]
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

Journal values (Henn 2012, α = 0), FCFS and IGNORE, in minutes. Completion: Tables 2 and 3 (S-shape), 5 and 6 (Largest Gap); turnover: Tables 4 and 7.

| Class | S-shape completion FIRST / SHORT / LONG / SAV | IGNORE FCFS | S-shape turnover FCFS/LONG, IGNORE FCFS | LG completion FIRST / SHORT / LONG / SAV | IGNORE FCFS | LG turnover FCFS/LONG, IGNORE FCFS |
|---|---|---|---|---|---|---|
| C/30/45 | 473 / 473 / 473 / 473 | 473 | 16, 16 | 472 / 472 / 472 / 472 | 472 | 15, 15 |
| C/30/75 | 473 / 473 / 473 / 473 | 473 | 16, 16 | 471 / 471 / 471 / 471 | 471 | 15, 15 |
| C/60/45 | 504 / 519 / 503 / 503 | 507 | 28, 31 | 502 / 514 / 501 / 501 | 504 | 26, 28 |
| C/60/75 | 497 / 499 / 497 / 497 | 521 | 24, 24 | 496 / 498 / 496 / 496 | 496 | 23, 23 |
| C/90/45 | 598 / 656 / 588 / 590 | 612 | 82, 91 | 588 / 645 / 577 / 580 | 601 | 76, 86 |
| C/90/75 | 509 / 555 / 508 / 508 | 521 | 43, 50 | 506 / 549 / 506 / 506 | 519 | 41, 48 |
| C/120/45 | 787 / 854 / 769 / 773 | 802 | 168, 179 | 773 / 837 / 752 / 761 | 785 | 157, 170 |
| C/120/75 | 618 / 699 / 613 / 614 | 637 | 90, 103 | 614 / 690 / 609 / 611 | 632 | 88, 99 |
| U/30/45 | 481 / 482 / 481 / 481 | 482 | 23, 24 | 478 / 479 / 478 / 478 | 478 | 21, 21 |
| U/30/75 | 479 / 479 / 479 / 479 | 479 | 23, 23 | 477 / 477 / 477 / 477 | 477 | 21, 21 |
| U/60/45 | 544 / 589 / 537 / 538 | 558 | 50, 61 | 541 / 578 / 536 / 538 | 553 | 50, 57 |
| U/60/75 | 508 / 521 / 508 / 508 | 511 | 34, 36 | 508 / 520 / 508 / 508 | 511 | 35, 37 |
| U/90/45 | 744 / 819 / 713 / 724 | 758 | 143, 168 | 743 / 805 / 719 / 726 | 754 | 146, 164 |
| U/90/75 | 548 / 635 / 544 / 546 | 578 | 64, 81 | 575 / 646 / 570 / 572 | 599 | 77, 92 |
| U/120/45 | 985 / 1061 / 942 / 956 | 999 | 247, 234 | 986 / 1053 / 951 / 963 | 997 | 255, 280 |
| U/120/75 | 700 / 806 / 685 / 691 | 725 | 126, 136 | 746 / 830 / 733 / 737 | 766 | 150, 168 |

One printed value is inconsistent within the journal: for C/60/75 with S-shape, Table 3 lists FCFS/LONG as 508 min and Table 2 as 497. The comparison takes Algorithm 1 from Table 2 and only IGNORE from Table 3, and reports the discrepancy.

CASIM needs: a seeded generator of these instances (layout with Henn's depot and aisle-exit geometry, class-based and uniform storage), the four selection rules as scheduling cards, the waiting rule of correction 4 with α (α = 1 is `HennWaiting`, α = 0 is no waiting), IGNORE as a configuration (commit every batch of a decision, decide again only when the picker is idle), `announced` stream closure, picker speed as an instance value, the turnover KPIs, and a Largest Gap router with Henn's definition. No published standard deviations exist, so the confidence interval comes from CASIM's own replications.

### B.5 L2 protocol check, 15 October 2026

The first approved 50-instance class (C, 30 orders, capacity 45, S-shape,
FCFS/FIRST) has a published completion mean that neither documented arrival
rate reproduces. The generator uses the same 50 seeds for the working-paper
and journal targets and for both n=30 rate interpretations. Its first order
arrives after one exponential interarrival. Values are means in minutes;
parentheses give approximate 95% confidence half-widths from these 50 runs.

| Target and arrival rate | Completion CASIM / published | Arrival turnover CASIM / published | Batch-start turnover CASIM | Kind and status |
|---|---:|---:|---:|---|
| Working paper α=1; stated 30/480 per min | 507.4 (22.0) / 467 | 42.6 (1.2) / 44 | 17.05 (0.17) | statistical; cause unresolved |
| Working paper α=1; printed 0.08625 per min | 380.6 (15.4) / 467 | 39.3 (1.2) / 44 | 17.25 (0.16) | statistical sensitivity; cause unresolved |
| Journal α=0; stated 30/480 per min | 500.2 (21.9) / 473 | 18.35 (0.74) / not published for FIRST | 13.37 (0.26) | statistical; cause unresolved |

The paper says the expected number of arrivals in eight hours is 30, which
implies 0.0625 per minute, but prints 0.08625 per minute for this class. The
two rates move the mean completion to opposite sides of the published value;
neither 50-instance interval includes it. All 153 L2 audit records captured so
far are internally consistent, so the cause has not been shown. Section 9 of
the correction plan requires a user decision before starting the remaining
L2 cells. The per-replication results are
[`working stated`](literature_benchmarks/l2_working_c30_45_first_stated.csv),
[`working printed`](literature_benchmarks/l2_working_c30_45_first_printed.csv),
and [`journal stated`](literature_benchmarks/l2_journal_c30_45_first_stated.csv).

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
| Citation | D'Haen, R., Braekers, K., Ramaekers, K. (2023), Integrated scheduling of order picking operations under dynamic order arrivals, International Journal of Production Research 61(10), 3205-3226, doi [10.1080/00207543.2022.2078747](https://doi.org/10.1080/00207543.2022.2078747); local copy placed by the user in `docs/research/dhaen/` (not committed) |
| Instances | OSF doi [10.17605/OSF.IO/7R4GN](https://doi.org/10.17605/OSF.IO/7R4GN), online instances restored by `scripts/fetch_data.py dhaen` into `scenarios/scenario_dhaen_online/data/osf/` (see its `SOURCE.md`) |
| Published values | `Results online problem instances.xlsx`: per instance and setting `OrderPickTime`, `Tardiness`, `Makespan`, `WaitingTime`, `Earliness`, `TotalSolutionTime` and the factor levels |
| Licence | CC BY 4.0 |
| Status | protocol verified against the paper; see correction 9 |

Times are in seconds (paper, Table 7); the distance matrix is in decimetres (OSF README). The LNS used a real-time budget (mean `TotalSolutionTime` 1,763 s per run), which CASIM does not spend; for a rule without search this does not matter.

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


## Approved subsets and stages

Approved by the user on 14 October 2026 with the changes below. Every benchmark runs in two stages: a smoke stage of 2 to 3 instances per set and pipeline, after which the stop rules of Section 9 apply, and then the full subset. The seed is 20261014 throughout. A CASIM run costs about 2.6 s of fixed time (explicit walk-or-wait example, three runs, 2.56 to 2.74 s; imports 2.0 s); estimates use 4 s per static run.

| # | Subset | Runs | Basis of the run time | Expected time |
|---|---|---|---|---|
| L1 | The initial approved subset is preserved in `l1_subset.csv`. The user's revised Task B.4 selects ten instances spread through the strata of each of the six distance sets, retaining the three smoke instances and prioritizing proven optima, and the full 295-instance Kris subset. Both pipelines use the same ten instances in each distance set. Exact list: `l1_runlist.csv`; smoke is its `smoke` column. | 415 runs, including 39 smoke | Measured smoke times and the 295 Kris instances dominate | Record actual time at the end of B.4 |
| L2 | Journal target (α = 0): 16 classes (C and U × n × W) × 2 routings × 5 strategies (FCFS with FIRST, SHORT, LONG, SAV; IGNORE FCFS). Working-paper target (α = 1): 8 classes × 2 routings × 4 rules. 50 generated instances per class, shared by both targets and all strategies of a class; λ = 0.0625 for n = 30, and λ = 0.08625 as a sensitivity variant for the n = 30 cells of both targets. Smoke: 2 instances of every cell. Both turnover definitions are computed | 224 cells + 56 sensitivity cells = 280 cells × 50 = 14,000 runs; smoke 560 | Henn online scenario, 100 orders over 8 h: 8.7 s per run | 34 h sequential; about 6 h on six parallel processes |
| L3 | All 128 instances with FCFS batching, the oldest-order selection, S-shape and no waiting; the 64 Henn instances twice, with the depot in the centre and bottom left. Smoke: 2 Henn instances per depot and 2 Albareda instances. List: `docs/literature_benchmarks/l3_instances.csv` | 192 | Henn online: 40 orders 5.1 s, 100 orders 8.7 s; Albareda up to 250 orders estimated at 15 s | 0.5 h |
| L4 | Replication 21 of each of the 96 OSF combinations (72 of the main design, 24 with uniform+ arrivals), forecasting off, 20 % of the trucks active, an earliest-due-date rule. Smoke: 3 instances | 96 | estimated 30 s for about 800 orders and 3 to 5 pickers | 0.8 h |
| L5 | First the 4D4L candidates (below); then all 100 standard-case instances × 5 waiting rules in CASIM with S-shape, and the decision-by-decision comparison on instances 12, 17, 45, 56, 60, 61, 62, 72, 86, 94. Smoke: instance 12 for every rule. List: `docs/literature_benchmarks/l5_instances.csv` | CASIM 500; original per candidate 2,000 | CASIM `instance_1`, wait-1, S-shape: 27.9 s | CASIM 3.9 h; original to be measured |

L5 runs last. The commit of `project_4D4L/2_Stochastic_Waiting` and the walking speed and pick time that produced `Results_Standard_Case.csv` are not recorded anywhere; candidates are the commits around the RADAR archive date (2026-04-29) and the paper, with speed and pick time from the paper, the RADAR metadata or each candidate's defaults, each run on the standard case and compared with the CSV.

Not pursued for now: L6 (companion not available), L7 and L8 (papers not available), L9 (decided by the user), L10 (vehicle routing, which CASIM does not model).

## B.4 smoke stage (L1), 14 October 2026

casim `736a8b8` plus this task's changes, ware_ops_algos `dd4ffc9`, ware_ops_pipes `554a153` (loaders only), CASOP values from the v0.1.0 results parquet. 39 runs: 3 instances per set and pipeline (`smoke` column of `l1_subset.csv`). Per-run table: `scenarios/scenario_static_benchmarks/outputs/l1_smoke.csv` (not committed).

Pipeline choice: CASIM has no integrated stages, so `RatliffRosenthalNF` (SPRP-SS) and `CombinedBatchingRoutingAssigning` (BahceciOencan) are a gap. The closest expressible pipeline over all instances with a reference value replaces them: SPRP-SS `MinMinIA+SingleOrderBatching+RR` (mean gap 13.99 %), BahceciOencan `GIA+LSBatchingRROrderNrFiFo+RR` (0.22 %, 0.02 s per instance, so all 45 instances instead of the approved 10, which were capped for the integrated pipeline's 151 s). The other choices are unchanged. Name mapping from v0.1.0 to today: `SavingsSShape` = `ClarkAndWrightSShape`, `LSOrdNrNN` = `LSBatchingNNFiFoOrderNr`, `LSBatchingRROrderNrFiFo` and `LSBatchingSShapeFiFoOrderNr` unchanged, `RR` = `RatliffRosenthal`, `NN` = `NearestNeighbourhood`, `GIA` = `GreedyIA`, `EDD` = `EDDScheduler`, `SingleOrderBatching` = a ware_ops_pipes template stage (one batch per order), expressed as `FiFo` on the one-order SPRP instances.

| Set | Pipeline | Runs | 1. CASIM = direct | 2. direct = v0.1.0 | 3. gap to reference | Invariant violations in CASIM |
|---|---|---:|---|---|---|---|
| SPRP | GIA + single + RR | 3 | not executable | 3 of 3 | 0 % (all optimal) | |
| SPRP | GIA + single + S-shape | 3 | 3 of 3 | 3 of 3 | 11.0 to 27.0 % | none |
| SPRP-SS | MinMinIA + single + RR | 3 | not executable | 3 of 3 | 0.7 to 52.0 % | |
| SPRP-SS | MinMinIA + single + S-shape | 3 | 3 of 3 | 3 of 3 | 85.7 to 275.5 % | none |
| BahceciOencan | LS(RR, OrderNr) + RR | 3 | not executable | 3 of 3 | 0 to 0.54 % | |
| BahceciOencan | LS(S-shape, OrderNr) + S-shape | 3 | 3 of 3 | 3 of 3 | 16.8 to 24.5 % | `overlap` in 1 run |
| HennWaescher | LS(RR, OrderNr) + RR | 3 | not executable | 3 of 3 | 0.5 to 4.2 % | |
| HennWaescher | Savings(S-shape) + S-shape | 3 | 3 of 3 | 3 of 3 | 12.5 to 21.8 % | none |
| MuterOencanWG | LS(RR, OrderNr) + RR | 3 | not executable | 3 of 3 | 3.8 to 10.2 % | |
| MuterOencanWG | Savings(S-shape) + S-shape | 3 | 3 of 3 | 3 of 3 | 15.5 to 20.5 % | `overlap` in 3 runs |
| Foodmart | LS(NN, OrderNr) + NN | 3 | 3 of 3 | 3 of 3 | 7.4 to 9.2 % | none |
| Foodmart | LS(S-shape, OrderNr) + S-shape | 3 | 3 of 3 | 3 of 3 | 6.9 to 18.3 % | none |
| Kris | GIA + DueDate + NN + EDD (total time) | 3 | 1 of 3 | 3 of 3 | 11.5 to 28.0 % | `overlap` in 1 run |

No value lies below a proven-optimal reference, and no local search or savings run reached its time limit (300 s), so every batching result is deterministic. Comparison 2 finds no drift on the smoke instances.

Findings, each with its evidence:

1. `RatliffRosenthalRouting` returns a distance and an item sequence but no node sequence (`annotated_route` is `None`; `routing.py`, `_run`), so CASIM cannot execute its tours (`TypeError` in `tour_manager.create_tour`). All RR pipelines (15 of 39 runs) are solved directly but not run in CASIM. This affects the closest pipeline of SPRP, SPRP-SS, BahceciOencan, HennWaescher and MuterOencanWG.
2. Kris, comparison 1: CASIM gives 138,504 against 137,352 (instances_208_1) and 122,874 against 120,570 (instances_31_1). The batches and the routes are identical. CASIM charges more for some consecutive nodes of the NN route than the graph-edge weights and the router's reported distance. In instances_31_1, the routes of orders 9 and 2 each have a reported and graph-edge distance 384 shorter than the sum of their matrix legs (2,406 vs. 2,790 and 1,482 vs. 1,866). With speed 1/3 this is 1,152 per route. The source of the conflicting edge and matrix weights is outstanding; see the continuation below.
3. `overlap` (`check_plan`): in the Muter-Öncan and Bahceci runs, a started tour's planned end exceeds the next tour's planned start by 5.7 × 10⁻¹⁴: on `TourStart`, `state.py` recomputes the planned end from the current time, and the strict comparison in `invariants.check_plan` has no tolerance. In Kris instances_31_1 the overlap is 1,152 s: the tour of finding 2 runs longer than the scheduler planned and nothing re-projects the queue. Both belong to the planned and executed times of Task 4.14.

Changes for this stage: ware_ops_pipes `554a153` as casim development dependency (git source; `ware-ops-algos`, `cosy-luigi` and `luigi` overridden to casim's own, since only the loaders are used), pyarrow for the parquet; casim's configured-component generator renders `LocalSearchBatching` and `ClarkAndWrightBatching` cards (12 generated components, replacing the hand-written `LSBatchingNNFiFo`, `LSBatchingSShapeFiFo`, `ClarkAndWrightNN` and `ClarkAndWrightSShape`, which called `LocalSearchBatching` without its required neighbourhoods); a `MinMinIA` component; the scenario `scenario_static_benchmarks`.

### B.4 continuation: outstanding distance representation

The prior smoke finding 2 correctly describes the **matrix-leg** distance CASIM charges, but its attribution to the NN router and its claim that this is the correct walked distance are unsupported. The graph edges of the emitted route for order 9 in Kris `instances_31_1_small` sum to the reported 2,406, while summing the distance matrix between consecutive emitted nodes gives 2,790. The corresponding figures for order 2 are 1,482 and 1,866. The two paths traverse `(96,1022)→(144,1022)` and `(144,1022)→(144,985)` in both directions. The graph gives these edges weights 48 and 37; the distance matrix gives 144 and 133. Four 96-unit differences explain the 384-unit tour gap. CASIM's `TravelEvent` calls `LayoutManager.get_distance`, which uses the matrix. Both edge pairs exist in the graph, so the graph-edge sum is independent of the router's stated distance.

The user directed that this conflict remain outstanding while the rest of the plan proceeds. B.4 records each NN route's reported, graph-edge and matrix-leg distances; comparison 1 may differ only by the measured matrix-leg gap. No router, matrix or simulator behaviour is changed here. `warehouse_info` is not named by any algorithm card, so the static domain's `OFFLINE` value cannot alter mapper applicability.

### B.4 full stage

`select_l1_runlist.py` fixes the full selection at 415 rows: ten distinct
instances for each of the six distance sets under both pipelines, all 295
selected Kris instances, and all 39 original smoke rows. The 355 distinct
instance files are restored and checked by `scripts/fetch_data.py
casop_instances`; `data/instances/SHA256SUMS.txt` pins their prepared bytes.
The initial full selection retained only 23 smoke rows, so the remaining 50
rows of the corrected selection were run and checked separately. All 415
direct results equal CASOP v0.1.0. Of 365 non-RR rows that CASIM can execute
at B.4, 260 equal the direct objective and the other 105 are Kris cases whose
CASIM/direct difference equals the measured matrix-leg gap. None of the 36
distance-set smoke distances changes with speed 2, pick time 7 and setup 11.
No batching run reaches its 300-second limit. The per-run CSV is ignored run
output in `scenarios/scenario_static_benchmarks/outputs/`.

Across 3,672 Kris NN routes, 202 matrix-leg sums differ from the router
report, by 71,904 distance units in total; 566 graph-edge sums differ from
the report. The matrix-based total-time gap to the literature is 0.45 to
48.70 %, compared with −2.81 to 48.70 % for the router-reported total time.
The sole direct result below a marked optimum is `instances_207_1_small`:
128,700 against 132,420, while CASIM gives 133,020. This is the same F64
question the user asked to leave outstanding. RR pipelines are compared
directly with CASOP here; Task B.4a makes them executable in CASIM.

### B.4a: RatliffRosenthalRouting executes its tours

Finding 1 is closed. `RatliffRosenthalRouting._run` now reconstructs its node
sequence when asked for a tour (`gen_tour=True`): it expands the Euler tour of
the dynamic program's multigraph (`_construct_picker_tour`) into network nodes
with the same shortest-path machinery as the other routers, then inserts a
`PICK` node at every visited pick position, in Euler order; the reconstruction
raises if a pick position is never visited. Without `gen_tour`, the returned
`Route` keeps `annotated_route=None`, so CASOP's behaviour is untouched by
construction: with `gen_tour=False` the reported distance is computed before
the reconstruction, and the applicable card-name sets of the seven CASOP data
cards (ware_ops_pipes `data/data_cards`) are identical to `main` (8, 13, 19,
19, 19, 15, 24 on sprp, sprp-ss, bahceci-oencan, henn-waescher, muter-oencan,
foodmart and kris).

The router used by comparison 1 now builds the RR router exactly as casim's
`RatliffRosenthal` component does (`start_node=network.start_node`,
`gen_tour=True`, `gen_item_sequence=True`). On all 15 RR smoke rows and all 50
full RR rows of the run list, every route's walked node sequence, its
underlying graph-edge sum and the router's reported distance are equal, so RR
has no F64-like gap. Comparison 1: CASIM equals the direct calls on all 50 RR
rows (as on the 260 equal non-RR rows); comparison 2: direct still equals
CASOP v0.1.0 on all 50 RR rows; no RR row lies below its proven optimum (49
rows at the optimum, `100_48_9` at its best-known reference).

The 14 RR rows on BahceciOencan and MuterOencanWG report `check_plan`
`overlap` violations, the same epsilon overlaps documented as finding 3 (the
planned end recomputed at `TourStart`); they also occur on 37 non-RR rows of
the same sets and belong to Task 4.14. ware_ops_algos commit `8b91419`
reconstructs the route with a test on the paper's synthetic instance; casim
wires the component and removes the RR entry from the runner's not-executable
list.

### B.5 smoke stage (L3), 15 October 2026

Protocol recheck against Gil-Borrás et al. (2020), pp. 301-302, 304 and
311-312: capacity is the sum of item counts for W5 and W1-W3, and of item
weights for W4; one setup of three minutes belongs to each batch's service
time; the next batch selected contains the oldest order. The paper's published
method repeatedly forms batches with GRASP and VND while the picker is away.
CASIM's FCFS batching is a comparator, so its tours are not expected to equal
the published tours. The paper gives no alternate capacity or setup unit that
removes the below-best-known smoke values. The W3 archive field `Tiempo de giro
(fuera y dentro) 20 0` specifies a turning time, but the paper's service-time
formula and Table 4 do not say how to charge it. The current runner omits it.
Every value below best known is labelled **protocol not reproduced** in the
full-run comparison, including turnover values affected by W3.

New scenario `scenarios/scenario_oobp` (runners `experiment_oobp.py` and
`run_l3.py`). The decision is FCFS batching, the oldest-order batch selection
(`FiFo` + `FIFOScheduler` under `SchedulingCommitmentPolicy(n_jobs=1)`),
S-shape and no waiting, under the protocol of Table 4 (speed 0.8 LU/s, 10 s
per item, setup 180 s; the archive's arrival CSVs are replayed). Both
objectives are recomputed from the executed tours (maximum completion time,
maximum turnover time from arrival to delivery).

Loaders, both new in `scenarios/scenario_oobp/loader.py`:

- `OobpHennLoader` reads the archive's `settNN.txt` (10 aisles, 45 locations
  per side, `cell_lengt` 1, aisle centre distance `2 * cell_width +
  aisle_widt` = 5, `dis_ais_wa` 1 from the depot to the front cross aisle) and
  the raw order files. Locations are 0-based in the archive; a location sits at
  `y = location`. With `depot=corner` (the benchmark page's "bottom left") the
  depot is `dis_ais_wa` below the front cross aisle in front of the leftmost
  aisle; with `depot=centre` (Table 3 "Center", Fig. 5a) it sits on the front
  cross aisle between the two middle aisles. It matches the archive's
  inconsistent arrival-file names on the separator-free stem.
- `AlbaredaLoader` reads `wsrp_input_layout_*` and `wsrp_input_pedido_*`:
  shared shelves, so the centre distance is shelf width plus aisle width; the
  depot sits on the front cross aisle at the leftmost aisle (0) or between the
  two middle aisles (1); W4's items carry their weights, and the cart is by
  weight. Pick nodes are the distinct item positions per aisle.

Smoke result (16 Henn runs, both depots; 8 Albareda runs), completion and
turnover in seconds against `data/oobp_reference.csv`; no run of the runner
raises, below-best-known is recorded per row:

| Instance | Depot | CASIM completion (bk) | CASIM turnover (bk) | Below bk |
|---|---|---|---|---|
| abc1 29s-40-30-0 | corner | 24,541 (21,109) | 15,541 (9,848) | - |
| abc1 29s-40-30-0 | centre | 24,223 (21,109) | 15,223 (9,848) | - |
| abc1 30s-40-45-0 | corner | 17,298 (17,541) | 8,299 (5,133) | completion -1.4 % |
| abc1 30s-40-45-0 | centre | 16,947 (17,541) | 7,948 (5,133) | completion -3.4 % |
| abc2 10l-40-45-0 | corner | 18,841 (18,170) | 10,070 (9,377) | - |
| abc2 10l-40-45-0 | centre | 18,487 (18,170) | 9,716 (9,377) | - |
| abc2 11l-40-60-0 | corner | 14,678 (15,708) | 5,908 (4,829) | completion -6.5 % |
| abc2 11l-40-60-0 | centre | 14,345 (15,708) | 5,574 (4,829) | completion -8.7 % |
| ran1 29s-40-30-0 | corner / centre | 30,091 / 29,647 (24,689) | 21,091 / 20,647 (12,838) | - |
| ran1 30s-40-45-0 | corner / centre | 19,418 / 19,109 (18,023) | 10,595 / 10,286 (8,715) | - |
| ran2 10l-40-45-0 | corner / centre | 21,751 / 21,440 (21,073) | 12,751 / 12,441 (8,990) | - |
| ran2 11l-40-60-0 | corner / centre | 16,241 / 15,976 (16,051) | 7,691 / 7,426 (5,874) | centre completion -0.5 % |
| W1 100 _000 / _030 | 0 | 25,381 (22,308) / 20,969 (18,371) | 13,740 / 9,339 | - |
| W2 100 _000 / _030 | 0 | 17,977 (17,203) / 15,185 (15,535) | 6,520 / 4,171 (2,702) | _030 completion -2.3 % |
| W3 100 _000 / _030 | 0 | 40,450 (39,902) / 34,040 (33,212) | 29,195 / 23,963 | turnover below bk (30,959 / 25,539) |
| W4 100 _000 / _030 | 0 | 129,629 (110,218) / 104,982 (92,936) | 118,055 / 93,298 | - |

Findings, each with its evidence:

1. Under the published depot (corner), CASIM's rule beats the published
   best-known completion on abc1 30s (-1.4 %) and abc2 11l (-6.5 %), and the
   centre depot on three instances. A hand check of the abc2 11l corner run:
   12 tours for 40 orders and 569 picks, every tour gap exactly the 180 s
   setup (the picker never idles after the first tour), per-tour travel 448 LU
   on average, and the first tour's walked node sequence reproduces the
   paper's S-shape by hand (368 LU computed, 368 LU executed). The paper's
   routing skips empty aisles and U-turns in the last aisle of an odd count
   (Section 3.3), which the implementation follows. The completion gap to the
   best known 15,707 s corresponds to about two additional tours at the same
   per-tour work, or to a systematically longer tour, neither of which the
   protocol text states. Per the 15 October user direction, the full run
   proceeds and these gaps are labelled **protocol not reproduced**.
2. W3 (25 aisles) has `Tiempo de giro (fuera y dentro) 20 0` in its layout
   files, a turning time the protocol tables do not mention and CASIM cannot
   charge. Its turnover beats the best known on both smoke instances while
   completion stays above; consistent with 20 s per aisle traversal missing
   from CASIM's tours. Not modeled; recorded here.
3. The epsilon `check_plan overlap` violations appear as before (planned end
   recomputed at `TourStart`, Task 4.14); counts 0 to 1,499 per run.
4. The paper's Table 2 warehouse values differ from the archive's layout
   files (for example W1: paper aisle length 50 and centre distance 4.3, file
   86.917 and 7.167). The runs use the files, which the published values
   correspond to; the discrepancy is recorded in the register's L3 row.

### B.5 full stage (L3), 15 October 2026

The full run covers 64 Henn instances under both centre and corner depots
(128 runs) and 64 Albareda-Sambola instances with the archive's depot
(192 runs total). The
per-instance numbers, flags and protocol status are in
[`l3_full_results.csv`](literature_benchmarks/l3_full_results.csv). Every
comparison has kind **bound**: CASIM runs a different batching method from the
published GRASP/VND search. The table gives means in seconds and counts of
individual results below the published best-known completion/turnover. A row
with either count above zero has status **protocol not reproduced** even if its
mean is above the best-known mean.

| Case | n | Completion CASIM / best known (s) | Turnover CASIM / best known (s) | Below best known C/T | Status |
|---|---:|---:|---:|---:|---|
| W1/100 | 4 | 23,057 / 20,167 | 11,435 / 8,331 | 0/0 | comparator only |
| W1/150 | 4 | 33,102 / 26,967 | 20,661 / 15,736 | 0/0 | comparator only |
| W1/200 | 4 | 43,988 / 35,698 | 31,292 / 23,909 | 0/0 | comparator only |
| W1/250 | 4 | 56,925 / 45,863 | 43,945 / 34,615 | 0/0 | comparator only |
| W2/100 | 4 | 15,777 / 16,235 | 4,562 / 4,138 | 3/1 | protocol not reproduced |
| W2/150 | 4 | 25,260 / 22,709 | 12,823 / 11,620 | 0/0 | comparator only |
| W2/200 | 4 | 33,021 / 29,240 | 20,474 / 18,613 | 0/0 | comparator only |
| W2/250 | 4 | 42,051 / 36,735 | 29,113 / 25,932 | 0/0 | comparator only |
| W3/100 | 4 | 38,704 / 37,039 | 27,653 / 28,471 | 0/3 | protocol not reproduced |
| W3/150 | 4 | 55,709 / 51,302 | 43,321 / 43,305 | 0/2 | protocol not reproduced |
| W3/200 | 4 | 74,418 / 67,579 | 62,086 / 60,692 | 0/1 | protocol not reproduced |
| W3/250 | 4 | 95,631 / 85,858 | 82,898 / 79,158 | 0/0 | comparator only |
| W4/100 | 4 | 110,178 / 93,861 | 98,522 / 80,852 | 0/0 | comparator only |
| W4/150 | 4 | 166,511 / 137,587 | 154,057 / 124,006 | 0/0 | comparator only |
| W4/200 | 4 | 218,712 / 180,012 | 206,017 / 166,726 | 0/0 | comparator only |
| W4/250 | 4 | 265,637 / 220,200 | 252,639 / 206,660 | 0/0 | comparator only |
| abc1/centre | 16 | 28,827 / 26,791 | 18,957 / 15,099 | 4/0 | protocol not reproduced |
| abc1/corner | 16 | 29,245 / 26,791 | 19,371 / 15,099 | 4/0 | protocol not reproduced |
| abc2/centre | 16 | 29,349 / 26,768 | 19,390 / 15,424 | 3/1 | protocol not reproduced |
| abc2/corner | 16 | 29,765 / 26,768 | 19,802 / 15,424 | 3/1 | protocol not reproduced |
| ran1/centre | 16 | 33,248 / 30,369 | 23,310 / 18,705 | 3/0 | protocol not reproduced |
| ran1/corner | 16 | 33,627 / 30,369 | 23,689 / 18,705 | 3/0 | protocol not reproduced |
| ran2/centre | 16 | 33,531 / 30,489 | 23,569 / 19,070 | 4/1 | protocol not reproduced |
| ran2/corner | 16 | 33,912 / 30,489 | 23,951 / 19,070 | 3/1 | protocol not reproduced |

Forty of 192 runs are below at least one best-known value: 30 below for
completion and 11 for turnover, with one run below on both. The largest
relative gaps are 21.02% for completion and 13.97% for turnover. These are
**protocol not reproduced**, as directed by the user; no documented
interpretation of capacity, setup, batch selection or W3 turning time removes
the smoke gaps. All 192 audit files have zero KPI recomputation differences.
The 118 files with violations contain only the previously documented
`check_plan overlap` issue (Task 4.14). Seven duplicate runs made while
partitioning the full sweep agreed on every result field except wall time.
