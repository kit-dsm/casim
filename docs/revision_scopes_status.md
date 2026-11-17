# Revision scopes status: Phase 4 done

Last pushed code commit: casim `c9d0207` (`4.16 (2): commented-out code out, library prints to logging`); ware_ops_algos `c7950f4`. Phase B is done and reported (`docs/phase_b_report.md`, `6789b39`); open for the user from it: the high-load L2 gap to the journal without a demonstrated cause and the decisions of Section 9.1. F64 is resolved by the user (`docs/kris_f64_report.md`): the cause is ware_ops_pipes' Kris loader, whose fix is the user's. Phase S is done and reported (`docs/phase_s_report.md`): the full suite passes in one process (574 tests) and `casim check --all` has no finding for any tracked scenario.

The two stashes of 15 October are applied and committed; they are kept as the local branches `wip/s4-henn2012` and `wip/s3-s4-docs` and remain in `git stash list`.

Waiting for the user: the 16 working-paper `truncated` L2 pilot baselines recorded at `cf2d4a8` before those cells moved to `l2_working` (F67) are to be re-recorded once the user accepts that difference.

Next steps, in order:

1. Phase 5, starting with Task 5.0 (the permanent CASOP applicability lock); the Phase 4 report is `docs/phase_4_report.md`. Task 4.17 follows the IJPE revision.
2. The two decisions of the Task 4.15 report: draft times after a delay, and a terminal check for orders still open.

The route-length check is done (16 November 2026, `docs/literature_benchmarks.md`, "Route-length check of the high-load gap"): CASIM's executed tours equal an independent S-shape computation on all 120 runs; α is applied through the waiting component and correctly leaves the instances unchanged; the journal's high-load values are not reachable with next-fit FCFS batches and match first-fit batches, which close about two thirds of the gap, leaving about 4 % common to both papers. First-fit FCFS is now a batching algorithm (ware_ops_algos `b650516`, card `FirstFitFiFo`) and a declared variant of the journal target in `scenarios/henn2012`; it does not close the gap to the +4.3 % residual (class-based +4.9 to +7.5 %, uniform −4.9 to +0.2 %), so the journal's high-load cells stay a statistical non-reproduction without demonstrated cause ("First-fit variant of the journal target" in the register). F68 (the OOBP loader's depth offset) is recorded and fixed (`80dbbc7`); the journal FIRST L2 runs with n = 90 and 120 are rerun and recorded again (`a35bb41`): every journal residual changes by at most 0.2 percentage points ("F68" in the register). After F68 all L2 runs and the 128 Henn-format L3 runs are recorded again (`d714c58`, `2ad6e22`); the count of L3 runs below a best-known value is unchanged at 40 of 192, the same runs before and after.

F64 is resolved (`docs/kris_f64_report.md`). The Kris instance files are consistent; ware_ops_pipes' `IBRSPLoader` moves every intersection to the nearest x with a product, so in instances where an aisle has no product several vertices become one coordinate node and the distance matrix keeps the entry of the vertex written last (166 of 243 small instances, 103,278 wrong entries). NearestNeighbourhood's reported distance is the shortest-path distance of its sequence and correct; CASIM's executed matrix-leg distance and the aisle-based routers' reported distances are inflated. The loader fix lives in ware_ops_pipes and is the user's decision; casim neither changes ware_ops_pipes nor works around it. Until then, a Kris CASIM/direct difference that this defect explains stays exempt from the B.4 stop rule; the exemption covers nothing else.

The B.4 measurements stand as measurements: at Kris `instances_31_1_small`, the NN route for order 9 reports 2,406 while the matrix legs CASIM executes sum to 2,790, because the matrix entry (96,1022)→(144,1022) holds the distance from vertex 40 to the merged vertex 49 (144) instead of vertex 43 (48); across 3,672 Kris NN routes, 202 matrix-leg sums differ from the report. The reported 2,406 is the correct distance. `instances_207_1_small` is not a case of F64: its optimum 132,420 requires every order to meet its due date, and the minimum without due dates is 118,044, so CASOP's 128,700 violates a due date. Kris results are compared with the best-known solution only when they are deadline-feasible, as in the CASOP paper; deadline-infeasible results are reported separately. `scripts/l1/run_benchmark.py` still applies the earlier F64 rule to this row and follows the corrected plan with the next L1 rerun.

This is an audit against the current `revision_scopes_corrections.md`, through Phase 3b. Evidence names are casim commits unless prefixed `ware:`. A row marked “moved to a later task” records an explicit phase boundary or user-approved sequencing deviation, not a completed check. The historical base results are diagnostics; Phase 5 establishes correctness.

## Baseline tiers and reruns

User directions of 16 and 17 November 2026, effective from then on:

- The baselines are a regression set, not a study (17 November 2026): one run per code path, 65 records instead of 1,654 (`863c8db`). Kept: every scenario configuration except 13 of the 15 Henn instances (26, of which the five IJPE cases are `not_runnable` until Task 6b.6); one L1 run per instance set and pipeline (13; Kris EDD keeps `instances_109_1_small`); one L2 cell per target, storage, rule and arrival mode at n = 30, capacity 45, replication 0, plus journal C/120/45 FIRST and C/60/75 SAV (10); one L3 run per stratum and depot (16). The literature tables keep their own runners (`docs/literature_benchmarks.md`).
- `scripts/baseline.py` runs `casim check` once per sweep (`sweep_key`), and a successful run no longer keeps its output folder (`aa02c80`).
- A rerun covers only what a change can affect; the report names the scenarios and benchmark cells first. `compare-benchmark --match <glob>` and `record-benchmark --match <glob>` restrict a benchmark.
- **Fast tier**, after every commit: `python scripts/baseline.py compare-fast --workers 4`, the 11 cases of `FAST`.
- **Full tier**, at the end of a task: `compare` and `compare-benchmark --benchmark l1`, `l2`, `l3`; 60 runs in 194 s on six workers (before the slimming and the two speedups below: about two hours).
- Run time (17 November 2026, idle machine): the audit checks the plan after the last event of every timestamp and after every decision instead of after every event, and a decision snapshot shares the layout, the articles and its freshly built decision data instead of deep-copying them. Henn default 8.6 → 4.0 s plain and 16.7 → 8.9 s with `--audit`; L3 abc1 centre 5.0 → 3.2 s and 11.1 → 6.5 s. About 3 s of a plain run is importing (ware's layout module imports pandas, matplotlib and networkx).

## Phase B progress

| Task | Status | Evidence and remaining work |
|---|---|---|
| B.1 | done | `feb7e69`; `docs/baseline.md`, 60 configurations, 45 run, 15 fail. |
| B.2 | done | `694830e`; `casim.baseline_capture` behind `experiment.audit`, `tests/verification/kpi_recomputation.py`, nine scenarios in `tests/verification/test_phase_b_scenarios.py`. |
| B.2 corrections | done | Distance between positions off the graph uses both aisle neighbours with a tolerance and measures one aisle segment directly (`tests/verification/test_baseline_capture.py`, four hand-derived cases, three of which fail with the earlier nearest-node rule). Verification scenario 6 is its own walk-or-wait scenario with prefix, suffix and distances derived in its docstring. The 61s-80-30-0 failure is traced to `ViolationCode.STATE_INFORMATION` on a `ReplaceSuffix` that takes an order from a draft cancelled in the same proposal (`docs/baseline.md`); not fixed. The three scenarios of Section 7.5 give the same KPIs with and without `experiment.audit`: Henn 30,669 makespan, 11,808 distance, 27 tours, 15 decisions; stochastic waiting 87 / 76; walk-or-wait 167 / 160; no invariant violations and every route leg recomputable. |
| B.3 | done; subsets approved | `docs/literature_benchmarks.md`: register L1 to L10 verified against the sources (Henn 2009 and 2012, Gil-Borrás 2020, D'Haen 2023, RADAR, Lorenz 2025), corrections to Section 7.4; the Henn journal version is a second L2 target (α = 0, uniform storage, IGNORE); approved two-stage subsets with lists in `docs/literature_benchmarks/`. Benchmark data is restored by `scripts/fetch_data.py` (`4da5840`). L6 to L10 not pursued for now. |
| B.4 | done; F64 resolved, loader fix by the user | `fd423b1`. 415 approved static rows and 39 smoke rows; all 415 direct results equal CASOP v0.1.0. The 365 non-RR rows have 260 CASIM/direct equalities and 105 Kris differences, all explained by the loader defect F64 (CASIM's executed distance is inflated, the router's report is correct). All 36 distance-set smoke runs keep distance under a second time parameter set. The single direct time below a marked optimum is `instances_207_1_small`, a deadline violation (`docs/kris_f64_report.md`), not F64; deadline-infeasible Kris results are not compared with the optimum. Data fetch and checksum verification cover 355 instances. RR execution follows in B.4a. |
| B.7 | scenario inventory, L1 and L3 done; L2 capture in progress | `1e2d053`: recorder uses relative commands and counts `decisions` as audit decision callbacks by decision name. Setup-inclusive occupation is a distinct definition, not a KPI inconsistency. All 45 runnable configurations re-recorded and deterministic comparison passed. `7cc47cc`: all 415 L1 audit baselines captured; zero KPI recomputation differences, 51 known `overlap` violations (21 Kris, 19 MuterOencanWG, 11 BahceciOencan), and CASIM objective equality to the corrected CSV. `c3f5eee`: all 192 L3 audit baselines captured; zero KPI recomputation differences, 118 known `overlap` files. L2: the baselines of every successful validation run are in `tests/baseline/benchmarks/l2/` (`<target>__<class>__SShape__<rule>__<replication>__conditioned.json`); failed SAV replications (F62) have none. Determinism: the 20 journal C/30/45 and C/60/45 FIRST files recorded at `cf2d4a8` (committed in `4ebbf95`) and rerun at `7f2f87c` and at `7e68bf6` differ only in the commit fields. The 20 working-paper files of the pilot change, as named: they now run `l2_working` (F67). |
| B.4a | done | ware:`8b91419` (RR route reconstruction from the DP's Euler tour, with a test on the paper's synthetic instance; `Route.annotated_route` stays `None` without `gen_tour`, so CASOP is unchanged; the applicable card-name sets of the seven CASOP data cards equal `main`, counts 8, 13, 19, 19, 19, 15, 24); casim wires the RR component like the other routers (`gen_tour=True`) and removes the RR not-executable entry. All 15 RR smoke rows and all 50 RR full rows: walked node sequence, graph-edge sum and reported distance are equal (no F64-like gap), CASIM equals direct (comparison 1), direct equals CASOP v0.1.0 (comparison 2), no row below its proven optimum. The 14 BahceciOencan/MuterOencanWG RR rows show the documented epsilon `overlap` violations, which also occur on 37 non-RR rows of the same sets; Task 4.14. See the B.4a section of `docs/literature_benchmarks.md`. |
| B.5 (L2, final) | done; reduced validation set reported | `7e68bf6`: working-paper cells use `decisions/l2_working.yaml` without `NbrOrdersCondition` (F67); C/30/45 FIRST 798.8 → 497.1 min against 467, the rest is a bound of the arrival process (`docs/literature_benchmarks.md`, "B.5 L2 validation set"). `tests/verification/test_l2_first_selection.py` shows that FIRST commits the first FCFS batch on every recorded FIRST run. The validation set (S-shape, FIRST and SAV, 20 replications, journal 16 classes, working paper 8) is in the same section with 95 % intervals and run time per run; journal FIRST matches at n ≤ 60 and is 4 to 14 % high at high load without a demonstrated cause (open, Section 9); the working paper's published turnover is arrival-based; SAV is blocked by F62 in most cells with n ≥ 60. Deferred L2 cells are listed there. |
| B.5 (history) | full L3 done; L2 matrix superseded by the validation set | `c3f5eee`: all 192 L3 runs complete; 40 are below at least one best-known value (30 completion, 11 turnover, one both), labelled **protocol not reproduced** per user direction. The paper confirms item-count/weight capacity, one setup per batch and oldest-order selection; GRASP+VND batch formation differs from FCFS; W3 turning-time charge remains unspecified. `8ffdcf2` and ware:`20d7ced` add the L2 generator, named `announced` closure option, SAV selector, audit baselines and initial 50-instance results. The first working-paper C/30/45 S-shape FIRST cell averages 507.4 min completion and 42.6 min arrival turnover versus published 467/44 under 30/480 arrivals per minute. The paper's printed 0.08625 rate yields 380.6 min completion and 39.3 min arrival turnover on the same 50 seeds. The journal α=0 target is 500.2 min completion versus 473 published under the stated rate. Neither rate fits the published completion and turnover together. The bounded arrival, tour, geometry and time-unit check found all generated picks charged, with setup once per tour; the unpublished source seeds leave a statistical nonreproduction. The user authorized the remaining matrix after this check. |
| B.9 | done | `b519ecb`: `docs/revision_scopes_baseline.md` points to `docs/baseline.md`; the unreferenced `docs/waiting_intervention_foundation.md` is removed. |

## Phase S progress

| Task | Status | Evidence and remaining work |
|---|---|---|
| S.1 | done | `793d765`: `casim.sources` with the `Source` protocol (`initial_state()`, `events()`), the library sources and `local` for a scenario's `source.py`. A source creates its initial events in the order of the old hooks and yields them sorted by time with a stable sort, so equal-time ties are unchanged. |
| S.2 | done | `793d765`: `casim run` (outputs `kpis.json`, `decisions.json`, `tours.csv`, `orders.csv`, `trace.json`; `audit.json` with `--audit`; `comparison.csv` with `reference.csv`; sweeps), `casim check <dir>`, `scenarios/minimal/`; `scripts/baseline.py` runs ported scenarios with `casim run`. |
| S.3 | done | `793d765`, `6dc732b`: `casim check --all` and its test on every tracked scenario; `AGENTS.md` and `CLAUDE.md`. |
| S.4 | done | `henn_waiting` `74c520a` (31 baselines equal), `walk_or_wait` `03f642f` (11 equal), `analytic_waiting` `16768e9` (3 equal), `oobp` `74f0214` (192 L3 runs equal), `casop_static` `6dc732b` (415 L1 runs equal; Kris rows carry the loader defect F64 until the user fixes the loader), grocery deleted `85a06c5`, `henn2012` `d12fb02` (738 of 754 L2 runs equal: all 714 validation runs and the 20 journal pilot runs; the 16 working-paper truncated pilot runs differ as named for F67). `scenarios/experiment_commons.py` stays for IJPE until Task 6b.6. |

## Phase 4 progress

| Task | Status | Evidence and remaining work |
|---|---|---|
| 4.0 | done | ware:`2676b78`, `ed97e5f`; casim:`2b482a0`; waiting, admission and positioning bases. Module consolidation remains in 4.12. |
| 4.1–4.2 | done | ware:`a05bf88`; casim:`1a1c7ab`, `0111acc`; whole-schedule waiting and injected Henn router. |
| 4.3 | done | ware:`a8b5142`; casim:`c07839d`; analytical precondition and cart capacity. |
| 4.4 | done | ware:`2ca5f67`; casim:`9458945`; ordinary positioning tours and return routing. |
| 4.5–4.7 | done | ware:`f29ff63`, `1c7aef9`; casim:`6387f0b`, `6853ab1`; incumbent router, card/base check and name sweep. |
| 4.8 | done | ware:`f2e467e`; casim:`3295930`; setup starts at scheduled start. |
| 4.9 | done | F42; `tests/verification/test_dock_fixture.py` compares the full prechange event, dock-fill, decision and KPI record for seeds 17 and 29 and truck capacities 2 and `null`. All four comparisons pass. ware adds optional `Dock`; casim owns `DockManager`, and the IJPE loader reads `simulation.dock.K_dock`. Grocery loader wiring is deferred to Phase 6b by the user's instruction. |
| 4.10 | done | ware:`20cb91e`; casim:`640b5ce`; one order-count waiting card and component. Task 4.12 replaces temporary component aliases. |
| 4.11 | done | Removed dead sequencing, renamed `PickListCreated` and the five decision files, restored alternative Henn strategies as `decisions` files, corrected stale references, and removed tracked outputs. One-instance sweep and 62 casim tests pass; comparison below. |
| 4.12 | done | Five configured CoSy components are generated from resolved CASOP cards; wait 0–4 and both Henn pipelines keep their KPIs. The seven CASOP applicability sets equal `main` (counts 19, 15, 19, 24, 19, 8, 13); Task 5.0 commits the permanent lock. Module layout, rerouting base, mapper docstrings, and `OrderSplittingSolution` are corrected. Four user decisions remain untouched. |
| 4.13 | done | Below, "Task 4.13 report". |
| 4.14 | done | Below, "Task 4.14 report". |
| 4.16 | done | Below, "Task 4.16 report". |
| 4.15 | done | Below, "Task 4.15 report"; narrowed, 4.17 deferred until after the IJPE revision (user decision, 16 November 2026). |

Task 4.12 makes `walk_or_wait_sshape_routing` and `fifo_scheduler` inapplicable to CASOP by requiring the `routing_origin` decision feature, which the seven CASOP data cards lack. CASIM's corresponding data cards provide it. A comparison of the applicable card-name sets for each of those seven data cards against `main` shows no additions or removals after this change. `IncumbentStartBatchingSShape` merges the SShape requirements through `configs.yaml`; `OrderCountWaiting1` through `OrderCountWaiting4` resolve the value option. The prechange committed code and current code have byte-identical `kpis.json` for `wait_0` through `wait_4` on `explicit_example` and for `repair_waiting_henn` on `29s-40-30-0.txt`. The baseline Henn pipeline still has makespan 22533.874, 30 tours, mean tour duration 563.566667 and distance 11057, as in Task 4.11. The task generated its components without scenario-data changes.

The shipped CASOP source has no reference to order splitting outside its implementation package and no order-splitting card; `OrderSplittingSolution` is now placed with the other algorithm interfaces. I propose removing order splitting from the CASOP library in a later task unless a supported CASOP use case is identified; I did not add a card with no consumer. The following Task 4.12 decisions remain untouched: whether to restore the `due_date` requirement of `spt_scheduling.yaml`, whether to change `Resources.get_features()` semantics, whether to change per-picker processing time in `earliest_free_assignment`, and whether to revise the cards' objective vocabulary.

Task 4.9 retained two known issues: a `TruckDeparture` with `capacity: null` releases no staged pallets, and grocery truck capacity counts orders while staging counts pallets per tour. The plan mentions a `TruckArrival` dock branch, but no such event exists in the current core; this factual slip changes no Task 4.9 work. The plan's original instruction to wire both loaders is superseded by the user's explicit direction to defer the missing grocery loader to Phase 6b.

Task 4.11 corrected the Section 6.1 decision-key status: it had been marked done because removal of `OBRSPW` and `ATIP` as problem classes was mistaken for renaming them as decision names. The five decision files, their runtime references, tests, and current documentation now use `OBRSP` and `active_tour_admission`. The historical `aaa8333` Henn configurations are alternatives under the same decision key. The unused `rolling_horizon` files remain; `casim check ... decisions=rolling_horizon --json` rejects `problem_class: rolling`.

The `29s-40-30-0.txt` Henn sweep compares as follows against `scripts/plots/waiting_strategy/summary.csv` (the old `scenario_henn_order_window_waiting` label denotes the order-window alternative, not a combined strategy):

| Strategy | Old → current makespan | Old → current mean tour duration | Old → current utilization | Tours | Dock mean / max, old → current |
|---|---:|---:|---:|---:|---:|
| Henn waiting | 22533.874 → 22533.874 | 563.566667 → 563.566667 | 0.764089 → 0.764089 | 30 → 30 | 0 / 0 → 15.860469 / 30 |
| Order window (5) | 22391.988 → 22438.988 | 594.142857 → 595.821429 | 0.773911 → 0.774405 | 28 → 28 | 0 / 0 → 14.790495 / 28 |

The order-window differences in makespan and total tour duration are both exactly 47 seconds; the tour count is unchanged. This is consistent with the final route changing when `OrderStreamClosed` is emitted at the last order arrival, whereas `aaa8333` flushed at event-heap exhaustion. Task 6.5 restores `at_exhaustion`; no closure rule was changed in 4.11. The dock numbers now count staging in every `State` since Task 4.9, including warehouses without a `Dock`; the historical plot was made before that change.

The Task 4.9 fixture recorded 298 events, 13 decisions, three completed tours, distance 126 and makespan 86 in each seeded run, both before and after the change. Dock fill ended at 1 with truck capacity 2 and at 3 with `capacity: null`; the full fill trace and tour records are locked in `tests/verification/dock_baseline.json`. Neither scenario config triggers a decision directly on `TruckDeparture`. The departure's emitted `PickerArrival` triggers grocery `ORSP` through `triggers_map`; IJPE has no `PickerArrival` trigger, so it triggers no decision there.

## Task 4.16 report

Commits: casim `19146d7` (files, functions, order splitting), `c9d0207` (commented-out code, logging). Each item was checked again for references before it went; the repo-review cleanup (`9a28ff3` to `1d431c7`) had already fixed the dummy-RL test, which now uses `scenarios/minimal` and asserts the value.

- **Deleted:** `pipelines/subproblems/assignment.py` (empty); `tests/verification/reference/center_before_task44.json`; `tests/data` (the committed cache `inital_domain.pkl` and the two Henn instances with their arrival files, read by no test or configuration); `tests/test_tour_model.py`; `io_helpers.load_json` and `dump_json`; `viz.app.picker_pos`; `viz.gantt_chart.plot_gantt_from_jobs`; `viz.kpis.active_picklist_order_ids`. Already gone before this task: `tests/test_henn_offline.py`, the Henn study plots and table under `scripts/plots/` (the reference `summary.csv` is in `docs/history/henn_waiting_strategy_aaa8333/`), and the tracked `tests/config/` (an untracked leftover directory is removed).
- **Order splitting removed** (`OrderSplitter`, `SimpleSplitter` with its hard-coded pallet size, `orders_provider_constraint`, the `OSBP` entry of `TAXONOMY`): no configuration used it, and the user did not decide that CASIM supports it (Section 9.1).
- **Commented-out code** in `operational_events`, `tour_manager`, `resource_manager` and `trackers` is gone.
- **Logging:** library `print` calls became module loggers: debug for pipeline synthesis, info for progress and the KPI summary, warning for an order missing from the buffer. The assertion in `TourManager.create_tour` has a message instead of `print()`. `casim run` prints the KPI summary itself (`casim.loggers.format_summary`). The CLI and the `__main__` entry points (viz launcher, component generator) keep their prints: they are command-line output.
- **CoSy components no configuration runs.** B.4, the scenarios and the tests name each component by its dotted path; these 14 concrete components appear in none of them, so nothing runs them: `batching.OrderNrFiFo`; the generated `ClarkAndWrightNN`, `ClarkAndWrightRR`, `LSBatchingNNDueDate`, `LSBatchingNNFiFo`, `LSBatchingRRDueDate`, `LSBatchingRRFiFo`, `LSBatchingSShapeDueDate`, `LSBatchingSShapeFiFo` and `SAVSchedulerLargestGap`; `picker_routing.Midpoint` and `TSPRouting`; `scheduling.ERDScheduler` and `HennLongScheduler` (the latter only in the deferred `docs/deferred/henn2012_long/`). They are kept, and on the user's decision `tests/test_component_smoke.py` now runs each of them once (about 60 s): swapped into the pipeline of `scenarios/casop_static` on Henn–Wäscher 21s-20-30-1, or on Kris `instances_9_1` where the card needs due dates or pick times, and `SAVSchedulerLargestGap` in `scenarios/henn_waiting` (one picker, pick times); each run must complete every order. 13 pass. `LSBatchingRRDueDate` needs due dates and Ratliff–Rosenthal aisle geometry, which no shipped data card has together; on the user's decision its generated component is deleted and the generator excludes the card (`EXCLUDED`), which stays in ware_ops_algos for CASOP.

Baseline comparison. Expected: none. Found: all 60 runs of the slim full tier equal (decisions, committed tours, KPIs, audit); `scenarios/minimal` keeps its trace digest. Tests: casim 133 passed with the 13 smoke tests, baseline test deselected by default; `casim check --all` reports only the untracked `scenario_dm`.

## Task 4.15 report

Scope as narrowed on 16 November 2026: the hidden rules that affect replanning and cancellation (F57, F67); the other rules are Task 4.17. Steps 4 and 5 contradicted `revision_scopes_design.md` (line 134 keeps both rules); the user decided on 17 November 2026 to follow the plan, and the design document is corrected accordingly.

Commits, in order: casim `31caeaf` (step 1), `82ac0cc` (step 3) and baseline `2267134`, `1f91d9d` (step 4) and baseline `0536416`, ware `a5b95c9` and casim `07ea7fd` (step 5), ware `d5d7a46` (`TourView.picker` takes no part in equality: comparing the live picker made `check_progress` report a done tour as changed, `test_positioning_invariants`).

| Step | Change | Expected | Found |
|---|---|---|---|
| 1 | `FlushRemainingOrders` deleted with the engine clause that let it bypass the conditions and its trigger entries in seven decisions | none | none: all 47 scenario configurations equal (no benchmark names the event) |
| 2 | `ReconsiderReason.WAIT` keeps bypassing the conditions until Task 6.4 | none | no change made |
| 3 | `NbrOrdersCondition` counts the work open to the decision: the order buffer and the orders of planned tours whose scope accepts `rebuild` (F67) | changes wherever a batch waits in a planned tour with an empty buffer | 231 of 888 runs differ (below) |
| 4 | A due-date window applies after the stream's end; `casim check` rejects a window `by: due_date` unless the data card declares `due_date` on orders | differences only with a due-date window | only `henn__release_only` (below) |
| 5 | The pickers of exposed started tours leave `snapshot.resources`; `TourView.picker` (ware) carries the picker and active-tour routing builds its router from it | none | none: all 15 configurations exposing started tours equal |

**Step 3 in detail.** Rerun: every configuration with `NbrOrdersCondition`, i.e. the henn and stochastic scenario baselines, the 660 L2 journal runs and the 192 L3 runs (L1, walk-or-wait and the L2 working-paper cells do not use it).

- Henn order window (14 of 15 configurations): before, the last orders of the stream (fewer than the window's 5) sat in a planned tour that the gate never released, so the run ended with orders unpicked: 57 of 60 (`37s-60-30-0`), 56 of 60 (`38s`, `39s`), 76 of 80 (`62s`, `63s`), 39 of 40 (`32s-40-75-0`). Every order is now completed; makespan rises by up to 13.3 % because that work is now done. The old records were `consistent`: no check reports orders still open at the end of a run (proposal below).
- Henn waiting (2 of 15): makespan −0.05 % and −0.19 %.
- L3 (116 of 192: every W1 and W2 run, W3 and W4 at depot 1 and partly 0, every centre-depot run of abc and ran): the same tours with the same distance, started earlier; the idle time before a planned tour disappears (e.g. `abc1__29s-40-30-0__centre`: idle 478 s → 0, makespan 24262 → 23784). Makespan −1.3 % on average, at most −4.6 %, never worse. The count of L3 runs below a best-known value (40 of 192 before) is not recomputed here; completion can only have fallen.
- L2 journal SAV (99 of 480): makespan −1.4 % on average (−5.3 % to +0.3 %, one cell worse); FIRST (180) unchanged.
- Stochastic scenarios: unchanged.

**Step 4 in detail.** `henn__release_only` (`release_only_example`) used a window `by: due_date` on Henn orders, which have no due dates: it showed no order until the stream closed, and then every order. Henn's data card declares no `due_date`, so the window goes. The run now plans from the first arrival: makespan 38492 → 29016, idle time 11638 → 45 s, 20 → 24 tours. It becomes `inconsistent` with 53 overlap findings: a planned draft keeps the times it was created with, while the projection of Task 4.14 moves the committed tours ahead of it when one runs late (first at 1843.75: draft 6 at 2699.75–3635.75 overlaps tour 4, projected to 1843.75–3149.75). Drafts are in no queue since Task 4.14, so nothing moves them; no other run creates drafts that outlive a delay.

Baseline comparison, full tier at the end of the task, on the slim set (`863c8db`) after the two speedups (`c02434e`, ware `c7950f4`): 60 runs in about 3 minutes; decisions, committed tours and KPIs equal in all of them. Three differ in audit counts only: `W1_100__wsrp_input_pedido_01_000__0` and `W2_100__wsrp_input_pedido_02_000__0` become `consistent`, because their epsilon overlaps existed only between two events of one timestamp, and `henn__release_only` keeps its draft overlaps with another count (re-recorded in `7a7f096`). Tests: casim 121 passed with the baseline test deselected by default (`1d431c7`); `test_l2_first_selection` now checks the FIRST rule on the 4 recorded FIRST runs instead of 480; ware 99 passed.

Open for the user:

1. Draft times after a delay (step 4): either project a resource's planned drafts after its queue, as the committed tours, or leave planned drafts out of `check_plan`'s overlap check, since a draft reserves nothing. Until then `henn__release_only` stays `inconsistent` for this reason.
2. Orders open at the end of a run: the audit has no terminal check, so runs that strand orders were recorded as `consistent` (step 3). I propose a terminal invariant: at the end of the run every order is completed or reported as open.

## Task 4.14 report

Commits, in order: ware `c98ef22` (state docstrings), `dc8bec6` (`earliest_start` on `CreateTour`, `Reschedule` and `TourView`); casim `936ce17` (derived resource state, presence, `Dispatch`, per-picker departures), `c7cbe78` (planned and actual times, queue projection, `start_times`), `00ee83c` (idle, busy and absent time from a state log; resource checks; two-picker tests), `74f2703` (projection check and overrunning tours). Baseline commit: `55bab7c`.

- **Resource state is derived.** `ResourceManager.resource_status` derives `BUSY > OFF_SHIFT > RESERVED > IDLE` from presence, the resource's current tour and its queue; nothing stores it. `ResourceManager` holds presence, a pending departure and its duration, a dispatch version per resource and the return time of an absence. `Resource.occupied`, `mark_picker_occupied`, `mark_picker_free`, `_busy_until`, `State.available_for_planning`, `state.is_break`, `state.break_duration`, `BreakCondition` and the `picker_available` argument are removed.
- **Queues hold committed tours only.** A tour enters its resource's queue when it is committed (`SCHEDULED`), ordered by planned start; a planned draft reserves nothing. `TourManager.advance` maintains the queues and validates a transition before changing anything.
- **One dispatch mechanism.** `Dispatch(picker_id, version, priority)` replaces `PickerTourQuery` and `SequencingDone`: it makes the first queued tour `pending` when the resource is present, free, without a pending departure, and the tour's earliest start is reached. `apply` emits one `Dispatch` per resource whose queue changed; `TourEnd`, `PickerArrival` and `PickerDeparture` emit one, and a `Dispatch` before the first tour's earliest start schedules another at that time; a `Dispatch` with an older version is skipped.
- **Presence.** `PickerArrival(time, picker_id)` and `PickerDeparture(time, picker_id, duration)` replace `ShiftStart`, `BreakStart` and `BreakEnd`. A departure takes effect at once for a free resource and after the current tour otherwise; with a duration it schedules the return, without one the resource is absent until the next arrival. `TruckDeparture` no longer emits `PickerArrival`. The IJPE configuration and hooks use the two events.
- **Planned and actual times.** `TourPlanningState` has `earliest_start`, `planned_start`, `planned_end`, `actual_start`, `actual_end` and `dispatched_at` instead of `start_time` and `end_time_planned`. `State.project_queue` moves every queued tour to the later of its earliest start and the time its predecessor (or the resource) is free, keeping its planned duration and the queue order; it runs at `Dispatch`, `TourStart`, `TourEnd` and admission. A running tour's planned end is its forecast; a done tour's is its actual end.
- **Binding or forecast start times.** The decision key `start_times` (`binding`, default, or `forecast`) decides whether a committed start is an earliest start (a timetable) or only a forecast (a sequence); `DecisionEngine.bind_start_times` sets `earliest_start` accordingly. All shipped configurations use `binding`, the behaviour before the task.
- **Idle, busy and absent time.** The tracker logs every change of derived state; idle, busy and absent time per resource are read from that log and sum to the horizon. Busy time includes setup.
- **Resource checks** (Section 7.2), after the last event of every timestamp, counted by the audit as `check_resources`: `concurrent_tours`, `early_dispatch`, `absent_dispatch`, `missed_dispatch`, `projection`, and `time_balance` at the end of the run. The projection check skips a resource whose running tour has passed its forecast end; `TourEnd` projects the queue again.
- **Tests:** `tests/verification/test_dispatch_two_pickers.py` (11): rescheduling earlier, moving a tour to another idle picker, an earlier commit while the picker waits for a later tour, a break with one picker busy and one idle, a shift end keeping queued tours, departure and arrival at one timestamp, an early tour under binding and forecast times, a late tour, and a tour overrunning its estimate.

Idle time before and after (one picker each; seconds):

| Configuration | Idle before | Idle after | Busy after | Absent after |
|---|---:|---:|---:|---:|
| `henn__default` | 4680.0 | 45.4 | 30578.0 | 45.4 |
| `henn__29s-40-30-0__henn_waiting` | 5220.0 | 113.4 | 22307.0 | 113.4 |
| `henn__29s-40-30-0__order_window` | 4860.0 | 602.6 | 21723.0 | 113.4 |
| `henn__release_only` | none | 11638.2 | 26808.0 | 45.4 |
| `stochastic__henn_waiting` | 4.0 | 6.0 | 12.0 | 0.0 |
| `stochastic__analytic_waiting` | 0.0 | 11.1 | 76.0 | 0.0 |
| `walk__default` | 0.0 | 0.0 | 167.0 | 0.0 |
| `walk__wait_4` | 0.0 | 12.0 | 135.0 | 0.0 |
| `walk__radar_wait_1` | 1548.6 | 1562.1 | 27300.0 | 0.0 |

The old idle time counted setup (180 s per Henn tour) as idle and missed some waiting before a binding start; the new idle time is the time present without a current tour. Absent time is the span before the pickers arrive at the first order arrival.

Do decisions select a different set of resources now that a draft no longer reserves its picker? In no recorded run: decisions and committed tours are equal in all 1,654 records.

Baseline comparison. Expected differences: idle time (redefined); busy and absent time (new KPIs); the new audit kind `check_resources`; event counts and trace digests (`Dispatch` replaces `PickerTourQuery` and `SequencingDone`, `PickerIdle` at arrival); runs turning `consistent` where the old reservation model produced overlap findings. Decisions, committed tours and every other KPI unchanged. Found, full tier over all 47 scenario configurations and 415 L1, 1,000 L2 and 192 L3 runs: decisions, committed tours and all other KPIs equal everywhere; 133 runs turn from `inconsistent` to `consistent`; no run has a resource finding. The first pass reported 47 Kris EDD L1 cells turning `inconsistent` with `projection` findings: a tour there runs 2,880 s past its estimated processing time (pre-existing; committed tours equal), so its forecast end lies in the past until it ends. The check now skips such a resource (`74f2703`, with a regression test); the Kris EDD cells were rerun and are `consistent`. The dock fixture (`tests/verification/dock_baseline.json`) is recorded again: one OBRSP decision fewer per run, at the second truck departure, which no longer emits `PickerArrival`. Tests: casim 591 passed, ware 99 passed.

Open for the user: Section 9.1 item 8, whether `Resource.available` and `Resource.occupied` are removed from ware's shared domain model; casim no longer uses either. The overrun of the Kris EDD tours (actual duration above the estimated processing time) is pre-existing and not investigated here.

## Task 4.13 report

Commits, in order: ware `9d1ec4f`, casim `7e91422` (F62); ware `dc10b50`, casim `ea1f13c` (TourProgress, admission); ware `8b3da8b`, casim `3886702` (Violation); ware `f4b4e16`, casim `7b8d458` (stage typing, `build_router`, tour filters, single-picker waiting). Baseline commits: `f6a51ea` (henn 61s-80-30-0), `d714c58` (L3 after F68), `2ad6e22` (L2 after F68 and F62).

- **F62, one rule for released work.** `validation.check` and casim's live check accept the same free orders for `CreateTour` and `ReplaceSuffix`: the order buffer, buffered pick lists and the orders of tours cancelled in the same command set. Applying a command that takes orders from a buffered pick list dissolves it; its other orders return to the order buffer. Tests: `test_validation_released_work.py` (ware), `test_apply_released_work.py` (casim).
- **TourProgress.** The execution facts of a started tour (active orders, cursor, pinned prefix, remaining route, routing origin, occupied boxes, pick in progress, holding) form one frozen `TourProgress`; `TourView.progress` is set for started and done tours (done tours keep it for the executed-prefix invariant) and `None` before.
- **Admission.** `AdmissionInput` holds started `TourView`s and `input_closed`; `AdmissionSolution` gains `return_tour_ids`. `AdmissionTour`, `to_admission_tour`, `ActiveTourBatch` and `ActiveTourBatchingSolution` are deleted. Both admission algorithms return a holding tour without orders once the input is closed; `AdmittedTourBatch` only assembles plain batches (`batch_id` = tour ID) and passes the decision on.
- **Violation.** `InvariantViolation` merges into `Violation`; `ViolationCode` gains schedule, route, capacity, overlap, time, status and terminal. The audit records `code.value`, the same strings as before.
- **CoSy stages.** `AbstractActiveTourRouting` (`batching_sol = AdmittedTourBatch`) replaces the mixin; depot routing takes `AbstractBatching` only; `AbstractBatchProvider` is removed; active-tour decisions end in `ActiveTourRoutingPlanCandidate`. `tests/test_cosy_stage_typing.py`: FiFo, `AdmittedTourBatch`, `SShape` and `ReroutableSShape` synthesize exactly the two valid families.
- **Enums and one filter.** `pipelines/tour_filters.py` (`admitting_tours`, `rebuildable_tours`) replaces the string comparisons of status and accepts in `IncumbentProvider`, `PickListProvider`, the admission nodes, `AdmittedTourBatch` and active-tour routing. The value `"revise"` the old comparisons named does not exist in `Accepts`.
- **Single picker.** The cards of `HennWaiting` and `AnalyticStochasticWaiting` require `n_resources = 1`; the three data cards using them declare it; the two waiting nodes fail with a clear error when they see another number of pickers. The CASOP applicability test is unchanged.
- **One router builder.** `pipelines/subproblems/routers.py`: `build_router(router_class, layout, pickers, **options)` builds every router (every routing node, `IncumbentBatchingNode`, `SAVScheduler`; configured batching uses `router_arguments`). Ratliff-Rosenthal now also receives `node_list`, which it does not use; the L1 RR rows are equal.
- **Kept:** `DynamicInfo`, `SimWarehouseDomain`, `DecisionContext`, `DecisionMemory`.
- **For the user to check:** ware's import-boundary test now ignores imports under `if TYPE_CHECKING:`, because `AdmissionInput.tours` names the planning `TourView` in an annotation; there is no runtime dependency of `algorithms` on `planning`.

Type counts (top-level classes in ware `planning/`, `algorithm_interfaces.py`, casim `problem_based_template.py`, the routing, admission, batching, scheduling and waiting nodes, `decision_engine/`, `state/apply.py`): 117 before, 115 after. `planning` 19 → 19 (+`TourProgress`, −`InvariantViolation`); `algorithm_interfaces` 28 → 25 (−`AdmissionTour`, −`ActiveTourBatch`, −`ActiveTourBatchingSolution`); `problem_based_template` 27 → 29 (−`AbstractBatchProvider`, +`_RoutingStage`, +`AbstractActiveTourRouting`, +`ActiveTourRoutingPlanCandidate`); routing nodes 12 → 11 (−`_ActiveTourRoutingNode`).

Baseline comparison. Expected differences: none, except configurations that failed on the F62 rejection. Found: both `henn__61s-80-30-0` configurations and 246 L2 SAV replications become runnable and `consistent`; every L1, L2 and L3 sweep has one candidate pipeline, so no recorded run could change, and none did. Full tier at casim `7b8d458`: all 47 scenario configurations, 415 L1, 1,000 L2 and 192 L3 runs equal (nine C/120/75 SAV runs timed out under load and are equal when rerun with three workers). Tests: casim 580 passed, ware 99 passed; `casim check --all` reports only the untracked `scenario_dm`. The F62 effect on the SAV cells is in `docs/literature_benchmarks.md` ("F62 and the L2 SAV cells"): SAV behaves like FIRST and does not explain the high-load gap.

## Tasks through Phase 3b

| Task | Status | Evidence and remaining work |
|---|---|---|
| 0.1 | done | Separate base checkout and `docs/revision_scopes_baseline.md`, Section 7.4 KPIs reproduced. |
| 0.2 | done | `1f1fa9a`, `af1a942`; `tests/test_solver_cache_isolation.py`, one-process full suites. |
| 1.1 | done | ware:`e66d4a8`, `3ce009d`; operational plan types deleted and `TourPlanningState` moved. |
| 1.2 | done | ware:`02d7885`; unused pipeline DAG executor deleted. |
| 1.3 | done | ware:`3faf07a`; mapper capability path removed, `AlgorithmCard.capabilities` retained. |
| 1.4 | done | ware:`8405404`; one mapper requirement path and mapper tests. |
| 1.5 | done | `adac710`; copied design document removed. |
| 2.1 | done | ware:`c0ed627`; `planning/lifecycle.py`, `test_lifecycle_exhaustive.py`. |
| 2.2 | done | `d1f1447`; `TourManager.advance`, AST and Hypothesis lifecycle tests. |
| 2.3 | moved to a later task | `b6eb7e6` moved `TourPlanningState`; user directed `positioning_mode` and `positioning_hold` to stay until 4.4. |
| 2.4 | moved to a later task | ware:`4af567c`, `5e190b1`; typed buffers/context complete; positioning fields remain until 4.4 by user direction. |
| 2.5 | done | `27826f4`, `d30b895`; typed `sim_domain.py` and loader constructors; ware:`54e7217` restores static domain models. |
| 2.6 | done | ware:`7fc4660`, `9a5002b`; admission component constructs `AdmissionTour`; user approved `TourView.active_order_ids` and removing unused `AdmissionTour.route_version`. |
| 2.7 | done | `d1f1447`; generic `simulation_engine/state_adapter.py`, manager methods and sorted orders. |
| 2.8 | done | `dedb385`; `OperationsView` has only the two operational facts. |
| 2.9 | done | ware:`1d95b16`; exhaustive lifecycle and hand-built invariant tests. User clarified duplicate detection in `check_plan` and loss detection in `check_progress`; full-run fixture is Phase 5. |
| 3.1 | done | `99baea8` and ware:`f652088`; old ranker compatibility shim removed in Phase 3c. |
| 3.2 | done | ware:`7d3a75b`, `4af567c`; `planning/scope.py`, exhaustive scope test and schema boundary. |
| 3.3 | done | `b6be5a5`, `d1f1447`; `commitment_policies.py`, `test_commitment_policies.py`. Ready-time meaning changes in 4.8. |
| 3.4 | done | ware:`7731b2a`; `decision_features.py` and `test_decision_features.py`. |
| 3.5 | done | ware:`54e7217` restores taxonomy byte-for-byte to `34bd157`; `7c388af`, `test_taxonomy_variant.py` hold online classes. |
| 3.6 | moved to a later task | ware:`0f64f90` defines six plan commands; temporary casim-internal `RecallPositioning` stays until 4.4 by user direction. |
| 3.7 | done | ware:`f652088`; `planning/conversion.py`, `tests/test_planning_conversion.py`. |
| 3.8 | done | ware:`9040494`, `54e7217`; pure `planning/validation.py`, `test_validation.py`. |
| 3.9 | done | `99baea8`, `d1f1447`; `decision_engine/solution_ranker.py`, setup checks for multiple batching candidates. |
| 3.10 | done | `99baea8`; `decision_engine/interfaces.py`, stable candidate names tested by solver tests. |
| 3.11 | done | `d1f1447`; named solver/ranker/policy maps and typed `DecisionResult`, import boundary test. |
| 3.12 | done | `d1f1447`; `config/schema.py`, `config/data_card.py`, committed `schemas/decision.schema.json`. |
| 3.13 | done | `d1f1447`; `setup.resolve`, routing-only rejection, write-set tests in `tests/test_revision_scope_contract.py`. |
| 3.14 | done | `d1f1447`; constructor collaborators and multi-decision trigger ordering in `simulation_engine.py`. |
| 3.15 | moved to a later task | `d1f1447` has two-phase `state/apply.py`; the temporary positioning recall apply path stays through 4.4 by user direction. |
| 3.16 | done | `d1f1447`; per-decision waiting/reconsideration in `State` and `DecisionMemory`. |
| 3.17 | moved to a later task | `d1f1447` adapts `BaseEnv`, scripts, CLI and schema; Phase 5 ports the base scenarios and Phase 6 ports IJPE and extension configs for the all-config `casim check` gate. |
| 3b.1 | done | `6836b50`; `scripts/oobp_reference.py`, `data/oobp_reference.csv`. Spreadsheet has no per-instance ILS field, per user clarification. |
| 3b.2 | done | `6836b50`; `docs/oobp_assumptions.md` maps each paper assumption and records conflicts. |
| 3b.3 | done | `6836b50`; comparison protocol in `docs/oobp_assumptions.md`, with runs explicitly assigned to 6.0. |

## Appendix A findings

| Finding | Status | Evidence or later task |
|---|---|---|
| F1 | moved to a later task | 5.1–5.3 Henn port and causal comparison. |
| F2 | done | `7c388af`, `d1f1447`; declared class and `variant`. |
| F3 | moved to a later task | Setup name rules removed in `d1f1447`; final algorithm-name sweep is 4.7. |
| F4 | done | `1a1c7ab`, ware:`a05bf88`; whole-schedule waiting stage. |
| F5 | moved to a later task | 4.3 restores the analytic precondition and card. |
| F6 | done | `d1f1447`; ranker and candidate check. |
| F7 | moved to a later task | Henn router injected in `0111acc`; incumbent router and cart are 4.5. |
| F8 | done | Phase 1 commits plus typed `DynamicInfo` in Phase 2. |
| F9 | moved to a later task | 4.4 removes positioning strings/recall/shortest path; 4.7 checks casim. |
| F10 | done | `1a1c7ab`, `d1f1447`; waiting stage and commitment policies. |
| F11 | done | ware:`a05bf88`; waiting retimes the picker schedule. |
| F12 | done | `d1f1447`; `ReconsiderReason`. |
| F13 | done | `d1f1447`; tuple trigger map and precedence. |
| F14 | done | ware:`c0ed627`; trimmed lifecycle. |
| F15 | done | `b6eb7e6`; simulator tour model back in casim. |
| F16 | moved to a later task | Flat `TourView`/admission bridge done; positioning fields intentionally stay until 4.4. |
| F17 | done | ware:`4af567c`, `5e190b1`, `dedb385`; typed snapshot and operations. |
| F18 | done | ware:`4af567c`; no active-candidate duplicate. |
| F19 | done | `99baea8`; solver protocol. |
| F20 | done | `d1f1447`; simulation constructor and repair removal. |
| F21 | done | `d1f1447`; data-card loading in casim config. |
| F22 | done | `d1f1447`; manager-backed two-phase apply. |
| F23 | done | `d1f1447`; tour status writer test. |
| F24 | done | ware:`0f64f90`; expected fields removed. |
| F25 | done | `1f1fa9a`, `af1a942`; two-solver regression and pinned cosy-luigi. |
| F26 | done | `d1f1447`; StateAdapter sorts `(order_date, order_id)`. |
| F27 | moved to a later task | Per-decision memory/policy done; Henn config/behavior comparison is 5.1–5.3. |
| F28 | moved to a later task | 4.8 time model. |
| F29 | moved to a later task | Copied design deleted in 1.5; baseline doc rewritten in 7.1. |
| F30 | done | ware:`8405404`; mapper tests. |
| F31 | moved to a later task | 6.1 labels synthetic rolling due dates. |
| F32 | moved to a later task | 4.4 positioning package. |
| F33 | done | `7c388af`; O2 rejection and exhaustive variant test. |
| F34 | done | ware:`54e7217`; homogeneous scheduler regression tests. |
| F35 | done | ware:`0f64f90`, `d1f1447`; waiting/considered data in result/memory. |
| F36 | done | ware:`0f64f90`; one ReplaceSuffix and merged Reschedule. |
| F37 | moved to a later task | Release gates removed; IJPE README note is 6.3. |
| F38 | moved to a later task | Lifecycle/planning comparisons converted in Phase 2–3; final algorithm/positioning sweep is 4.4–4.7. |
| F39 | done | `d1f1447`; routing-only setup rejection, no RoutingDone event. |
| F40 | done | ware:`54e7217`; `git diff 34bd157 -- src/ware_ops_algos/taxonomy/taxonomy.py` is empty. |
| F41 | done | `27826f4`, `d30b895`; simulator domain moved back and re-export removed. |
| F42 | done | Task 4.9, typed dock and exact synthetic trace comparison. |
| F43 | done | ware:`20cb91e`; casim:`640b5ce`; one parameterized waiting card. |
| F44 | done | ware:`0d47fe0`; casim:`c94c410`; residual admission after completed picks. |
| F45 | done | Task 4.11 follow-ups; waiting aliases move to 4.12 and dispatch checks to 4.14 as directed. |
| F46 | awaiting user decision | `spt_scheduling.yaml` due-date requirement left untouched in Task 4.12. |
| F47–F49 | done, with named decisions pending | Task 4.12 card applicability, configured components, module layout and mapper docstrings; the four named decisions above remain untouched. |
| F50 | moved to a later task | Task 4.13 type consolidation. |
| F51 | moved to a later task | Task 6.5, timing of stream closure. |
| F52 | moved to a later task | Tasks 4.11 and 4.13, admission choice and CoSy stage typing. |
| F53 | moved to a later task | Task 5.0, one stage registry. |
| F54 | done | Task 4.11 removed dead sequencing and tracked Henn output; `rolling_horizon` is retained for Task 6.1 and rejected by `casim check`. |
| F55 | moved to a later task | Task 4.14, dispatch and resource availability. |
| F56 | moved to a later task | Task 4.11 restored the alternative Henn sweep and corrected stale references; Task 5.0 checks all configurations. |
| F57 | moved to a later task | Task 4.15, hidden decision rules. |
| F58–F59 | moved to a later task | Phase 6b scenario structure and IJPE study. |

## Section 5 rules

| Rule | Status | Evidence or later task |
|---|---|---|
| 5.1 static domain ownership | done | ware static `domain_models`; `tests/test_import_boundaries.py`. |
| 5.1 algorithm inputs independent of planning | done | ware algorithm interfaces and import boundary test. |
| 5.1 planning owns lifecycle, scope, commands and invariants | done | ware `planning/`, exhaustive/validation tests. |
| 5.1 CASOP taxonomy and mapper unchanged from base | done | ware taxonomy exact base diff; mapper tests and user approval for 1.3–1.4. |
| 5.1 casim owns both engines and live state | done | casim engine/state packages and import boundary test. |
| 5.1 only conditions read `context.cause` | done | `tests/verification/test_import_boundaries.py`. |
| 5.1 every module respects forbidden import directions | done | Both repositories' AST import tests. |
| 5.2 ware algorithm interfaces | moved to a later task | Waiting/admission/active-tour inputs live in `algorithm_interfaces.py`; positioning mode removal is 4.4. |
| 5.2 ware waiting base and policies | done | ware:`2676b78`, `a05bf88`, `test_subproblem_bases.py`. |
| 5.2 ware admission base | done | ware:`2676b78`, `test_subproblem_bases.py`. |
| 5.2 ware positioning package | moved to a later task | 4.0/4.4; current flat module is temporary. |
| 5.2 ware optional depot router | moved to a later task | 4.4 first tests ReroutableSShapeRouting and adds a router only if needed. |
| 5.2 ware incumbent batching injection | moved to a later task | 4.5. |
| 5.2 ware planning/lifecycle/models/scope/features | done | `planning/`, lifecycle/scope/features tests. |
| 5.2 ware planning commands/conversion/validation/invariants | done | `planning/`, command/conversion/validation/invariant tests. |
| 5.2 ware taxonomy restoration | done | Exact diff against `34bd157` is empty. |
| 5.2 casim sim_domain/tour_model | moved to a later task | Domain and tour state moved back; positioning fields stay until 4.4. |
| 5.2 casim decision engine/ranker/interfaces/policies | done | `decision_engine/`, import and policy tests. |
| 5.2 casim decision process events | done | `SequencingDone` and `PickListCreated`; `RoutingDone` deleted. |
| 5.2 casim state managers/apply/views | moved to a later task | Generic apply/views in place; temporary recall apply path stays until 4.4. |
| 5.2 casim simulation engine/adapter/conditions | done | `simulation_engine/`, adapter and decision tests. |
| 5.2 casim schema/data card/setup/CLI | done | `config/`, `setup.py`, `cli.py`, generated schema. |
| 5.2 casim CoSy taxonomy/runner/waiting/admission | done | `pipelines/`, online variant and waiting tests. |
| 5.2 casim positioning component | moved to a later task | 4.4. |
| 5.2 listed removals | moved to a later task | All listed non-positioning modules are gone; recall/positioning paths await 4.4 by user direction. |

## Section 6 names

| Rule | Status | Evidence or later task |
|---|---|---|
| 6.1 selection/state information | done | ware `StateInformation`, casim `DecisionConfig.state_information`. |
| 6.1 commitment configuration | done | `config/schema.py`, `commitment_policies.py`. |
| 6.1 planned/scheduled lifecycle | done | ware `planning/lifecycle.py`, exhaustive tests. |
| 6.1 ReleaseTour/CommitTour | done | ware `planning/commands.py`. |
| 6.1 InsertOrders/ReplaceSuffix | done | ware `planning/commands.py`, conversion/validation tests. |
| 6.1 Reassign/Reschedule | done | ware `planning/commands.py`. |
| 6.1 membership_ops removal | done | ware `planning/decision_features.py`. |
| 6.1 waiting service-time field | done | ware `WaitingInput`, `HennWaiting` injected router. |
| 6.1 OBRSPW/ATIP config keys | done | Task 4.11 renamed all five decision files and their runtime, test, and current documentation references. Previously marked done by confusing problem-class removal with decision-key renaming. |
| 6.1 reconsider cause enum | done | casim `sim_domain.py`, `operational_events.py`. |
| 6.1 ranker move | done | casim `decision_engine/solution_ranker.py`; Phase 3c removes old shim. |
| 6.1 command/conversion moves | done | ware `planning/`; no active imports from old paths. |
| 6.1 lifecycle/planning split | done | ware `planning/`, casim `sim_domain.py`. |
| 6.1 simulator domain/tour moves | done | casim `sim_domain.py`, `tour_model.py`. |
| 6.1 DecisionSolver move | done | casim `decision_engine/interfaces.py`. |
| 6.2 protected names | done | Named API remains in casim and ware; contract tests cover `DecisionEngine.on_trigger`, `SimulationEngine.run/step` and planning enums. |
| 6.3 TourStates.ASSIGNED | done | ware `planning/lifecycle.py`. |
| 6.3 Accepts.REVISE | done | ware `planning/lifecycle.py`. |
| 6.3 TimeAttribute.ORDER_DATE | done | ware `planning/lifecycle.py`. |
| 6.3 PickListStates | done | ware `planning/lifecycle.py`. |
| 6.3 RevisionRights/PlanEntry/OperationalPlan | done | ware Phase 1 deletion. |
| 6.3 on_violation/waiting/release_gates/solver.output keys | done | Active schema rejects them; historical generated outputs are inventoried below. |
| 6.3 command expected fields | done | ware `test_commands.py`. |
| 6.3 Wait/RecordConsidered | done | ware commands and casim decision result. |
| 6.3 RecallPositioning | moved to a later task | Temporary casim-internal command retained through 4.4 by user direction. |
| 6.3 PlanView.active_candidate_ids | done | ware `PlanView`. |
| 6.3 loose DynamicInfo fields | done | casim `DynamicInfo(context, plan, memory)`. |
| 6.3 positioning_mode/positioning_hold | moved to a later task | User directed removal in 4.4 with center return. |
| 6.3 decision_engine/waiting.py | done | Deleted; CoSy waiting stage restored. |
| 6.3 RoutingDone | done | Deleted; setup rejects routing-only output. |

## Section 7.1–7.2 verification rules

| Rule | Status | Evidence or later task |
|---|---|---|
| 7.1 transitions forward/reachable/terminal | done | ware `test_lifecycle_exhaustive.py`. |
| 7.1 PHYSICAL monotonicity | done | ware `test_lifecycle_exhaustive.py`. |
| 7.1 every scope × exposed-status set | done | ware `test_lifecycle_exhaustive.py`. |
| 7.1 accepts never increases with status | done | ware `test_lifecycle_exhaustive.py`. |
| 7.1 variant total-or-named-rejection/ordered | done | casim `test_taxonomy_variant.py`. |
| 7.1 every pair of decision write sets | done | casim `test_revision_scope_contract.py`. |
| 7.2 `check_plan` order membership | done | ware `planning/invariants.py`, `test_plan_invariants.py`; user clarified no unseen-order completeness check. |
| 7.2 `check_plan` picker tour nonoverlap | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 `check_plan` committed tour assignment/times | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 `check_plan` started route and occupied boxes | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 `check_progress` legal status changes | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 `check_progress` executed prefix and completed picks | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 `check_progress` terminal tours immutable | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 `check_progress` time and order conservation | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 fixture after every step and event of every scenario | moved to a later task | 5.2. |
| 7.2 Hypothesis instances/configurations | moved to a later task | 5.2. |
| 7.2 command/event RuleBasedStateMachine | moved to a later task | 5.2 extends the existing TourManager machine. |

## Untouched tracked scripts and docs

The `scripts/` directory does not yet exist; `scripts/compare_with_base.py` is assigned to Phase 5. Generated image/CSV/TeX files under `scenarios/scenario_henn_online/scripts/plots/` are historical plot outputs, not executable scripts. All other tracked files under `scenarios/*/scripts/` and `docs/` that Phases 2–3b did not change are listed here. “Holds” means the file remains useful as a historical result or source and is not an active configuration.

| File | Verdict |
|---|---|
| `docs/revision_scopes_baseline.md` | Pointer to `docs/baseline.md` after B.9. |
| `docs/waiting_intervention_foundation.md` | Removed in B.9 because no file links to it. |
| `scenarios/scenario_grocery_retailer/scripts/__init__.py` | Still imports as a package marker. |
| `scenarios/scenario_grocery_retailer/scripts/experiment_plots.py` | Compiles; figure helpers hold until given tracker data. |
| `scenarios/scenario_henn_online/scripts/make_waiting_plots.py` | Compiles; holds for recorded multirun output and writes plots when executed. |
| `scenarios/scenario_ijpe/scripts/make_results_domain_validation.py` | Compiles; Phase 3c fixes its deleted `build_data_loader` import. Full execution awaits IJPE data validation in 6.3. |
| `scenarios/scenario_ijpe/scripts/make_results_scenarios_multirun.py` | Compiles; holds for recorded multirun output. |
| `scenarios/scenario_ijpe/scripts/make_summary_scenarios_multirun.py` | Compiles; holds for recorded multirun output. |
| `scenarios/scenario_henn_online/scripts/plots/waiting_strategy/scaling_makespan_orders.png` | Holds historical plot. |
| `scenarios/scenario_henn_online/scripts/plots/waiting_strategy/scatter_avg_utilization.png` | Holds historical plot. |
| `scenarios/scenario_henn_online/scripts/plots/waiting_strategy/scatter_final_makespan.png` | Holds historical plot. |
| `scenarios/scenario_henn_online/scripts/plots/waiting_strategy/summary.csv` | Holds historical plot source. |
| `scenarios/scenario_henn_online/scripts/plots/waiting_strategy/summary_table.tex` | Holds historical plot source. |

In ware_ops_algos there is no tracked `scripts/` or `scenarios/*/scripts/` tree. Its untouched tracked documentation is:

| File | Verdict |
|---|---|
| `docs/Makefile` | Still holds as the Sphinx build entry point. |
| `docs/make.bat` | Still holds as the Windows Sphinx build entry point. |
| `docs/source/_static/favicon.png` | Still holds as a documentation asset. |
| `docs/source/_static/favicon.svg` | Still holds as a documentation asset. |
| `docs/source/algorithms.rst` | Still holds; API directives name present algorithm modules. |
| `docs/source/authors.rst` | Still holds as author metadata. |
| `docs/source/conf.py` | Still holds as Sphinx configuration; documentation was not rebuilt in this audit. |
| `docs/source/examples.rst` | Still holds; links the current example notebook. |
| `docs/source/extending.rst` | Still holds; references the present `examples/custom_batching.py`. |
| `docs/source/getting_started.nblink` | Still holds; links `examples/getting_started.ipynb`. |
| `docs/source/index.rst` | Still holds as the documentation table of contents. |

Untracked user files remain untouched: casim `.env`, `.hydra/`, `.worktrees/`, `docs/research/`, the two user-owned revision markdown files, `docs/technical_report/`, `docs/visualizations/`, `scenarios/scenario_dm/`, `scenarios/scenario_ijpe/generator/private/`, `tmp/`, `tmp_output/`; ware `.worktrees/` and the three `examples/stochastic_waiting_*.py` files. The ignored ware `docs/build/` contains generated documentation, including an old loader import, and is left untouched. The ignored `domain_models/operational/__pycache__/` is the only remaining physical content at that removed package path. The ware `pyproject.toml` edit predates this audit and is excluded from its commits.

## Phase 3c checks

`ruff check --select F401,F811,F841` passed over casim source/tests/scenario scripts and ware source/tests. `vulture --min-confidence 80` reported no high-confidence dead code in the changed packages. Public imports in ware `__init__.py` files are intentionally re-exports. The cleanup commits are ware:`102dee4` and casim `75fa182`. The original Henn archive and per-instance comparison are in `scenario_henn_online/data/oobp/` and `docs/oobp_source_comparison.md` (`6eaf5ee`); all 194 committed source blobs were checked byte-for-byte against the ZIP. The archive loader gap is recorded there for Task 6.0.

The pre-existing grocery/IJPE hooks attach an optional dock manager to `State` after construction. Phase 3c preserves that behavior and its existing optional dock lookup; changing the hook contract would be a separate design decision during the IJPE work in 6.3.
