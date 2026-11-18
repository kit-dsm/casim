# IJPE revision plan (PROECO-D-26-03325)

Revision due 27 December 2026. This plan answers the reviews of 4 August (reviewer 1, "Rev1") and 16 October 2026 (reviewer 2, "Rev2"). Phases are numbered P0 to P5 so that they cannot be confused with the reviewers.

The plan is organised by what the response letter has to show, and every task names the comment it answers. Tasks are marked **must** (the response is not credible without it), **should** (strengthens a claim the reviewers question) or **could** (only if time remains after the checkpoint of 6 December). A task that answers no comment and is not a prerequisite of one is not part of this plan.

## 1. Ground rules

The WMS data and everything derived from it at order, line or picker level are confidential. They live below `scenarios/ijpe/data/` (and the original export wherever the user places it), are listed in `.gitignore`, and are never committed, pushed, pasted into an issue or written into a committed document. Committed documents contain aggregates only. A figure or table that shows individual orders or pickers is written to an ignored directory and goes into the paper only by the user. The manuscript and the reviews stay local.

Calibration and validation use different data. Leave-one-day-out is the primary estimate; for figures, parameters are estimated on Monday to Wednesday and reported on Thursday to Saturday, with Saturday reported separately. No number of the revision is validated on the data it was calibrated on (Rev2).

Every assumption of the model is stated as a hypothesis with a test on the data, and its result is recorded whether it confirms or refutes the assumption. A result that contradicts the submitted paper is reported, not explained away.

Do not change ware_ops_pipes. Commit each task separately and update `docs/ijpe/status.md` after every commit.

## 2. Code base and core freeze

The revision runs on the branch `codex/ijpe-revision`, created once from `codex/revision-scopes` at a commit the user names (the base). It is not rebased onto later core work.

The core is frozen for the revision. A change to `src/casim` or ware_ops_algos is made on the revision branch only if a task of this plan needs it, and then with a test, with the baselines of all other scenarios identical, and with the task named in the commit. The deferred core work in `docs/development.md` waits until after 27 December. The core changes this plan needs are listed in Section 4 and are small.

The submitted results are reproduced from casim `main` (identical to `ijpe`), tagged `ijpe-submitted`, and its outputs are kept in an ignored directory as the reference. The port does not imitate defects of the old code; every difference between reference and port is classified in P2.2.

## 3. What the submitted model assumes

Read from `main`, `scenarios/scenario_ijpe/`, not from the manuscript. Each item is a candidate cause of a reviewer comment.

- Historic processing time of an order is the last line's end minus the first line's start (`make_results_domain_validation.py`). Modelled processing time is the U-shape route of the order at 2,306 mm/s plus 19 s per line; the 215 s setup that the simulation charges is not in the validation's model, and quantity, weight and volume do not enter.
- The route uses `GreedyItemAssignment`, not the recorded pick locations. A tour holds one order. The loader builds 25 identical pickers.
- Due times are the end of the hourly bucket in which the order was completed, so the historical schedule meets them by construction (Rev2).
- One transport unit is 1.6 orders (`palett_te_factor`); the warehouse's workbook defines 2.6.
- The pipeline repository has one batching rule (DueDate), one router (UShape) and three schedulers (EDD, SPT, LPT) (Rev2: "one routing heuristic and three scheduling rules").
- `TruckDisruption.handle` (Scenarios 1 and 2) cancels planned and scheduled tours itself, before any decision runs. It selects them in the order of the tour dictionary by `earliest_due_date >= from_time` until the TU volume is reached, not by store, and searches the pick-list buffer only if no tour was cancelled. The affected orders are therefore not the stores of the warehouse's list (Rev2: "two stores, 50 orders").
- Scenario 3 injects orders generated from a calibration file with a seed, not the warehouse's list. Scenario 4 rewrites future `OrderArrival` events at setup time.
- Section 4.1 of the manuscript describes fixed zones and several pickers on one store order; the model has neither.

## 4. Core changes this plan needs

Each is a prerequisite of a reviewer answer and small on the current code.

| Change | Why | Where |
|---|---|---|
| C1. Disruptions as exogenous information | `TruckDisruption` cancels tours outside a decision, so a policy without cancellation or without replanning cannot be configured (Rev2, counterfactual policies). The event may only change exogenous state: due dates of the affected orders, new orders. Cancellation happens only through the replanning decision's `scope`. This also closes the open point of the design document. | `events/operational_events.py`, IJPE source |
| C2. Affected-order flag | Full KPIs for affected and unaffected orders (Rev2). The source marks each order it changes or adds; `orders.csv` gets the column. | `run.py`, source protocol |
| C3. Decision log | `decisions.json` keeps only pipeline counts. `SolutionRanker.select_best` already computes every candidate's score and `DecisionResult` carries `solve_seconds`; both are written per decision: time, decision name, trigger, candidates with scores, chosen pipeline, solve and conversion time (Rev2: selected pipelines, changes after disruptions; computation times). | `decision_engine.py`, `trackers.py`, `run.py` |
| C4. Pick table and service measures | `on_pick_end` exists in the tracker but no table is written. Write `picks.csv`; compute on-time rate, total tardiness, maximum lateness, completed and open orders per group, makespan per day, busy, idle and absent time per picker relative to presence, departures with delay and orders carried. | `run.py`, scenario `metrics.py` |
| C5. Dock and departures | The two open results of Task 4.9: a `TruckDeparture` with `capacity: null` releases no pallets; truck capacity counts orders while staging counts pallets. Both can produce the departure counts of Table 6 (Rev2). | `DockManager`, `TruckDeparture` |

## 5. Phase P0: data audit (done)

`docs/ijpe/data_audit.md`. Findings the later phases build on: six days, Saturday 11 April and Monday 13 to Friday 17 April 2026 (Rev1: data period; Rev2: day labels of Figures 11 to 14); 1,525 orders, 1,422 after dropping aisle 10 and one article, 1,419 after three zero-span orders (Rev2: n = 1419); an order is one zone order of a store-day, picked by one picker in 99.5 % of cases, so zones and parallel pickers exist at the store-day level (Rev2: unit of analysis); observed busy share 62 to 70 % (Rev2: 47.75 %); TU = 2.6 × orders; Scenario 2 is 38 orders of 37 stores; Scenario 3 differs from the warehouse's list; the export has no departure data.

Open from P0, asked now because P3 depends on it: the departure plan per store (truck, departure time, staging lead time) and, if available, the article master data and the meaning of TYP, LHM_TYP and VOLUMEN_KOLLI.

## 6. Phase P1: time model and validation (Rev2: validation)

Goal: a time model calibrated and validated on separate days, and validation at system level, with CASIM's own infrastructure (Phase S structure, `casim run`, outputs of C4). Analysis code in `scenarios/ijpe/analysis/` reads CASIM's outputs and the data; it never computes model times itself.

- **P1.1 must.** Source. `scenarios/ijpe/source.py` emits layout, storage, pickers and the events of a day: picker arrival at the first recorded line and departure after the last, the recorded pauses as `PickerDeparture` with duration (pause threshold proposed from the gap distribution; three thresholds reported until the user fixes one), orders available from the shift start. Days are assigned by timestamp.
- **P1.2 must.** Order level. Modelled order processing time (route plus per-line pick time plus setup, as the simulation charges it) against the historic span, per order size, with the submitted parameters and with parameters calibrated on the calibration days. Report RMSE, MAE and bias by order size, leave-one-day-out and on the held-out days. This replaces the in-sample numbers of the submitted Figure 8.
- **P1.3 must.** System level. CASIM runs each historic day with the WMS batches, the recorded picker presence and its own routing and scheduling. Compare per day and per picker-day: completion profile of lines and orders over the day, makespan, busy share relative to presence (same definition on both sides), number of tours. Report the submitted and the calibrated parameters.
- **P1.4 should.** Recorded assignment. A scheduling component `RecordedAssignment` in `src/casim/pipelines` assigns each order to its recorded picker in recorded start order (one order per tour, routing by the configured router). The difference to P1.3 is the effect of CASIM's scheduling, the difference to the data the effect of the time model and the router. No new router, no change to cart capacity.
- **P1.5 should.** Line attributes. Regress the recorded line duration (which includes the walk to the line) on distance, aisle change, quantity and the first line of an order, leave-one-day-out. If a line attribute changes the out-of-sample order error by more than the user's threshold, add it to the calibrated pick time as a per-line formula in the picker resource, used identically in planning and execution; otherwise report it as residual.
- **P1.6 could.** Sensitivity of the validation to the submitted simplifications: greedy item assignment versus recorded locations, aisle 10 excluded versus included, one order per tour versus the recorded 11 % interleaving.

Output: `docs/ijpe/validation.md` with hypotheses, tests, parameters and residuals. Stop and report; the user fixes the time model.

Not part of the revision (moved to after 27 December): replay of the recorded line sequence (`RecordedPlan`, `GivenSequenceRouting`), tours with several interleaved orders, zone eligibility of pickers. The response letter states the unit of analysis (zone order of a store-day, one picker) with P0 as evidence and names these as limitations.

## 7. Phase P2: port and scenarios

- **P2.1 must.** Disruptions as source events (C1). Scenarios 1 and 2: the affected orders are the stores of the warehouse's list on the corrected day; their due time moves; nothing is cancelled by the event. Scenario 3: the warehouse's list instead of generated orders, or, if the list cannot be used, the generated orders stated as such in the paper. Scenario 4: the orders moved across days as a source event. TU = 2.6 × orders. Report each scenario's difference from the submitted one (Rev2: Scenario 2 stores and TU ratios).
- **P2.2 must.** Reproduction. Run the submitted configuration as a variant of `scenarios/ijpe/` (submitted parameters, hourly due times, submitted scenario definitions where they are expressible) and compare with the reference of `ijpe-submitted` (Table 4, Table 6, Figures 9 to 14). Classify every difference: defect of the old code (for example C5, the tour selection of `TruckDisruption`), defect of the port, or changed definition. This answers Rev2 on Table 6 and on late unaffected orders in Figure 11. Delete `scenarios/scenario_ijpe/` once the classification is complete.
- **P2.3 must.** Full KPIs (C4) for baseline and every scenario, affected and unaffected orders separately, with departures (Rev2).

## 8. Phase P3: due times (Rev2: feasible by construction)

- **P3.1 must.** Departure-based due times: departure of the store's truck minus the staging lead time, from the departure plan requested in P0. If the plan does not arrive by 1 December, use the delivery windows and staging times of the manuscript's Section 4.1 and state that in the paper.
- **P3.2 must.** Baseline on-time rate, lateness and departure delays under the hourly bucket of the submitted paper, the departure-based due time, and the departure-based due time tightened by a margin. State that the first is feasible for the historical schedule by construction.

## 9. Phase P4: experiments for the reviews

All runs use the calibrated model and the departure-based due time; the hourly bucket is a variant.

- **P4.1 must.** Policies (Rev2). Every policy is a `decisions` file of `scenarios/ijpe/`; a policy that cannot be expressed through scope, commitment policy and triggers is reported, not built ad hoc.

  | Policy | Replanning decision on disruption | Scope | Commitment |
  |---|---|---|---|
  | No reaction | none; affected orders are planned at the next regular trigger | as ORSP | CommitAll |
  | Append only (full commitment, no cancellation) | ORSP on the disruption triggers | `freeze_from: planned, frozen_accepts: none` | CommitAll |
  | Submitted policy | ORSP on the disruption triggers | `freeze_from: pending, frozen_accepts: none` | CommitAll |
  | Rolling, one tour per picker | ORSP on `PickerIdle` and the disruption triggers | `freeze_from: pending, frozen_accepts: none` | `SchedulingCommitmentPolicy(n_jobs: 1)` |

  Check first that the append-only scope composes with a decision that sees planned tours (`casim check`, one hand-derived test). Report full KPIs per scenario and policy, affected and unaffected orders separately.
- **P4.2 must.** Load (Rev2: spare capacity at 48 % utilisation). Repeat P4.1 with picker presence reduced in steps until the baseline busy share reaches the observed 62 to 70 % and beyond.
- **P4.3 must.** Pipeline selection (Rev2). From the decision log (C3): per decision the candidates with scores, the chosen pipeline, whether the choice changes after a disruption, and how often each decision runs. The manuscript's "revising decisions from tour to tour" is replaced by what the runs show. The user decides whether to add further applicable routers or batching rules to the repository or to narrow the claim in title and abstract.
- **P4.4 must.** Combined Scenario 2 and 4 (Rev2), as a variant with both source events.
- **P4.5 must.** Sensitivity and computation time (Rev2). Walk speed, pick time and setup over the ranges of the P1 estimates, picker presence, due-time margin; computation time per decision and per run from the decision log. If P1 leaves substantial unexplained variation, add replications with times sampled from the residual distribution and report intervals (should).
- **P4.6 should.** Benchmarks (Rev1). A paragraph and table from `docs/validation.md`: L1 equality with CASOP v0.1.0, L2 (Henn 2012) and L3 (D'Haen) with the open points stated. No new benchmark runs.

## 10. Phase P5: manuscript and release (user, with scripts)

- Novelty table (Rev1, Rev2): what is in Bischoff et al. (2026a, 2026b), what is new here. The user decides whether the revision scope (cancellation and commitment as a configured property of a decision instead of the `ReORSPAdapter`) is claimed as a contribution of this paper. If yes, Section 3 presents it from `docs/concepts.md`; if no, Section 3 cites it and the contribution is the case study and the policy comparison.
- Section 3 rewritten against the code of the base commit (Rev1: algorithmic mechanics): decision properties, state information, scope, commitment, the decision flow. Figure 3 is generated from the `decisions` files actually run (Rev2 minor: OnlineCoSySolver versus CoSySolver; RORSP no longer a problem class).
- Financial analysis (Rev2): reframed as an illustrative break-even analysis with sensitivity ranges from P4, not annualised from a stress test; or shortened.
- Figures and tables from scripts in `scenarios/ijpe/plots.py` (Rev1: Gantt notation, Figure 2 font size). Text fixes: ROI equation number, NPV, "14:00 PM", the typos listed by Rev2; the 70 % speed reduction is stated as applied or removed (Rev2 minor).
- Release (the manuscript claims a public framework): the README describes the current scenarios and setup, `uv sync --locked` works on a fresh clone (at present it fails because ware_ops_algos added `alns`), and `scenarios/ijpe/` is runnable without the confidential data on a synthetic day whose aggregates match the data. The user decides what is released.

## 11. Schedule

| Week | Work |
|---|---|
| 17–22 Nov | Base commit fixed; C2, C3, C4; P1.1, P1.2; departure plan requested |
| 23–29 Nov | P1.3, P1.4; C1, C5; P2.1 |
| 30 Nov–6 Dec | P1.5; P2.2, P2.3; P3 |
| 7–13 Dec | P4.1 to P4.5 |
| 14–27 Dec | P4.6, P5, response letter |

Checkpoint on 6 December: tasks marked should or could that are not started are dropped, and the paper states what is open.

## 12. Stop and ask

- The base commit of `codex/ijpe-revision`.
- The pause threshold, after P1.1.
- The time model, at the end of P1.
- Whether aisle 10 enters the layout or is reported as an exclusion with its share.
- If the departure plan per store is not available by 1 December.
- If a number of the submitted paper cannot be reproduced and the cause is not a demonstrated defect.
- Whether the revision scope is claimed as a contribution (P5), and whether the pipeline repository is extended or the claim narrowed (P4.3).
- Before any core change not listed in Section 4.
- Before any committed file would contain data at order, line or picker level.

## 13. Reviewer comments and where they are answered

| Comment | Task |
|---|---|
| Rev1: novelty, literature, research gaps | P5 (user) |
| Rev1: algorithmic mechanics of CASIM | P5, Section 3 from the design document |
| Rev1: data period | P0 |
| Rev1: Gantt notation, Figure 2 | P5, plots from scripts |
| Rev1: benchmarks | P4.6 |
| Rev1: language | P5 |
| Rev2: novelty relative to 2026a/b; cancellation core versus contribution | P5 (novelty table, scope decision) |
| Rev2: counterfactual policies; spare capacity | C1, P4.1, P4.2 |
| Rev2: selected pipelines, changes after disruptions, tour-to-tour revision | C3, P4.3 |
| Rev2: due times feasible by construction | P3 |
| Rev2: validation, calibration on same data, RMSE/MAE, Figure 8 bias, system level, 47.75 % | P0, P1.2, P1.3, P1.5 |
| Rev2: full KPIs per scenario, unaffected orders, Figure 11 | C2, C4, P2.2, P2.3 |
| Rev2: Table 6 departures | C5, P2.2 |
| Rev2: Scenario 2 stores, orders and TU ratios | P0, P2.1 |
| Rev2: day labels of Figures 11 to 14 | P0, P5 |
| Rev2: combined Scenario 2 and 4 | P4.4 |
| Rev2: replications, sensitivity, computation times | C3, P4.5 |
| Rev2: financial analysis | P5 |
| Rev2: zones, order splitting, unit of analysis | P0, P1 (limitation) |
| Rev2 minor: OnlineCoSySolver in Figure 3 | P5 |
| Rev2 minor: ROI equation, NPV, typos | P5 |
| Rev2 minor: 70 % speed reduction | P1.1, P5 |
| Rev2 minor: n = 1419 versus 1,422 | P0 |
| Rev2 minor: due-time profile versus Section 4.1 | P3 |
