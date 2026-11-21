# IJPE revision plan (PROECO-D-26-03325)

Revision due 27 December 2026. This plan answers the reviews of 4 August (reviewer 1, "Rev1") and 16 October 2026 (reviewer 2, "Rev2"). Phases are numbered P0 to P5 so that they cannot be confused with the reviewers.

The plan is organised by what the response letter has to show, and every task names the comment it answers. Tasks are marked **must** (the response is not credible without it), **should** (strengthens a claim the reviewers question) or **could** (only if time remains after the checkpoint of 6 December). A task that answers no comment and is not a prerequisite of one is not part of this plan.

## Positioning of the revised paper (decided 20 November)

The revision presents the case as a practitioner-oriented evaluation in three parts, each answering what a warehouse has to see before it uses a simulation for decisions. The word "usability" is avoided, since it implies a user study. The three parts answer the novelty comments of both reviewers: the contribution is not simulation as such but the path from a credible model to what-if analysis to decision support, which also maps onto the maturity levels of the digital-twin literature Rev1 asks for.

1. **Credibility** (P1.7). What the model reproduces and what it does not, per component, with held-out metrics. The result defines the validity envelope: throughput and on-time completion up to midday, compared between policies with common random numbers; no picker-level predictions. Parts 2 and 3 report only measures inside it.
2. **What-if analysis** (P4.2, with P4.1 and P4.4). The main result is a sweep per scenario (S1 to S4 and S2+S4) over the advanced volume, 0 to 4 times the volume scaled to the area, crossed with staffing in pickers (recorded roster, −2, −4). The realistic volumes are marked on it, and the onset of a bottleneck is a band over the replications, not one threshold. The commitment and replanning policies are compared on it. Phrased as the operational question of a shift lead: up to which volume, at which staffing, can an advancement be absorbed?
3. **Decision support** (P4.3, P4.7). The pipeline selection actually made by the ranker, and one cost-based decision study whose simulated per-event savings become the inputs of the investment appraisal (P5).

Policy levers remain the sequencing rule, the commitment policy and the replanning scope and triggers. One order per tour and U-shape routing are fixed by the warehouse. Staffing is exogenous: it enters parts 2 and 3 only as a sensitivity (and in part 3 as a priced sensitivity), never as a decision alternative. Truck staging times are not optimised: the dispatch plan is synthetic (`docs/ijpe/departure_fallback.md`), so an optimised staging time would optimise an assumption; this is reconsidered only if the partner supplies the departure plan.

## 1. Ground rules

The WMS data and everything derived from it at order, line or picker level are confidential. They live below `scenarios/ijpe/data/` (and the original export wherever the user places it), are listed in `.gitignore`, and are never committed, pushed, pasted into an issue or written into a committed document. Committed documents contain aggregates only. A figure or table that shows individual orders or pickers is written to an ignored directory and goes into the paper only by the user. The manuscript and the reviews stay local. The one exception is the public data package of P5 (release), committed only after the partner has approved its content in writing.

Calibration and validation use different data. Leave-one-day-out is the primary estimate; for figures, parameters are estimated on Monday to Wednesday and reported on Thursday to Saturday, with Saturday reported separately. No number of the revision is validated on the data it was calibrated on (Rev2).

Every assumption of the model is stated as a hypothesis with a test on the data, and its result is recorded whether it confirms or refutes the assumption. A result that contradicts the submitted paper is reported, not explained away.

Do not change ware_ops_pipes. Commit each task separately and update `docs/ijpe/status.md` after every commit.

## 2. Code base and core freeze

The revision runs on the branch `codex/ijpe-revision`, created once from `codex/revision-scopes` at a commit the user names (the base). It is not rebased onto later core work.

The core is frozen for the revision. A change to `src/casim` or ware_ops_algos is made on the revision branch only if a task of this plan needs it, and then with a test, with the baselines of all other scenarios identical, and with the task named in the commit. The deferred core work in `docs/development.md` waits until after 27 December. The core changes this plan needs are listed in Section 4 and are small.

The submitted code is the tag `ijpe-2026` (casim `7b6a1f3`, ware_ops_algos `32cdff5`, tagged the same). Its results are reproduced on `main` only (commit `8756ffd`, `scenarios/scenario_ijpe/README.md`); nothing of the submitted code is ported to or compared within this branch, since the two versions rest on different assumptions.

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
| C5. Dock and departures | The two open results of Task 4.9: a `TruckDeparture` with `capacity: null` releases no pallets; truck capacity counts orders while staging counts pallets. Not needed (20 November): Table 6's departures are hourly due-time groups computed from order completions by the submitted results script, not dock events (P2.2), and the truck waiting of P4.2 and P4.7 is computed from the order completions too. | `DockManager`, `TruckDeparture` |
| C6. One definition of revisable tours | The pipelines receive every tour the replanning decision's scope lets it rebuild, scheduled tours included, but the comparison that turns a solution into commands compared only planned tours. Under `freeze_from: pending` with scheduled tours observed, a replanned scheduled tour became a second tour for its orders, every candidate was rejected and the run stopped; with only planned tours observed (the current configuration) nothing was revisable, so "re-plan unstarted" behaved like "append only" once C1 removed the event's own cancelling. The comparison now uses the same set; a candidate holding exactly a revisable tour's orders revises it, revisable tours missing from the candidate are cancelled, the rest are created (the overlap rule remains for decisions that batch anew). One test per policy of P4.1. The decision log of C3 records per decision how many tours were kept, changed, cancelled and created. | ware_ops_algos `planning/conversion.py`, `tests/test_replanning_policies.py` |
| C7. Departures in request order | A picker's departure requested while it is in a tour replaced an earlier pending one, so a pause could be lost or replace the end of a shift (5 requests, 0.6 h of pause in six IJPE days with recorded presence). Requests now take effect one after another at the picker's next free point; a request made during an absence waits for the return; an arrival of a present picker still cancels its pending departures. Approved by the user on 18 November. | `state/resource_manager.py`, `Dispatch`, tests in `tests/verification/test_dispatch_two_pickers.py` |
| C8. Process truth | The fitted IJPE day draws each line's time and each setup of a picker from distributions, the same for every policy of a replication, instead of absences that were not observed (reviewer objection). The source fixes these realized times in W_0 as a `ProcessTruth`; execution (`Dispatch`, `NodeArrival`) takes them, planning keeps the resource's time model (in ware_ops_algos, `ClassPickTime`: the expected line time by quantity class), and the decision snapshot never carries the truth. Without a truth nothing changes. `tours.csv` reports the setup as executed (from dispatch). Approved by the user on 19 November (option B, after a first version that bypassed the resource's time model was reverted). | `domain_objects/process_truth.py`, `SimWarehouseDomain`, `State`, `SimulationEngine.reset`, `events/operational_events.py`, `outputs/facts.py`, ware_ops_algos `domain_models/resources/models.py`, `tests/verification/test_process_truth.py` |
| C9. A pull that routes once | The FIFO pull of the calibrated day routed every open pick list again and copied every open order and the planner information into each of its about 220 decision snapshots a day: 75 s per day instead of 9 s for the push. `PlanOnlyPolicy` lets a decision plan without committing, `PlannedRoutes` lets a scheduling decision keep the routes of the tours it may revise, and the decision snapshot copies each order and the planner information once per run (like batches and routes). The calibrated day now plans once (OBRSP) and pulls (ORSP); its tours and orders are identical on all six days, 14 s per day. Approved by the user on 19 November. | `decision_engine/commitment_policies.py`, `config/schema.py`, `pipelines/subproblems/picker_routing.py`, `simulation_engine/state_adapter.py`, `tests/test_route_once_pull.py` |
| C10. Canonical order streams and exogenous rules (approved by the user, 19 November) | The experiments need the week as one order stream with a fixed roster, and disruptions declared like the submitted model's hooks, without scenario code reading the raw export. A library source reads canonical order and line tables; a scenario declares roster, due-date rules, disruptions and process times under `exogenous`, applied to any source with an epoch. | `casim/sources/order_stream.py`, `casim/sources/exogenous.py`, `run.py`; `process_truth.drawn` moved from the IJPE source |
| C11. One tour for every idle picker | `SchedulingCommitmentPolicy(n_jobs: 1)` commits one tour per decision; pickers that become idle at the same time (a fixed shift start) waited for the next trigger, so most of them never started. `per_resource: true` commits the first `n_jobs` tours of each resource; the default is unchanged. | `commitment_policies.py` |
| C12. Run time of the sweep (approved by the user, 20 November) | A week run took 75 to 147 s; the sweep of P4.2 needs thousands. Three changes, none altering a result: (a) the storage is shared in the decision snapshot's copy instead of copied 1,344 times per week (`State._SHARED_FIELDS` already declares it never mutated); (b) resource states are logged only after events that can change them, not after each of 1.47 million travel, arrival and pick events; (c) `travel: pick` walks to the next pick node in one event instead of one per graph edge, adding times and distances edge by edge; a decision's view of a tour inside a leg is the edge walked at that time (`edge_at`). `travel: pick` is refused if a decision is triggered by a travel or node arrival, or may revise a started tour. The week gives identical tours, picks, orders and KPIs with 0.20 instead of 1.47 million events; `scenarios/ijpe_week` uses it, the default stays `edge`. Edge-mode digests are unchanged. | `simulation_engine.py`, `state_adapter.py`, `events/`, `tour_model.py`, `facts.py`, `audit.py`, `run.py`; tests `test_resource_state_log.py`, `test_travel_granularity.py` |

## 5. Phase P0: data audit (done)

`docs/ijpe/data_audit.md`. Findings the later phases build on: six days, Saturday 11 April and Monday 13 to Friday 17 April 2026 (Rev1: data period; Rev2: day labels of Figures 11 to 14); 1,525 orders, 1,422 after dropping aisle 10 and one article, 1,419 after three zero-span orders (Rev2: n = 1419); an order is one zone order of a store-day, picked by one picker in 99.5 % of cases, so zones and parallel pickers exist at the store-day level (Rev2: unit of analysis); observed busy share 62 to 70 % (Rev2: 47.75 %); Scenario 2 is 38 orders of 37 stores; Scenario 3 differs from the warehouse's list; the export has no departure data.

Findings of 20 November (`docs/ijpe/scenarios.md`): the 2.6 TE per order is a constant of the export pivot, not a measurement; by goods volume an area order is about 2.45 TE. The area's share of the warehouse's volume is about 0.20 (the 32 % of the mail is a share of picks). Scaled to the area, the scenarios are 3 to 4 times smaller than the submitted ones and than the warehouse's lists, which count warehouse TE as area TE. The submitted S2 was configured with 120 TE, not the 100 TE named for it.

Closed: the article master data were supplied (`scenarios/ijpe/data/SOURCE.md`); VOLUMEN_KOLLI is litres per package and MENGE_IST counts single units, so the question on TYP, LHM_TYP and VOLUMEN_KOLLI is closed. Carrier types 39 (roll container) and 40 (pallet of about 1.8 roll containers) are a stated assumption, used only as a check. The departure plan per store is still open (Section 12).

## 6. Phase P1: time model and validation (Rev2: validation)

Goal: a time model calibrated and validated on separate days, and validation at system level, with CASIM's own infrastructure (Phase S structure, `casim run`, outputs of C4). Analysis code in `scripts/ijpe/` reads CASIM's outputs and the data; it never computes model times itself.

- **P1.1 must.** Source. `scenarios/ijpe/source.py` emits layout, storage, pickers and the events of a day: picker arrival at the first recorded line and departure after the last, the recorded pauses as `PickerDeparture` with duration (pause threshold proposed from the gap distribution; three thresholds reported until the user fixes one), orders available from the shift start. Days are assigned by timestamp.
- **P1.2 must.** Order level. Modelled order processing time (route plus per-line pick time plus setup, as the simulation charges it) against the historic span, per order size, with the submitted parameters and with parameters calibrated on the calibration days. Report RMSE, MAE and bias by order size, leave-one-day-out and on the held-out days. This replaces the in-sample numbers of the submitted Figure 8.
- **P1.3 must.** System level. CASIM runs each historic day with the WMS batches, the recorded picker presence and its own routing and scheduling. Compare per day and per picker-day: completion profile of lines and orders over the day, makespan, busy share relative to presence (same definition on both sides), number of tours. Report the submitted and the calibrated parameters.
- **P1.4 should.** Recorded assignment. A scheduling component `RecordedAssignment` in `src/casim/pipelines` assigns each order to its recorded picker in recorded start order (one order per tour, routing by the configured router). The difference to P1.3 is the effect of CASIM's scheduling, the difference to the data the effect of the time model and the router. No new router, no change to cart capacity.
- **P1.5 should, covered.** The fitted line time by quantity class with an order factor (`docs/ijpe/validation.md`, fitted non-picking time) answers this task; no further regression on line attributes is run. The text below is kept as the original task. Line attributes. Regress the recorded line duration (which includes the walk to the line) on distance, aisle change, quantity and the first line of an order, leave-one-day-out. If a line attribute changes the out-of-sample order error by more than the user's threshold, add it to the calibrated pick time as a per-line formula in the picker resource, used identically in planning and execution; otherwise report it as residual.
- **P1.6 dropped (20 November)**, named as a limitation instead. Sensitivity of the validation to the submitted simplifications: greedy item assignment versus recorded locations, one order per tour versus the recorded 11 % interleaving.
- **P1.7 must.** Credibility profile (part 1 of the positioning; Rev2: validation, Figure 8, system level, 47.75 %). One table of model components with metric, held-out result and verdict (reproduced, partly reproduced, not identified, not reproduced), built from P1.2 to P1.5 and `docs/ijpe/validation.md`, no new fitting: graph and U-shape routing (recorded-sequence distance over U-shape distance by order size), service time per order (ECDF of one-person order durations net of breaks), throughput up to midday (cumulative completed orders per day, WMS against the 90 % band of the replications), staffing per day (not identified), end of day, picker finish times and busy share (not reproduced, not used). The order generator enters only if it stays a claimed contribution (P5) or is used for S3 (P2.1), checked on a held-out day for lines per order, aisle access and volume per order. Volume per order comes from the package dimensions, with packages per line from GEWICHT_SOLL and the package weight; the check states that MENGE_IST counts single units, not packages (quantity × package volume overstates volume by the units per package). Figures from scripts, aggregates only: the profile table, the routing ratio by order size, the duration ECDF (replaces the submitted Figure 8), the cumulative completion curves. The section closes with the validity envelope that P4 reports in, stated by clock time from the 08:00, 10:00, 12:00 and 14:00 comparisons of the capacity validation, not only "up to midday". Consistency check before P4 is written up: every KPI reported for a scenario is tagged inside or outside the envelope. Candidates outside it are the 13:30 and 15:30 dispatch deadlines and the whole-day effect of Scenario 4 on the receiving day. A KPI outside the envelope is either dropped, or reported as a comparison between policies only and marked as such.

Output: `docs/ijpe/validation.md` with hypotheses, tests, parameters and residuals. Stop and report; the user fixes the time model.

Not part of the revision (moved to after 27 December): replay of the recorded line sequence (`RecordedPlan`, `GivenSequenceRouting`), tours with several interleaved orders, zone eligibility of pickers. The response letter states the unit of analysis (zone order of a store-day, one picker) with P0 as evidence and names these as limitations.

## 7. Phase P2: port and scenarios

- **P2.1 must.** Disruptions as source events (C1), defined by the scenario spec `docs/ijpe/scenarios.md`. The list-based variants of 20 November (`scenarios/ijpe_week/variants/`) are replaced; an event only changes exogenous state (due times, new or moved orders) and cancels nothing.
  - Volume target: the TE the partner's mail names for each scenario × 0.20, the area's share of the warehouse's volume. The capacity workbook motivates day and hour only; no workbook number is quoted in the paper or in committed documents.
  - TE per order by goods volume: packages per line are GEWICHT_SOLL divided by the package weight, the volume follows from the package dimensions of the article master data, and 0.445 m³ is one TE. Carrier types 39 (1 TE) and 40 (1.8 TE) are a stated assumption, used as a check only.
  - Selection unit: whole stores (S2, S3, S4) and whole dispatch groups (S1). The synthetic plan has no trucks, so a dispatch group is represented by stores of its slot. Stores are taken from the slot of the original departure on the scenario day until the target is reached; achieved TE, stores and orders are reported next to the target.
  - S3: one order of a new store from the order generator conditioned on the target volume; fallback, stated as such, a copy of the template order.
  - S4: whether the warehouse picks a delivery day's volume on the day before is measured from the export (release practice), and S4 and S2+S4 follow the measured practice.
  - Slots and deadlines: the "13:00" origin is the 14:00 slot, Thursday 10:00 the 10:00 slot; the 30 minutes of staging apply to every scenario deadline as to every baseline deadline.
  - Selections: a fixed rule (largest stores first) and alternative selections, one per replication, the same for every policy (common random numbers extend to the selection), written as named lists into `lists.csv` without a core change.

  Report each scenario's difference from the submitted one (Rev2: Scenario 2 stores and TU ratios).
- **P2.2 must, cut down (20 November).** The submitted results are reproduced on `main` with the submitted code (`scenarios/scenario_ijpe/README.md` there): Tables 4 and 5 match, Table 6 matches in structure, with Scenario 2 at 47 instead of 50 affected orders and slightly different waits, cause not determined. Rev2's questions on the submitted results are answered from that code, not by porting it: Table 6's "delayed departures" are hourly due-time groups (latest completion per due time against the due time, averaged over all due times, empty ones included), so Scenario 1's delayed departures come from late unaffected orders in other groups and Scenario 2's late orders share one group; Figure 11's late unaffected orders are real in the submitted run (4 late orders while all 72 affected orders are on time); Scenario 2's 50 orders follow from `TruckDisruption`'s tour selection, which fills the TU volume by tours, not by store (Section 3). No port comparison is made. The `submitted` variant of `scenarios/ijpe` stays only for comparing the submitted assumptions inside the new model (validation, model A). The untracked `scenarios/scenario_ijpe/` folder of this checkout can be deleted.
- **P2.3 must.** Full KPIs (C4) for baseline and every scenario, affected and unaffected orders separately, with departures (Rev2).

## 8. Phase P3: due times (Rev2: feasible by construction)

- **P3.1 must.** If the partner supplies a departure plan, use each store's actual truck departure minus its staging lead time. Otherwise use the fixed synthetic dispatch plan in `docs/ijpe/departure_fallback.md`. Its store-to-slot rule is independent of completed picking times and is held constant across scenarios. State clearly that the fallback does not identify actual trucks or observed departure delays.
- **P3.2 must.** Baseline on-time rate, lateness and simulated departure delays under the hourly bucket of the submitted paper, the fixed departure-based due time, and a stated tighter margin. State that the hourly bucket is feasible for the historical schedule by construction and that the synthetic departure plan is an assumption whose deadlines are binding on the recorded orders.

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
- **P4.2 must.** Volume and staffing sweep, the main what-if result (Rev2: spare capacity at 48 % utilisation). Per scenario, the advanced volume V as a multiple of the P2.1 target (0, 0.5, 1, 1.5, 2, 3, 4) crossed with staffing in pickers per day (recorded roster, −2, −4), with the alternative selections and process-time seeds as replications and the baseline of the same seed as the pair. The realistic volumes are marked on the sweep: V = 1 (volume share 0.20) and V = 1.5 (about the 0.31 pick share as an upper sensitivity); V = 3 to 4 is about the volume of the submitted scenarios and of the warehouse's lists. The onset of a bottleneck is reported as a band over the replications (the V at which affected orders start to miss their deadline, or unaffected orders due by 12:00 become late, from the lower to the upper quantile), not as one threshold. Busy share is not a load target: it lies outside the validity envelope of P1.7. The policies of P4.1 are compared on this sweep at selected points.
- **P4.3 must.** Pipeline selection (Rev2). From the decision log (C3): per decision the candidates with scores, the chosen pipeline, whether the choice changes after a disruption, and how often each decision runs. The manuscript's "revising decisions from tour to tour" is replaced by what the runs show. The user decides whether to add further applicable routers or batching rules to the repository or to narrow the claim in title and abstract.
- **P4.4 must.** Combined Scenario 2 and 4 (Rev2), as a variant with both source events.
- **P4.5 must.** Sensitivity and computation time (Rev2). Walk speed, pick time and setup over the ranges of the P1 estimates, picker presence, due-time margin; computation time per decision and per run from the decision log. If P1 leaves substantial unexplained variation, add replications with times sampled from the residual distribution and report intervals (should).
- **P4.7 must (slim), could (full grid).** Cost-based decision study (part 3 of the positioning; Rev2: financial analysis, Scenario 3 valuation not performed; manuscript Section 6.3 claims the trade-off between fulfillment cost, delay of unaffected orders and lost sales). It is a must because the financial section has nothing defensible without it. The slim version: an S2-type early-delivery request at one fixed request time, the alternatives reject, accept with the submitted policy, and accept plus pull forward, each at 1×, 2× and 4× the P2.1 target volume, recorded roster. The full grid below is could. Fallback if even the slim version slips past 13 December: the per-event saving is the difference in simulated truck waiting and tardiness between the best and the default policy of P4.1 on Scenario 2, stated as such. Its output for P5 is the cost saved per event, by event type (S2, and S3 for the promised valuation). The full grid: the request swept over request time and volume. Alternatives: reject (the store keeps its slot), accept without replanning, accept with the submitted policy, accept with rolling replanning, accept and pull volume of the following day forward (S4 type, the amount as search variable). Each alternative is simulated with common random numbers; its cost is read from the run: 40 EUR per hour of simulated truck waiting (the partner's figure in the submitted Section 5.5; a dispatch group leaves when its last order is done, and its waiting is the excess of that completion over the group's staging deadline, computed from `facts/orders.csv` after the run; this does not depend on C5) plus a tardiness penalty on unaffected orders plus a reject or lost-sales penalty. Prices the partner has not given (lost-sales value, tardiness penalty) are not invented: the result is the cost-minimal alternative per request time and volume over a range of the price ratio to the truck idle rate. Staffing is a sensitivity layer: one map per staffing level of P4.2 (recorded roster, −2, −4 pickers), the picker-hour price used only to state what extra capacity would be worth against the cheapest alternative. Scenario 3 is valued the same way (urgent orders absorbed against lost sales). All synthetic elements (dispatch groups, staging lead) are stated as such.
- **P4.6 should.** Benchmarks (Rev1). A paragraph and table from `docs/validation.md`: L1 equality with CASOP v0.1.0, L2 (Henn 2012) and L3 (D'Haen) with the open points stated. No new benchmark runs.

## 10. Phase P5: manuscript and release (user, with scripts)

- Novelty table (Rev1, Rev2): what is in Bischoff et al. (2026a, 2026b), what is new here. The user decides whether the revision scope (cancellation and commitment as a configured property of a decision instead of the `ReORSPAdapter`) is claimed as a contribution of this paper. If yes, Section 3 presents it from `docs/concepts.md`; if no, Section 3 cites it and the contribution is the case study and the policy comparison.
- Section 3 rewritten against the code of the base commit (Rev1: algorithmic mechanics): decision properties, state information, scope, commitment, the decision flow. Figure 3 is generated from the `decisions` files actually run (Rev2 minor: OnlineCoSySolver versus CoSySolver; RORSP no longer a problem class).
- Financial analysis (Rev2): proposed to keep the investment appraisal as a break-even NPV on simulated inputs. The cost saved per event comes from P4.7 (or its P4.1 fallback). The yearly saving is that figure times the number of events of that type per year (an explicit input from the partner, not a daily stress test). The result is the break-even: the framework pays back if events of type S2 occur at least k times per year, shown over ranges of implementation cost, idle rate and discount rate. This removes the daily annualisation, the flat "top 40 %" factor and the unsimulated mitigation, and gives Scenario 3 its promised valuation. The final form of the section is the main author's decision (Section 12).
- Paper structure follows the positioning section: credibility (P1.7), what-if analysis (P4.1, P4.2, P4.4), decision support (P4.3, P4.7). The user decides whether title and abstract change accordingly.
- Figures and tables from scripts in `scenarios/ijpe/plots.py` (Rev1: Gantt notation, Figure 2 font size). Text fixes: ROI equation number, NPV, "14:00 PM", the typos listed by Rev2; the 70 % speed reduction is stated as applied or removed (Rev2 minor).
- Release (the manuscript claims a public framework and promises a public canonical order stream and a calibrated generator): the README describes the current scenarios and setup, `uv sync --locked` works on a fresh clone (at present it fails because ware_ops_algos added `alns`), and `scenarios/ijpe_week` runs without the confidential data from one public data package (decided 20 November):
  - **Package** (in the repository, a few MB, own README with license and citation): `layout.csv` (already public); `orders.csv` with order, weekday and store; `lines.csv` with order, pick position, packages and volume per line; `times.json`, the fitted time model in the format of `week.json` but with each sample list replaced by its percentiles (setup, line time per quantity class) plus the order factor and the breaks, so no individual sample is published and `process_times` reads it unchanged. Not in the package: pickers, timestamps, raw article and store numbers, the article dimensions as a table. Pseudonyms come from a private seed, not the public seed of `anonymize.py`, which shuffles the sorted real numbers and could be inverted by anyone who knows them.
  - **Content: a generated week** (B). The partner does not allow the recorded orders to be published, even pseudonymised (20 November), but allows a generated or fitted stream. Its fidelity to the recorded week is measured before release (stream statistics and the P4 measures on both streams). The calibration itself is not published; it would disclose the article table, as B does implicitly.
  - **The paper's experiments (P4) run on the recorded stream with the package's time model and are repeated on the generated week**; the confidential tables also run the validation (P1, P1.7, whose aggregated results the repository reports). The paper reports both where they differ, so a reader can reproduce the generated-week results exactly.
  - **Scenario inputs from the package:** the generator is fitted on the package with the public code, S3's new store order is drawn at run time with a seed in the configuration, and the store selections of S1, S2 and S4 follow the rules of P2.1 on the package. No request to the authors is needed for any experiment.
  - `scenarios/ijpe_week` reads the package by default; the confidential tables are an override (`source.tables=...`).
  - **The generator is published with the package** (decided 21 November): the code (`scripts/ijpe/order_generator.py`, G3) and the generated week, not the calibration fitted on the recorded data. Anyone can refit it on the generated week and draw further weeks or demand levels. Before release: a generator refitted on the generated week still matches the recorded week's statistics (second generation), and the README names the seed and code version of the published week, which only the authors can regenerate.
  - **Names** (decided 21 November): the case is called `grocery_retailer`, by role. `scenarios/ijpe_week` becomes `scenarios/grocery_retailer` (public experiments, reads the package), `scenarios/ijpe` becomes `scenarios/grocery_retailer_validation` (confidential replay and validation, with the confidential data store under its `data/`), `scripts/ijpe` becomes `scripts/grocery_retailer` (data building separated from analysis; one-off diagnostics not cited by `docs/ijpe/validation.md` archived after 27 December), and the baseline cases `ijpe_week__*` and `ijpe_replay__*` follow. `docs/ijpe/` keeps its name (it documents the revision), and `main` keeps `scenario_ijpe` (the submitted record). The partner is not named. The rename is one commit together with the package build, when no session has uncommitted work on these paths.

## 11. Schedule

| Week | Work |
|---|---|
| 17–22 Nov | Base commit fixed; C2, C3, C4; P1.1, P1.2; departure plan requested |
| 23–29 Nov | P1.3, P1.4; C1, C5, C6; P2.1 |
| 30 Nov–6 Dec | P1.5; P2.2, P2.3; P3; the rename to `grocery_retailer` and the release package built (generated week, local until approved), since P4 runs on it |
| 7–13 Dec | P1.7; P4.1 to P4.5; P4.7 slim. All experiments finished by 13 December |
| 14–22 Dec | P4.6; result-dependent parts of P5; full draft with the co-authors before the holidays |
| 23–27 Dec | Co-author comments, response letter |

The P5 parts that do not depend on results start now, alongside P2 to P4: the novelty table, Section 3 rewritten against the code, the literature Rev1 asks for, and the language and typo fixes.

Checkpoint on 6 December: tasks marked should or could that are not started are dropped, and the paper states what is open.

## 12. Stop and ask

- The base commit of `codex/ijpe-revision`.
- The pause threshold, after P1.1.
- The time model, at the end of P1.
- Whether aisle 10 enters the layout or is reported as an exclusion with its share. Decided 18 November: aisle 10 is not part of the study.
- If the departure plan per store becomes available, compare it with the documented synthetic fallback before replacing the latter in the main experiment.
- If a number of the submitted paper cannot be reproduced and the cause is not a demonstrated defect.
- Whether the revision scope is claimed as a contribution (P5), and whether the pipeline repository is extended or the claim narrowed (P4.3).
- To the partner now. No question blocks a task; each has a stated fallback:
  - departure plan per store: the synthetic four-slot plan (`docs/ijpe/departure_fallback.md`);
  - the area's share of the warehouse's volume: 0.20 from the data, 0.31 (pick share) as the marked upper point of the sweep;
  - picker-hour cost, lost-sales value per order or TE, contractual lateness penalty, price of an extra trip: price-ratio ranges and break-even prices in P4.7;
  - yearly frequency of each disruption type: the break-even in events per year only;
  - whether the warehouse pre-picks the next day's volume (S4): the release practice measured from the export (P2.1);
  - the remaining questions of `docs/ijpe/scenarios.md` (workbook period and capacity formula, how a truck's stores and the early-request timing work): the stated assumptions there;
  - the public data package (P5, release): answered on 20 November, a generated week is allowed, the recorded orders are not; the built package goes to the partner for written approval before it is committed.
- To the main author now: the P0 and P2 findings change the submitted results (busy share 62 to 70 % instead of 47.75 %; an area order about 2.45 TE by volume instead of 1.6; the area about 0.20 of the warehouse's volume, so the scaled scenarios are 3 to 4 times smaller than submitted; the submitted S2 configured with 120 TE; Scenario 3 not the warehouse's list), and the story of the paper may change with them: whether a realistic advancement creates a bottleneck is answered by the sweep of P4.2, not by one scenario.
- The final form of the financial section (break-even NPV on simulated per-event savings, a shorter version, or removal): the main author decides.
- Before any core change not listed in Section 4.
- Before any committed file would contain data at order, line or picker level.

## 13. Reviewer comments and where they are answered

| Comment | Task |
|---|---|
| Rev1: novelty, literature, research gaps | Positioning, P5 (user) |
| Rev1: algorithmic mechanics of CASIM | P5, Section 3 from the design document |
| Rev1: data period | P0 |
| Rev1: Gantt notation, Figure 2 | P5, plots from scripts |
| Rev1: benchmarks | P4.6 |
| Rev1: language | P5 |
| Rev2: novelty relative to 2026a/b; cancellation core versus contribution | P5 (novelty table, scope decision) |
| Rev2: counterfactual policies; spare capacity | C1, C6, P4.1, P4.2 |
| Rev2: selected pipelines, changes after disruptions, tour-to-tour revision | C3, P4.3 |
| Rev2: due times feasible by construction | P3 |
| Rev2: validation, calibration on same data, RMSE/MAE, Figure 8 bias, system level, 47.75 % | P0, P1.2, P1.3, P1.5, P1.7 |
| Rev2: full KPIs per scenario, unaffected orders, Figure 11 | C2, C4, P2.2, P2.3 |
| Rev2: Table 6 departures | P2.2 |
| Rev2: Scenario 2 stores, orders and TU ratios | P0, P2.1 |
| Rev2: day labels of Figures 11 to 14 | P0, P5 |
| Rev2: combined Scenario 2 and 4 | P4.4 |
| Rev2: replications, sensitivity, computation times | C3, P4.5 |
| Rev2: financial analysis; Scenario 3 valuation | P4.7, P5 |
| Rev2: zones, order splitting, unit of analysis | P0, P1 (limitation) |
| Rev2 minor: OnlineCoSySolver in Figure 3 | P5 |
| Rev2 minor: ROI equation, NPV, typos | P5 |
| Rev2 minor: 70 % speed reduction | P1.1, P5 |
| Rev2 minor: n = 1419 versus 1,422 | P0 |
| Rev2 minor: due-time profile versus Section 4.1 | P3 |
