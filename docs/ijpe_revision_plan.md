# IJPE revision plan (PROECO-D-26-03325)

Revision due 27 December 2026. This plan answers the reviews of 4 August (reviewer 1) and 16 October 2026 (reviewer 2). It starts from the WMS data as ground truth: before any scenario is rerun, CASIM has to reproduce what happened in the warehouse, and every gap between model and data has a demonstrated cause. The plan is self-contained; `docs/revision_scopes_corrections.md` remains the plan for the core and is referenced only where a task here depends on it.

## 1. Ground rules

The WMS data and everything derived from it at order, line or picker level are confidential. They live in the local checkout below `scenarios/ijpe/data/` (and the original export wherever the user places it), are listed in `.gitignore`, and are never committed, pushed, pasted into an issue or written into a committed document. Committed documents contain aggregates only (counts, distributions, error statistics, regression coefficients); a figure or table that shows individual orders or pickers is written to an ignored directory and goes into the paper only by the user. The submitted manuscript and the reviews stay local as well.

Do not change ware_ops_pipes. Do not change existing scenario files outside `scenarios/scenario_ijpe/` and `scenarios/ijpe/`.

Calibration and validation use different data. Split by day: parameters are estimated on days chosen by the user (proposal: days 1 to 4) and every validation number is reported on the remaining days, and in addition with a leave-one-day-out estimate. Reviewer 2 points out that pick and setup times were calibrated on the data they were validated on; no number of the revision may repeat that.

Every assumption of the model is stated as a hypothesis with a test on the data, and its result is recorded whether it confirms or refutes the assumption. A result that contradicts the submitted paper is reported, not explained away.

Commit each task separately. Update `docs/ijpe/status.md` after every commit with what was done, what was found and what is next. Stop at the end of each phase and at the stop points of Section 9.

## 2. What the submitted model assumes

The following is read from the code that produced the paper (casim `main`, `scenarios/scenario_ijpe/`), not from the manuscript. Each item is a candidate cause of the validation error (RMSE 15.3 min, MAE 9.3 min, overprediction of short and underprediction of long orders).

- Historic processing time of an order is the last line's end minus the first line's start over all lines of the order, regardless of how many pickers picked it (`make_results_domain_validation.py`, `build_historic_order_times`). Negative spans get 24 hours added.
- Modelled processing time is the U-shape route of the whole order at 2,306 mm/s plus 19 s per order line. Quantity, weight and volume of a line do not enter. The tour setup time of 215 s, which the simulation charges, is not in the validation's model.
- The route is computed on the item assignment of `GreedyItemAssignment`, not on the pick locations recorded in the data.
- A tour holds exactly one order (`PickCart` capacity 1 in dimension orders). The loader builds 25 identical pickers; the per-day picker counts come from the simulation configuration.
- The historic replay (`_build_jobs_replay`) assigns each order to the first picker that appears in its lines.
- Due times are the end of the hourly bucket in which the order was completed (`plot_due_windows` and the order stream), so the historical schedule meets them by construction.
- Section 4.1 of the manuscript describes fixed zones and several pickers working on one store order in parallel; the model has neither zones nor order splitting.

## 3. Phase R0: data audit (done)

Goal: know exactly what the data contains before modelling it. Output: `docs/ijpe/data_audit.md` (aggregates only) and the reading code in `scenarios/ijpe/analysis/`.

Findings that the later phases build on: the model drops aisle 10 and one article (1,422 of 1,525 orders); 99.5 % of WMS orders have one picker, because the WMS already splits a store's daily demand into one order per zone; pickers have two or more orders open 11 % of the time; 83 % of lines start when the previous line ends, so a line's duration includes the walk to it; 295 lines are booked one day after they were picked; long gaps are spread over the day, not at the configured breaks; pickers are busy 62 to 70 % between their first and last line; the workbook defines TU = 2.6 × orders while the simulation uses 1.6; Scenario 2 is 38 orders from 37 stores; Scenario 3 differs from the warehouse's list; the export has no departure data. Days are assigned by timestamp, not by booking date, in everything that follows.

- **R0.1** Inventory. Columns, types and units of the export; date range and weekdays of the six days (reviewer 1 asks for the exact period; reviewer 2 notes that the day labels of Figures 11 to 14 do not match the weekdays); orders, lines, customers, articles, pick locations, pickers per day. Explain the difference between 1,422 orders and the 1,419 of Figure 8 (which filter drops which orders and why).
- **R0.2** Who picks an order. Per order the number of distinct pickers, of houses or zones and of aisles; whether the pickers of one order work in parallel (overlapping intervals) or one after another; whether a picker interleaves lines of several orders. Distribution over orders and by order size.
- **R0.3** Picker timelines. Per picker and day the sequence of lines in time order: inter-line gaps, their distribution, gaps that are breaks (compare with the configured breaks at 8:00 and 11:00), gaps between orders (candidate setup), first and last activity of the day. From this, the picker's present and busy time per day, which is the basis of the utilisation question (the paper reports 47.75 %).
- **R0.4** Timestamps. Resolution, missing values, zero-length lines, lines across midnight, lines whose end precedes their start, duplicate lines.
- **R0.5** Stores, trucks and transport units. The relation between customer (store), order, truck departure and transport units in the data. Reviewer 2 notes that Scenario 2 is described as two stores but affects 50 orders, and that the ratios of transport units to orders vary widely (120 TU to 72 orders, 200 TU to 20 orders). Establish from the data what an order is (one store per order or not) and what a TU is, and whether the departure schedule per store can be derived (needed for R3).

Stop and report.

## 4. Phase R1: the historic picking days in CASIM

Goal: reproduce each historic picking day with CASIM and find out why the submitted model does not. The replay fixes everything the WMS decided (which picker picks which lines, in which order, from which locations) and lets CASIM compute the times. Every remaining difference is then a difference of the time model, not of decisions.

The replay uses CASIM's own infrastructure throughout: a scenario in the Phase S structure with its own `source.py`, decisions configured like every other scenario, pipeline components in `src/casim/pipelines`, algorithms in ware_ops_algos with cards, execution by CASIM's events and resources, and CASIM's outputs (trace, tour and order tables). Analysis code in `scenarios/ijpe/analysis/` only reads CASIM's outputs and the data and compares them; it never computes model times itself. No simulation loop, pipeline component or time model lives in the scenario. If something the replay needs is missing from CASIM, it is added to CASIM through its interfaces, with a test, and the baselines of all other scenarios stay identical; if that is not possible without a design decision, stop and ask.

The work happens on the branch `codex/ijpe-revision`, created from `codex/revision-scopes` and rebased onto it whenever the core work commits. The replay scenario joins the baseline, so every later core task shows its effect on the replay.

- **R1.1** Facts tables. CASIM writes, from its trace, the realised order table, tour table and pick table (one row per picked line with its start and end) of `docs/proposals/measures_and_ranking.md`. Check them on a hand-derived scenario.
- **R1.2** Source. `scenarios/ijpe/source.py` emits the initial state (layout, storage with the recorded pick locations, pickers) and the events of a day: picker arrival at the first recorded line and departure after the last, orders available from the shift start, and the recorded pauses as picker unavailability. A pause is a recorded gap longer than the pause threshold (Section 9). Days are assigned by timestamp.
- **R1.3** Recorded decisions. A pipeline component `RecordedPlan` in `src/casim/pipelines` returns, for one decision at the shift start, the recorded tours of every picker in recorded order, each with its recorded lines in recorded sequence. A recorded tour is a picker's run of lines between two pauses or order changes; interleaved orders form one tour with several orders, with the cart capacity that this requires. Routing through the recorded sequence uses a router `GivenSequenceRouting` in ware_ops_algos (shortest paths between consecutive positions in the given order, returning the node sequence), with a card that makes it applicable only where a sequence is given. The decision commits everything at once and never replans. Add a hand-derived verification scenario: two pickers, a few lines, one pause, with every start and end derived in its docstring.
- **R1.4** Replay with the submitted model (2,306 mm/s, 19 s per line, 215 s setup) on every day. Compare CASIM's pick, tour and order tables with the data at three levels: per line (duration error), per order (span error, as in Figure 8, and per order size), per picker-day (end of the last line, busy time, completion profile of lines and orders over the day). Run it twice: anchored, with all recorded gaps between tours as unavailability, so that each tour starts at its recorded time and its error is isolated; and chained, with only the pauses as unavailability, so that errors accumulate over the day as they would in a simulation.
- **R1.5** Causes. Explain the per-line error of R1.4 by the line's attributes: walk distance, aisle change, quantity, weight, volume, first line of a tour, interleaving, position in the day. Since a recorded line's duration contains the walk to it, estimate walk and pick jointly: regress the recorded line duration on distance, aisle change and the line attributes. Report the coefficients with intervals, estimated leave-one-day-out.
- **R1.6** The submitted model's other assumptions, each as a configuration variant of the CASIM replay and judged by the change of the picker-day error and the order span error:
  - U-shape routing over an order versus the recorded sequence (router choice);
  - greedy item assignment versus the recorded locations (item assignment component);
  - one order per tour versus the recorded interleaving (tour definition and cart capacity);
  - the configured breaks at 8:00 and 11:00 versus the recorded pauses (source option);
  - aisle 10 excluded versus included (layout variant, if the layout data allows it);
  - the historic order span over several pickers versus single-picker spans, for the 0.5 % of orders it concerns.
  The order of replacement can change the attribution; report two orders and state which effects interact.
- **R1.7** Pick-time model. If R1.5 shows that pick time depends on the line, add a typed pick-time model to the picker resource (constant per line as the default, which keeps every existing baseline) and use it wherever CASIM computes a tour's duration, in execution and in planning, so that the time model of the design document stays one formula. Calibrate it with the walk speed on the calibration days, replay all days, and report the three levels of R1.4 on the held-out days and leave-one-day-out, next to the submitted model. Record what it still does not explain.
- **R1.8** Zone affinity. Per picker and day, the share of lines in the picker's main zone, and whether pickers change zones during the day. This decides whether the model needs picker-to-zone eligibility (Section 9).

Output: `docs/ijpe/replay.md` with the hypotheses, their tests and results, the decomposition, the final parameters and the residual. Stop and report; the user decides the time model of R2 (Section 9).

## 5. Phase R2: the paper, CASIM's decisions and the scenarios

Goal: reproduce the submitted results with the port, then let CASIM decide on the historic days, then rebuild the scenarios from the warehouse.

- **R2.1** Service measures. The measures this plan needs, on the tables of R1.1: completed and open orders, late orders, on-time rate, total tardiness and maximum lateness, each over a selectable group of orders (all, affected, unaffected); makespan per day; utilisation and idle time per picker relative to presence; departures with their delay and the orders they carry. The ranking part of the proposal is not part of this plan.
- **R2.2** Paper reproduction. Resolve the two open results of Task 4.9 first: a `TruckDeparture` with `capacity: null` releases no pallets, and the truck capacity counts orders while staging counts pallets; both can produce the departure counts of Table 6 that reviewer 2 questions. Then run the submitted configuration as a variant of `scenarios/ijpe/` and compare with the old code's outputs in the paper (Table 4, Table 6, Figures 9 to 14). Every difference has a cause: a defect of the old code, a defect of the port, or a changed definition. This answers reviewer 2 on Table 6 and on late unaffected orders in Figure 11. Delete `scenarios/scenario_ijpe/` once the port reproduces what it should.
- **R2.3** CASIM's own decisions on the historic days. CASIM plans routing and scheduling as in the paper (batching from the WMS), with the calibrated model; compare with the replay of R1.7 and with the data. The difference is the effect of CASIM's decisions, not of the model.
- **R2.4** Scenarios from the warehouse. Define the disruption scenarios from the warehouse's disruption lists and the workbook (TU = 2.6 × orders, the affected orders and stores as listed, the day of each list after correcting booking dates), not from the submitted configuration. Report each scenario's difference from the submitted one.

Stop and report.

## 6. Phase R3: due times

- **R3.1** Derive due times from the staging and departure logic of the manuscript's Section 4.1 and the store-to-truck relation of R0.5: due time is the departure of the order's truck minus the staging lead time. If the data does not give the relation, ask the user for the departure plan per store.
- **R3.2** Report the baseline on-time rate, lateness and departure delays of the replay (R1.7) and of CASIM's decisions (R2.3) under three due-time definitions: the hourly completion bucket of the paper, the departure-based due time, and the departure-based due time tightened by a margin. State that the first is feasible for the historical schedule by construction (reviewer 2).

## 7. Phase R4: experiments for the reviews

All runs use the calibrated model and the departure-based due time unless stated otherwise; the hourly bucket is reported as a variant.

- **R4.1** Policies. For each of the four disruption scenarios and the baseline, compare: the paper's policy (CommitAll with trigger-based replanning and cancellation of planned, unstarted tours); a fixed shift-start schedule without replanning; a rolling commitment of one tour per picker; full commitment without cancellation. Use the existing commitment policies and triggers; a policy that cannot be expressed is reported, not built ad hoc. Report full KPIs (R2.1) for every scenario and policy, affected and unaffected orders separately, and departures.
- **R4.2** Load. Repeat R4.1 with picker availability reduced until the baseline utilisation is high, to separate the policy's effect from spare capacity (reviewer 2: at about 48 % utilisation, absorbing a disruption may reflect capacity, not the policy).
- **R4.3** Pipeline selection. Record for every decision the candidate pipelines with their scores (the ranker computes them; `decisions.json` keeps only the chosen one), the chosen pipeline and whether the choice changes after a disruption. Report how often and when replanning happens, so that the manuscript's "revising decisions from tour to tour" is stated as what the runs do.
- **R4.4** Combined Scenario 2 and 4, as reviewer 2 asks before claiming that Scenario 4 resolves the bottleneck of Scenario 2.
- **R4.5** Sensitivity. Travel speed, pick time model and picker availability over ranges from the spread estimated in R1 (not arbitrary percentages); setup time; due-time margin. The model is deterministic, so this answers the request for replications; if R1 shows substantial unexplained variation in pick or travel times, add replications with sampled times drawn from the estimated residual distribution and report their intervals.
- **R4.6** Computation time per decision and per run.
- **R4.7** Benchmarks (reviewer 1). Summarise from `docs/phase_b_report.md` what CASIM reproduces on published benchmarks (L1 equality with CASOP, L2 and L3 online benchmarks), with the open points stated. No new benchmark runs.

Stop and report.

## 8. Dependencies and schedule

R1 runs on CASIM as it is at the head of `codex/revision-scopes` (Phase S structure) and starts now, on its own branch; it does not wait for the core tasks. The core work of `docs/revision_scopes_corrections.md` continues in parallel: Phase S, Tasks 4.13 to 4.15 and Phase 5 reduced to Tasks 5.2 and 5.3. Since the replay scenario is in the baseline, each core task reports its effect on the replay; a change there is explained before the task is accepted. R2 to R4 run after the core prerequisites. Task 4.16, Task 5.0, Task 5.1 and the rest of Phases 6, 6b and 7 wait until after the revision. Task 6b.6 is replaced by R1.2 and R2.2 of this plan.

| Week | IJPE track | Core track |
|---|---|---|
| 16–22 Nov | R0 (done); R1.1 to R1.4 | Phase S complete; Task 4.13 |
| 23–29 Nov | R1.5 to R1.8; user decides the time model | Tasks 4.14, 4.15; Phase 5 (5.2, 5.3) |
| 30 Nov–6 Dec | R2 | rebase of the IJPE branch; replay rerun |
| 7–13 Dec | R3, R4 | |
| 14–27 Dec | paper tables and figures from scripts; writing and response letter (user) | |

Checkpoint on 6 December: if the core prerequisites are not done, R2 to R4 run on the branch as it is and the paper states which verification is open.

## 9. Stop and ask

Stop and ask the user:

- before fixing the calibration and validation days and the pause threshold of the replay; until the user answers, use leave-one-day-out as the primary estimate, calibration on Monday to Wednesday with validation on Thursday, Friday and Saturday for figures, Saturday reported separately, and report results for three pause thresholds proposed from the gap distribution;
- at the end of R1, with the decomposition, before the time model of R2 is fixed;
- whether to model picker-to-zone eligibility or to justify its absence, with the evidence of R1.8; order splitting is not modelled, since the WMS already splits store demand by zone;
- whether aisle 10 enters the layout or is reported as an exclusion with its share;
- if R0.5 does not give the store-to-truck relation;
- if a number of the submitted paper cannot be reproduced by the port with the submitted model and the cause is not a demonstrated defect;
- before any committed file would contain data at order, line or picker level.

## 10. Reviewer comments and where they are answered

| Comment | Task |
|---|---|
| R2: novelty relative to Bischoff et al. 2026a/b; tour cancellation core versus contribution | writing (user), with the design document |
| R2: counterfactual policies; spare capacity | R4.1, R4.2 |
| R2: selected pipelines, changes after disruptions, tour-to-tour revision | R4.3 |
| R2: due times feasible by construction; alternative definitions | R3 |
| R2: validation only at order level; calibration on the same data; RMSE, MAE, Figure 8 bias; system level; 47.75 % utilisation | R0.3, R1 |
| R2: Scenario 2 and TU ratios; Scenario 3 | R0.5, R2.4 |
| R2: full KPIs per scenario, unaffected orders, Figure 11 | R2.1, R2.2, R4.1 |
| R2: Table 6 departures | R2.2 |
| R2: Scenario 2 stores, orders and TU ratios | R0.5 |
| R2: day labels of Figures 11 to 14 | R0.1 |
| R2: combined Scenario 2 and 4 | R4.4 |
| R2: replications, sensitivity, computation times | R4.5, R4.6 |
| R2: financial analysis | writing (user), using R4 for sensitivity ranges |
| R2: zones and order splitting; unit of analysis | R0.2, R1 (unit of analysis), Section 9 |
| R2 minor: OnlineCoSySolver versus CoSySolver in Figure 3 | R2.2 (configuration of the port), writing |
| R2 minor: 70 % speed reduction | R1.2 |
| R2 minor: n = 1419 versus 1,422 | R0.1 |
| R2 minor: due-time profile versus Section 4.1 | R3 |
| R1: novelty, literature review, research gaps | writing (user) |
| R1: algorithmic mechanics of CASIM | writing (user), with the design document and the decision model |
| R1: data period | R0.1 |
| R1: Gantt notation, Figure 2 readability | figures from the scripts of R4 |
| R1: comparison with benchmarks | R4.7 |
