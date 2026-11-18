# Status of the IJPE revision

Plan: `docs/ijpe/plan.md`. Newest entry first.

## 2026-11-18: C1, disruptions as exogenous information

**Done.** `TruckDisruption(time, order_ids, due_date)` sets the due date of
the given orders wherever they are still open (order buffer, pick lists,
the batches of unfinished tours, the order history) and cancels nothing.
Which orders a truck carries is the source's choice. A replanning decision
triggered by `TruckDisruption` revises the plan within its revision scope,
so a policy without cancellation (`freeze_from: planned`) can now be
configured. The snapshot cache of the state adapter is cleared when a known
order changes in place, so the next decision sees the new due date. Tests:
`tests/test_truck_disruption.py`. All 60 baseline runs are equal; 136 tests
pass; `casim check --all` passes.

**Changed definition** (for P2.2). The submitted event cancelled planned and
scheduled tours in the order of the tour dictionary whose earliest due date
was at or after the disrupted departure, until the TU volume was reached,
and used the pick-list buffer only if no tour was cancelled. Now the source
names the orders, and the replanning decision of `scenario_obp.yaml`
(`freeze_from: pending`) makes every unstarted tour revisable, so it can
change other tours than the submitted event cancelled. The old
`scenario_ijpe` hook now selects the orders known at the disruption with a
due time at or after the disrupted departure, earliest due first, until the
TU volume is reached. It cannot run on the base (its order stream file is
missing), so this hook is untested.

**Open for P2.1.** The affected orders of Scenarios 1 and 2 are the stores
of the warehouse's list; this needs the IJPE source of P1.1, which is not on
this branch. The R1 replay work of the old worktree (recorded replay source,
facts tables, `scripts/ijpe/` diagnostics, replay baselines and the status
entries of 16 and 17 November) is preserved on the local branch
`wip/ijpe-replay-r1`, not pushed. Its core changes need the user's approval
before they enter this branch.

## 2026-11-16: Phase R0 (data audit) done, stopped for the user

**Done.**

- R0 setup: the WMS export was untracked but not ignored in
  `scenarios/scenario_ijpe/data/`. `.gitignore` now ignores
  `scenarios/ijpe/data/*` (except `SOURCE.md`, `SHA256SUMS.txt`),
  `scenarios/ijpe/outputs/`, the workbook and `generator/private/` of
  `scenario_ijpe`, and `docs/ijpe/private/`. A copy of the workbook sits in
  `scenarios/ijpe/data/` under a neutral name, with its checksum; the original
  is untouched. `scenarios/ijpe/analysis/export.py` reads it (parquet cache,
  `openpyxl` via `uv run --with`, no new dependency). `ijpe` is in
  `NOT_PORTED` of `casim check` until R2.1 gives it the scenario structure.
- R0.1 to R0.5: `scenarios/ijpe/analysis/audit.py` and
  `docs/ijpe/data_audit.md` (aggregates only); tests on a hand-built table in
  `tests/test_ijpe_audit.py`.

**Found** (details and numbers in `data_audit.md`):

- 1,525 orders → 1,422 (aisle 10 and one article dropped) → 1,419 (three
  single-line orders with span 0). Saturday 11 April is ISO week 15 and the
  workbook calls it Sunday; the scripts' D1 is Saturday, Monday is D2.
- 99.5 % of orders have one picker. The WMS splits a store-day into one order
  per zone, picked by different pickers, often in parallel.
- Pickers have two or more orders open 11 % of the time; a line's interval
  includes the walk to it; 295 lines are booked a day after they were picked.
- Long gaps are spread over the day, not concentrated at 8:00 and 11:00.
  Observed busy share 62 % (line-time union over presence) or 70 % (gaps
  > 5 min idle); the paper's 47.75 % is a simulation output.
- TU = 2.6 × orders in the warehouse's workbook; the simulation's 1.6 and its
  due-time selection produce the 72 and 20 orders reviewer 2 questions.
  Scenario 2's list is 38 orders of 37 stores. The submitted configuration
  differs from the warehouse's lists in TU (Scenario 2) and day and content
  (Scenario 3).
- The export has no departure or truck information.

**Next.** The user decides (plan Section 9): calibration and validation days
and the tour gap threshold; zones and order splitting; the departure plan per
store; and supplies, if available, the article master data and the meaning of
TYP, LHM_TYP and VOLUMEN_KOLLI. Then Phase R1.
