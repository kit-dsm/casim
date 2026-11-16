# Status of the IJPE revision

Plan: `docs/ijpe_revision_plan.md`. Newest entry first.

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
