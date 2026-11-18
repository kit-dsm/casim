# Status of the IJPE revision

Plan: `docs/ijpe/plan.md`. Newest entry first.

## 2026-11-18: Calibrated day rerun after C7; validation cut to two runs

**Done.** The system-level validation now runs only the calibrated day
(recorded presence, five-minute pauses, setup inside the recorded gaps),
with recorded assignment (the validation) and with FIFO assignment (the base
of the policy comparisons): 12 day-runs instead of 54. The nine diagnostic
settings stay in `docs/ijpe/validation.md` as the record of how the
calibration was found, marked as before C7, and are not rerun.

**Found.** With recorded assignment, orders completed by 10:00 and 12:00
are within 17 of the WMS on every day (+43 by 10:00 over six days, 47
orders open). Against the pre-C7 run: the day's last line is +14 min
instead of +23, picker finish RMSE is 80 instead of 85 min, open orders
are 47 instead of 31. FIFO: within 31 orders, 91 open. The decision of the
P1.3 entry stands.

## 2026-11-18: C7, departures take effect in request order

**Defect.** A departure requested while a picker was busy replaced any
departure still pending for that picker. With recorded presence and the
five-minute pauses, this lost 5 requests on the six days: 0.6 h of pause,
and in one case a pause was replaced by the shift end.

**Fix (core, approved by the user).** `ResourceManager` keeps the pending
departures of each picker in request order. Each one takes effect at the next
free point. Departures requested during an absence wait until the picker
returns, and `free_at` adds them up. Two new hand-derived tests cover this
(`tests/verification/test_dispatch_two_pickers.py`), and the expected absent
time in a recorded-replay test is re-derived.

**Checks.** pytest (165 passed), `casim check --all`, and every scenario
and benchmark baseline (27 scenario runs, l1, l2, l3) are unchanged. The
defect only affects chained recorded-presence runs, which have no baseline
record. The system-level numbers in `docs/ijpe/validation.md` came from runs
before the fix and move by at most the 0.6 h of lost pause.

## 2026-11-18: P1.3 and P1.4 system-level validation, and why pickers idle

**Done.** Six-day runs with recorded batches, U-shape routing and either
FIFO or recorded assignment, under three presence settings, with and
without setup, and with the fitted pick time; a new presence setting
`arrival` (recorded arrival, no forced departure) with a hand-derived test.
Aggregate tests on the export of why recorded pickers idle. Results and
tables in `docs/ijpe/validation.md`.

**Found.** Presence dominates: with every picker present from 05:30, CASIM
completes 365 more orders by 10:00 than the WMS. With recorded presence,
recorded assignment and the 215 s setup, orders completed by 10:00 and 12:00
are within 14 of the WMS on every day. The end of the day is not reproduced:
with setup, orders stay open or the day overruns; without setup, the day's
end matches but the morning runs ahead. Picker-level finish times are not
reproduced in any run. The fitted pick time makes every system measure
worse. The export shows no dock limit, work-in-progress cap or release
waves: work was waiting at the end of every long gap, and open orders show
no ceiling. A dock limit is not used to reproduce the WMS.

**Decision.** Disruption and policy results are comparisons between policies
on the calibrated day (recorded presence, five-minute pauses), measured on
throughput and on-time completion up to midday, with the 215 s setup inside
the recorded gaps (`setup_in_gap`; the export shows about 4 to 5 minutes per
tour, so zero setup is not supported). Picker-level effects and end-of-day measures are not claimed.
Aisle 10 is not part of the study (user, 18 November).

## 2026-11-18: P1.2 order-level validation completed

`docs/ijpe/validation.md` reports six-day leave-one-day-out and held-out
order-span errors by order size. The fixed-speed interior-line calibration
reduces pooled six-day RMSE from 16.27 to 16.12 minutes but raises held-out
RMSE from 14.95 to 15.39 minutes and makes bias more negative. It does not
justify replacing the submitted pick-time constant for the paper's main
model. P1.3 and P1.4 remain before the P1 time-model decision.

## 2026-11-18: Pause threshold selected and departure fallback specified

The user selected a five-minute inferred pause threshold after a new six-day
24-run sweep on the C6 base. Across 95 picker-days, the 2-, 5- and 10-minute
thresholds yield respectively 957, 378 and 217 pauses; 176.0, 142.5 and
125.2 hours of inferred absence; and 81.0, 80.5 and 82.7 minutes of pooled
picker-day last-line RMSE. Between-tour pauses do not change order spans.
The five-minute threshold is the selected chained replay setting;
`source.pause_threshold_s=null` remains the anchored diagnostic.

The partner has not supplied a departure plan. `docs/ijpe/departure_fallback.md`
specifies a fixed synthetic schedule, with 10:00, 12:00, 14:00 and 16:00
dispatch groups, a stated 30-minute staging assumption, and a stable
store-to-slot assignment independent of picking outcomes. An aggregate audit
on the replay cohort finds recorded late shares of 16.3% to 51.9% by day.
These are missed synthetic deadlines, not observed truck delays. The
`fixed_dispatch` variant now puts the deadlines on orders; its Friday smoke
run completed 81 orders with a 54.3% simulated on-time rate. C5 is required
before departure events and truck delay measures use this schedule.

## 2026-11-18: C2, C3 and order and picker measures of C4

The source may expose `affected_order_ids`. Both `facts/orders.csv` and the
legacy `orders.csv` mark those orders, and all other scenarios write false.
`decisions.json` now records each accepted decision's trigger, feasible
candidate scores, selected pipeline, solve, conversion and apply times, and
counts of visible kept, changed, cancelled and created tours. These counts
refer to the decision-visible plan and accepted commands; no individual
orders or tours are committed.

`service.json` reports completed and open orders, on-time rate among completed
orders with due dates, total and maximum tardiness, and makespan for all,
affected and unaffected orders. It reports each picker's busy, idle and
absent seconds, with busy share divided by busy plus idle presence. The
existing `facts/picks.csv` remains the pick table. Departure delay and
orders-carried measures still need C5's consistent dock and manifest
accounting; the current event does not identify a truck's orders.

Focused tests pass for affected and unaffected denominators, candidate and
command logging, and picker presence. The one-process suite passed with
160 tests, 3 skipped and 1 deselected. `casim check --all` passed. The fast
and full scenario comparisons, including all six IJPE days, and all L1, L2
and L3 benchmark comparisons show no differences. The full scenario comparison
initially found the RADAR case unrunnable because its ignored selected input
was absent from the new worktree; copying that public local input and
rerunning the case made it equal.

## 2026-11-18: C6, one definition of the tours a replanning decision may revise

**Found.** The pipelines receive every tour the decision's scope lets it
rebuild (`rebuildable_tours`), scheduled tours included under
`freeze_from: pending`; the comparison that turns a solution into commands
(`_schedule_commands` in ware_ops_algos) compared only planned tours. A
replanned scheduled tour therefore became a second `CreateTour` for its
orders, validation rejected every candidate and the run stopped ("no
feasible candidate solution"); reproduced in a test. The configuration on
this branch avoids the error only because RORSP observes planned tours, of
which there are none after ORSP's commit-all: since C1, "re-plan unstarted"
replans nothing and behaves like "append only". No other configuration
makes scheduled tours rebuildable.

**Done.** The comparison uses the same set of tours as the pipelines. A
candidate holding exactly a revisable tour's orders revises it (reschedule,
new route); revisable tours missing from the candidate are cancelled and
their pick lists return to the buffer; the rest are created. The overlap
rule remains for decisions that batch anew. A rescheduled committed tour
re-enters its picker's queue by planned start (this already held). Tests:
four in ware_ops_algos `tests/test_planning_conversion.py`, and
`tests/test_replanning_policies.py` for the policies of P4.1 (append only,
re-plan unstarted, rolling with one committed tour, no replanning); the
re-plan and rolling tests fail with the old comparison. Every baseline is
unchanged (27 scenario runs including the six IJPE days, 39 benchmark
runs); 159 tests pass; `casim check --all` passes.

**Open.** For P2.2 and P4.1, RORSP must observe scheduled tours
(`tours: [planned, scheduled]`) for "re-plan unstarted" to replan. The
counts of kept, changed, cancelled and created tours per decision go into
the decision log with C3.

## 2026-11-18: P1.1, the IJPE source, with the replay work of 16 and 17 November

**Done.** The replay work of the old worktree is on this branch, on top of
the base and C1: the IJPE scenario (`scenarios/ijpe/`) with its source over
`casim.sources.wms_export`, `casim.sources.grid_layout` and
`casim.sources.recorded`; the facts tables (`facts/tours.csv`, `orders.csv`,
`picks.csv`, from `casim.facts`); the recorded-plan components (`RecordedIA`,
`RecordedTours`, `GivenSequence`, `RecordedPlan`); a line pick time that may
depend on the quantity; the recorded pick order at shared nodes; the
`recorded_replay` verification scenario; and the diagnostics in
`scripts/ijpe/`. The ware_ops_algos branch `codex/ijpe-revision` (the
recorded algorithms and the line pick-time model) is merged with
`core-ijpe-base`; its 115 tests pass. Its old presence model (picker
absences in the resource manager, `PickerReturn`) was dropped: the base's
`PickerDeparture` already defers an absence until the running tour ends.

New for P1.1: `source.presence: recorded` makes a picker arrive at the setup
start of its first recorded tour and leave after its last recorded line;
`variants/recorded_presence.yaml` combines it with the 5 minute pause
threshold and lets the decision see pickers not yet arrived. The default
(`shift`) keeps every picker present from 5:30, as before. Hand-derived
tests in `tests/verification/test_recorded_replay.py`. On Friday 17 April,
recorded presence leaves 3 of 81 orders open: those pickers' recorded day
ends while CASIM still has their tours queued.

**Reproduced on the base.** The default replay of all six days keeps every
scalar KPI of the old worktree's baselines (makespan, mean completion time,
tours, orders, distance). The records differ only by the base's new busy,
absent and idle times per picker and by about 180 fewer events per day
(the base's dispatch). The anchored six-day order-span RMSE is again
16.273 min on 1,414 orders (single-order tours 14.145 min). Every other
baseline is unchanged (21 scenario and 39 benchmark runs); 154 tests pass;
`casim check --all` passes.

**Open.** P1.2 to P1.4 (order level, system level with CASIM's own
decisions, recorded assignment) build on this source. The WMS gap analysis
of 17 November continues as before; its scripts are on this branch.

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
of the warehouse's list, selected by the IJPE source (P1.1, entry above).

## 2026-11-17: Current WMS reproduction diagnosis

WMS line starts are partly order-local bookkeeping: on 3,036 returns to an
open order, the start matches that order's preceding completion within one
second yet precedes the picker's previous completion on a different order.
An opt-in completion-time sequence variant reorders 5,860 lines and lowers
six-day anchored RMSE only from 16.273 to 16.079 min. Combining that sequence
with the fixed-speed pick fit gives 15.864 min pooled and 16.112 min on later
Thursday/Friday. Thus correcting sequence interpretation alone does not
reproduce the WMS order spans. The original start-order run remains default.
Combining one-order tours with completion order and that same fit still gives
15.865 min pooled and 16.024 min later-day RMSE. No simple tour/sequence/time
combination tested so far closes the full trace gap.
Completion-to-completion evaluation avoids the order-local start field:
308 matched within-tour WMS gaps exceed 300 s, versus none in the submitted
CASIM model. The fitted handling formula improves ordinary later-day gap MAE,
but the long unlabelled gaps remain. `scripts/ijpe/completion_gaps.py` reports
only aggregate counts and errors.

With the original, correspondence-supported 15 m cross-aisle geometry and
the exact recorded line sequence, the six-day anchored order-span RMSE is
16.273 min for 1,414 retained orders. Tours containing several orders account
for 378 orders and 44.6% of squared error. WMS intervals overlap at 2,539 of
2,871 transitions that change both order and aisle; the replay cannot treat
those starts and ends as consecutive physical activity. The single-order
tour RMSE is still 14.145 min. Long WMS line intervals and gaps inside orders
explain much of that residual, but the export does not label their causes.
Ordering single-order tours by WMS position number instead of booking time
saves only about 10.5 s of graph travel per order at the submitted speed, so
the router's line order cannot account for the minute-scale RMSE. A fitted
same-aisle line model and a separate order-residual prototype improve later
days only modestly; neither identifies a physical pause or tour process.
An exact-sequence six-day CASIM variant with one order per tour scores 15.575
min pooled, a small improvement that does not justify assuming the WMS never
interleaves orders.
For a secondary trace-quality sensitivity screen (single-order tour, no WMS
line over 300 s, no internal idle gap over 300 s), the unchanged submitted
model has 4.360 min RMSE on 789/1,414 orders. Later Thursday/Friday validation
is 4.401 min on 198/353 orders, compared with 16.205 min on all 353. This is
an outcome-filtered sensitivity analysis with explicit coverage, not a
replacement for the full-cohort result or evidence of pause reconstruction.
On the same 198 later-day orders, the Monday--Wednesday ordinary-line fit
lowers RMSE from 4.401 to 3.639 min, while full later-day RMSE barely moves
(16.205 to 16.109 min). Its duration-selected training sample prevents
interpreting its speed coefficient as a measured truck speed.
Keeping speed fixed at the submitted 2.306 m/s and fitting only handling
gives 3.632 min on that same 198-order later-day screen, essentially equal to
the free-speed variant. Its full later-day RMSE is 16.262 min. This fixed-speed
variant is in `scenarios/ijpe/variants/fixed_speed_pick_fit.yaml`; it is a
diagnostic, while the submitted constant remains default.
The current diagnostic is `scripts/ijpe/trace_diagnosis.py`; `blockers.py`
crosses tour type, idle gaps and long lines.

## 2026-11-16: R1.3 exact pick sequence committed in ware_ops_algos

Commit `4b0eb38` records the ordered pick positions on a given-sequence
route, so CASIM can select the correct order and article when consecutive
recorded picks share a graph node. The full ware_ops_algos suite passes. The
CASIM replay executes the recorded tour sequences and its focused tests pass.
The CASIM worktree and its scalar/digest IJPE baselines remain uncommitted.

## 2026-11-16: Reproduction diagnosis in progress

The IJPE worktree preserves the recorded pick order when several orders share
one graph node. A review of Jonas Koreis's layout correspondence overturned
an attempted aisle-column rescaling: the submitted loader's column scale
already yields the reported aisle-change distance across the cross aisle.
The original scale has been restored and its travel-edge distance is tested.
The aggregate replay check confirms the recorded tour sequences. The WMS
intervals still conflict with a sequential physical pick trace, and the
submitted time model still has substantial order-span error. The fitted
same-aisle line model improves line timing and only slightly improves later-day
order spans. The current analysis is checking WMS timestamp semantics and latent
waiting inside tours before selecting a revision time model. Focused replay
tests pass. Broad repository gates and task commits remain pending.
The numerical entries below predate the exact pick-order correction; use the
current diagnostic script for the corrected replay.

The export's pallet identifier was also tested as a possible missing tour
boundary. It usually identifies one handling unit within an order, but pallet
switches are often adjacent in WMS time. Charging the reported setup mean for
every additional pallet gives only a modest forward-day improvement and is
not yet justified as a physical event. Pallet count is retained as an
explanatory diagnostic, not silently added to the simulator.

## 2026-11-16: Corrected departure sweep and defect-aware evaluation

The source now emits `PickerDeparture(picker_id, duration)` and the core defers
an absence until an active tour ends. The hand-derived replay tests and
`casim check --all` pass. Six-day anchored and chained runs at 2, 5 and 10
minute thresholds conserve the recorded lines. The 1,414-order pooled
anchored RMSE is 16.554 min; order spans are unchanged by between-tour pause
thresholds. Chained picker-day timing is substantially worse. The 284 WMS
lines over 300 s account for 92.6% of squared line error; both busy and idle
components contribute to order-span error. The clean 50-order diagnostic
cohort has 2.167 min RMSE but is too selective for paper validation. Details
are in `docs/ijpe/replay.md`. The 18-run assumption sweep and 24-run reverse
sweep completed: U-shape raises pooled order-span RMSE to 19.756 min with
recorded tours; one-order tours lower it to 15.575 min but change the recorded
work organization. The order of replacements matters. The full CASIM suite
and benchmark comparisons are running; `casim check --all` and the standalone
all-scenario baseline comparison pass with no differences. This work remains
uncommitted.

## 2026-11-16: R1.3 recorded item assignment committed in ware_ops_algos

Commit `8a3e5ea` adds a recorded item assigner through the existing algorithm
interface. Its full suite passes (107 tests). The CASIM replay now selects it
instead of greedy assignment; the old greedy behavior remains a named variant.
Across six anchored days all 62,985 retained lines use their recorded pick
nodes, versus 22 moved picks previously. The pooled order-span RMSE changes
only from 16.56 to 16.55 min, ruling out those moved picks as the principal
cause. CASIM wiring, the greedy comparison variant, and the one changed IJPE
scalar/digest baseline are in the worktree but uncommitted. `casim check --all`
passes, the ware_ops_algos suite passes, and six anchored CASIM runs conserve
all 62,985 lines. The full CASIM test and all-scenario baseline gates were
started but stopped at the Task 4.14 core-interface boundary. No CASIM commit
or push was made.

## 2026-11-16: R1.3 routing extension committed in ware_ops_algos

Commit `e973651` adds an optional recorded-tour start node and fixes the
zero-distance first-leg case in `GivenSequenceRouting`. Its regression test
and the full ware_ops_algos suite pass (104 tests). CASIM's diagnostic source,
analysis and report remain uncommitted. The one-process CASIM suite passed
(587 tests), as did the standalone all-scenario baseline comparison and
`casim check --all`. The recorded-pause correction still depends on Task 4.14:
the current core has `BreakStart` but no `PickerDeparture` interface.

## 2026-11-16: R1 replay diagnosis in progress

The six-day anchored replay and a hand-derived verification scenario run in
CASIM. `docs/ijpe/replay.md` records the aggregate baseline and diagnostic
results. Replacing the assumed fixed-depot origin with an aisle-head or
first-pick origin leaves pooled order-span RMSE near 16.6 min. A zero-distance
first-leg routing defect was fixed and verified. A fixed-speed pick-time fit on
13–15 April improves line MAE on held-out days but raises their pooled
order-span RMSE from 14.97 to 15.41 min. WMS interval extremes and overlap
prevent literal physical timestamp reproduction. A setup-in-gap diagnostic
improves anchored picker-day timing slightly but exposes more missing idle in
the chained replay. No CASIM R1 task has yet been committed; the remaining
replay corrections are pending.

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
