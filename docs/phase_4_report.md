# Phase 4 report

17 November 2026, casim `9239c94`, ware_ops_algos `c7950f4`. Phase 4 of
`docs/revision_scopes_corrections.md` (Tasks 4.0 to 4.16) is done; Task 4.17 is deferred until
after the IJPE revision (user decision, 16 November 2026). Evidence per task is in
`docs/revision_scopes_status.md`, which has a report for each of Tasks 4.13 to 4.16.

The plan's completion criteria:

| Criterion | Result |
|---|---|
| The tests pass | casim 133 passed (the baseline test is opt-in, `pytest -m baseline`), ware_ops_algos 99 passed; `casim check --all` reports only the untracked `scenario_dm` |
| The name sweep test passes | `tests/test_no_algorithm_dispatch_strings.py` passes: no algorithm, waiting-policy or positioning name outside the subproblem components |
| Every baseline difference with its cause | below |
| Every card whose CASOP applicability differs from `main` | one card added, none removed (below) |
| The type counts of Task 4.13 | 117 before 4.13, 115 after, 114 now (below) |

## What Phase 4 changed

- **4.0 to 4.12** (before a baseline existed; Phase B checked them afterwards): subproblem base
  classes, waiting as a pipeline stage over the whole schedule, injected routers, the analytical
  precondition, positioning as ordinary tours, the incumbent router from its card, the card and
  base check, the name sweep, setup counted once (F28), `Dock` and `DockManager` (F42), one
  order-count waiting card (F43), residual admission after picks (F44), configuration fixes with
  the Henn sweep restored as decision alternatives (F45, F54, F56), CASOP conventions with
  configured cards (F46 to F49).
- **4.13** One rule for released work (F62); `TourProgress`; admission over tour views;
  `Violation` merged; typed CoSy stages; one router builder; one tour filter; the single-picker
  requirement of the two analytic waiting rules.
- **4.14** One dispatch mechanism (F55): resource state is derived, queues hold committed tours
  only, `Dispatch`, `PickerArrival` and `PickerDeparture` replace five events, planned and actual
  times with queue projection, binding or forecast start times, idle, busy and absent time from a
  state log, and the resource checks of Section 7.2.
- **4.15** The hidden rules that affect replanning and cancellation (F57, F67):
  `FlushRemainingOrders` deleted; `NbrOrdersCondition` counts the work open to the decision; a
  due-date window applies after the stream's end and needs due dates; started tours' pickers come
  with the tour view (`TourView.picker`), not the decision's resources.
- **4.16** Unused files, functions and order splitting removed (F63); library prints became
  logging; the 14 components no configuration ran got smoke tests (13 run;
  `LSBatchingRRDueDate`, which no data card can apply, is deleted).
- **Alongside** (user directions of 16 and 17 November 2026): F68, the OOBP depth offset;
  one `casim check` per sweep; a fast tier; the baselines slimmed from 1,654 to 65 runs; the audit
  once per timestamp and after each decision instead of after every event; decision snapshots
  that copy routes once. A full tier takes about 3 minutes instead of about 2 hours; Henn default
  runs in 5.0 s instead of 8.6 s.

## Every baseline difference and its cause

All differences were recorded in commits titled "Baseline: …". Runs not listed were equal at
every step.

| Commit | Runs | What changed | Cause | Expected |
|---|---|---|---|---|
| `a35bb41`, `d714c58`, `2ad6e22` | journal L2 FIRST n = 90, 120; 128 Henn-format L3 runs; the other L2 runs | route lengths +0.1 to 1.6 LU per tour, completion up to +1.1 min; L3 count below a best-known value unchanged at 40 of 192 | F68: the OOBP loader placed location *l* at depth *l* instead of cell *l* + 1 | yes |
| `f6a51ea`, `2ad6e22` | both `henn__61s-80-30-0` configurations, 246 L2 SAV replications | runnable and `consistent` (before: no feasible candidate) | F62: casim's live check rejected a suffix adding orders of a draft cancelled in the same candidate | yes |
| `55bab7c` | all 1,654 records | idle time redefined (setup now counts as busy; Henn default 4680 → 45 s), busy and absent time added, the audit kind `check_resources` (no finding), event counts and trace digests; 133 runs `inconsistent` → `consistent`; decisions, committed tours and every other KPI unchanged | 4.14: derived resource state and `Dispatch`; the old reservation model's overlap findings disappear | yes |
| `74f2703` (code) | 47 Kris EDD L1 cells | `projection` findings removed | 4.14: a tour running 2,880 s past its estimate left the projection stale until it ended; the check now skips such a resource | found by the comparison, fixed |
| `2267134` | 16 Henn scenario configurations, 99 L2 journal SAV cells, 116 L3 runs | Henn order window completes every order (before up to 4 orders of the last window were never picked; makespan up to +13.3 % because that work is now done); Henn waiting −0.05 to −0.19 %; L3 makespan −1.3 % on average (same tours, no idle time before a planned tour); L2 SAV −1.4 % on average | 4.15, F67: the order-count gate ignored orders in planned tours | yes; broader than the plan's list (it expected L3 and the journal cells unchanged) |
| `0536416` | `henn__release_only` | plans from the first arrival: makespan 38492 → 29016, idle 11638 → 45 s, 20 → 24 tours; becomes `inconsistent` with overlaps between planned drafts and committed tours | 4.15: the due-date window no longer lifts at the stream's end, so `release_only_example` loses a window on orders without due dates; drafts are in no queue since 4.14, so their times are not projected after a delay | yes for the run; the overlaps are an open decision |
| `863c8db` | 1,594 records removed (1,568 benchmark, 26 Henn), 60 kept | — | the regression set is slimmed to one run per code path | user direction |
| `7a7f096` | `W1_100__wsrp_input_pedido_01_000__0`, `W2_100__wsrp_input_pedido_02_000__0`, `henn__release_only` | audit counts only: the two W runs become `consistent`, `henn__release_only` keeps its overlaps with another count | the audit runs once per timestamp; the W overlaps existed only between two events of one timestamp | yes |

Tasks 4.13 (apart from F62) and 4.16 changed no recorded run. The 4.15 steps 1 and 5 and the
two speedups changed no decision, committed tour or KPI.

## CASOP applicability against `main`

For each of the seven CASOP data cards (`scenarios/casop_static/data_card/`), the applicable
algorithm cards with ware_ops_algos' `DomainAlgorithmMapper` and the CASOP taxonomy, at the
current branch and at `origin/main` (`4e697c4`):

| Data card | `main` | now | Added | Removed |
|---|---:|---:|---|---|
| bahceci_oencan | 23 | 24 | `FirstFitFiFo` | — |
| foodmart | 18 | 19 | `FirstFitFiFo` | — |
| henn_waescher | 23 | 24 | `FirstFitFiFo` | — |
| kris | 24 | 25 | `FirstFitFiFo` | — |
| muter_oencan | 23 | 24 | `FirstFitFiFo` | — |
| sprp | 8 | 8 | — | — |
| sprp_ss | 13 | 13 | — | — |

The one difference is the card added on purpose for the L2 route-length check (ware `b650516`,
first-fit FCFS batching). No card became inapplicable. The counts differ from those of Task 4.12
(19, 15, 19, 24, 19, 8, 13), which were taken from CASOP's instance-derived data cards rather than
these files; the comparison against `main` is what the criterion asks. Task 5.0 commits the
permanent lock.

## Type counts (method of Task 4.13)

Top-level classes in ware `planning/` and `algorithm_interfaces.py`, and in casim
`problem_based_template.py`, the routing, admission, batching, scheduling and waiting nodes,
`decision_engine/` and `state/apply.py`. The method reproduces 115 at the 4.13 commits.

| Group | Before 4.13 | After 4.13 | Now |
|---|---:|---:|---:|
| `planning` | 19 | 19 | 19 |
| `algorithm_interfaces` | 28 | 25 | 25 |
| `problem_based_template` | 27 | 29 | 28 (−`OrderSplitter`) |
| nodes (routing, admission, batching, scheduling, waiting) | 33 | 32 | 32 |
| `decision_engine` | 9 | 9 | 9 |
| `state/apply` | 1 | 1 | 1 |
| **Total** | **117** | **115** | **114** |

## Open for the user

1. Draft times after a delay (Task 4.15): project a resource's planned drafts after its queue, or
   leave drafts out of the overlap check. Until then `henn__release_only` stays `inconsistent`.
2. A terminal check for orders still open at the end of a run (Task 4.15): the old order-window
   records were `consistent` with orders never picked.
3. Section 9.1 item 8: whether `Resource.available` and `Resource.occupied` leave ware's domain
   model; casim no longer uses them.
4. The four Task 4.12 decisions (F46 and the named decisions of F47 to F49).
5. The 16 working-paper `truncated` L2 pilot records of `cf2d4a8`, re-recorded once the
   difference is accepted; the slim set keeps one working-paper `truncated` record.
6. From Phase B: the journal's high-load L2 cells remain a statistical non-reproduction without
   demonstrated cause.
