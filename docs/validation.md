# Validation

This document states what has been checked about CASIM, how, and what remains open. It separates
three questions: whether the simulation is internally sound, whether it reproduces published
results, and whether a change altered behaviour. No earlier output of this project counts as a
reference for correctness; a number counts as confirmed only when one of the checks below
confirms it.

## Internal checks

| Check | What it verifies | Where |
|---|---|---|
| Exhaustive checks over finite domains | Tour statuses only move forward and every status is reachable; the physical bound never increases along the lifecycle; a revision scope is accepted exactly when it respects the bound for every exposed status; a tour's accepted level never increases; `variant` is defined or rejects with a reason for every feature combination; overlapping write sets need distinct precedences | `ware_ops_algos/tests/verification/`, `tests/verification/test_taxonomy_variant.py` |
| Plan invariants | Every order in exactly one place; no overlapping tours of a picker; status changes follow the lifecycle; executed prefixes and finished tours never change; time never decreases | `ware_ops_algos.planning.invariants`, run by `--audit` after every timestamp and decision |
| Resource checks | At most one running tour per resource; no dispatch before the earliest start or while absent; no startable tour left waiting; queue times equal their projection; idle, busy and absent time sum to the horizon | `casim/state/resource_checks.py`, run by `--audit` |
| Open orders | The orders not completed at the end of a run, by status | `--audit`, recorded in every baseline |
| KPI recomputation | Every tour KPI recomputed from the recorded tours and route legs with the time model, independently of the trackers | `tests/verification/kpi_recomputation.py`, run on every baseline record |
| Hand-derived scenarios | Nine cases small enough to derive every value by hand: one picker and one order (end 39); wait-0 against wait-2; Henn's release time 2r + st_i − st_j; one-job commitment keeping a draft; a rejected edit of a frozen tour; admission keeping the executed prefix; window commitment; periodic decisions stopping at closure; the analytic waiting time 10 ln(4/e) | `tests/verification/test_phase_b_scenarios.py`, `scenarios/minimal` |
| Dispatch with two pickers | Rescheduling earlier, moving a tour to another picker, breaks with one busy and one idle picker, shift ends, binding and forecast starts, tours that end early or late, drafts behind a late tour | `tests/verification/test_dispatch_two_pickers.py` |
| Component smoke tests | Every CoSy component that no scenario or benchmark runs completes a static instance | `tests/test_component_smoke.py` |
| Static structure | No algorithm or policy name outside the subproblem components; import boundaries between the packages | `tests/test_no_algorithm_dispatch_strings.py`, `tests/verification/test_import_boundaries.py` |

## Literature benchmarks

Each benchmark has a kind: exact (CASIM must give the same value), statistical (the published
value must lie in CASIM's confidence interval), bound (CASIM runs a simpler rule than the
published method and must not beat the published value), or reproduction (the value of a
specific earlier implementation). The benchmark run lists, results and selection scripts are in
`benchmarks/`.

| Benchmark | Kind | Scenario | Status |
|---|---|---|---|
| L1: seven CASOP sets, static | exact | `casop_static` | run on 415 instance-pipeline pairs; equal |
| L2: Henn (2012), online FCFS batching | statistical | `henn2012` | run on the validation set; low and medium load reproduced, high load open |
| L3: OOBP, Gil-Borrás et al. (2020) | bound | `oobp` | run on 192 instances; 40 runs below a best-known value |
| L4: D'Haen, Braekers and Ramaekers (2023) | bound | `scenario_dhaen_online` (data only) | not run; needs two-block layouts, trucks and departures |
| L5: Walking vs. Waiting, RADAR standard case | reproduction | `walk_or_wait` | not run; needs the original simulator commit and its speed and pick time |
| L6 to L10: Lorenz et al.; Gil-Borrás et al. 2021, 2023; van Gils et al. 2019; D'Haen et al. 2024 | bound | none | not pursued: companion material or papers not available, or out of scope (vehicle routing) |

### L1: CASOP sets, exact

The seven CASOP benchmark sets (Bahçeci and Öncan, Foodmart, Henn and Wäscher, Muter and Öncan,
SPRP, SPRP-SS, Kris) run as static instances: every order and picker at t = 0, one decision,
every tour committed. Three comparisons per run:

1. CASIM against the same ware_ops_algos algorithms called directly: equal on 310 of 415 rows.
   The other 105 are Kris rows that the Kris loader defect below explains: CASIM executes the
   distance of the loader's faulty matrix, while the nearest-neighbour router reports the
   correct shortest-path distance.
2. The direct calls against the published CASOP release v0.1.0: equal on all 415 rows.
3. Against the literature: no CASIM result lies below a proven optimum. The one direct result
   below a marked optimum (Kris `instances_207_1_small`, 128,700 against 132,420) violates a due
   date. Kris results are compared with the best-known solution only when they meet every due
   date.

Ratliff and Rosenthal's routing now returns its node sequence; on all 65 of its rows the walked
distance, the sum of graph edges and the reported distance are equal.

### L2: Henn (2012), statistical

The protocol is taken from Henn's working paper (2009, FEMM 34) and checked against the journal
version: one block with front and rear cross aisle, 10 aisles of 2 × 45 cells, depot in front of
the leftmost aisle, 48 length units per minute, 10 s per item, 3 min setup, order sizes uniform on
5 to 25 items, capacity 45 or 75 items, 30 to 120 orders over 8 h with exponential interarrival
times, class-based (52/36/12 % of demand in aisles 1, 2 to 4, 5 to 10) or uniform storage,
S-shape routing. CASIM generates the instances with a seeded generator, 20 replications per
class, and runs the journal target (no waiting) and the working-paper target (Henn's waiting
rule) with the FIRST and SAV batch selection rules.

Verified:

- Journal, n ≤ 60: every FIRST cell lies within ±1.6 % of the published completion time and its
  95 % interval contains the published value (8 of 8).
- FIRST is Henn's first FCFS batch: shown by the code path and by
  `tests/verification/test_l2_first_selection.py` on the recorded FIRST runs.
- Henn's published turnover is the time from arrival to completion (C/30/45: 47.1 against 44),
  not the start-to-completion time the working paper's text states (about 18 min).
- The instances do not depend on the waiting parameter α, by construction: α is a parameter of
  the decision, not of the generator.
- SAV behaves like FIRST: the journal cells with n ≤ 60 contain the published value, and SAV
  completes at most 15 min earlier than FIRST in the same cell.

Open:

- **Journal, n = 90 and 120.** Every FIRST cell is above the published value and outside its
  interval, by 2.7 to 13.7 % (completion in minutes after the loader correction below):

  | Class | CASIM | Published | Gap |
  |---|---:|---:|---:|
  | C/90/45 | 680.7 | 598 | +13.8 % |
  | C/90/75 | 541.2 | 509 | +6.3 % |
  | C/120/45 | 888.5 | 787 | +12.9 % |
  | C/120/75 | 679.6 | 618 | +10.0 % |
  | U/90/45 | 773.9 | 744 | +4.0 % |
  | U/90/75 | 561.5 | 548 | +2.5 % |
  | U/120/45 | 1023.8 | 985 | +3.9 % |
  | U/120/75 | 724.1 | 700 | +3.4 % |

  At C/120/45 the picker is busy 883 of 887 min, so the gap lies in setup and travel. CASIM's
  executed tours equal an independent S-shape computation on all 120 checked runs. The journal's
  values are consistent with first-fit FCFS batching, the working paper's with next-fit, which
  CASIM implements and both papers' text describes. A first-fit variant (declared in
  `henn2012/sweep/variant_journal_first_fit.yaml`, labelled as inferred) closes about half of
  the class-based gap but falls below the published values on uniform storage, so no single
  fill rule fits the journal. This remains a statistical non-reproduction without a
  demonstrated cause.
- **Working paper, n = 30.** Completion is 6.7 to 9.4 % above the published value. This is a
  bound, not a defect: under the stated arrival process the expected last arrival is 464.5 min
  and the last batch needs at least about 3.8 min, so 467 min is not reachable.
- After the order-count gate was corrected (it ignored orders in planned tours), the recorded
  journal SAV runs changed by −1.4 % on average; FIRST is unchanged. The cell tables have not
  been recomputed over all 20 replications since.

### L3: OOBP, bound

The 128 instances of the OOBP archive (64 of Albareda-Sambola et al., warehouses W1 to W4; 64 of
Henn, warehouse W5) run with FCFS batching, selection of the batch with the oldest order, S-shape
routing and no waiting: 48 length units per minute, 6 items per minute, 3 min setup, capacity by
weight. The paper places the Henn depot in the centre, the benchmark page at the bottom left;
both are run. CASIM's rule is not the published GRASP with variable neighbourhood descent, so it
must not beat a best-known value.

Of the 192 runs, 40 lie below at least one best-known value (30 in completion time, 11 in
turnover time, one in both); they carry the status "protocol not reproduced". No documented
reading of capacity, setup, batch selection or turning time removes these gaps. All 192 runs
have no KPI recomputation difference. The correction of the order-count gate shortened the
completion time of 116 runs by 1.3 % on average; the count of runs below a best-known value has
not been recomputed since.

### Data provenance

- The Henn instances of `henn_waiting` are converted files of Heßler and Irnich. Their arrival
  times are identical to the OOBP archive, but their item positions differ (between 0 and 6 of
  40 to 100 orders match per file). Results on them are not comparable with published OOBP
  values. The OOBP benchmark reads the archive itself.
- The OOBP archive numbers storage locations from 0; the protocol puts the 45 cells 1 to 45 length
  units from the front cross aisle. CASIM maps location l to cell l + 1.
- The Kris instance files are consistent: every published batch distance is reproduced as a sum
  of shortest paths. ware_ops_pipes' loader moves every aisle intersection to the nearest x with
  a product; in 166 of 243 small instances an aisle without products makes several vertices one
  node and leaves wrong distance-matrix entries (103,278 entries). The fix belongs in
  ware_ops_pipes and is open.

## Defects found and fixed

| Defect | Effect | Where fixed |
|---|---|---|
| Setup time counted twice, in the planned start and again at dispatch | Every executed tour one setup too long | `earliest_free_assignment` (ware_ops_algos), dispatch |
| Snapshot reordered the order buffer | FIFO batching saw a different input order | `StateAdapter` sorts by arrival and order id |
| Waiting rule saw one candidate, and the engine duplicated Henn's rule | Wrong release decisions | Waiting stage over the whole schedule |
| A waiting delay retimed only the first job | Inconsistent schedules | Waiting delays every job of the picker |
| Ratliff and Rosenthal's routing returned no node sequence | Its tours could not be executed | ware_ops_algos routing |
| OOBP loader used location indices as depths | Tours 0.1 to 1.6 length units short | `casim.sources.oobp` |
| Validation rejected a suffix adding orders of a draft cancelled in the same decision | 246 of 480 L2 SAV runs and two Henn configurations failed | `ware_ops_algos.planning.validation`, `casim.state.apply` |
| A planned draft reserved its picker; dispatch had several mechanisms | Overlap findings, idle time misattributed | Derived resource state and one dispatch event |
| The order-count gate ignored orders in planned tours | Pickers idled until a stale reconsideration; up to 4 orders of the last order window were never picked | `NbrOrdersCondition` |
| A due-date window was switched off when the order stream closed | A decision saw no order until closure, then every order | `StateAdapter` |
| Planned drafts kept their times after a delay | Commitment policies read infeasible starts | Queue projection includes drafts |
| Stream end signalled by an event no code emitted, and that bypassed conditions | Dead configuration | Removed |

Open defects are listed in `docs/development.md`, among them the Kris loader, the disruption
event of the IJPE study that cancels tours itself, and the analytic waiting rule's precondition,
which is encoded as a decision condition.

## Comparison with the base version

The base version (casim `67f9f31`, ware_ops_algos `34bd157`) is a point of comparison, not a
target. `scripts/compare_with_base.py` compares the three scenarios that existed then.

| Scenario | Orders | Tours | Makespan | Distance | Mean order completion |
|---|---:|---:|---:|---:|---:|
| Walk-or-wait, small example (base) | 6 | 3 | 167 | 160 | 107.167 |
| Walk-or-wait, small example (now) | 6 | 3 | 167 | 160 | 107.167 |
| Analytic waiting (base) | 4 | 2 | 87.085 | 76 | 54.835 |
| Analytic waiting (now) | 4 | 2 | 87.085 | 76 | 54.835 |
| Henn waiting, `72s-100-75-0` (base) | 100 | 25 | 34,702 | 11,603 | 14,584.174 |
| Henn waiting, `72s-100-75-0` (now) | 100 | 27 | 30,668.75 | 11,808 | 12,572.582 |

Walk-or-wait and analytic waiting are identical to the base. Henn differs, and its values have
not changed since the regression baseline was first recorded, so the difference comes from
changes made before that. Four changes alter the Henn decisions, each a corrected defect or a
change of the model:

1. The setup time is counted once instead of twice, so every executed tour is one setup (180 s)
   shorter.
2. The waiting rule receives the whole schedule instead of one candidate.
3. The decision's order buffer is sorted by arrival and order id; the base passed it in another
   order, which changed FIFO batching.
4. The uncommitted part of a solution stays in the plan as planned tours that the next decision
   may revise (one committed tour per decision instead of a solution that contained one job),
   and a due reconsideration solves again instead of committing an old planned tour.

The difference has not been decomposed among these four changes; that decomposition is on the
backlog (`docs/development.md`). The decision counts are not comparable: the base counted
decisions that committed a tour (26), CASIM now counts every decision call (128).

## Baseline tiers

The baselines detect behaviour changes. They are a regression set with one run per code path,
not a study: 26 scenario configurations (the five IJPE configurations are not runnable yet), 13
L1 runs (one per instance set and pipeline), 10 L2 runs (one cell per target, storage, rule and
arrival mode, plus a high-load and a capacity-75 cell) and 16 L3 runs (one per stratum and
depot). Each record holds the KPIs, the decisions, the committed tours, the audit findings, the
open orders, the event count and the trace digest. All 60 records are consistent: no invariant
or resource finding and no KPI recomputation difference.

| Tier | Command | When | Runs, time |
|---|---|---|---|
| Fast | `python scripts/baseline.py compare-fast --workers 4` | after every commit | 11 runs, about 1 min |
| Full | `python scripts/baseline.py compare` and `compare-benchmark --benchmark l1`, `l2`, `l3` | at the end of a change | 60 runs, about 3 min on six workers |

`record --case <name>` and `record-benchmark --match <glob>` record runs again. A change is
committed together with the baseline difference it causes, explained; records are re-recorded
in a separate commit. The literature tables above are produced by the scenarios' own sweeps and
`plots.py`, not by the baseline.
