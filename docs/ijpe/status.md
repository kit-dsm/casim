# Status of the IJPE revision

Plan: `docs/ijpe/plan.md`. Newest entry first.

## 2026-11-21: P2.1 at the two volumes of the lean design

**Done.** `scripts/ijpe/disruptions.py` writes one selection per scenario
and volume, largest stores first: V = 1 (`*_v1_fixed`, the mail's TE × 0.20)
and V = 5 (`*_v5_fixed`, the mail's full TE all in the area). The random
selections are dropped (plan P4.2). The V = 1 lists are unchanged; S3's V = 1
order is a new draw (25 lines, −2 % of the target), because the random draws
no longer come before it.

**Targets against achieved TE.** S1 24 / +12 % and 120 / +1 %; S2 20 / +21 % and
100 / +1 %; S4 40 / +12 % and 200 / −24 %; S3 1.2 / −2 % and 6 / +2 % (one
order of 82 lines). At full volume Thursday's 10:00 slot holds 152 TE, so S4
takes the whole slot (45 stores, 72 orders) and reports the shortfall.

**Check.** One run of S4 at full volume (seed 0, EDD; deadlines still under
discussion): none of the 72 moved orders is late, while late orders over the
week rise from 136 to 220. `tests/verification/test_ijpe_week_variants.py`: 9
pass. `casim run` on the week fails after C13 (`No module named 'scenarios'`);
`python -m casim run` works; reported to the due-dates session.

## 2026-11-21: P1.7, credibility profile and validity envelope

**Done.** `docs/ijpe/credibility.md` holds the profile table (10 components
with metric, held-out basis, result and verdict), the four figures in
`docs/ijpe/figures/` (profile table, routing ratio by order size, ECDF of
one-person order durations net of breaks replacing the submitted Figure 8,
cumulative completed orders per day against the 90 % band of C), the
validity envelope by clock time, the pre-picking caveat, and every planned
P4 KPI tagged inside or outside the envelope. `scripts/ijpe/credibility.py`
adds no fitting and no runs. It reads `calibrated_recorded` and
`capacity_fifo` and reproduces the numbers of `validation.md` exactly
(routing medians 0.81 / 0.99 / 1.00 / 1.05; durations 18.5 / 5.4 / 15.7 /
35.2 min; 600 and 956 orders by 10:00 and 12:00). Figures show quantiles
and daily counts only.

**Envelope.** Up to 12:00, as comparisons between policies with common
random numbers. Six-day levels are +2 %, −4 % and −7 % at 08:00, 10:00 and
12:00; the day pattern is not reproduced at any clock time (WMS inside the
day band on 2 of 6 days). From 14:00 it is outside: the total is close only
because the days run out of work. Outside: the 13:30 and 15:30 groups, the
whole day, truck waiting of the 14:00 and 16:00 groups, S4's whole-day
effect, end-of-day and picker measures, and busy share.

**G3 held out** (`scripts/ijpe/generator_holdout.py`, fitted Mon–Wed,
generating Thu, Fri and Sat). Partly reproduced. In sample it matches its
days. Held out, it gives the calibration mix: Thursday 46.8 against 43.3
lines per order, 3.38 against 3.73 aisles, 1.06 against 1.01 m³, mostly
outside the seed band. Saturday and Friday orders are smaller (0.84 and
0.81 m³ against 0.98 to 1.24 on Mon–Wed). Quantity times package volume
would overstate volume 15.8 times; G3 uses the package-based volume.

**For the user.** The paper should claim G3 as a generator of a calibration
weekday, not as a forecast of an unseen day. The deadline rework of C13's
session does not change the tags, which follow the deadline, not the
scenario.

## 2026-11-21: C13, exogenous rules back into the scenario

**Decision (user).** `casim.sources.exogenous` is removed from the core: its
rules served only the IJPE week and worked only with `order_stream`. A
scenario's source produces its exogenous events (AGENTS rule 3,
`docs/scenarios.md`); a helper moves into the core only when a second scenario
needs it.

**Done.** `scenarios/ijpe_week/source.py` wraps `order_stream` and produces
the roster, due dates, process times and disruptions, with the parameters
under `source`; the variants name `source.disruptions.<name>` with a `type`.
The store slot (`store_slot`) and the fitted times are shared with the day
source. `scripts/ijpe/disruptions.py` reads the slot rule from there. A
configuration with an `exogenous:` block is refused. `casim.sources.wms_export`
moved to `scripts/ijpe/`. P2.1's uncommitted work was committed first as it
stood, at the user's request.

**Checks.** Week default and replay-day baselines identical; P2.1's variants
give the same facts under the old and the new code; `compare-fast` equal for
every runnable case; `casim check --all` passes. The 13 failing component smoke
tests fail on the base commit too (CASOP instance folders empty in this
checkout).

## 2026-11-21: P2.1, the scenario spec in the week (deadlines under discussion)

**Done, without a core change.** `scripts/ijpe/canonical.py` adds `weight`
and `volume` (m³: the line's weight over the package weight of the article
file, times VOLUMEN_KOLLI) to the canonical lines; every line has a package
weight, the 1,414 orders hold 1,465 m³ (2.3 TE per order); old columns and
all IJPE baselines unchanged. `scripts/ijpe/disruptions.py` writes the
selections: for V = 0.5 to 4 times the target (mail TE × 0.20), whole stores
of the original departure's slot (S1 Monday and S2 Thursday from the 14:00
slot, S4 Thursday from the 10:00 slot), largest first (`*_fixed`) and 30
random selections within 15 % of the target (`*_rNN`), as named lists; S3 as
one new store order drawn by G3 within 15 % of the target, as templates. A
store's slot comes from the week's own dispatch rule. The variants use plain
deadlines with staging (S1 09:30, S2 10:30 requested at 09:30, S3 and S4
09:30); `scenario2_short` is the request at 10:30. The order generator now
writes a volume per line (0.99 m³ per order against 1.04 recorded).

**Findings.** The fixed rule overshoots small targets (S1 +67 % at V = 0.5,
S2 +21 % at V = 1), so results are plotted against the achieved TE. Thursday's
10:00 slot holds 152 TE, less than S4's target at V = 4 (160 TE).

**Correction (same day).** An earlier version of this entry said the export shows no
pre-picking. That check only found orders split across booking dates; the export
has no delivery date, so it cannot show pre-picking at all, and the manuscript
(Section 4.1) says early windows are picked the day before. S4 rests on the
stated assumption of own-day release; the partner is asked.

**Runs** (seed 0, EDD, V = 1; affected / affected late / all late / orders on
time in the baseline that become late): baseline 0 / 0 / 136 / 0; S1 6 / 0 /
133 / 2; S2 5 / 1 / 141 / 5; S2 short notice 5 / 5 (up to 60 min) / 132 / 3;
S3 1 / 0 / 135 / 1; S4 12 / 0 / 124 / 9; S2+S4 17 / 1 / 124 / 10. At the
realistic volume the scenarios barely bite; the sweep of P4.2 carries the
result.

**Checks.** `tests/verification/test_ijpe_week_variants.py` (9 tests: whole
stores of the slot, deadlines, the new order's volume, S2+S4 disjoint, random
selections within 15 %). The baseline case `ijpe_week__scenario2` changes with
the new Scenario 2 and needs re-recording. Another session is redesigning
`casim.sources.exogenous`; P2.1 uses only the existing rule kinds.

## 2026-11-21: Name and generator release (plan only)

**Decision (user).** The case is named `grocery_retailer`: the public
experiment scenario `scenarios/grocery_retailer` (now `ijpe_week`), the
confidential validation `scenarios/grocery_retailer_validation` (now
`ijpe`), and `scripts/grocery_retailer` (now `scripts/ijpe`). The rename is
one commit with the package build, when no session has uncommitted work on
these paths (the C12 work touches `scenarios/ijpe_week`, `scripts/baseline.py`
and tests at present). The generator (G3) is published with the generated
week; the calibration fitted on the recorded data is not.

## 2026-11-21: How close a generated week comes (P5 release)

**Done (not committed).** `scripts/ijpe/order_generator.py` fits three
generators on the canonical tables and writes generated weeks in the same
format to the ignored `scenarios/ijpe/data/generated/`. G0 is the submitted
generator's logic (articles by frequency over the whole area) with stores
added; G1 is zone-aware; G2 is G1 fitted per day with articles drawn within
the zone's aisles. Zones come from the data: aisles 1 to 6 and 7 to 8; a
store-day has one order per zone (377 of the two-order store-days), and
aisles 7 and 8 hold 52 % of the lines.

**Stream** (mean of three seeds against recorded): aisles per order G0 7.07,
G2 3.44, recorded 3.48; orders within two aisles G0 6 %, G2 62 %, recorded
61 %; lines per order p10/p50/p90 equal within one line. Still off in G2:
quantity per line p90 36 against 47, the largest tenth of store-days holds
22 % of lines against 24 %, total lines +2 %.

**Week run** (EDD, fitted times, the same three time seeds): route per tour G0
1,065 m, G2 622 m, recorded 628 m; tour 25.6 against 25.6 min. Lines done by
08:00, 10:00, 12:00 and 14:00 in G2 are within 1 % of the recorded week; lines
carried or open +23 %, which is G2's +2 % load. Orders done by 10:00 are 9 %
fewer, because the hashed dispatch slots put smaller real stores into the
09:30 slot by chance (40.7 against 45.9 lines per order there). G0 is not
usable (57 % late against 10 %).

**G3** (same day). G2 plus: store-day profiles (zone orders and their line
counts) resampled from the same day, keeping the draw closest to the day's
recorded lines; articles included with probability proportional to their
share (systematic sampling); each line's quantity a quantile of its article's
recorded quantities through a normal score with a per-store effect (18 % of
the score variance lies between stores). Stream: lines 62,988 against
62,985, quantity per line p90 48 against 47, largest tenth of store-days 24 %
against 24 %, top 50 articles 23 % against 24 %, aisles per order 3.46
against 3.48. Week runs with ten seeds each: lines done by 08:00, 10:00,
12:00 and 14:00 are 1.06, 1.03, 1.02 and 1.02 times the recorded week, lines
left over 0.92, late share 11.7 against 13.1 %; every G3 mean lies inside the
recorded week's 5 to 95 % band over seeds. The gap in leftover work seen
with three seeds was noise (recorded band 3,294 to 10,557 lines). Friday's
tours are longer in G3 (24.6 against 21.6 min, three seeds): its article mix
is pooled over the week.

**Next.** Build the package from G3 (one fixed seed) with the percentile time
model, and send it to the partner for approval.

## 2026-11-20: Public data package (plan only)

**Decision (user).** The release (P5) is one public data package in the
repository, read by `scenarios/ijpe_week` by default, and the paper's
experiments run on it; the confidential tables only build it and run the
validation. Content: the pseudonymised recorded orders (A, as SLAPStack
published WEPAStacks) if the partner approves, otherwise a generated week
(B). The time model is published as percentiles in the format of
`week.json`; pseudonyms use a private seed. The generator calibration is not
published; S3's order and the store selections come from the package with
public code. The plan's ground rule names the package as its one exception,
committed only after written approval.

**Partner (same day).** The recorded orders may not be published, even
pseudonymised; a generated or fitted stream may. The package is therefore a
generated week (B); its fidelity is measured next.

## 2026-11-20: Submitted results reproducible on main; P2.2 cut down

**Decision (user).** The submitted version is reproduced on `main`, and this
branch builds only the revised version; the two are not compared here, since
they rest on different assumptions.

**Done.** `main` `8756ffd` (pushed) adds the converter that built the
submitted stream (it was in no repository), an ignore rule for the partner's
data and `scenarios/scenario_ijpe/README.md` with inputs, commands and the
reproduced tables. The submitted code stays at the published tag `ijpe-2026`;
ware_ops_algos `32cdff5` has the tag `ijpe-2026` too (pushed). Run with the
paper's own scripts: Tables 4 and 5 match (utilisation 47.65 against
47.75 %), Table 6 matches in structure; Scenario 2 has 47 affected orders
instead of 50, and the waits of Scenarios 1 and 2 differ slightly, cause not
determined. Table 6's "delayed departures" are hourly due-time groups, which
answers Rev2's Table 6 questions; C5 is not needed. The reproduction tooling
committed on this branch earlier today was removed again (unpushed commit
dropped).

**Run time** (same laptop, alternating, CPU time per week run): submitted
median 30 s, current with `travel: pick` median 49 s; the difference is the
pull's 1,344 decisions per week against 12.

## 2026-11-20: C12, run time of a week run; IJPE runs out of pytest

**Decision (user).** The core change C12 (plan, Section 4) and moving the
IJPE scenario runs from pytest to the baseline tier.

**Profile** (one baseline week run, 75 s on a quiet machine; the same run took
up to 147 s with the laptop in a slower state): decision snapshot 31 %
(the storage copied for each of 1,344 decisions), resource-state logging 19 %
(after each of 1.47 million events), solving 16 %, the rest mostly handling
the 1.47 million travel and arrival events, two per graph edge.

**Done.** C12 (a) to (c). The week with `travel: pick` has 197,384 events
instead of 1,467,924, and `facts/tours.csv`, `picks.csv`, `orders.csv`,
`service.json` are byte-identical to edge travel; `kpis.json` and
`tracker.json` have equal values (key order differs). Same for Scenario 2.
Edge-mode digests of the week and Scenario 2 are unchanged by (a) and (b).
`scenarios/ijpe_week` sets `travel: pick`. `ijpe` and `ijpe_week` are no
longer run by `tests/test_scenarios.py` (confidential data, minutes per run;
the week with the per-event audit took hours); `ijpe_week__default` and
`ijpe_week__scenario2` are new data-dependent baseline cases, recorded with
`travel: pick`.

**Checks.** `tests/test_resource_state_log.py` (logging after every event gives
the same trace and KPIs, three configurations of `recorded_replay`);
`tests/test_travel_granularity.py` (edge and pick travel give equal facts,
KPIs, audit legs, invariant checks and decision views on three
configurations; `edge_at` inside a leg; refusal where in-between nodes
matter). `compare-fast`: all runnable cases equal; six cases not runnable
because their data files are missing in this checkout (before the change too).
`casim check --all`: only the untracked `scenario_ijpe` and `scenario_dm`.

**Open.** Not yet fixed: `_copy_memo` costs about 7 s per week run, and
`build_jobs` recomputes every buffered tour's processing time at each decision.

## 2026-11-20: Plan aligned with the scenario spec; runtime of the sweep (plan only)

**Decision (user).** P2.1 follows `docs/ijpe/scenarios.md`: volume target =
the mail's TE × 0.20, TE per order by goods volume, whole stores and dispatch
groups, S3 from the volume-conditioned generator with the copy as fallback,
S4 following the release practice measured from the export, the "13:00"
origin as the 14:00 slot, staging on every scenario deadline, no workbook
number quoted, alternative store selections per replication. The main
what-if result (P4.2) is the sweep of advanced volume (0 to 4 × target) ×
staffing (recorded, −2, −4 pickers), realistic volumes marked, bottleneck
onset as a band; busy share is no longer a load target. P4.7 slim: three
alternatives at 1×, 2×, 4× target, request time fixed, recorded roster; truck
waiting from dispatch-group completion against the staging deadline,
computed from `facts/orders.csv`, so P4.7 no longer depends on C5 (C5 stays
for P2.2). P1.7's generator check adds volume per order and the MENGE_IST
unit check. Partner questions are non-blocking, each with a fallback; the
TYP/LHM_TYP/VOLUMEN_KOLLI question is closed.

**Runtime estimate** (re-timed after C12, `travel: pick`, 197k events per
week run). Measured on this laptop (8 cores, 16 logical, 28 GB), EDD,
current list-based variants. Single runs: 31 to 32 s for the baseline, S2 and
S4 with FIFO, one outlier at 52 s. In parallel, seeds 1 to 8 or 1 to 12 of
the baseline: 8 runs in 50 s and 12 runs in 69 s wall clock, so about 5.7 s
per run at 12 in parallel; all runs completed. A stale pytest from 19
November that had run the week with the per-event audit for almost a day was
stopped first; the earlier 2.1 to 2.6 min per run were measured with edge
travel and that process competing. The sweep's runs per replication: per
staffing level one baseline plus 5 scenarios × 6 non-zero volumes = 31, × 3
staffing levels = 93.

| Part | Runs | CPU hours | Wall clock, 12 in parallel |
|---|---:|---:|---:|
| P4.2 sweep, submitted policy, 10 replications | 930 | 8 | about 1.5 h |
| P4.2 sweep, submitted policy, 30 replications | 2,790 | 24 | about 4.5 h |
| P4.1 policies (3 more) at V = 1, 2, 4, recorded roster, 5 scenarios, 30 replications | 1,440 | 12 | about 2.5 h |
| All 4 policies on the full sweep, 30 replications | 11,160 | 96 | about 18 h |
| P4.7 slim (only "accept plus pull forward" is new), 30 replications | 90 | 0.8 | about 10 min |

Runs with the new volume-based selections are not timed yet; larger advanced
volumes may take longer, and single-run times on this laptop vary (52 s
outlier here; 75 to 147 s for identical edge-travel runs in another
session). Proposed: the sweep with the submitted policy and the policies at
selected points, about 4,200 runs or 7 h wall clock for 30 replications (one
night); all policies on the full grid would also fit in one day.

## 2026-11-20: Positioning in three parts (plan only)

**Decision (user).** The revised paper is organised as credibility,
what-if analysis and decision support (plan, "Positioning of the revised
paper"). New tasks: P1.7 credibility profile (must), P4.7 cost-based
decision study on an S2-type early-delivery request (slim version must,
full grid could; fallback from the P4.1 Scenario 2 differences). Its
per-event savings feed a break-even NPV (events per year needed to pay
back). The final form of the financial section is the main author's
decision. P1.5 is covered by the fitted line time; P1.6 is dropped. The
schedule now finishes the experiments by 13 December and starts the P5 parts
that do not depend on results now. The KPIs of P4 are checked against the
validity envelope of P1.7. Staffing enters only as a sensitivity, also as a
priced one in P4.7, never as a decision alternative. Truck staging times are
not optimised while the dispatch plan is synthetic.

**Open.** From the partner: departure plan, picker-hour cost, lost-sales
value, any lateness penalty, and the yearly frequency of each disruption
type. To the main author: the changed results from P0 and P2, and the form
of the financial section.

## 2026-11-20: Week scenarios on the warehouse's lists (P2)

**Decision (user).** The week's disruption variants are the warehouse's
lists S1, S2, S3 and S5 as defined in `scenarios/ijpe_week/variants/`:
the submitted due times as picking deadlines, a deadline applied only if
earlier than the order's slot deadline, Scenario 3 on Wednesday.

**Checks.** `tests/verification/test_ijpe_week_variants.py` (on the data):
S1 advances 30 orders, S2 27 (listed orders due earlier keep their
deadline), S3 adds three template copies on Wednesday 08:30, S4 moves 63
Thursday orders of S5 to Wednesday 06:30, S2+S4 marks the union.

**Runs** (seed 0, EDD, one each; affected orders late / all orders late of
1,414): S1 0 of 30 / 113; S2 15 of 27 / 133; S3 0 of 3 / 110 (1,417 orders);
S4 0 of 63 / 294; S2+S4 7 of 76 / 307. Orders still open at the end of the
stream are not counted as late here.

## 2026-11-20: Disruptions only on the week

**Decision (user).** The experiments use only the week (`scenarios/ijpe_week`,
one order stream, exogenous rules), not single days.

**Done.** The single-day disruptions of P2.1 are removed from
`scenarios/ijpe` (`DISRUPTIONS`, `source.disruption`, the arriving orders,
`listed_orders`, `template_positions`, `tests/verification/test_ijpe_disruptions.py`);
the single-day replay and validation are unchanged (the `OrderArrival`
trigger of `decisions/pull.yaml` stays, harmless). P2.1 had found, on the
warehouse's lists: S1 (Mon 07:30, due 10:00) affects 30 orders, S2 (Thu
10:30, due 11:00) 27, S3 (Wed 08:30) three copies of the template (6 TU /
2.6), S4 (Wed 06:30) 63 Thursday orders of S5; the week variants reproduce
these rules. The user accepted these definitions (entry above).

## 2026-11-19: Everything reads the canonical tables

**Done.** `scripts/ijpe/canonical.py` is the only reader of the WMS workbook
(besides the data audit). It writes `scenarios/ijpe/data/canonical/`:
`orders.csv`, `lines.csv` (the cohort with recorded login, start, end and
sheet row), `records.csv` (every record, for the work the fit counts outside
the replayed lines), `excluded.csv`, `lists.csv`, `templates/S3.csv`,
`layout.csv`. The IJPE source (`day_lines`, `listed_orders`,
`template_positions`), the fit, the validation and diagnostic scripts and the
week (`order_stream`, `source.tables`) read only these tables;
`casim.sources.wms_export` remains for the conversion and the audit.

**Identical results.** The fit files (six days and the week) are
byte-identical to those from the workbook; the six ijpe_replay baselines are
equal; trace digests of a calibrated day, a fitted day with Scenario 4 and
the week with Scenario 4 equal those of the previous commit. The cohort
frames agree column by column (quantities are integers instead of floats,
with equal values).

## 2026-11-19: The week as one order stream; canonical data and exogenous rules (C10, C11)

**Decision (user).** The experiments simulate the week as one order stream,
as the submitted model did, with a fixed shift (05:30 to 14:30) and break
roster; disruptions are declared in the configuration, as the submitted
hooks were, but as a core capability for any scenario; no simulation code
reads the workbook.

**Done.** C10: the library source `order_stream` reads canonical tables
(`orders.csv`, `lines.csv`, layout, named lists, template orders), and the
`exogenous` rules of `scenario.yaml` (`roster`, `dispatch_slots`,
`due_change`, `order_move`, `order_injection`, `process_times`) wrap any
source with an epoch (`casim/sources/exogenous.py`, `docs/concepts.md`).
`process_truth.drawn` moved from the IJPE source into the core, unchanged.
C11: `SchedulingCommitmentPolicy(per_resource: true)` gives every idle picker
its next tour; with one tour per decision, pickers arriving at the same
second never started (13 of 21 worked on a single day with a common shift
start, 3 of 21 in the week). The default is unchanged. `scripts/ijpe/canonical.py`
writes the IJPE tables; `scenarios/ijpe_week` is the week with the paper's
scenarios as variants (`scenario1` to `scenario4`, `scenario2_4`).

**Staffing.** Pickers per day are the record's picker-hours divided by the
9-hour shift (12, 13, 14, 16, 13, 4; the record has 21, 17, 16, 19, 15, 7
logins, many present for part of the day). With every login on the full
shift the week ran 259 orders ahead of the WMS by 10:00.

**Result** (seed 0, EDD, no disruption). 632 orders by 10:00 against the
WMS's 626 over six days, 938 by 12:00 against 1,031; Monday and Tuesday
within 13 orders at both times, Saturday 36 and Wednesday and Thursday 23
and 27 behind by 12:00; 108 orders carry over to the next morning, 30 Friday
orders are open when the stream ends. Under EDD, Scenario 1's 30 orders are
on time, Scenario 2 leaves 15 of 27 late, Scenario 4's 63 moved orders are on
time while 127 other Wednesday orders are late.

**Open.** The pull of `scenarios/ijpe` does not yet use `per_resource`.

**Checks.** `tests/test_exogenous.py` (synthetic order stream, every rule),
a commitment test for `per_resource`; the decision schema regenerated;
compare-fast and the six ijpe_replay baselines are equal; pytest has the 13
`test_component_smoke` failures that HEAD has; `casim check --all` reports
only the untracked `scenario_ijpe` and `scenario_dm` folders.

## 2026-11-19: Code structure: one output record, packages instead of loose files

**Done (no change in results).** Completed tours are a typed `CompletedTour`
instead of an 8-element tuple read by position. Every output file has one
writer (`casim/outputs/write.py`): `kpis.json`, `tracker.json`,
`decisions.json`, `service.json`, `trace.json` and `facts/`. The duplicates
are gone (`kpis/` with `kpis.pkl`, a second `kpis.json`, `tracker.json` and
`decisions.json`; the top-level `tours.csv` and `orders.csv`). `facts/tours.csv`
now holds every committed tour with its status, also one that never started,
and names its columns by what they mean (`start` of the setup, `route_distance`
planned, `distance` walked). The top level of `casim` is `cli.py`, `run.py`
and `__main__.py`; the rest moved into `casim/outputs/` (tracker, kpis, facts,
service, trace, audit, reference, write), `casim/config/` (setup,
scenario_check), `casim/simulation_engine/event_logger.py`,
`casim/viz/logger.py` (the Dash recorder) and `casim/sources/base.py`
(`DataLoader`); `runner.py` is in `run.py`, `io_helpers.py` is gone. The old
submitted scenario `scenarios/scenario_ijpe` is removed (it no longer ran; tag
`archive/scenario-ijpe-submitted`). Baselines, the default IJPE replay,
`submitted` and the calibrated day have unchanged trace digests.

## 2026-11-18: Fewer IJPE settings, variants and scripts

**Done.** Removed the settled diagnostics: `source.origin_mode` and
`source.sequence_clock` (and their support in `casim/sources/recorded.py`);
15 of 20 variants, kept are `calibrated`, `submitted`, `fixed_dispatch`,
`recorded_presence` and `system_fifo`; the sweeps `variants` and
`second_order`; `cosy_repo/recorded_greedy`; the scripts `pause_sweep`,
`origin_diagnostic`, `blockers`, `completion_gaps` and `trace_diagnosis`.
`source.breaks` and `source.absence` are one setting, `source.absences`:
`recorded`, `fitted` or a list of fixed breaks; all become timed
`PickerDeparture`s. Baselines, the calibrated day and `submitted` are
unchanged (same trace digests).

## 2026-11-18: FIFO as a pull; push and pull in the core

**Found.** The open orders of the calibrated FIFO day (141 over six days)
were not missing capacity: the FIFO decision planned the whole day at 05:30
into fixed picker queues and never revised it, also for pickers not yet
arrived or leaving early. Tours stayed in the queues of pickers who had
left while others stood idle (Monday: 52 orders open, 11 pickers idle more
than 10 minutes before their recorded end, 22 h in total). The graph and
U-shape routing are not the cause: for tours of 11 to 60 lines the
recorded walk through the lines has the same length as CASIM's route.

**Done.** FIFO as a pull with the existing concepts, batching and assignment
separated as in the paper (`decisions/pull.yaml`): OBP makes the recorded
tours into pick lists once; ORSP, triggered by `PickerIdle` and
`PickerArrival`, sees the idle pickers, routes the open pick lists and commits
one tour. Setup rejects a component that plans for every picker in a decision
triggered by `PickerIdle` (`supports_pull = False`: `RecordedPlan`). A first
version added a separate pull commitment policy; it gave the same days and
was removed. `RecordedIA` assigns the open orders
only, so the recorded batches can be planned again at each decision. Every
run reports in `service.json` the committed tours stranded with a picker who
is away at the end (`stranded`). Tests in `tests/test_pull_assignment.py`.
IJPE: `decisions/pull.yaml`; the calibrated day uses it; the recorded
decision now sees every picker itself (`off_shift`), so the variants no
longer set it. All baselines are unchanged.

**Result.** Calibrated FIFO as a pull: 5 open orders over six days (push:
141), picker finish RMSE 28 minutes (push: 74; one Monday picker whose
recorded absences run back to back accounts for most of it), the day's last
line +4 minutes on average, orders by 12:00 within 1 % of the WMS. Fitted absences
(breaks plus allowance by start-up, regular time and wind-down) with pull:
1,085 orders by 12:00 against 1,031 (Monday's unusually high absence is most
of the gap), 12 open orders over six days (`validation.md`).

## 2026-11-18: Fitted absences for the experiments

**Decision.** The paper validates in two steps: trace-driven (recorded
decisions and absences, isolating the time model) and distribution-driven
(fitted absences). The experiments use the second: the roster stays an
input, absences during the shift are drawn from a model fitted on the other
days, with replications and common seeds across policies.

**Done.** `source.absence=stochastic` with `source.absence_seed`
(`casim/sources/absence.py`; the recorded source takes given absences;
tests in `tests/verification/test_ijpe_absence.py`): a Poisson process of
absence requests per picker with a rate per clock hour, and durations drawn
from recorded absences near the same clock time. Validation with 30
replications per day (`scripts/ijpe/absence_validation.py`).

**Result.** With FIFO assignment the fitted absences reproduce the six days
as well as the recorded ones: orders by 12:00 1,041 (90 % band 980 to
1,075) against 1,031 in the WMS, open work as in the trace-driven run. At
10:00 both runs are 6 to 8 % ahead. Single days with an unusual absence
level (Monday, Friday) are not reproduced. With recorded assignment the
fitted absences do not fit, because the recorded work plan is matched to each
picker's own absences; that combination is not used. Details in
`docs/ijpe/validation.md`.

**For the user.** Ask the use-case partner what pickers do when not picking
(the peaks around 08:00 and 12:00 look like scheduled breaks or other tasks).
If they are scheduled breaks, the model can take them as fixed and keep the
rest random, and the paper can explain the 35 % instead of only measuring it.

## 2026-11-18: Stops inside tours; the calibrated day

**Found.** On the calibrated day after C7, pickers still finished about an
hour early, although per-order time was unbiased. Matched tour by tour, the
missing time is stops inside tours: 28 % of recorded tours contain a stop
of five minutes or more (a gap between lines or one long line), 99 h on the
six days. Between tours such gaps were already absences; inside tours they
were dropped. Without them, CASIM's tours are slightly too slow: the
submitted 19 s per line had absorbed part of the stops. The earlier "an
hour early" also mixed two groups: with recorded presence a picker who runs
late leaves with work open instead of finishing late, so picker finish
times are censored.

**Done.** `source.stops_in_tours` (the recorded source, default off, so no
baseline changes; hand-derived test) makes such stops timed absences that
begin when the running tour ends. The line time is refitted net of stops:
16.26 s on Monday to Wednesday, tour bias -0.15 min and RMSE 6.0 min on the
held-out days (`scripts/ijpe/line_time.py`). `variants=calibrated` is the
calibrated day. `scripts/ijpe/system_validation.py` runs it with recorded
and FIFO assignment and, as the uncensored picker check, without forced
departures.

**Result.** Orders completed by 10:00 and 12:00 are within 15 of the WMS on
every day (recorded assignment; FIFO within 10). The picker-day bias is
-10 minutes without forced departures (-78 before), RMSE an hour. The end
of the day is not reproduced. The earlier entry of today, with the
pre-stops numbers, is superseded.

**Decision.** `variants=calibrated` is the base of the disruption and
policy scenarios. Claims stay as in the P1.3 entry: policy comparisons on
throughput and on-time completion up to midday, no picker-level or
end-of-day effects.

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
