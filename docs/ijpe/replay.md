# IJPE recorded replay: what limits reproduction

**Current-run note (17 November 2026).** The numerical replay and variant
results below were produced before a correction that preserves the exact
recorded pick order when several orders share one graph node. A proposed
aisle-column rescaling was rejected after checking Jonas Koreis's stated
aisle-change distance: the submitted loader's original scale reproduces it.
The earlier numbers are retained as experiment history and must not be cited
as the current replay result. The new aggregate diagnosis is in
`scripts/ijpe/trace_diagnosis.py`; it checks the measured cross-aisle edge,
exact tour sequence, physical feasibility of the WMS line intervals and the
held-out line-time fit. The full-order replay still has substantial error,
which fitting ordinary line times alone does not remove.

The corrected exact-sequence run has 16.273 min pooled order-span RMSE for
1,414 orders (16.205 min on the later Thursday and Friday). Its 1,172 inferred
tours reproduce the WMS line order exactly. The day RMSEs in date order
(11, 13, 14, 15, 16, 17 April) are 12.808, 15.263, 14.380, 20.517, 17.430
and 11.146 min. Single-order tours contain 1,036 orders with 14.145 min RMSE;
378 orders in multi-order tours have 21.029 min
RMSE and contribute 44.6% of total squared error. At a transition changing
both order and aisle, 2,539 of 2,871 next-line starts precede the previous
line's end. This is evidence that the export's interleaved booking intervals
cannot be read as a fully sequential physical route. An order-span lower-bound
check using only interior graph legs and zero pick time fails for 29 orders,
but only one of these is in a single-order tour; interleaving causes most of
that contradiction. A separate diagnostic crosses interleaving, long recorded
intervals and order-internal idle gaps in `scripts/ijpe/blockers.py`. These
outcome-defined cohorts are for diagnosis, not a filtered paper benchmark.

For the single-order tours, sorting lines by WMS order-position number instead
of their timestamps reduces summed interior graph travel from 372.5 to 347.3
km. At the submitted 2.306 m/s this is about 10.5 s per order on average,
far below the 14.145 min order-span RMSE; route ordering is therefore not the
main explanation. Of the 3,660 same-order aisle changes, 859 recorded line
intervals are shorter than the graph walk alone at 2.306 m/s, and 407 remain
shorter even at 4 m/s. Their path or timestamp interpretation needs care.
Among those 407, 246 reverse the WMS position numbering, which supports a
sequence or booking-order explanation for many but not all of them.
The near-zero pooled bias also hides opposite errors: single-order tours with
long recorded lines but no >5 min idle gap tend to be underpredicted, while
multi-order tours without such gaps tend to be overpredicted. Changing one
global pick-time constant cannot resolve both mechanisms. Re-running the
corrected six-day CASIM variant with one order per tour gives 15.575 min pooled
RMSE, versus 16.273 min for inferred interleaved tours. It changes the WMS
work-organization assumption and still leaves most of the gap.

A transparent sensitivity screen retains only single-order tours with no
recorded line longer than 300 s and no order-internal idle gap longer than
300 s. It retains 789/1,414 orders (55.8%) and the unchanged submitted model
has 4.360 min RMSE on them. On the later Thursday and Friday it retains
198/353 orders (56.1%) with 4.401 min RMSE, versus 16.205 min on all 353.
The retained orders contribute only 4.0% of full-cohort squared error (4.1%
on Thursday/Friday); the omitted 44% of orders carry nearly all the gap.
Across the full cohort, 244 orders contain a recorded line over 300 s and
account for 49.5% of squared order error; 244 orders have over five minutes
of internal recorded idle and account for 44.1%. These groups overlap.
At a 120 s line limit the forward coverage falls to 117/353 (33.1%) and RMSE
is 4.804 min. These screens use WMS outcomes and omit much of the workload;
they support a conditional trace-quality sensitivity analysis, not a claim
that the simulator predicts the excluded orders or faithfully reconstructs
unobserved pauses. The full-cohort result remains the primary reference.
The graph's maximum shortest path is 189.4 m, equivalent to 82.1 s of walking
at 2.306 m/s; with a 19 s constant pick, the submitted model cannot explain
a line over 300 s through travel and one ordinary pick alone. The threshold
is conservative, but it does not prove whether the excess is a pause, handling
work or a WMS booking artifact.

The ordinary same-aisle line-time fit, trained on Monday--Wednesday, lowers
the Thursday/Friday RMSE on that fixed 300 s quality screen from 4.401 to
3.639 min for the same 198 orders. Its full Thursday/Friday RMSE improves only
from 16.205 to 16.109 min. Thus the time formula helps interpretable trace
segments, while it cannot account for the omitted, operationally ambiguous
segments. The constant 19 s per line remains the scenario default.
Across all six days the fitted variant has 15.896 min pooled RMSE, compared
with 16.273 min for the submitted constant.
The training rows were selected using their observed duration, so the fitted
walking coefficient is a diagnostic estimate and should not be presented as
an independently measured physical speed.

Fixing speed at the submitted 2.306 m/s and fitting only handling on the
same Monday--Wednesday ordinary lines yields 14.217 s per line plus 0.15189 s
per package. Its later-day quality-screen RMSE is 3.632 min for the same 198
orders, essentially equal to the free-speed fit's 3.639 min. On all later-day
orders it gives 16.262 min, slightly worse than the unchanged submitted
model's 16.205 min; pooled across six days it gives 16.032 min. This is the
more defensible diagnostic formula because it does not reinterpret WMS
booking durations as a measured walking speed. It also does not solve the
full-cohort discrepancy.

## WMS line-start semantics and completion-order test

The default replay sorts a picker's lines by WMS start time, but a line start
often belongs to that order's own interval history. Across six days, 3,036
returns to an already open order start within one second of that order's
previous completion while preceding the picker's immediately prior completion
on another order. Those windows overlap the intervening work by a combined
23.77 hours; summing them as independent picker activity double-counts time.
A global start-time sort therefore cannot be treated as a
physical picker sequence. Sorting within each inferred tour by completion
time reorders 5,860 lines, but changes summed picker-day interior graph travel
only from 728.3 to 716.8 km. The corresponding six-day CASIM order-span RMSE
falls from 16.273 to 16.079 min; with the fixed-speed pick fit, it is 15.864
min pooled and 16.112 min on later Thursday/Friday. The remaining gap is not
resolved by sequence choice. Even completion order contains 6,457 of 62,890
consecutive picker intervals shorter than the graph walk alone at 2.306 m/s,
so the export still cannot uniquely specify a physical trace.
Even combining one-order tours, completion-time sequence and the fixed-speed
handling fit gives 15.865 min pooled and 16.024 min on later Thursday/Friday.
This changes tour membership as well as timing, yet leaves the main gap.

An evaluation using successive picker **completion** gaps avoids treating
order-local line starts as physical starts. In the completion-ordered variant,
61,508 consecutive within-tour gaps match their WMS line signatures. The WMS
has 308 of these gaps above 300 s; the submitted CASIM time model has none.
On later Thursday/Friday within-tour gaps at most 120 s, WMS median is 14 s,
versus 20.73 s submitted and 18.75 s with the fixed-speed handling fit. The
fit lowers ordinary-gap MAE from 13.98 to 12.41 s, while the full within-tour
gap RMSE remains about 93 s because long unlabelled intervals persist.
`scripts/ijpe/completion_gaps.py` reports this without exporting line data.

For the paper revision, report the full 1,414-order WMS comparison as the
primary external check, with the 300 s quality screen and its coverage as a
sensitivity analysis. Use the fixed-speed formula only as a labelled
calibration variant; retain the submitted constant in the reference run.
Results generated by CASIM can support counterfactual experiments if labelled
as synthetic simulation outputs. The available export cannot identify the
exact within-tour pause, handling, physical interleaving or first-leg process
needed to reproduce every observed timestamp. A lower RMSE obtained by using
those timestamps as future scheduling inputs would not be a predictive
reproduction.

This report compares CASIM's facts tables with the local WMS export. Only
aggregates appear here. The replay fixes picker, line sequence and pick
location; CASIM executes the tours with its own routing, events and time model.
The WMS export has no measured tour origin, tour identifier or pause event.
Jonas Koreis's 8 December 2025 layout email specifies 0.8 m between storage
places and 15 m for an aisle change. The submitted loader's graph has these
two travel-edge lengths. Its schematic CSV boxes are not measured rack faces.

## Cohort and reference

The six-day anchored replay has 62,985 lines and 1,414 orders after the
layout and single-picker filters. The manuscript's Figure 8 has 1,419 orders:
the replay additionally excludes five retained orders with several pickers.
Its order-span definition is first line start to last line end, in both the
WMS and CASIM facts. The submitted-model anchored order-span RMSE is 16.56 min
pooled; by day it is 12.81, 16.00, 14.52, 20.99, 17.48 and 11.15 min in
date order (11, 13, 14, 15, 16, 17 April).

The WMS timestamps do not prove the path actually walked. Jonas Koreis wrote
on 12 December 2025 that picking runs in a U shape within an aisle, and the
January correspondence says heavy products are picked first in practice.
The recorded-sequence router reproduces the export's timestamp order exactly;
it should be described as a **booking-order replay** until a physical route
trace is available. U-shape remains a separate hypothesis test.

The [9 February 2026, *IWSPE: Update FFZ/Cobot Parameter*] correspondence
reports 215 s setup as the cleaned-data mean. The 6 February message calls
8.3 km/h (2,306 mm/s) a plausible average manual truck speed, not a measured
speed from this six-day export. The 19 s pick-time value was an average across
manual and Cobot operation, not a per-line timestamp rule for every April
line. These emails are evidence about parameter provenance, not calibration
data for the April validation days.

## What the first-leg experiment showed

Jonas Koreis described tours as starting in an aisle in the December 2025
layout correspondence. The current replay begins each tour at the layout's
fixed aisle-1 depot. To measure the effect, two diagnostic CASIM variants
instead begin at the lower cross aisle of the first pick's rail, or exactly at
the first pick. Neither position is asserted to be the true WMS tour origin.
All other parameters and recorded decisions are identical.

| Origin hypothesis | Pooled order-span RMSE, 1,414 orders |
|---|---:|
| Fixed depot | 16.56 min |
| Lower aisle head | 16.63 min |
| First pick | 16.65 min |

Thus the fixed depot is an unsupported physical assumption, but replacing it
does not solve the measured order-span error. This corrects the earlier
first-leg hypothesis. The graph still maps the retained pick locations and
uses the layout CSV's 0.8 m location spacing; the actual tour origins remain
unknown. The first-pick variant also exposed a routing defect: a zero-distance
first leg dropped the first pick. The router now retains the origin as a route
node, and the hand-derived replay tests verify all five picks.

## Where the 215-second setup belongs

The 215 s setup mean was estimated from gaps between tours. In the April
replay, 1,077 inter-tour gaps have a median of 242 s; 438 are shorter than
215 s. The initial source holds a tour until its next first-line start and
then charges another 215 s of setup. That likely counts setup within the gap
twice. A diagnostic variant instead starts setup 215 s before the recorded
first-line start and ends each inferred pause there.

Anchored picker-day last-line bias moves exactly 3.58 min earlier; on all six
days its absolute mean error improves modestly. The earlier chained comparison
used a pause event that did not respect an active tour and is superseded by
the corrected departure sweep below. The WMS does not label the setup interval
of any individual tour, so the variant is a test of the interpretation, not an
observed event schedule.

## Line intervals and pauses

WMS intervals are not a clean log of uninterrupted physical actions. Of the
62,985 replayed lines, 7,030 last at most 1 s, 9,844 last under 4 s, 4,197
last over 59 s, and 284 last over 300 s. The 284 longest lines account for
92.6% of the submitted model's squared *line-duration* error. Across the
broader export, the data audit finds 4,639 line intervals overlap the same
picker's previous line, 3,096 by more than 10 s. A single sequential picker
cannot have two physical picks in progress simultaneously. Exact reproduction
of these recorded starts and ends therefore requires a WMS accounting model,
which the export does not define.

On lines lasting at most 59 s, the submitted model's line RMSE is 15.68 s;
its median modeled duration is about 21 s against a recorded median of 14 s.
For all lines the line RMSE is 94.38 s. Recorded idle inside inferred tours
totals 43.2 h across the six days. The export does not say which gaps were
breaks, waiting for work, loading or other tasks. CASIM currently does not
interrupt a running tour for an inferred pause.

## A time-model test on held-out days

To avoid fitting an impossible speed to unobserved first legs, an alternative
fit fixes speed at 2,306 mm/s and estimates pick time only from interior lines
lasting 4–59 s on 13–15 April. It gives **12.7535 s + 0.1480 s per package**
for a line (29,348 training lines), retaining 215 s tour setup. The submitted
19 s constant remains the scenario default. The held-out days are 11, 16 and
17 April; the full WMS cohort, including anomalous intervals, is evaluated.

| Held-out metric, 586 orders | Submitted | Interior fit |
|---|---:|---:|
| Pooled order-span RMSE | 14.97 min | 15.41 min |
| Line MAE, 11 April | 20.63 s | 18.62 s |
| Line MAE, 16 April | 21.41 s | 19.23 s |
| Line MAE, 17 April | 22.45 s | 19.84 s |

The alternative improves ordinary line prediction and the picker-day end
bias, but it does not improve order spans. A fitted per-line formula alone is
therefore insufficient. The unconstrained regression's 4.3–5.6 m/s estimate
should not be installed as a truck speed; it conflates unobserved tour starts,
long line intervals and pick behavior.

Order errors also grow with order size: pooled RMSE is 7.56 min for 1–20
lines and 25.96 min for over 80 lines. The 244 replayed orders with over 5 min
of recorded idle within their order spans account for 42.5% of squared
order-span error. The remaining orders account for 57.5%, so removing only
long pauses cannot close the gap.

Decomposing each order span into the union of its line intervals (busy) and
the gaps inside the span (idle) gives a busy-component error RMSE of 11.54 min
and an idle-component error RMSE of 11.02 min. Both contribute materially to
the combined 16.56 min RMSE. A replay that changes only the pick-time formula
or only the pause schedule cannot address both components.

The raw-log cohorts make this limit concrete. Of 1,414 orders, 1,154 contain
a line longer than 59 s and 1,255 contain a line shorter than 4 s. For the 225
orders with no line over 59 s and no recorded idle over 5 min, order-span RMSE
is 8.17 min. For the stricter 50 orders with every line between 4 and 59 s,
no overlapping line intervals and at most 5 min of recorded idle, it is
2.17 min. This smaller cohort is a diagnostic, not a replacement for the
paper's validation cohort; selecting it favors easy orders.

## Current limits and next experiments

Verified: the recorded replay executes the retained sequence; its fixed
depot assumption does not materially explain Figure 8; ordinary-line pick
calibration improves line but not order error; timestamp extremes, same-picker
overlap and within-tour idle are present in the WMS data. Assumptions: exact
tour boundaries, physical tour origins, and the operational meaning of a WMS
line's start and end. The current fixed-depot and aisle-origin variants are
explicit hypotheses, not observed routes.

The later defect-aware comparison below separates overlapping lines and long
idle; the corrected picker-departure sweep measures chained day completion.
The remaining physical unknowns are the meaning of WMS line timestamps,
true tour origins, and the work or waiting inside long intervals. Exact line
timestamp matching is not a valid success criterion for overlapping intervals
unless the WMS semantics become available.

## Geometry and route-policy check (16 November)

The layout CSV is 68 by 24 cells and produces 947 storage locations, 996
graph nodes and 1,025 edges. Its three cross-aisle levels are at 0.4, 28.4
(manually inserted) and 54.0 m; the retained pick nodes span 51.2 m along an
aisle. The paper rounds aisle travel length to 50 m. The CSV and the loader
are the only available physical dimensions; exact cross-aisle width and tour
origins have no independent measurement. Every one of Monday's 9,935
interior-line graph shortest paths matched the distance charged by CASIM.
For 1,496 Monday consecutive lines at neighbouring nodes on the same rail,
the graph distance was 0.8 m. Their recorded median interval was 9 s, or
15 s for intervals between 4 and 59 s; 35.8% were shorter than 4 s. A
five-consecutive-line hand check matched CASIM's `walk_distance / 2306 + 19`
formula exactly. These checks support graph consistency, not the interpretation
of WMS intervals as measured driving time.

On all six anchored days, replaying the recorded sequence used 924.59 km of
tour routes and gave 16.56 min pooled order-span RMSE before the assignment
correction. Substituting the paper's U-shape router on the same tours and
locations used 723.01 km but increased the RMSE to 19.76 min. U-shape is a
policy variant; it cannot be called the recorded route. The recorded item
assigner removes the 22 pick-node changes made by greedy assignment, with
16.55 min RMSE after correction. Thus location assignment and fixed-depot
routing do not explain the bulk of the discrepancy.

With speed fixed at 1.0, 1.5 and 2.306 m/s in a regression on lines of at most
300 s, training RMSE is 28.44, 25.68 and 24.49 s respectively; an unconstrained
fit reaches 24.03 s by implying 5.62 m/s. As speed rises, the fitted first-line
term moves from -48.19 through -19.96 to -0.23 s; at the unconstrained speed it
is +21.42 s. The quantity coefficient stays near 0.20 s per package. Distance
and first-line terms correlate at 0.53, while quantity is nearly uncorrelated
with either. The small fit gain from an impossible speed is therefore mainly
absorbing first-leg and observation effects, not a truck-speed measurement.

## Corrected picker departures and defect-aware comparison

`PickerDeparture(picker_id, duration)` begins at a recorded gap's start; if
the picker is finishing a tour, the absence begins when the tour finishes.
The first tour of each picker has a binding WMS start in both modes. Across
six days, anchored and chained replays at 2, 5 and 10 minute pause thresholds
have the same per-day order-span RMSE: 12.81, 16.00, 14.52, 20.99, 17.43 and
11.15 min (16.554 min pooled). Each retained order lies within one inferred
tour. A gap between tours therefore cannot alter its span. Chained picker-day
last-line RMSE at the 5-minute threshold rises to 57.33–109.63 min by day,
versus 19.65–30.03 min anchored. Inferring hard breaks from the export does
not reproduce the day timeline; it is not evidence for a new travel speed.

The corrected recorded-location anchored runs have 62,985 lines and 1,414
orders. Line-duration RMSE is 94.38 s overall but 15.67 s among the 58,788
lines at most 59 s. The 284 lines over 300 s cause 92.6% of squared line
error. Order-span RMSE is 16.554 min on all orders. A diagnostic subset of 50
orders whose every line lasts 4–59 s, whose line intervals do not overlap and
whose recorded idle is at most 5 min has RMSE 2.167 min. That subset contains
3.5% of orders and is **not** a replacement validation cohort: its conditions
use observed WMS outcomes. Any revision figure must show the full cohort and
the defect flags alongside it.

The order error decomposes into 11.538 min RMSE for recorded busy time versus
modeled busy time, and 11.014 min RMSE for idle inside order spans. Both
matter. For 380 orders with overlapping recorded lines, order RMSE is 18.994
min. Overlapping same-picker timestamps cannot represent uninterrupted,
sequential physical picks. Without WMS timestamp semantics, exact line-level
reproduction is not identifiable from this export. The submitted 2.306 m/s
speed and 19 s constant remain the operational reference model; a fitted
4.3–5.6 m/s speed is not defensible.

## Zone affinity (R1.8)

Across the full WMS week, 89 of 106 picker-days have all orders in one main
area. Seventeen picker-days change an order's main area at least once; the
mean is 0.415 changes per picker-day, with a maximum of eight. Only 3.08% of
consecutive order changes also change main area. The median share of a
picker-day's lines in its dominant area is 100% (10th percentile 88.5%).
Day-level zone eligibility is therefore a plausible policy variant for later
CASIM decisions, but an absolute per-picker zone ban would contradict the
observed switches. The recorded replay already fixes each picker's decisions,
so adding such eligibility cannot reduce its time-model RMSE.

## Submitted-assumption variants on the corrected replay

With recorded locations and all other defaults held fixed, U-shape routing
raises the six-day pooled order-span RMSE from 16.554 to 19.756 min. Replacing
the recorded interleaved tours with one order per tour lowers it to 15.575 min.
The latter changes the observed work organization, so its lower error does not
validate it as a WMS replay. The configured 8:00 and 11:00 breaks leave the
within-tour order-span RMSE at 16.554 min but make the six-day picker-day
last-line RMSE 76.43–137.81 min by day, compared with 19.65–30.03 min in the
anchored reference. The first two variants remain anchored; the configured
break variant uses breaks in place of WMS pause inference. These experiments
show that matching one outcome by changing a decision can harm the physical
fidelity or the day timeline.

The reverse replacement order shows an interaction. Starting with U-shape,
one-order tours and configured breaks together gives 15.469 min pooled RMSE.
Restoring the recorded sequence alone gives 15.575 min; restoring recorded
interleaved tours alone gives 19.756 min; restoring recorded pauses alone leaves
15.469 min. The pooled picker-day last-line RMSE falls from 97.59 to 29.53
min when recorded pauses replace the configured breaks in that combination;
the breaks matter for the day clock even though they cannot change an order
span within a tour. U-shape hurts substantially when the observed interleaved tours
are retained, but slightly helps when tours are split into one order. There
is therefore no additive attribution of the route and tour choices. None of
these counterfactual combinations reproduces the WMS work organization more
faithfully than the recorded-decision replay.

## Revision dataset and validation claim

Use all 1,414 replayable single-picker orders as the primary WMS validation
cohort, with the same order-span definition as Figure 8 and an explicit note
that five otherwise retained multi-picker orders are outside this replay.
Report RMSE and MAE on that cohort together with line-duration tails,
same-picker overlap and within-tour idle. The quality-stratified subsets are
diagnostics and must retain their counts; selecting the 50 cleanest orders as
the primary result would bias the claim. A generated dataset for policy
experiments may use the physically bounded CASIM time model, but it must be
labelled simulated and evaluated separately from WMS observations. The six-day
export supports a comparison of observed decision sequences and order spans;
it does not identify an exact physical speed or a unique pause schedule.
