# IJPE scenario: current scope and replay error

This report describes the `codex/ijpe-revision` scenario as of 18 November
2026. The numbers below come from the six-day `days_thresholds` sweep and
`python -m scripts.ijpe.current_state`. The script matches every inferred
recorded tour to one model tour by its orders, checks its picker and line
count, and writes only aggregate tables to ignored `outputs/current_state/`.
All differences are model minus WMS. A negative time difference means the
model is earlier or shorter.

## What the package runs

`scenarios/ijpe/` is a confidential-data scenario. Its input is the WMS
workbook `data/wms_export_kw16.xlsx`, the warehouse `data/layout.csv`, and
ignored parquet caches made by `scripts/ijpe/convert.py`. The WMS export has
line start and end timestamps, order, article, quantity, pick location,
picker and store identifiers. It has no observed tour start at the depot,
picker break labels, truck assignment, loading record or departure time. The
source maps recorded locations onto the layout graph and retains located
single-picker orders. The raw six days have 1,525 orders and 65,291 lines;
the replay has 1,414 orders and 62,985 lines. The paper's 1,419-order
Figure 8 cohort has a different filter; see `docs/ijpe/data_audit.md`.

The six replay dates are Saturday 11 April and Monday 13 through Friday 17
April 2026. The source creates one resource per picker actually seen on each
date. There are 95 picker-days in total, from 45 distinct pickers in the raw
export. Pickers are numbered by their first retained line each day. The
default creates 21, 17, 16, 19, 15 and 7 pickers, respectively; it does not
use the submitted model's fixed 25-picker assumption. A picker has a cart
large enough for the largest inferred tour in that day's recorded plan.

| Day, April 2026 | Orders | Lines | Pickers | Inferred tours |
|---|---:|---:|---:|---:|
| 11 | 233 | 9,756 | 21 | 230 |
| 13 | 259 | 10,141 | 17 | 206 |
| 14 | 263 | 12,818 | 16 | 218 |
| 15 | 306 | 15,671 | 19 | 219 |
| 16 | 272 | 11,775 | 15 | 219 |
| 17 | 81 | 2,824 | 7 | 80 |
| All | 1,414 | 62,985 | 95 picker-days | 1,172 |

Every retained order arrives and is available at 05:30 on its date; the
order stream closes at the same time. Recorded WMS order creation or release
times are not replayed. The recorded plan is committed in one decision at
05:30 and never revised. It supplies pick location, the inferred batching of
overlapping orders, the start-ordered line sequence, and the recorded picker
for every tour. CASIM calculates travel, picking and setup time. This is a
conditional replay of a given plan, not a test of CASIM's batching,
scheduling, pipeline choice or response to disruption.

The default time model is 2,306 mm/s walk speed, 19 seconds per line and 215
seconds of setup per tour. There is no quantity-dependent pick time. It uses
the depot as origin and return point, permits interleaved orders in a tour,
and has no configured breaks or due dates. The default picker is present from
05:30. In the default **anchored** diagnostic, every tour is held until at
least its recorded first-line start. This uses future WMS timing and makes
the default useful for within-tour checks, not for forecasting a day.

The selected **five-minute chained** setting uses the same tours and time
model. Only each picker's first tour is held to its recorded start. A recorded
gap between tours longer than five minutes becomes a timed picker departure;
shorter gaps disappear. The six days produce 378 inferred departures totaling
142.5 hours. These gaps are not verified breaks. Idle intervals inside an
inferred tour are not inserted into the simulation. `recorded_presence`
also moves each arrival to the first recorded setup start and adds a final
departure. `submitted` instead uses one order per tour, U-shape routing and
fixed 08:00–08:30 and 11:00–11:30 breaks. Neither variant is the default.

`fixed_dispatch` uses the five-minute setting and gives every order a
synthetic picking deadline. A stable hash of its store identifier chooses a
10:00, 12:00, 14:00 or 16:00 dispatch group; a stated 30-minute staging
lead makes the deadline 09:30, 11:30, 13:30 or 15:30. The choice does not
read completion time, so the deadlines are binding on some recorded orders.
There are no identified trucks or simulated truck departure events yet. See
`docs/ijpe/departure_fallback.md`.

## How much the replay differs

The anchored order-processing span, first line start to last line end, has
16.27 minutes RMSE, 9.84 minutes MAE and -0.65 minutes bias over 1,414
orders. A leave-one-day-out pick-time fit only changes pooled RMSE to 16.12
minutes, and worsens held-out RMSE from 14.95 to 15.39 minutes. These are
order spans, so changing only the between-tour pause threshold leaves them
unchanged. See `docs/ijpe/validation.md` for size classes and calibration.

A recorded tour window runs from its first line start to its last line end.
The comparable model window runs from its first line's walk start to its last
line's pick end. In 1,172 matched tours, model window minus recorded window
has -2.78 minutes mean bias, 8.95 minutes MAE and 15.47 minutes RMSE. The
median is +1.00 minutes: 671 tours are longer and 501 shorter in the model.
The 10th and 90th percentile errors are -21.79 and +6.66 minutes. Thus the
model is longer for a majority of individual tours but underpredicts total
window time because some short predictions have large errors. The pooled
recorded windows total 481.32 hours versus 426.96 modeled hours.

For a closer *work proxy*, the recorded side is the union of the WMS line
intervals inside each tour, and the model side sums travel-to-line plus pick
time for each line. The union avoids counting overlapping WMS bookings twice.
It does not prove how long a picker physically worked: WMS intervals can
contain waiting and bookkeeping. The recorded proxy totals 438.12 hours and
the modeled line work 426.96 hours. Model minus recorded is -0.57 minutes
per tour on average, with 8.01 minutes MAE and 13.64 minutes RMSE. The median
is +1.79 minutes; 765 tours are longer and 407 shorter in the model. The
10th and 90th percentile errors are -14.53 and +7.41 minutes. The recorded
tour windows contain another 43.20 hours outside the union of line intervals,
which is not labeled as work or break in the export. Model tours including
setup and depot return total 513.54 hours, but that full duration has no
like-for-like WMS comparator.

For the five-minute chained setting, the **mean picker-day last-pick error**
is -20.17 minutes across 95 picker-days: 52 pickers finish earlier and 43
later. Its picker-day MAE is 60.60 minutes and RMSE is 80.51 minutes. This
large error shows that preserving within-tour work does not reproduce the
timing of a picker's day. The anchored diagnostic has +8.26 minutes mean
picker-day error, 17.30 minutes MAE and 27.36 minutes RMSE, but its recorded
start anchors make that day-level fit conditional on future information.

| Day, April 2026 | Mean picker finish, chained model minus WMS | Picker finish RMSE | Last pick of whole day, chained model minus WMS |
|---|---:|---:|---:|
| 11 | -19.7 min | 55.2 min | +87.6 min |
| 13 | -48.8 min | 111.9 min | +94.5 min |
| 14 | +6.8 min | 90.6 min | +160.7 min |
| 15 | -18.8 min | 73.3 min | +71.3 min |
| 16 | -21.6 min | 78.1 min | +159.3 min |
| 17 | -14.6 min | 44.7 min | +49.1 min |

The last pick of the whole modeled day is later on all six dates. These
figures use last **line** completions on both sides. A model tour's depot
return is later still and has no WMS return timestamp for comparison.

## What can be studied now

The package supports replay error analysis by line, order, tour and picker
day; sensitivity to the submitted route, tour and break assumptions; the
selected pause threshold; held-out order-time calibration; and service
against explicitly synthetic picking deadlines. `facts/picks.csv`,
`facts/tours.csv`, `facts/orders.csv`, `decisions.json` and `service.json`
provide the model outputs. The default has no due dates, so its on-time rate
is undefined. In the current replay the decision log contains one accepted
recorded pipeline and no policy comparison, and the affected-order flag is
false for all orders. The model's picker absence measure includes time before
05:30 and inferred departures; it is not an observed absence measure.

System validation (P1.3) and recorded-assignment isolation (P1.4) are now
in `docs/ijpe/validation.md`. Truck and dock accounting
(C5), actual or simulated departure delays, disruption experiments and
policy comparisons remain open. The fixed synthetic deadlines answer whether
deadlines can be set independently of observed completion; they do not
validate real transport service. No result here establishes that CASIM
predicts a new day's completion profile or the effect of a disruption.
