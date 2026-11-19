# IJPE order-level time validation (P1.2)

The cohort is the 1,414 located orders picked by one recorded picker on six
April 2026 days. CASIM replays the inferred recorded tours and line sequence,
but calculates setup, travel and pick times itself. An order's processing span
is from its first line start to its last line end. The WMS timestamps contain
overlaps and unlabelled idle, so this is a conditional replay check, not a
predictive reconstruction of every physical action. The cohort and trace
limitations are detailed in `docs/ijpe/replay.md`.

The submitted model uses 2,306 mm/s travel, 19 seconds per line and 215
seconds setup per tour. The calibration keeps travel and setup fixed and fits
a base pick time plus seconds per package from interior lines recorded as 4
to 59 seconds. This fit sample is selected by the recorded line duration;
all retained orders, including those with extreme line times, remain in
validation. `scripts/ijpe/calibrate.py` calculates the parameters, and
`scripts/ijpe/order_validation.py` runs CASIM with each out-of-day parameter
set and writes aggregate size-class errors to ignored outputs.

For leave-one-day-out, each day's fit excludes that day. For the held-out
comparison, the fit uses Monday 13 to Wednesday 15 April and is evaluated on
Saturday 11, Thursday 16 and Friday 17 April. The held-out fit is 12.755
seconds per line plus 0.148 seconds per package. The six leave-one-day-out
fits range from 12.518 to 12.825 seconds per line and 0.135 to 0.147 seconds
per package. These values are calibration coefficients, not independently
measured handling times.

| Cohort and model | Orders | Bias, min | MAE, min | RMSE, min |
|---|---:|---:|---:|---:|
| Six days, submitted | 1,414 | -0.65 | 9.84 | 16.27 |
| Six days, leave-one-day-out fit | 1,414 | -3.73 | 9.32 | 16.12 |
| Held-out days, submitted | 586 | -2.39 | 8.74 | 14.95 |
| Held-out days, Monday–Wednesday fit | 586 | -4.94 | 8.67 | 15.39 |

Bias is model span minus recorded span. Negative bias means CASIM finishes an
order's span too quickly. The fit slightly lowers six-day pooled MAE and
RMSE, but worsens held-out RMSE and negative bias. It does not validate a
claim that calibration improves out-of-sample order duration.

| Lines per order | Orders, six days | Submitted RMSE, min | Leave-one-day-out RMSE, min | Held-out submitted RMSE, min | Held-out fitted RMSE, min |
|---|---:|---:|---:|---:|---:|
| 1–20 | 176 | 7.71 | 7.82 | 8.12 | 8.30 |
| 21–40 | 472 | 14.92 | 14.45 | 13.88 | 13.97 |
| 41–60 | 450 | 15.63 | 15.32 | 14.31 | 14.88 |
| 61–80 | 223 | 19.78 | 20.00 | 20.51 | 21.34 |
| Over 80 | 93 | 25.75 | 26.07 | 24.55 | 26.03 |

Order error increases strongly with size in both models. The held-out fitted
RMSE is higher in every size class. No time model is fixed by this result;
P1.3 must still check completion profiles, picker presence and tours at the
system level, and P1.4 must isolate the effect of recorded assignment.

# System-level validation (P1.3, P1.4)

All runs keep the recorded batches (inferred tours) and recorded pick
locations, route with U-shape and use the submitted time model unless
stated. Measures are model minus WMS over the 1,414 orders and 95
picker-days; `scripts/ijpe/system_validation.py` writes them to ignored
outputs.

The validation has two steps (trace-driven, then distribution-driven; Law
and Kelton):

1. Trace-driven: everything the WMS decided or recorded is replayed (the
   recorded tours, each to its recorded picker, the recorded presence and
   absences); CASIM computes only setup, travel and picks. This isolates the
   time model, the graph and the routing (`calibrated_recorded` below).
2. Distribution-driven: the setting of the experiments. CASIM's FIFO
   assignment as a pull and absences drawn from a model fitted on the other
   days (Fitted absences, below).

FIFO with recorded absences is neither: it combines recorded absences with
work the WMS did not record. A recorded absence sat between the tours its
picker actually did; with FIFO the picker does other tours, so an absence
meant for the gap between two tours can fall due inside a longer one. It is
kept as a diagnostic (it showed that push strands work, see below), not as
validation.

## The calibrated day

The calibrated day (`variants=calibrated`) has five parts, each measured on
its own and none tuned on the system-level results below:

- presence: a picker arrives at the setup start of its first recorded tour
  and leaves after its last recorded line;
- pauses: a recorded gap between tours over five minutes is a timed absence;
- stops inside tours: a gap between two lines, or a line's excess over 19 s,
  over the same five minutes is a timed absence too
  (`source.stops_in_tours`). CASIM does not interrupt a tour, so the absence
  begins when the tour ends; the day keeps its recorded absence;
- setup: 215 s per tour inside the recorded gap before it
  (`source.setup_in_gap`); the short gaps between tours have a median of
  188 s;
- time per line: 16.26 s, so that CASIM's walking plus the line time covers
  each recorded tour net of its stops. It is fitted on Monday to Wednesday;
  on the other three days the tour bias is -0.15 min and the RMSE 6.0 min
  per tour; the six days separately give 13.7 to 17.2 s
  (`scripts/ijpe/line_time.py`).

Why the stops matter. Without them, CASIM's pickers finished an hour early
although per-order time was unbiased: 28 % of the recorded tours contain a
stop of five minutes or more (99 h on the six days), and the submitted 19 s
per line had absorbed part of it. With the stops as absences and the line
time fitted net of them, the per-picker bias without forced departures goes
from -78 to -10 minutes.

Three runs, all after C7 (departures in request order), 18 November. The
recorded assignment is a push: the recorded plan is committed into each
picker's queue at the start. FIFO is a pull (`decisions/pull.yaml`): when the orders are in,
OBRSP makes the recorded tours into pick lists, routes them (U-shape) and
schedules them, and commits nothing; ORSP keeps their routes and gives the
next tour to each picker that becomes free (about 220 decisions a day). The
routes are computed once, which gives the same tours as routing all open pick
lists at every decision.

| Run | Assignment | Presence | Open orders | Orders by 10:00 / 12:00 | Largest day error by 10:00 / 12:00 | Picker finish bias, min | Picker finish RMSE, min | Day's last line, mean error, min |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `calibrated_recorded` | recorded | recorded | 86 | +32 / -13 | 11 / 15 | -26 | 52 | +19 |
| `calibrated_fifo` (diagnostic) | FIFO, pull | recorded | 5 | +44 / +10 | 14 / 8 | -6 | 28 | +4 |
| `calibrated_recorded_arrival` | recorded | no departure | 0 | +32 / -12 | 11 / 14 | -10 | 61 | +89 |

- Throughput up to midday is reproduced: on every day the orders completed
  by 10:00 and 12:00 are within 15 of the WMS count with recorded
  assignment and within 10 with FIFO; over six days CASIM is 32 to 37
  orders ahead at 10:00 and level at 12:00.
- Picker finish times are unbiased but noisy. The picker-day bias is -10
  minutes when no picker is sent home. With recorded presence it reads -26
  minutes because the measure is censored: a picker who runs late leaves at
  the recorded time with work open (86 orders, 6 %) instead of finishing
  late, so only early finishes remain. The spread (RMSE 61 minutes) comes
  from tour errors adding up over about 12 tours per picker; picker-level
  effects are not claimed.
- The end of the day is not reproduced: either orders stay open (recorded
  presence) or the last picker finishes late (+89 minutes on average without
  departures).
- FIFO as a pull reproduces the day: 5 open orders over six days (all on
  Friday), picker finish RMSE 28 minutes (8 to 19 on five days) and the
  day's last line within 4 minutes on average. Monday's picker RMSE is 58
  minutes because of one picker: with FIFO it gets a longer tour than
  recorded, four of its recorded absences fall due during it, and absences
  take effect in request order, so they run back to back from 09:35 until
  it leaves at 13:16. A free picker takes over work a late
  picker would otherwise leave open, as on the floor. The recorded
  assignment cannot: it fixes every tour to its picker, so the tour errors
  add up per picker (RMSE 61 minutes without departures).
- Pushed FIFO, as run before 18 November, planned the whole day at 05:30
  into fixed queues, also for pickers not yet arrived or leaving early: 141
  orders stayed in the queues of pickers who had left, while others stood
  idle (on Monday 11 pickers stopped more than 10 minutes before their
  recorded end, 22 h in total, with 52 orders open). `service.json` now
  reports such stranded tours (`stranded`); with pull there are none.

Graph and routing, checked against the recorded line sequence. For tours of
11 to 60 lines (765 of 1,172 tours, mostly one order) the shortest-path
distance through the recorded sequence of lines equals CASIM's U-shape
distance (median ratio 0.98 to 1.00); for tours of up to 10 lines U-shape is
longer (median ratio 0.82, 77 tours). In tours over 100 lines (several orders, 26 % of lines)
the recorded sequence jumps back and forth between orders (about 13 jumps
of more than 40 m within 20 seconds per tour), which no picker can walk:
there the confirmation order is not the walking order, and the comparison is
not informative.

## Fitted absences (distribution-driven validation)

The calibrated day takes every absence from the record. The experiments
cannot: they need absences that are modelled, not replayed. The roster (when
each picker arrives and leaves) stays an input, as a planner knows it; the
absences during the shift come from a model (`source.absences=fitted`,
`casim/sources/absence.py`). The recorded absences are 229 h, 34 % of the
665 present picker-hours of the six days. The model has two parts:

- scheduled breaks around 08:00 and 12:00: in each window (07:00 to 09:30,
  10:30 to 13:30) a picker's longest absence of at least 10 minutes is its
  break. Every present picker takes the morning break (median 30 minutes,
  around 09:00), 84 % the midday break (median 40 minutes, around 12:15);
  together 110 h, 17 % of present time. Time and duration are drawn together
  from the recorded breaks of the window;
- allowance, every other absence (personal needs, fatigue, delays; the
  record does not say what): a Poisson process per picker with its own rate
  and durations in three parts of the picker's presence. In the first hour
  1.05 absences per hour (median 9 minutes), 28 % of the time: start-up; in
  the regular time 0.80 per hour (median 8 minutes), 17 %; in the last hour
  0.80 per hour (median 6 minutes), 11 %: wind-down. A single rate over the
  whole presence put too little absence before 08:00 (43 h drawn against 58
  h recorded) and too much at the end, and the days ran 8 % ahead by 12:00;
- each day is fitted on the other five days; the draws depend only on seed
  and roster, so all policies of a replication see the same absences. As
  recorded absences, they begin when the running tour ends.

Thirty replications per day, FIFO as a pull, the calibrated time model
(`scripts/ijpe/absence_validation.py`):

| Assignment | Orders by 10:00, six days | Orders by 12:00, six days | Open orders | WMS inside the 90 % band at 12:00 |
|---|---:|---:|---:|---:|
| WMS | 626 | 1,031 | 0 | |
| FIFO, recorded absences (diagnostic) | 670 | 1,041 | 5 | |
| FIFO, fitted absences, mean (90 % band) | 682 (662 to 703) | 1,085 (1,057 to 1,113) | 12 (0 to 26) | 3 of 6 days |

- Open work stays small with fitted absences (12 orders over six days), as
  with recorded absences.
- By 12:00 the fitted days are 5 % ahead of the WMS. Most of it is Monday
  (+38 of +54 orders): Monday had 51 h of absence, the model fitted on the
  other days draws about 36 h. Without Monday the five days are within 2 %.
  Thursday goes the other way (about 10 h more drawn than recorded). A model
  fitted on other days reproduces the average day, not each day's level, and
  the paper says so.
- With recorded and with fitted absences the days are ahead at 10:00 (7 to 9 %);
  this comes from the time model, not from the absences.
- Recorded assignment is not combined with fitted absences: the recorded
  work plan gave each picker the work it managed given its own absences,
  which drawn absences do not respect.

## How the calibration was found (diagnostics, before C7)

The runs below each change one assumption, with the submitted 19 s per line
and without stops inside tours. They were run before C7, so runs with
recorded presence lose up to 0.6 h of pause over the six days. They are not
rerun. Their conclusion on presence holds; their picker-level numbers are
superseded by the calibrated day, and the finding that the fitted pick time
makes every measure worse held only while the stops inside tours were
missing. Selected five-minute pauses are timed departures. Presence is either
`shift` (every picker from 05:30), `recorded` (arrival at the setup start of
the first recorded tour, departure after the last recorded line) or
`arrival` (recorded arrival, no forced departure). "Orders by 10:00" is the pooled difference in
orders completed by 10:00 over six days.

| Run | Assignment | Presence | Setup | Open orders | Picker finish bias, min | Picker finish RMSE, min | Day's last line, mean error, min | Orders by 10:00 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `system_recorded_assignment` | recorded | recorded | 215 s | 47 | -44 | 76 | +30 | +28 |
| `system_recorded_nosetup` | recorded | recorded | 0 | 6 | -85 | 105 | -4 | +135 |
| `system_recorded_setup_in_gap` | recorded | recorded | 215 s, in the gap | 31 | -55 | 85 | +23 | +55 |
| `system_fifo` | FIFO | recorded | 215 s | 89 | -55 | 84 | +15 | +57 |
| `system_fifo_arrival` | FIFO | arrival | 215 s | 0 | -37 | 99 | +80 | +57 |
| `system_fifo_shift` | FIFO | shift | 215 s | 0 | -113 | 142 | -20 | +365 |
| `system_fifo_shift_nopause` | FIFO | shift, no pauses | 215 s | 0 | -178 | 190 | -205 | +503 |
| `system_fifo_calibrated_recorded` | FIFO | recorded | 215 s, fitted pick | 58 | -91 | 120 | +3 | +156 |

Findings:

- Presence dominates. With every picker present from 05:30, CASIM completes
  365 more orders by 10:00 than the WMS and pickers finish about two hours
  early; without pauses, more than three hours early. With recorded
  presence, recorded assignment and the 215 s setup, the orders completed by
  10:00 and by 12:00 are within 14 of the WMS count on every day (at most
  12 % on Saturday to Thursday, 119 to 222 orders; Friday has only 8 and 36).
  FIFO stays within 30 orders; without setup or with the fitted pick time
  the morning runs up to 38 and 42 orders ahead.
- The setup per tour is visible in the export. Between inferred tours, the
  695 gaps under five minutes have a median of 188 s, and a tour's first line
  lasts 48 s against 15 s for other lines (medians; means 120 s and 25 s).
  Together about 4 to 5 minutes per tour, so the submitted 215 s is in range
  and slightly low; zero setup is not supported. In chained runs the pause
  before a tour should end at the tour's setup start (`source.setup_in_gap`),
  or the 215 s are charged twice after every long gap.
- Total work is close. With recorded assignment and `setup_in_gap`, CASIM
  spends 413 h on routes and picks and 68 h on setup; the WMS has 438 h of
  line time inside tours (about 407 h without the longer first lines) and
  about 70 to 90 h of setup-like time (short gaps between tours and longer
  first lines). CASIM does the day's work in 5 to 10 % less time.
- The end of the day and picker finish times are not reproduced because
  tour errors add up along each picker's day: matched tours have 15 min RMSE
  and a picker works about 12 tours, so finish errors of an hour are
  expected even without bias. With recorded presence a picker that runs
  behind leaves work open (31 orders with `setup_in_gap`); without forced
  departures the day ends late. Assignment is not the cause: recorded and
  FIFO assignment give the same picture.
- Picker-level timing is not reproduced in any run (finish RMSE 76 to 190
  minutes), including with recorded assignment (P1.4). CASIM's pickers are
  busy 76 to 92 % of their presence against 53 to 73 % in the WMS.
- The fitted pick time (12.8 s per line plus 0.15 s per package) makes every
  system-level measure worse.

## Why recorded pickers idle

Pickers have no order open for 31 % of their time between first and last
line (data audit, R0.3). Aggregate tests on the export, with aisle 10
excluded:

- Work was waiting. At the end of each of the 432 gaps longer than five
  minutes, a median of 124 orders of that day had not yet started; at none
  was the day's work exhausted.
- No release over the day. 96 % of orders carry their pick date as creation
  date, and orders numbered late in the day could start as early as those
  numbered early (rank correlation of number and first start 0.09 to 0.20).
- No ceiling on active work. Open orders reach their daily maximum in 0.4 to
  1.3 % of minutes, and fewer pickers idle when many orders are open (1.5 to
  3.1) than otherwise (2.2 to 4.9).
- Idle is synchronised across pickers beyond chance on three of five full
  days (Saturday, Monday, Tuesday), not on Wednesday and Thursday. Long gaps
  do not end together more than by chance (0.94 other gap ends within 2.5
  minutes against 0.83 if uniform), which a dock limit released by truck
  departures would produce.

The export therefore gives no evidence of a dock limit, a work-in-progress
cap or release waves, in line with the partner's statement that work can
always be started. The idle time is unrecorded activity or breaks. A dock
limit is not used to reproduce the WMS.

## What the case supports

- Order-level time with the submitted model: unbiased on average
  (-0.65 min), 16 min RMSE (order spans include the stops inside tours).
- Tour time net of stops with the calibrated line time: bias -0.15 min,
  RMSE 6.0 min per tour on held-out days.
- Daily throughput up to midday on the calibrated day: within 15 orders of
  the WMS count at 10:00 and 12:00 on every day with recorded assignment,
  within 10 with FIFO. This is the window of the disruption scenarios (07:30
  to 13:00).
- Picker-day time without bias (-10 minutes without forced departures).
- Not validated: the end of the day, individual picker finish times (RMSE
  an hour) and busy shares.

- With fitted absences and FIFO assignment, the six-day throughput by
  12:00 within 1 % of the WMS (90 % band of 30 replications contains it).

Disruption and policy results are therefore reported as comparisons between
policies on the calibrated day with fitted absences and the recorded roster
(`variants=calibrated source.absences=fitted`, replications with common
seeds across policies), measured on throughput and on-time completion up to
midday, not as predicted picker-level effects in this warehouse.
