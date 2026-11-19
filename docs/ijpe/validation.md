# IJPE order-level time validation (P1.2)

The cohort is the 1,414 located orders picked by one recorded picker on six
April 2026 days. CASIM replays each recorded order as one tour with its
recorded line sequence, each tour held to its recorded start (anchored,
`sweep=anchored`), but calculates setup, travel and pick times itself. An order's processing span
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
Saturday 11, Thursday 16 and Friday 17 April. The held-out fit is 13.843
seconds per line plus 0.152 seconds per package. The six leave-one-day-out
fits range from 13.249 to 13.540 seconds per line and 0.135 to 0.151 seconds
per package. These values are calibration coefficients, not independently
measured handling times.

| Cohort and model | Orders | Bias, min | MAE, min | RMSE, min |
|---|---:|---:|---:|---:|
| Six days, submitted | 1,414 | -5.86 | 9.19 | 15.58 |
| Six days, leave-one-day-out fit | 1,414 | -7.82 | 9.28 | 16.30 |
| Held-out days, submitted | 586 | -5.33 | 8.64 | 14.73 |
| Held-out days, Monday–Wednesday fit | 586 | -6.77 | 8.52 | 15.13 |

Bias is model span minus recorded span. Negative bias means CASIM finishes an
order's span too quickly. Both models are about 6 minutes too quick, since
the recorded span of an order the picker worked on together with another one
also holds the other order's lines, which the one-order tour does not. The
fit worsens six-day MAE, RMSE and bias, and on the held-out days lowers MAE
slightly but worsens RMSE and bias. It does not validate a claim that
calibration improves out-of-sample order duration.

| Lines per order | Orders, six days | Submitted RMSE, min | Leave-one-day-out RMSE, min | Held-out submitted RMSE, min | Held-out fitted RMSE, min |
|---|---:|---:|---:|---:|---:|
| 1–20 | 176 | 9.42 | 9.47 | 10.38 | 10.33 |
| 21–40 | 472 | 14.64 | 15.16 | 14.42 | 14.80 |
| 41–60 | 450 | 14.24 | 14.91 | 13.72 | 14.20 |
| 61–80 | 223 | 18.28 | 19.29 | 18.33 | 18.84 |
| Over 80 | 93 | 25.51 | 27.12 | 23.56 | 24.20 |

Order error increases strongly with size in both models. The held-out fitted
RMSE is higher in every size class but the smallest. No time model is fixed by this result;
P1.3 must still check completion profiles, picker presence and tours at the
system level, and P1.4 must isolate the effect of recorded assignment.

# System-level validation (P1.3, P1.4)

All runs make one tour of each recorded order (the WMS order is the picking
tour) with its recorded pick locations, route with U-shape and use the
submitted time model unless stated. Measures are model minus WMS over the 1,414 orders and 95
picker-days; `scripts/ijpe/system_validation.py` writes them to ignored
outputs.

The validation has two steps (trace-driven, then distribution-driven; Law
and Kelton):

1. Trace-driven: everything the WMS decided or recorded is replayed (the
   recorded tours, each to its recorded picker, the recorded presence and
   pauses); CASIM computes only setup, travel and picks. This isolates the
   time model, the graph and the routing (`calibrated_recorded` below).
2. Distribution-driven: the setting of the experiments. CASIM's FIFO
   assignment as a pull and non-picking time drawn from distributions fitted
   on the other days (Fitted non-picking time, below).

FIFO with recorded pauses is neither: it combines recorded pauses with work
the WMS did not record. A recorded pause sat between the tours its picker
actually did; with FIFO the picker does other tours, so a pause meant for the
gap between two tours can fall due inside a longer one. It is
kept as a diagnostic (it showed that push strands work, see below), not as
validation.

## The calibrated day

The calibrated day (`variants=calibrated`) has five parts, each measured on
its own and none tuned on the system-level results below:

- presence: a picker arrives at the setup start of its first recorded tour
  and leaves after its last recorded line;
- pauses: a recorded gap between tours over five minutes is a timed
  departure (counting only time in which the picker has no line open);
- stops inside tours: a gap between two lines, or a line's excess over 19 s,
  over the same five minutes is a timed departure too
  (`source.stops_in_tours`). CASIM does not interrupt a tour, so the
  departure begins when the tour ends; the day keeps its recorded pauses;
- setup: 215 s per tour inside the recorded gap before it
  (`source.setup_in_gap`); the short gaps between tours have a median of
  188 s;
- time per line: 16.26 s, so that CASIM's walking plus the line time covers
  each recorded tour net of its stops, fitted on Monday to Wednesday when a
  tour was a picker's lines cut where no order was open (interleaved orders
  shared one tour). With one tour per order the same method
  (`scripts/ijpe/line_time.py`) gives 18.74 s (held-out bias +0.47 min, RMSE
  6.3 min per tour; the six days separately 13.9 to 20.4 s), because an
  order's window now also holds the lines of the other orders its picker had
  open at the same time. The calibrated day keeps 16.26 s; the fitted
  non-picking model does not use it (its line times come from its own fit).

Why the stops matter, measured when interleaved orders still shared a tour.
Without them, CASIM's pickers finished an hour early although per-order time
was unbiased: 28 % of the recorded tours contained a stop of five minutes or
more (99 h on the six days), and the submitted 19 s per line had absorbed part
of it. With the stops as timed departures and the line time fitted net of
them, the per-picker bias without forced departures went from -78 to -10
minutes; with one tour per order it is +14 minutes (below).

Three runs on the current code (19 November, one tour per recorded order).
The recorded assignment is a push: the recorded plan is committed into each
picker's queue at the start. FIFO is a pull (`decisions/pull.yaml`): when the
orders are in, OBRSP makes the recorded tours into pick lists, routes them
(U-shape) and schedules them, and commits nothing; ORSP keeps their routes
and gives the next tour to each picker that becomes free (about 220 decisions
a day). The routes are computed once, which gives the same tours as routing
all open pick lists at every decision.

| Run | Assignment | Presence | Open orders | Orders by 10:00 / 12:00 | Largest day error by 10:00 / 12:00 | Picker finish bias, min | Picker finish RMSE, min | Day's last line, mean error, min |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `calibrated_recorded` | recorded | recorded | 150 | -24 / -86 | 12 / 28 | -16 | 43 | +19 |
| `calibrated_fifo` (diagnostic) | FIFO, pull | recorded | 87 | -30 / -76 | 15 / 28 | +6 | 13 | +18 |
| `calibrated_recorded_arrival` | recorded | no departure | 0 | -24 / -85 | 12 / 28 | +14 | 72 | +137 |

| Day | Orders | Orders that overlap another order of their picker | FIFO, orders by 12:00 minus WMS | Recorded assignment, orders by 12:00 minus WMS |
|---|---:|---:|---:|---:|
| Saturday 11 | 233 | 5 | -3 | -16 |
| Monday 13 | 259 | 83 | -18 | -16 |
| Tuesday 14 | 263 | 72 | -16 | -17 |
| Wednesday 15 | 306 | 133 | -28 | -28 |
| Thursday 16 | 272 | 81 | -9 | -7 |
| Friday 17 | 81 | 2 | -2 | -2 |

- Throughput up to midday is not reproduced on every day. Over six days
  CASIM is 24 to 30 orders behind at 10:00 and 76 to 86 behind at 12:00 (7
  to 8 %). The error follows the interleaving: pickers worked on several
  orders at once (376 of 1,414 orders overlap another order of the same
  picker, almost all Monday to Thursday), and one tour per order with a cart
  for one order does them one after another, with the line time fitted when
  interleaved orders shared a tour. On the two days without interleaving the
  FIFO day is within 3 orders of the WMS at 12:00.
- Picker finish times are biased by +14 minutes when no picker is sent home,
  with an RMSE of 72 minutes from tour errors adding up over about 15 tours
  per picker. With recorded presence the bias reads -16 minutes because the
  measure is censored: a picker who runs late leaves at the recorded time
  with work open (150 orders, 11 %) instead of finishing late. Picker-level
  effects are not claimed.
- The end of the day is not reproduced: either orders stay open (recorded
  presence) or the last picker finishes late (+137 minutes on average
  without departures).
- FIFO as a pull leaves 87 orders open over six days (Monday 25, Tuesday 18,
  Wednesday 28, Thursday 10, Friday 6, Saturday none), with a picker finish
  RMSE of 13 minutes and the day's last line 18 minutes late on average. A
  free picker takes over work a late picker would otherwise leave open, as
  on the floor; the recorded assignment fixes every tour to its picker, so
  the tour errors add up per picker.
- A FIFO push plans the whole day into fixed queues, also for pickers not yet
  arrived or leaving early, so work stays in the queues of pickers who have
  left while others stand idle (`tests/test_pull_assignment.py`).
  `service.json` reports such stranded tours (`stranded`); with pull there
  are none.

Graph and routing, checked against the recorded line sequence of each of the
1,414 one-order tours: the shortest-path distance through the recorded
sequence equals CASIM's U-shape distance for tours of 11 to 60 lines (992
tours, median ratio 0.99) and of 61 to 100 lines (278 tours, 1.00); it is 5 %
longer for tours over 100 lines (38 tours) and 19 % shorter for tours of up
to 10 lines (106 tours, median ratio 0.81), where U-shape walks whole aisles.
The recorded sequence within an order is walkable: about one jump of more
than 40 m within 20 seconds per tour. The back-and-forth jumps reported for
the earlier tours (about 13 per tour over 100 lines) came from interleaved
orders merged into one tour.

## Fitted non-picking time (distribution-driven validation)

The calibrated day takes its pauses and stops from the record. The experiments
cannot: they need non-picking time that is modelled, not replayed. The roster
(when each picker arrives and leaves) stays an input, as a planner knows it.
The recorded pauses and stops mixed three things: gaps between tours, gaps
inside tours and the excess of long scanner lines over the pick time, the
last of which is not observed absence. The fitted model has three parts
instead, each measured on the picker's time without any WMS record and each
fitted, for a day, on the other five days (`scripts/ijpe/fit_non_picking.py`,
which writes the fit to `data/non_picking/<day>.json`; `source.non_picking=fitted`
reads it).

Only what one person did is a sample. The system lets a login work on one
order at a time (the use case partner), yet 374 of 1,414 orders (26 %) overlap
another order of their login in scan window (first to last line start): two
people under one login or wrong records, and the scans do not tell how many
people. An order that overlaps another order of its login gives no line
sample, a gap next to one no setup sample, and a picker-day with one (53 of
95, almost all Monday to Thursday) no break sample. Without this, 31 % of
setup samples were zero (the next order starting before the last one ended,
the second person's work) against 15 % now; non-zero setups (351 and 340 s)
and line samples (21.2 and 21.8 s on average) are the same. Overlapping
orders measured on their own order's scans take 2 s longer per line at the
median, not twice as long: they look like one person per order.

- breaks, tied to the clock: in each window (07:00 to 09:30, 10:30 to 13:30)
  a picker's longest spell without a record of at least 10 minutes is its
  break, whether it lies between tours, inside a tour or is a line that long.
  Fitted on five days (Tuesday left out), all present one-person picker-days
  take the morning break (47 minutes on average, 30 samples) and 72 % the
  midday break (32 minutes, 26 samples); start and duration are drawn
  together. The samples are few, since only 42 picker-days are one person.
  Breaks are timed departures, as before;
- tour setup: the time without a record between two tours of a picker, net of
  a break in it and of the walk to and from the depot, which the simulation
  walks itself. Mean 289 s, median 128 s, 90th percentile 741 s, 99th
  percentile 51 minutes, 14 % zero (Tuesday left out). A picker's first setup of the day stays the fixed 215 s that defines
  its arrival;
- line time: a line's duration plus the time without a record before it
  inside its tour, net of the walk from the previous line, by quantity class
  (up to 6, 7 to 12, 13 to 24, over 24 packages: means 16.4, 18.5, 23.1 and
  46.1 s). Lines that hold another record of the picker inside them, or
  overlap a record outside the cohort, are bookkeeping and left out. The
  lines of one order are slow or fast together: every line time of an order
  is multiplied by an order factor, lognormal with mean 1 and a standard
  deviation of 0.25 to 0.39 (fitted on orders with at least 10 line samples),
  and the line pools hold the variation within an order;
- time covered by the picker's WMS records outside the cohort (aisle 10,
  shared orders) is work and left out of every fit (7.1 h over six days);
- the source fixes the realized times of a run in W_0 as the process truth
  (C8): each line's time is drawn from a stream seeded by seed, day and order
  (one draw per position), each picker's setups from a stream seeded by seed,
  day and picker, so every policy of a replication meets the same times.
  Execution takes these times; the resources plan with the means (setup, and
  line time by quantity class, `ClassPickTime`), and no decision sees the
  truth. The empirical distributions keep their observed tails; no
  parametric tail is added. There is no separate allowance.

Thirty replications per day, FIFO as a pull, recorded roster (arrival and
departure), against the previous model of breaks plus allowance per working
hour (`scripts/ijpe/non_picking_validation.py`; "recorded pauses" is the
trace-driven run, unchanged as a diagnostic):

| Six days | Orders by 10:00 | Orders by 12:00 | Open orders | WMS inside the 90 % band, days at 10:00 / 12:00 |
|---|---:|---:|---:|---:|
| WMS | 626 | 1,031 | 0 | |
| Recorded pauses (diagnostic) | 596 | 955 | 87 | |
| Breaks plus allowance, mean (90 % band) | 611 (590 to 633) | 985 (956 to 1,017) | 81 (65 to 110) | 2 / 2 |
| Breaks, setup and line time, mean (90 % band) | 600 (578 to 618) | 945 (915 to 974) | 89 (66 to 112) | 2 / 2 |

- With one-person service times and one picker per login the model is 4 %
  behind by 10:00 and 8 % by 12:00 over six days, and leaves 89 orders open
  at the recorded departures. Before the fit was restricted to one person it
  matched the WMS by 12:00 (1,026): the zero setups next to overlapping
  orders gave the logins the capacity of the second person as free setup
  time. The shortfall is the staffing question (next section), not a time to
  fit.
- Per day by 12:00: Saturday -1, Monday +9, Tuesday -22, Wednesday -51,
  Thursday -38, Friday +16 orders. The days that fall behind are the days with
  the most overlapping orders (132 and 80 of 306 and 272 on Wednesday and
  Thursday); Saturday and Friday have almost none.
- Tour durations are compared on orders that overlap no other order of their
  login, net of the breaks inside them (next section): the model reproduces
  them.

## Capacity and staffing (system level)

The roster was the second input taken from the record. A picker's recorded
departure is its last line, which depends on when the work ran out, and a
login is not always one person. So the system-level validation takes only
the arrivals from the record (`source.presence=arrival`: a picker arrives at
its first tour and stays until the work is done) and treats staffing as a
sensitivity (`scripts/ijpe/capacity_validation.py`). All six days, full
workload, one tour per order, FIFO as a pull:

- A: the paper's time model (19 s per line, 215 s setup, breaks at 08:00 and
  11:00 of 30 minutes for everyone), one run;
- B: fitted line time and setup, no breaks, 30 replications;
- C: B plus the fitted breaks, 30 replications;
- D+k: C with k pickers added to the logins (`source.extra_pickers`,
  arriving at the median arrival, never leaving), 10 replications; Dpaper:
  the pickers per day of the submitted configuration (25, 17, 19, 20, 18, 7),
  which are the logins of the whole export, not an independent staff count
  (4, 0, 3, 1, 3 and 0 more than the logins of the cohort).

| Six days, orders done by | WMS | A | B | C | D+2 | D+4 | D+6 | Dpaper |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 08:00 | 319 | 370 | 334 | 324 | 328 | 329 | 329 | 328 |
| 10:00 | 626 | 702 | 704 | 600 (578 to 618) | 613 | 622 | 630 | 610 |
| 12:00 | 1,031 | 1,059 | 1,099 | 956 (924 to 986) | 1,000 | 1,034 | 1,074 | 995 |
| 14:00 | 1,285 | 1,393 | 1,380 | 1,261 | 1,318 | 1,362 | 1,391 | 1,321 |

| Orders by 12:00 minus WMS | Sat | Mon | Tue | Wed | Thu | Fri | Mean absolute day error |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | +23 | +24 | +10 | -20 | -17 | +8 | 17 |
| C | +7 | +11 | -20 | -50 | -38 | +16 | 24 |
| D+4 | +17 | +29 | -2 | -36 | -20 | +16 | 20 |
| Dpaper | +17 | +8 | -5 | -48 | -24 | +15 | 19 |

| One-person orders, minutes, six days | Orders | Mean | 10th | 50th | 90th percentile |
|---|---:|---:|---:|---:|---:|
| WMS, first line to last | 1,040 | 21.1 | 6.0 | 16.6 | 46.1 |
| WMS, net of the breaks inside the order | 1,040 | 18.6 | 5.9 | 16.1 | 34.0 |
| WMS, orders without a break | 961 | 18.0 | 5.7 | 15.8 | 31.8 |
| A | | 16.4 | 7.3 | 15.6 | 27.0 |
| C (fitted times) | | 18.5 | 5.4 | 15.7 | 35.2 |

- Service time is reproduced. A break taken inside an order stretches its
  WMS duration, while CASIM takes breaks between tours; 79 of the 1,040
  one-person orders hold a break the fit finds. Net of it, the fitted model
  matches the WMS in mean, median and 90th percentile. The paper's model is
  too narrow (27 against 34 minutes at the 90th percentile).
- Capacity is not reproduced day by day. With one picker per login (C) the
  model is 4 % behind by 10:00 and 7 % by 12:00 over six days; adding the
  breaks costs about 100 orders by 10:00 against B. By day it is ahead on
  Saturday, Monday and Friday and well behind on Wednesday and Thursday; the
  last line is 49 and 46 minutes early on Monday and Friday and 83 and 84
  minutes late on Wednesday and Thursday.
- Staffing does not explain the day pattern. Four pickers more matches the
  six-day total by 12:00 (1,034 against 1,031) but lowers the mean absolute
  day error only from 24 to 20 orders: Wednesday and Thursday catch up while
  Saturday, Monday and Friday move further ahead. Choosing k on five days and
  testing on the sixth picks 4 or 8 and leaves held-out errors of +17, +49,
  +14, -36, -20 and +16 orders by 12:00. The paper's counts act like two
  pickers more. The staffing level of a day is therefore not identified from
  these data, and calibrating it per day would make each day a calibration
  day.
- The paper's model has the smallest day errors at 12:00 but the wrong shape:
  76 orders ahead by 10:00 and 108 by 14:00, with too-short long orders; its
  12:00 match is too-fast picking against too-little staff.
- Nothing is added back for the day pattern. What a day's staffing was needs
  the partner's shift plan or clock-in times.
- Recorded assignment is not combined with fitted non-picking time: the
  recorded work plan gave each picker the work it managed given its own
  pauses, which drawn breaks do not respect.

## How the calibration was found (diagnostics, before C7)

The runs below each change one assumption, with the submitted 19 s per line
and without stops inside tours. They were run before C7, so runs with
recorded presence lose up to 0.6 h of pause over the six days, and before one
tour per order, when interleaved orders shared a tour. They are not rerun. Their conclusion on presence holds; their picker-level numbers are
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

- Order-level time with the submitted model: 6 minutes too quick on average
  (-5.86 min), 16 min RMSE (recorded order spans include the stops inside
  tours and the lines of interleaved orders).
- Tour time net of stops with a fitted line time: held-out bias +0.47 min,
  RMSE 6.3 min per one-order tour (18.74 s per line).
- Daily throughput up to midday on the trace-driven calibrated day: within
  3 orders of the WMS on the days without interleaving, up to 28 orders
  (13 %) behind on the days with many interleaved orders, 7 to 8 % behind
  over six days at 12:00. This is the window of the disruption scenarios
  (07:30 to 13:00).
- Not validated: the end of the day, individual picker finish times (RMSE
  over an hour without departures, bias +14 minutes) and busy shares.

- Service time with fitted line time (order factor included) and setup:
  one-person order durations match the WMS net of breaks (mean 18.5 against
  18.6 minutes, 90th percentile 35.2 against 34.0).
- Capacity with one picker per login and fitted breaks: 4 % behind by 10:00
  and 7 % by 12:00 over six days, single days from +16 to -50 orders by
  12:00. Extra pickers shift the level but not the day pattern; a day's
  staffing is not identified from the WMS records.

Disruption and policy results are therefore reported as comparisons between
policies on the calibrated day with fitted non-picking time and the
recorded roster (`variants=calibrated source.non_picking=fitted`, a shared
break roster through `source.breaks` where policies are compared, and
replications with common random numbers per order line across policies), measured on throughput and on-time completion up to
midday, not as predicted picker-level effects in this warehouse.
