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
stated. Selected five-minute pauses are timed departures. Presence is either
`shift` (every picker from 05:30), `recorded` (arrival at the setup start of
the first recorded tour, departure after the last recorded line) or
`arrival` (recorded arrival, no forced departure). Measures are model minus
WMS over the 1,414 orders and 95 picker-days; `scripts/ijpe/system_validation.py`
writes them to ignored outputs. "Orders by 10:00" is the pooled difference in
orders completed by 10:00 over six days.

| Run | Assignment | Presence | Setup | Open orders | Picker finish bias, min | Picker finish RMSE, min | Day's last line, mean error, min | Orders by 10:00 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `system_recorded_assignment` | recorded | recorded | 215 s | 47 | -44 | 76 | +30 | +28 |
| `system_recorded_nosetup` | recorded | recorded | 0 | 6 | -85 | 105 | -4 | +135 |
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
- The end of the day is not reproduced by any single setting. With the 215 s
  setup per tour, recorded presence leaves orders open (the assigned picker
  leaves before reaching them) or, without forced departures, the day ends
  29 to 169 minutes late. Without setup, the day's last line is within 10
  minutes on five of six days, but the morning runs ahead and pickers finish
  85 minutes early on average. WMS gaps at order changes have a median of
  -2 s, so the WMS shows no 215 s setup between orders; the true setup lies
  between the two settings and is not identified by the export.
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

- Order-level time: unbiased on average (-0.65 min), 16 min RMSE.
- Daily throughput up to midday with recorded presence and setup: within
  14 orders (at most 12 % on Saturday to Thursday) of the WMS count at 10:00
  and 12:00 with recorded assignment, within 30 with FIFO. This is the window
  of the disruption scenarios (07:30 to 13:00).
- Not validated: the end of the day (setup and assignment at the day's end),
  picker-level finish times and busy shares.

Disruption and policy results are therefore reported as comparisons between
policies on the same calibrated day (recorded presence, inferred pauses),
measured on throughput and on-time completion up to midday, with both setup
settings as bounds, not as predicted picker-level effects in this warehouse.
