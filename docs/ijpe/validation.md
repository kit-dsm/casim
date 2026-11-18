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
