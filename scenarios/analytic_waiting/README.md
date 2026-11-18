# Analytic waiting

## Question

Does the analytic waiting rule for single-line orders under exponential arrivals give the waiting
time its closed form predicts, inside a full CASIM run? This is a small integration example of
`AnalyticStochasticWaiting`, not the wait-k study of `walk_or_wait`.

## Data source

None. The scenario's own `source.py` (`source: {name: local}`) emits the realised arrivals written
in `scenario.yaml`: four single-line orders at t = 0, 1, 2 and 12, all in aisle 2, no pick time.
The forecast the waiting rule uses (an exponential arrival process) is the data card's process
information and is separate from the realised stream.

## Decisions

- `decisions/waiting.yaml` (default): one OBRSP decision. The analytic rule needs exactly q − 1
  known orders for a cart of q = 4 bins, so the decision solves only when three orders are open
  (`NbrOrdersCondition(threshold=3)`, which also lets a final partial batch through once the
  stream is closed). The rule requires one picker (`n_resources: 1` on the data card).
  `cosy_repo/analytic_waiting.yaml` uses the analytic rule; `henn_waiting` and `no_waiting`
  replace it by Henn's rule or no waiting, under the same condition.
- `decisions/remaining_route_admission.yaml`: adds a second decision that admits an arrived order
  to the running tour when a bin is free and all its picks lie on the remaining route.

The condition encodes the rule's precondition in the decision, so the no-waiting and Henn
variants also wait for three orders. Moving the precondition into the algorithm is on the
backlog (`docs/development.md`). The two-phase method of the original derivation (a second phase
that compares the detour with the next batch) is not implemented.

## How to run

```bash
python -m casim run scenarios/analytic_waiting
python -m casim run scenarios/analytic_waiting cosy_repo=henn_waiting
python -m casim run scenarios/analytic_waiting cosy_repo=no_waiting
python -m casim run scenarios/analytic_waiting decisions=remaining_route_admission
```

## Expected outputs

The standard outputs. In the default run the decision waits when the third order arrives at
t = 2 and releases the first tour (orders 0, 1, 2) at t ≈ 11.085; the fourth order, arriving at
t = 12, goes into a second tour (`tours.csv`). With `decisions=remaining_route_admission` it is
admitted to the running tour instead.

## Reference result

With a mean interarrival time of 10 s and a tail cost of 40/e, the closed form gives a waiting
time of w* = 10 ln(4/e) = 3.86294 s; `tests/verification/test_phase_b_scenarios.py` checks it.
Recorded regression values (`tests/baseline/stochastic__*.json`):

| Configuration | Makespan (s) | Mean order completion (s) | Tours | Distance |
|---|---:|---:|---:|---:|
| analytic waiting | 87.1 | 54.8 | 2 | 76 |
| Henn waiting | 18.0 | 6.8 | 2 | 12 |
| no waiting | 18.0 | 6.8 | 2 | 12 |
