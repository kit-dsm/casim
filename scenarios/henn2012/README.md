# Henn (2012): online order batching (L2)

## Question

Does CASIM reproduce, within their confidence intervals, the completion and turnover times Henn
publishes for online FCFS order batching, with and without his waiting rule?

## Data source

No instances are published. The scenario's own `source.py` (`source: {name: local}`) generates
them from the papers' protocol with fixed seeds: 10 aisles of 45 cells per side, orders of 5 to
25 items, n orders over eight hours, capacity W items, 48 length units and 6 items per minute,
3 min setup, one picker, class-based or uniform storage. Parameters: `source.storage`,
`source.n_orders`, `source.capacity`, `source.replication`, and `source.arrival_mode`
(`conditioned`: exactly n orders spread over eight hours; `truncated`: a Poisson process cut at
eight hours). The stream closes at 480 min.

## Decisions

Two targets:

- The journal (Henn 2012, Computers & Operations Research 39, α = 0, no waiting):
  `decisions/l2_journal.yaml` with `cosy_repo/journal_sshape_first`, `journal_sshape_sav`, or
  `journal_sshape_ignore` (with `decisions/l2_ignore.yaml`).
- The working paper (Henn 2009, FEMM WP 34, α = 1, Henn's waiting rule):
  `decisions/l2_working.yaml`, which has no order-count gate because Henn's algorithm has none,
  with `cosy_repo/working_sshape_first` or `working_sshape_sav`.

`sweep/variant_journal_first_fit.yaml` is a declared variant of the journal target at n = 90 and
120 with first-fit FCFS batching (`cosy_repo/journal_sshape_first_fit.yaml`); the fill rule is
inferred from the published values, not stated in the paper.

## How to run

```bash
python -m casim run scenarios/henn2012
python -m casim run scenarios/henn2012 decisions=l2_working cosy_repo=working_sshape_first
python -m casim run scenarios/henn2012 +sweep=validation_journal_first
python scenarios/henn2012/plots.py scenarios/henn2012/outputs/run
```

## Expected outputs

The standard outputs per replication. `metrics.py` gives the completion time and both turnover
definitions in minutes; `plots.py` builds the table of cell means with 95 % intervals against
`reference.csv`, which holds the published values.

## Reference result

Journal, n ≤ 60: all eight FIRST cells lie within ±1.6 % of the published value, inside the 95 %
interval. Journal, n = 90 and 120: CASIM is 2.5 to 13.8 % above the published values, without a
demonstrated cause. Working paper: Henn's turnover is the arrival-to-completion time; completion
is within ±1.9 % at n = 60 and above elsewhere, at n = 30 because of an arrival bound. Details and
all cells in `docs/validation.md` (L2).
