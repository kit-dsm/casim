# Henn (2012): online order batching with FCFS

Literature benchmark L2 of `docs/literature_benchmarks.md`, kind
**statistical**: no instances are published, so `source.py` regenerates them
from the papers' protocol (10 aisles of 45 cells per side, orders of 5 to 25
items, n orders over eight hours, capacity W items, 48 length units and 6
items per minute, 3 min setup, one picker) with fixed seeds, and the averages
over replications are compared with the published averages within their
confidence intervals.

Two targets: the journal (Henn 2012, Computers & Operations Research 39,
α = 0, no waiting; `decisions/l2_journal.yaml`) and the working paper (Henn
2009, FEMM WP 34, α = 1, Henn's waiting rule; `decisions/l2_working.yaml`,
which has no order-count gate because Algorithm 4.1 has none, F67).

## Source, decisions, runs

- Source: the scenario's own `source.py` (`source: {name: local}`), one class
  (`storage`, `n_orders`, `capacity`) and `replication`; `arrival_mode` is
  `conditioned` (exactly n orders, uniform over eight hours) or `truncated`
  (a Poisson process truncated at eight hours). The stream closes at 480 min.
- Decisions and repositories: `l2_journal` with `journal_sshape_first`,
  `journal_sshape_sav` or `journal_sshape_ignore` (with `l2_ignore`);
  `l2_working` with `working_sshape_first` or `working_sshape_sav`.
- Runs: `sweep/validation_<target>_<rule>.yaml`, the reduced validation set
  of Phase B. `sweep/variant_journal_first_fit.yaml` is a declared variant of
  the journal target at n = 90 and 120: FCFS with the first-fit fill rule
  (`cosy_repo/journal_sshape_first_fit.yaml`, card `FirstFitFiFo`), a fill
  rule inferred from the bound, not stated in the paper; the working paper
  stays on next-fit (`FiFo`). `reference.csv` holds the published values; `metrics.py` the
  completion and both turnover definitions in minutes; `plots.py` the table
  with 95 % intervals.

Henn's LONG rule used a scenario-local scheduling card and solver; LONG is
deferred, and those files wait in `docs/deferred/henn2012_long/`.

## Running

```bash
python -m casim run scenarios/henn2012
python -m casim run scenarios/henn2012 decisions=l2_working cosy_repo=working_sshape_first
python -m casim run scenarios/henn2012 +sweep=validation_journal_first
python scenarios/henn2012/plots.py scenarios/henn2012/outputs/run
```

A sweep records a failed replication (`error.txt`, F62 for SAV) and goes on.
The recorded runs are `tests/baseline/benchmarks/l2/`;
`python scripts/baseline.py compare-benchmark --benchmark l2` reruns them.
