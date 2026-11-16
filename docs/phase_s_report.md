# Phase S report

16 November 2026, casim `5baad1c`, ware_ops_algos `0bd0a7e`. Phase S of
`docs/revision_scopes_corrections.md` (Section 8.2b, F66) is done. Evidence per task is in
`docs/revision_scopes_status.md`, "Phase S progress".

## What a scenario is now

`scenarios/<name>/` with `README.md`, `scenario.yaml` (the source and the groups `data_card`,
`decisions`, `cosy_repo`, `sweep`, `variants`), optionally `source.py`, `metrics.py`,
`plots.py`, `reference.csv`, and `data/SOURCE.md` with `data/SHA256SUMS.txt`. A scenario runs
no simulation loop; `casim run` owns it. `AGENTS.md` and `CLAUDE.md` state the rules;
`scenarios/minimal/` is the template.

- **S.1** `casim.sources`: the `Source` protocol (`initial_state()`, `events()`) and the
  library sources `hessler_irnich`, `oobp`, `casop`, `foodmart`, `kris`, `radar`; `local` for a
  scenario's own `source.py`.
- **S.2** `casim run <dir> [overrides] [--audit] [--output-dir]`: the loop, sweeps (a failed
  combination is recorded in `error.txt`), and the outputs `kpis.json`, `decisions.json`,
  `tours.csv`, `orders.csv`, `trace.json`, `audit.json` (with `--audit`) and `comparison.csv`
  (with `reference.csv`). `scripts/baseline.py` runs ported scenarios with `casim run`;
  `compare-benchmark` reruns recorded benchmark runs.
- **S.3** `casim check --all`: required and allowed files, the five groups, the source
  protocol, checksums of present data, no simulation import inside a scenario, every group
  file composing with the defaults or with one variant. The test suite runs it on every
  tracked scenario and runs each scenario's default configuration under the audit flag.

## Ports (S.4): every recorded run reproduces

| Scenario | From | Source | Recorded runs compared | Result |
|---|---|---|---:|---|
| `minimal` | new | `local` | hand-derived values | equal |
| `henn_waiting` | `scenario_henn_online` | `hessler_irnich` | 31 | 31 equal |
| `walk_or_wait` | `scenario_walk_or_wait` | `radar` | 11 | 11 equal |
| `analytic_waiting` | `scenario_stochastic_waiting` | `local` | 3 | 3 equal |
| `oobp` | `scenario_oobp` | `oobp` | 192 (L3) | 192 equal |
| `casop_static` | `scenario_static_benchmarks` | `casop` | 415 (L1) | 415 equal |
| `henn2012` | `scenario_henn2012` | `local` | 754 (L2) | 738 equal; 16 differ as named |
| grocery | `scenario_grocery_retailer` | — | — | deleted (decision 5 of 9.1) |

The 16 differing L2 files are working-paper `truncated` pilot runs recorded at `cf2d4a8`,
before those cells moved to the decision without `NbrOrdersCondition` (F67, `7e68bf6`), the
difference named then; all 714 validation runs and the 20 journal pilot runs are equal.
`scenario_ijpe` keeps its structure until Task 6b.6, and with it `scenarios/experiment_commons.py`.

## Open points for the user

1. **Re-record the 16 L2 pilot baselines** above, if you accept the difference.
2. **Additions that the plan and the design document do not name**, each needed by a port: the
   outputs `orders.csv` and the `tours.csv` columns `setup`, `distance`, `picks`,
   `orders_at_start`; `metrics.py` functions as `reference.csv` KPIs; the composed key
   `choices` (the file chosen per group) as a `reference.csv` selector; `data_card` as a
   scenario group; `casim run --output-dir`; `scripts/baseline.py compare-benchmark`.
3. **Moved out of scenarios** because they do not compose: the Henn rolling-horizon files
   (`docs/deferred/task_6_1_henn_rolling/`, Task 6.1) and Henn's LONG rule with its
   scenario-local card and solver (`docs/deferred/henn2012_long/`); the `aaa8333` plots to
   `docs/history/`.
4. **Events are queued at reset.** A source yields its events in time order, but the runner
   reads them all at reset: consuming them lazily would change the order of equal-time events
   (the event id breaks ties) and with it every recorded trace. This fits a live feed only once
   Task 6b.1 settles the tie order.
5. **F64** stays in the L1 results until you fix the loader in ware_ops_pipes; the Kris L1
   baselines are then recorded again. `scripts/l1/run_benchmark.py` still applies the earlier
   F64 rule to `instances_207_1` and does not yet compare only deadline-feasible Kris results
   with the best-known solution, as the corrected plan requires; that changes with the next L1
   rerun.
6. **Local leftovers that `casim check --all` reports on this machine**: your untracked
   `scenarios/scenario_dm/`. The old scenario directories keep only ignored outputs and caches.
7. **The decisions of Section 9.1** remain untouched (decision 5 applied as the plan directs).

## Checks at `5baad1c`

`python -m pytest -q` in one process: 574 passed, `tests/test_baseline.py` included (every recorded scenario configuration reproduces). `python -m casim check --all`: no finding for any tracked scenario; it reports only the untracked `scenarios/scenario_dm/` (point 6).
