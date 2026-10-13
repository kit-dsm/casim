# Revision-scope implementation baseline

The `codex/revision-scopes` branches began from CASIM
`67f9f31c6eae578b84dedd869d09d40ab3ffa341` and `ware_ops_algos`
`34bd157089e1746e3a29bff83115526b30b5c3ae`. Both checkouts also had
uncommitted work, which remains in place. The source design's claim that both
repositories were at `67f9f31` does not match the sibling checkout.

The following commands were run from the CASIM root with its existing `.venv`
before changing decision behavior. They use the configured deterministic example
inputs and `viz.launch=false`.

| Scenario command | Result | Baseline |
| --- | --- | --- |
| `.venv\\Scripts\\python.exe -m scenarios.scenario_walk_or_wait.experiment simulation=explicit_example viz.launch=false` | success | 6 orders, 3 tours, makespan 167, distance 160, mean order completion 107.1666667, no tardiness |
| `.venv\\Scripts\\python.exe -m scenarios.scenario_stochastic_waiting.experiment_stochastic_waiting viz.launch=false` | success | 4 orders, 2 tours, 3 decisions, makespan 87.0850348, distance 76, mean order completion 54.835 |
| `.venv\\Scripts\\python.exe -m scenarios.scenario_henn_online.experiment_henn viz.launch=false` | success after correcting the existing input path | 100 orders, 25 tours, 26 decisions, makespan 34702, distance 11603, mean order completion 14584.174 |

The Henn input files already existed under `scenarios/scenario_henn_online/data`.
Its default configuration pointed to the repository-root `data` directory, and
the loader resolved the arrival CSV relative to the process working directory.
Those paths were corrected to make the experiment entry point runnable.

The starting runs captured KPIs and decision counts. They did not serialize
the old per-decision projections, so a retrospective field-by-field snapshot
comparison is unavailable. Projection isolation and command behavior are
covered by the contract tests; the reference KPIs remained aligned through
the behavior-preserving setup work. Changes from steps 6–8 are explained below.

## Current implementation checkpoint

Run these commands from the CASIM root with the existing virtual environment:

```powershell
.venv\Scripts\python.exe -m casim check scenarios/scenario_henn_online/config henn_online_config
.venv\Scripts\python.exe -m casim check scenarios/scenario_stochastic_waiting/config stochastic_waiting_config
.venv\Scripts\python.exe -m casim check scenarios/scenario_walk_or_wait/config walk_or_wait_config
.venv\Scripts\python.exe -m scenarios.scenario_henn_online.experiment_henn viz.launch=false
.venv\Scripts\python.exe -m scenarios.scenario_stochastic_waiting.experiment_stochastic_waiting viz.launch=false
.venv\Scripts\python.exe -m scenarios.scenario_walk_or_wait.experiment simulation=explicit_example viz.launch=false
.venv\Scripts\python.exe -m scenarios.scenario_walk_or_wait.experiment simulation=explicit_wait0 policy=wait_0 cosy_repo=paper_patrol viz.launch=false
.venv\Scripts\python.exe -m scenarios.scenario_henn_online.experiment_henn cosy_repo=repair_waiting_henn +decisions.OBRSPW.solver.solution_ranker.change_penalty=10 viz.launch=false
```

The final isolated runs are in `outputs/revision_scope_final/{henn_full_views,henn_repair_views,stochastic_views,walk_views,wait0_views}`. Each directory contains the normal scenario trace, `kpis/kpis.json`, and `kpis/decisions.json`. The latter records selected pipeline counts and solve, conversion, and application times. Pass `hydra.run.dir=outputs/revision_scope_final/<name>` to reproduce an isolated run.

| Run | Orders | Tours | Makespan | Distance | Solves | Difference from baseline |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Henn, full rebuild | 100 | 26 | 32,197 | 11,402 | 12 | Timestamp grouping, persistent drafts, and release-only decisions reduce solves and change tour grouping. |
| Stochastic waiting | 4 | 2 | 87.085 | 76 | 4 | KPI parity; solving can occur while a draft waits for its minimum order count. |
| Walk-or-wait explicit example | 6 | 3 | 167 | 160 | — | KPI parity; insertion and planning follow precedence at a shared timestamp. |
| Walk-or-wait wait-0 | 3 | 2 | 87 | 84 | — | One empty patrol starts and accepts one order; positioning is separate from waiting. |
| Henn, full plus incumbent repair | 100 | 23 | 30,345 | 10,270 | 11 | Four full and seven repair pipelines selected across decisions. |

For the isolated Henn full run, one pipeline was constructed and selected 12
times. Aggregate measured solve time was 0.118 s, conversion 0.235 s, and
transactional application 1.732 s. For the repair run, two pipelines were
constructed and selected four full and seven repair times; measured solve time
was 0.218 s, conversion 0.281 s, and application 1.748 s. The repair trace
contains five `InsertOrders`/`ReplaceSuffix` pairs and 63 `Reschedule`
commands. The stochastic run measured 0.009 s solve, 0.084 s conversion, and
0.016 s application; walk-or-wait measured 0.022 s, under 0.001 s, and 0.023 s;
wait-0 measured 0.014 s, under 0.001 s, and 0.020 s respectively. These
wall times describe this local run, not an algorithmic bound.

The previous per-event decision path completed the Henn baseline with 25 tours
and 26 decisions. Step 6 grouped triggers by timestamp and produced 25 tours,
makespan 34,152, distance 11,151, and 26 decisions. Later draft and
release-only handling produced the figures above. Stochastic and walk-or-wait
retained their fixed-seed KPIs through this checkpoint.

## Positioning and rolling extension, 2026-10-13

The current branches add two positioning components and one rolling Henn
configuration. The new runs used `viz.launch=false` and separate
`hydra.run.dir=tmp_output/<name>` directories. Each completed run wrote the
normal trace, `kpis/kpis.json`, and `kpis/decisions.json`.

| Run | Command overrides | Orders | Tours | Makespan | Distance | Pipelines | Aggregate solve / conversion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Henn rolling | `decisions=rolling_horizon data_card=henn_rolling cosy_repo=rolling_horizon` | 100 | 25 | 32,218 | 11,271 | 2, selected 35 full and 79 repair | 3.88 s / 0.17 s |
| Wait-0 patrol | `simulation=explicit_wait0 policy=wait_0 cosy_repo=paper_patrol` | 3 | 2 | 87 | 84 | one planning and one insertion | 0.017 s / under 0.001 s |
| Center relocation | `simulation=explicit_wait0 policy=center cosy_repo=paper_center insertion_repo=center` | 3 | 2 | 91 | 88 | one planning and one insertion | 0.029 s / under 0.001 s |
| Optional solver-free release | `decisions=release_only_example data_card=henn_rolling cosy_repo=rolling_horizon` | 100 | 23 | 31,149 | 10,418 | two planning, zero release | release-only decision ran 55 times |

The established Henn, stochastic-waiting, and walk-or-wait scripts were rerun
after integration. Their order and tour counts and KPI values match the
checkpoint table above: Henn 100/26, stochastic 4/2, and walk-or-wait 6/3.
The rolling result differs intentionally because it adds due dates 5,400
seconds after arrival, selects a one-hour due-date horizon, revises drafts
every ten minutes and on operational events, and freezes work five minutes
before its planned start. The center example carries an empty cart and admits
an order from its observed position; it is a CASIM modeling choice rather
than a full Reopt* reproduction. A focused no-order run holds at the center
and returns the unused cart at closure without a pick.
