# Henn waiting on the converted Henn instances

Henn's (2012) waiting rule for online order batching, compared with an
order-window strategy, on sixteen Henn instances in the Heßler-Irnich format
with Henn's arrival times. The item positions of these files differ from the
published OOBP archive (F60, `docs/oobp_source_comparison.md`), so the results
are not comparable with published OOBP values; the `oobp` scenario runs the
published archive.

## Source, decisions, runs

- Source: `hessler_irnich` (`casim.sources.hessler_irnich`), one instance file
  named by `source.filepath`, arrival files in `data/HennOnline/`. Orders and
  the closure of the stream at the last arrival come from the source; the
  picker arrives with the first order.
- Decisions: `decisions/henn_waiting.yaml` names the triggers, conditions,
  state information, revision scope, solver and commitment policy.
  `cosy_repo/baseline_waiting_henn.yaml` selects item assignment, FiFo
  batching, S-shape routing, SPT scheduling and Henn waiting;
  `cosy_repo/repair_waiting_henn.yaml` adds incumbent-start batching as a
  second candidate. `decisions/order_window.yaml` waits for five buffered
  orders and has no waiting component. `decisions/release_only_example.yaml`
  is kept for Task 6.1; the rolling-horizon files, which `casim check`
  rejects (`problem_class: rolling`), wait in `docs/deferred/task_6_1_henn_rolling/`.
- Runs: `sweep/waiting_strategy_sweep.yaml` runs both strategies on fifteen
  instances. `plots.py` reads the sweep's outputs.

Known: the decision uses `NbrOrdersCondition(1)`, which ignores orders in
planned tours (F67, fixed by Task 4.15); its last batch can keep a stale wait.

## Running

```bash
python scripts/fetch_data.py henn_waiting
python -m casim run scenarios/henn_waiting
python -m casim run scenarios/henn_waiting cosy_repo=repair_waiting_henn +decisions.OBRSP.change_penalty=10
python -m casim run scenarios/henn_waiting +sweep=waiting_strategy_sweep
python scenarios/henn_waiting/plots.py
```

Outputs go to `scenarios/henn_waiting/outputs/run/` (one directory per sweep
combination). The recorded baselines are in `tests/baseline/henn__*.json`
(`docs/baseline.md`).

## Sources

1. K. Heßler and S. Irnich (2022), *Modeling and Exact Solution of Picker
   Routing and Order Batching Problems*, LM-2022-03.
2. S. Henn (2012), *Algorithms for on-line order batching in an order picking
   warehouse*, Computers & Operations Research 39(11), 2549-2563.
