# Lorenz reoptimization study

This study contains only the paper's depot-level comparison:

- `cios`: complete-information Lorenz DP; paper-example makespan `76`;
- `reopt`: Pcart-N, solving all currently visible orders and committing the
  first batch; paper-example makespan `117`;
- `no_wait`: `GreedyIA → FIFO → TSP → SPT` as a configured CoSy baseline.

Active-route and active-batch experiments live in
`scenario_intervention_stress`. Henn remains a separate study because its
capacity, objective, routing, and benchmark semantics differ.

## Run

```powershell
python -m scenarios.scenario_reopt.experiment_reopt variant=cios
python -m scenarios.scenario_reopt.experiment_reopt variant=reopt
python -m scenarios.scenario_reopt.experiment_reopt variant=no_wait
python -m scenarios.scenario_reopt.experiment_reopt -m variant=cios,reopt,no_wait
```

The loader is selected under `config/input/`; the DataCard describes static
problem context only. Each run writes `result.json`, and online runs also
write `decision_trace.jsonl`. Decisions are synchronous: computation time is
reported but does not advance simulation time.
