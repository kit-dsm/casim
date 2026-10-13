# Henn online waiting scenario

The active experiment is configured by `config/henn_online_config.yaml`. Its `decisions/henn_waiting.yaml` entry names the triggers, feasibility conditions, lifecycle selection, revision scope, solver, waiting rule and release rule. `cosy_repo/baseline_waiting_henn.yaml` selects item assignment, FiFo batching, S-shape routing and SPT scheduling. `cosy_repo/repair_waiting_henn.yaml` adds incumbent-start batching as another candidate pipeline.

The generic `StateAdapter` projects buffered orders and selected draft tours. The router supplies batch and single-order service times for Henn's release threshold. CASIM stores complete plans as scheduled drafts, releases a tour explicitly, and uses versioned `Reconsider` events when the policy waits. The finite benchmark hook emits `OrderStreamClosed` when the arrival stream ends.

The experiment composes Hydra config, resolves each decision once, seeds order and picker arrivals, then runs and reports. The source instance and arrival files come from the Heßler–Irnich and Henn benchmarks described below. To run with a local data directory, set `instances_base` and `data_card.source.filepath` through Hydra. The 40-order fixture under `tests/data/instances` supports an integration run.

Run `python -m scenarios.scenario_henn_online.experiment_henn viz.launch=false` for the full pipeline, or add `cosy_repo=repair_waiting_henn +decisions.OBRSPW.solver.solution_ranker.change_penalty=10` to compare full and repair candidates. Both write a trace and `kpis/decisions.json` under Hydra's output directory. The fixed-seed baseline and current measurements are in [the revision-scope baseline](../../docs/revision_scopes_baseline.md).

## Sources

1. K. Heßler and S. Irnich (2022), *Modeling and Exact Solution of Picker Routing and Order Batching Problems*, LM-2022-03.
2. S. Henn (2012), *Algorithms for on-line order batching in an order picking warehouse*, Computers & Operations Research 39(11), 2549–2563.
