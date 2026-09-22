# Henn online waiting scenario

The active experiment is configured by `config/henn_online_config.yaml`. Its `engines/henn_waiting.yaml` file selects the trigger events, conditions, `HennWaitingAdapter`, CoSy solver, and `CommitAllPolicy`. `cosy_repo/baseline_waiting_henn.yaml` selects item assignment, FiFo batching, S-shape routing, SPT scheduling, and the shared waiting stage. The waiting stage selects `HennWaiting` through its algorithm card and returns the same `WaitingSolution` type as the other waiting policies.

`HennWaitingAdapter` projects only currently buffered orders. It neither retains a snapshot nor changes live state. The selected router computes batch and single-order service times for Henn's release threshold. CASIM commits a released job; a `WaitExpired` event triggers reconsideration when the policy waits. The finite benchmark hook emits `OrderStreamClosed` at the end of the realized arrival stream, so the policy learns closure as an operational event.

The experiment keeps the usual `setup_scenario` / `setup_decision_engine` loop. Its hooks seed order and picker arrivals. The source instance and arrival files come from the Heßler–Irnich and Henn benchmarks described below. To run with a local data directory, set `instances_base` and `data_card.source.filepath` through Hydra. The 40-order fixture under `tests/data/instances` supports an integration run.

The files under `config/scenario` and the historical plots under `scripts/plots/waiting_strategy` document the earlier comparison with a five-order window. They are not evidence of numerical parity for this branch. See [the implementation boundary](../../docs/waiting_intervention_foundation.md) for supported claims and tests.

## Sources

1. K. Heßler and S. Irnich (2022), *Modeling and Exact Solution of Picker Routing and Order Batching Problems*, LM-2022-03.
2. S. Henn (2012), *Algorithms for on-line order batching in an order picking warehouse*, Computers & Operations Research 39(11), 2549–2563.
