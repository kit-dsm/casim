# Maintained research studies

A scenario is a self-contained study, not a CASIM engine mode. Select the
closest study along the experimental axes below, then change its input,
timing, planning window, commitment, algorithms, or replanning scope.

| Study | Order knowledge | Trigger | Planning input | Commitment | Replanning | Physical context |
|---|---|---|---|---|---|---|
| `scenario_reopt` | arrivals only / complete information | picker idle | all visible | first batch | depot future work | one picker, order-bin cart |
| `scenario_henn` | arrivals only | waiting policy | all visible | dispatch or wait | depot future work | Henn/Gil W5 |
| `scenario_intervention_stress` | arrivals only | event/blocking | bounded dispatch or active residual | first job / active suffix | active route or batch | eight pickers, centerline graph |
| `scenario_dynamic_operations` | backlog plus arrivals | nightly/periodic | due window | all, per-picker, or time fence | buffered, unstarted, optional active route | seeded multi-day e-commerce |
| `scenario_ijpe` | historic/generated backlog plus arrivals | WMS, planning, disruptions | batch/due window | configured schedule prefix | unstarted work | two-sided grocery pallet graph |

The remaining axes are resource/cart configuration, congestion approximation,
disruption source, and drain versus fixed-horizon termination. Intervention is
a replanning scope and can be combined with push or pull timing when the
selected StateAdapter and algorithms are applicable.

## Applying CASIM elsewhere

1. Start from the nearest maintained study.
2. Reuse its loader or add a scenario-local loader using existing domain types.
3. Select files or a seeded generator in `config/input/`.
4. Describe static technical context in a handwritten DataCard.
5. Configure triggers, planning windows, commitment, and replanning in
   `config/engines/`.
6. Select actual algorithm components in `config/cosy_repo/`.
7. Let `DomainAlgorithmMapper` reject technically incompatible components
   before execution.
8. Add scenario hooks only for exogenous operational events.

See [the execution model](../docs/execution_model.md) for ownership and buffer
semantics.
