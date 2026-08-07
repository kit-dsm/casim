# Henn release-timing RL feasibility study

This study learns only whether to wait or dispatch. Item assignment, FIFO
batching, first-route selection, and S-shape routing are fixed concrete code;
Luigi, CoSy, and CASIM's decision engine are not imported by this scenario.

The finite-episode reward is the negative incremental order flow time. PPO
uses `gamma=1.0`, so undiscounted return is exactly normalized total flow time.

Run the four-size counterfactual timing pilot from PowerShell:

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
uv run python -m scenarios.scenario_henn_rl.experiment_henn_rl `
  experiment.mode=pilot
```

Run the configured feasibility study:

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
uv run python -m scenarios.scenario_henn_rl.experiment_henn_rl
```

Run the one-seed PPO checkpoint diagnosis:

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
uv run python -m scenarios.scenario_henn_rl.experiment_henn_rl `
  experiment.mode=diagnosis
```

The default uses two worker processes for independent instance evaluation and
counterfactual generation. PPO runs remain sequential to avoid CPU contention.

Run the structured online-batching pilot:

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
uv run python -m scenarios.scenario_henn_rl.experiment_structured_batching
```

The structured study uses no-wait dispatch. Its action is a capacity-feasible
subset of the visible raw orders. It does not select from heuristic-generated
candidate batches. The scenario uses an exact additive knapsack oracle,
`ware_ops_algos.BatchObject`, S-shape routing, and the objective-derived flow
reward.

## Structured-batching code map

Start at `experiment_structured_batching.py`. It only composes Hydra, selects
the data split, and dispatches the configured mode.

| Responsibility | Module |
|---|---|
| Actor, critics, decoding, candidate construction, structured loss | `structured_policy.py` |
| On-policy and imitation training loops | `structured_training.py` |
| Learned-policy and existing-policy evaluation | `structured_evaluation.py` |
| Exact alternative continuations and saved-audit scoring | `counterfactuals.py` |
| Supervised critic calibration and memorization | `critic_fitting.py` |
| Ordinary pilot, temperature, and evaluation modes | `policy_experiments.py` |
| Baseline comparison and objective study | `objective_experiments.py` |
| Critic reconstruction audit | `critic_audit_experiment.py` |
| Return-to-go and exact-label calibration | `critic_calibration_experiment.py` |
| Memorization diagnostic | `critic_memorization_experiment.py` |
| Nested 8/16/32-state coverage study | `critic_coverage_experiment.py` |
| Candidate operational-feature sufficiency study | `candidate_feature_experiment.py` |
| JSON, split, paired-summary, gate, and spawn-worker helpers | `experiment_support.py` |

The older binary release-timing PPO study remains in `experiment_henn_rl.py`
and `learning.py`; it is separate from the structured batching path.

For canonical CoSy/applicability-validated Henn baselines, retain the existing
`scenario_henn` matrix. CoSy and Luigi are deliberately absent from the SRL
training loop.
