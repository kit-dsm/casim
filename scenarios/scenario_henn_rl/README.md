# Structured batching on generated Henn data

This study combines generated warehouse instances, CASIM's controlled batching
loop, and the Structured-RL implementation. The supported objective is total
order flow time.

The complete decision path is:

```text
CASIM StateAdapter -> (planning snapshot, resolved orders)
  -> make order features -> actor scores -> feasible decoder
  -> selected order IDs -> DecisionEngine.commit -> CASIM events
```

`OrderBatchingEnv` controls only OBP. The downstream ORSP decision still uses
CASIM's configured S-shape router and FIFO scheduler. Reward is the negative
increment in CASIM's accrued total-flow-time KPI divided by the number of
orders and a fixed `objective_scale`:

```text
reward = -delta_flow_time / (n_orders * objective_scale)
```

The `objective_scale` is the mean per-order flow time of a C&W/SAV reference
policy on the validation split, so the total normalized episode return is near
minus one under the reference policy. The scale is computed once before
training and held fixed for the run. The raw flow-time objective is reported
independently of the normalized RL return.

## Commands

```powershell
python -m scenarios.scenario_henn_rl.experiment_structured_batching `
  experiment=generate data=henn_lorenz

python -m scenarios.scenario_henn_rl.experiment_structured_batching `
  experiment=train data=henn_lorenz decoder=knapsack

python -m scenarios.scenario_henn_rl.experiment_structured_batching `
  experiment=train data=henn_lorenz decoder=route_aware_greedy tracking=wandb

python -m scenarios.scenario_henn_rl.experiment_structured_batching `
  experiment=evaluate data=henn_lorenz decoder=route_aware_greedy `
  checkpoint=outputs/henn_rl/structured/train/<run>/best.pt `
  experiment.split=test
```

Tracking is disabled by default. Add `tracking=wandb` for an online W&B run,
or add `tracking.mode=offline` when running without a connection. Training runs
record flow-time objective and reward consistency, distance/tours/decisions,
batch fill and size, replay and gradient diagnostics, actor/decoder/update
timings, validation checkpoints, the selected episode, and run artifacts.
Evaluation runs record the same comparable outcome and decision-time metrics
for the learned policy and every configured baseline.

The knapsack decoder exactly maximizes additive actor scores under cart
capacity. The route-aware decoder greedily includes positive marginal choices
using actual pick positions and `router.score()`; it is not an exact pricing
oracle.

## Code map

| Responsibility | Location |
|---|---|
| Generated data and splits | `scenarios/scenario_henn_rl/data.py` |
| Concrete CASIM composition | `scenarios/scenario_henn_rl/runtime.py` |
| Controlled batching/flow reward | `src/casim/envs/order_batching.py` |
| Observation, actor, critic, decoding | `learning/structured_batching/policy.py` |
| Candidate generation, targets, Fenchel-Young loss | `learning/structured_batching/learning.py` |
| Checkpoint I/O and compatibility | `learning/structured_batching/checkpoint.py` |
| Rollout, replay, updates, training | `learning/structured_batching/train.py` |
| Learned and FIFO/CW evaluation | `learning/structured_batching/evaluate.py` |
| Optional W&B experiment tracking | `learning/structured_batching/tracking.py` |

Checkpoints are versioned (format version 4) and retain the actor, critic,
decoder, feature count, route-cost scale, and reward normalization scale. A
checkpoint is rejected only on a real structural or semantic mismatch
(observation schema, actor/critic architecture, decoder semantics, or
incompatible reward normalization); the evaluation dataset need not match the
training dataset.
