# Henn reinforcement-learning scenario

This package contains the structured-batching workbench: the learned action is a
capacity-feasible subset of currently visible orders, decoded from per-order
scores by a configured decoder. Batching actions use the existing CASIM/Henn
domain, `ware_ops_algos` objects, S-shape routing, and Henn release logic. The
training path does not import Luigi or CoSy.

## Decoders

The decoder is selected explicitly via the `decoder` Hydra config group. There
is exactly one supported path from actor scores to a decoded action; training,
validation, evaluation, checkpoint evaluation, and audit all use the same
configured decoder.

| `decoder=` | Guarantee | Route cost | Notes |
|---|---|---|---|
| `knapsack` | Exact additive-score 0/1 knapsack | Ignored | Default |
| `route_aware_greedy` | Greedy marginal-acceptance (approximate) | Fixed S-shape route distance | Not exact pricing |

The `route_aware_greedy` decoder approximates the route-aware pricing problem;
it is **not** an exact profitable-SPRP oracle. An exact Gurobi-based SPRP
pricing implementation exists elsewhere in the repository but is too slow for
RL training and is deliberately kept offline-only; it is never invoked from
the RL hot path.

## Structured-batching quick start

Generate and inspect the deterministic Lorenz-inspired Henn corpus:

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
uv run python -m scenarios.scenario_henn_rl.experiment_structured_batching `
  experiment=generate data=henn_lorenz
```

### Knapsack training

```powershell
uv run python -m scenarios.scenario_henn_rl.experiment_structured_batching `
  experiment=train data=henn_lorenz decoder=knapsack objective=sla tracking=wandb
```

### Greedy route-aware training

```powershell
uv run python -m scenarios.scenario_henn_rl.experiment_structured_batching `
  experiment=train data=henn_lorenz decoder=route_aware_greedy objective=sla tracking=wandb
```

Use `tracking.mode=offline` when the machine has no network access. Tracking is
disabled by default; local JSON, JSONL, manifests, and checkpoints are always
written and remain the authoritative results.

The command prints its current stage, the selected decoder, and the objective at
startup, and renders progress for initial validation, training episodes, and
checkpoint validation. Set `progress=false` for quiet batch jobs. The
simulator's internal event progress is hidden because it would create one noisy
bar per episode.

### Evaluate a selected checkpoint

The checkpoint stores the decoder; evaluation reconstructs the exact
actor-decoder combination and rejects an incompatible command-line decoder
instead of silently evaluating a different policy.

```powershell
uv run python -m scenarios.scenario_henn_rl.experiment_structured_batching `
  experiment=evaluate data=henn_lorenz decoder=route_aware_greedy objective=sla `
  checkpoint=outputs/henn_rl/structured/train/<run>/best.pt `
  experiment.split=test tracking=wandb
```

### Audit

```powershell
uv run python -m scenarios.scenario_henn_rl.experiment_structured_batching `
  experiment=audit data=henn_lorenz decoder=route_aware_greedy objective=sla `
  checkpoint=outputs/henn_rl/structured/train/<run>/best.pt
```

Hydra overrides are the experiment interface. For example,
`data.orders_per_four_hours=110`, `data.location_policy=class_based`, or
`learner.actor_learning_rate=0.0005` changes one controlled factor. Use Hydra
multirun for grids; W&B receives one run per Hydra job.

## Reward scale

Training rewards are the per-decision negative cost increment divided by an
instance scale `max(1, n_orders) * objective_scale`. By default the scale is
computed before training as the reference policy's mean per-order objective
(the `cw`/`sav`/`no_wait` baseline on validation), so the total normalized
episode reward is near minus one under the configured objective (flow time or
tardiness) regardless of instance size. Set `objective_scale=5000` to pin a
fixed scale instead; the computed value is stored in `result.json` and in the
`best.pt` checkpoint so evaluation and audit reuse the training scale.

The discount factor is `gamma = 1`: flow time and SLA tardiness are total-cost,
finite-horizon objectives, so the critic learns the exact remaining cost and
the optimizing policy also minimizes the stated objective. This matches the
official DVSP agent in the Structured-RL repository (`gamma = 1.0`), where
discounted values (`0.99`) are used only for the stochastic DAP/GSPP
environments.

## Checkpoints

Every advertised checkpoint file (`best.pt`, `episode_NNN.pt`) uses the same
versioned, self-describing schema:

- checkpoint format version;
- feature schema;
- actor architecture and state;
- critic architecture and state;
- decoder type and parameters;
- route-cost coefficient/scaling definition;
- reward/objective configuration;
- relevant data configuration identity;
- selected training episode.

When validation selects the best episode, the actor **and** critic from the
same training point are saved. Evaluation and audit obtain the policy-defining
configuration from the checkpoint and reject incompatible objective, feature
schema, data configuration, or decoder.

## Code map

| Responsibility | Location |
|---|---|
| Supported command dispatch | `experiment_structured_batching.py` |
| Generated instances and manifests | `structured/data.py` |
| Typed structured decision state | `structured/state.py` |
| Decoder interface, factory, and implementations | `structured/decoders.py` |
| Structured policy (actor + decoder) | `structured/policy.py` |
| Versioned checkpoint I/O and compatibility | `structured/checkpoint.py` |
| Simulator state and structured episode | `structured/environment.py` |
| Actor, critics, and structured losses | `structured/models.py` |
| SRL collection and optimization | `structured/training.py` |
| Learned and existing-policy evaluation | `structured/evaluation.py` |
| Exact alternative continuations | `structured/audit.py` |
| Optional W&B integration | `structured/tracking.py` |
| Legacy knapsack-only SRL diagnostics | `studies/srl_failure_diagnosis.py` |

The normal entry point does not import `studies`. The single diagnostic module
reproduces the canonical SRL-failure evidence (reward identity, exact-Q actor
audit, representability) from the checkpoint in
`outputs/henn_rl/final_diagnosis/`; it is knapsack-specific and rejects
route-aware checkpoints. See
`docs/structured_batching_srl_diagnosis.md`.

Historical seven-feature checkpoints are intentionally incompatible with the
current `deadline_v1` checkpoint schema. This prevents accidental evaluation of
a model that never observed due-date slack.

## Limitations

- The greedy route-aware decoder is approximate; it is not an exact
  profitable-SPRP pricing oracle.
- The exact Gurobi SPRP pricing implementation remains offline-only and is never
  invoked from the RL hot path.
- The learned policy currently cannot wait: both decoders produce a nonempty
  batch whenever any feasible order exists (waiting is not decoded).
