# Henn reinforcement-learning scenario

This package contains the structured-batching workbench: the learned action is a
capacity-feasible subset of currently visible orders, decoded by an exact
additive knapsack oracle over per-order scores. Batching actions use the
existing CASIM/Henn domain, `ware_ops_algos` objects, S-shape routing, and
Henn release logic. The training path does not import Luigi or CoSy.

## Structured-batching quick start

Generate and inspect the deterministic Lorenz-inspired Henn corpus:

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
uv run python -m scenarios.scenario_henn_rl.experiment_structured_batching `
  experiment=generate data=henn_lorenz
```

Train the deadline-aware actor with exact SLA-tardiness reward and W&B:

```powershell
uv run python -m scenarios.scenario_henn_rl.experiment_structured_batching `
  experiment=train data=henn_lorenz objective=sla tracking=wandb
```

Use `tracking.mode=offline` when the machine has no network access. Tracking is
disabled by default; local JSON, JSONL, manifests, and checkpoints are always
written and remain the authoritative results.

The command prints its current stage and renders progress for initial
validation, training episodes, and checkpoint validation. Set `progress=false`
for quiet batch jobs. The simulator's internal event progress remains hidden
because it would create one noisy bar per episode.

Evaluate a selected checkpoint on the held-out test split:

```powershell
uv run python -m scenarios.scenario_henn_rl.experiment_structured_batching `
  experiment=evaluate data=henn_lorenz objective=sla `
  checkpoint=outputs/henn_rl/structured/train/<run>/best.pt `
  experiment.split=test tracking=wandb
```

Run a small exact counterfactual critic audit on validation states:

```powershell
uv run python -m scenarios.scenario_henn_rl.experiment_structured_batching `
  experiment=audit data=henn_lorenz objective=sla `
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

## Code map

| Responsibility | Location |
|---|---|
| Supported command and four-action dispatch | `experiment_structured_batching.py` |
| Generated instances and manifests | `structured/data.py` |
| Simulator state and exact knapsack action | `structured/environment.py` |
| Actor, critics, and structured losses | `structured/models.py` |
| SRL collection and optimization | `structured/training.py` |
| Learned and existing-policy evaluation | `structured/evaluation.py` |
| Exact alternative continuations | `structured/audit.py` |
| Optional W&B integration | `structured/tracking.py` |
| SRL failure diagnostics | `studies/srl_failure_diagnosis.py` |

The normal entry point does not import `studies`. The single diagnostic module
reproduces the canonical SRL-failure evidence (reward identity, exact-Q actor
audit, representability) from the checkpoint in
`outputs/henn_rl/final_diagnosis/`; see
`docs/structured_batching_srl_diagnosis.md`.

Historical seven-feature checkpoints are intentionally incompatible with the
current `deadline_v1` checkpoint schema. This prevents accidental evaluation of
a model that never observed due-date slack.
