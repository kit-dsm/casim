# Structured-RL failure diagnosis for online order batching

Single consolidated report for the Structured-RL (SRL) order-batching workbench.
Supersedes the earlier fragment reports (`structured_batching_srl_*`,
`structured_critic_counterfactual_audit`, `structured_batching_*_study`,
`structured_rl_for_dummies`, `structured_rl_order_batching`,
`structured_batching_loop_run_01`). The evidence is reproduced by one diagnostic
module and stored in one canonical directory.

- **Diagnostic module:** `scenarios/scenario_henn_rl/studies/srl_failure_diagnosis.py`
  (modes: `reward-identity`, `collect`, `actor-update`, `representability`).
- **Canonical evidence:** `outputs/henn_rl/final_diagnosis/` — the N2 checkpoint
  (`checkpoint/best.pt`), the exact-Q candidate dataset (`n2/dataset.json`), the
  actor-update audits (`n2/`, `n2_exactq/`, `n2_target/actor_update.json`), and the
  representability audit (`n2_repr/representability.json`).

The N2 policy is the best Structured-RL run: interaction critic, rank labels
`w=0.5`, selected episode 64, validation objective 8 325, test 8 462, flow
objective (power 1.0, gamma 1.0, no due dates), `objective_scale=8709.8`,
feature schema `deadline_v1` (8 features). All probes below use it.

## 1. Reward identity is exact (not the confound)

`run_reward_identity` rolls out full episodes with the production knapsack
oracle and checks `sum(rewards) == -objective_cost / scale`. The identity holds
to `< 1e-9` for `OrderCostReward` with power 1.0 and zero threshold (total flow
time). The learning signal is not corrupted by the reward model; this is
verified by `test_srl_diagnosis_reward_identity_is_exact` and the per-episode
`reward_identity_error` reported in every training run.

## 2. The critic-driven actor update never improves exact Q

`run_actor_update_audit` takes the N2 actor+critic at each of 16 selected
states (8 validation instances, quantiles 0.33/0.67) and applies **one**
Fenchel-Young update to a **fresh deep copy** of the checkpoint actor. Exact Q
is rolled out under the **frozen checkpoint actor** as continuation policy, so
`ΔQ` isolates the value of the action change at that state.

| target | quality source | fraction improving (ΔQ>0) | mean ΔQ | mean full-policy ΔQ |
|---|---|---|---|---|
| current (critic, τ=0.001) | learned critic | **0/16** | −0.0026 | −0.038 |
| exact_q (exact rollout Q, τ=0.1 standardized) | exact counterfactual | **1/16** | −0.0046 | −0.068 |

- The learned-critic target moves the actor to a worse exact-Q action in **0 of
  16** states (mean ΔQ −0.0026). The target is near-argmax (entropy 0.05, max
  weight 0.97) because the critic's Q-spread (≈13) is ~600× the exact objective
  spread (≈0.017), so the configured τ=0.001 is effectively hard-argmax on the
  critic's scale.
- The exact-Q target is genuinely soft (entropy 0.89, max weight 0.68,
  effective count 3.1) and uses trustworthy values (candidate Q-spread 0.018,
  the true objective scale). It **still** improves only **1 of 16** states
  (8/16 worsen, mean ΔQ −0.0046, slightly worse than the critic control).

**Conclusion:** the learned critic is *not* the binding error. Supplying the
target with exact counterfactual Q under the identical frozen continuation does
not make the Fenchel-Young update improve the chosen batch. The failure is in
the mapping `candidate values → FY update → additive order scores → knapsack
batch`, not in the Q source.

## 3. Additive selection is representable; the ordering is not

`run_representability` is actor-free and critic-free: per state it grants the
additive model a completely free per-order score vector, then asks whether the
production decoder and an additive batch utility can match the exact-Q
evidence.

### Test A — the best batch is always representable

| metric | value |
|---|---|
| states where the production decoder selects the exact-Q-best batch `B*` | **16/16** |
| states with positive margin | 16/16 |
| margin (all states) | exactly 1.0 (the theoretical ceiling) |

The additive action space can **express the argmax decision** in every state
with the maximal structurally-possible margin. Selection of the best batch is
never the problem.

### Test B — the exact-Q ordering is largely non-additive

| metric | value |
|---|---|
| states with complete pairwise ordering reproducible (positive margin) | 5/16 |
| mean achievable Spearman (LSQ fit) | 0.576 |
| mean achievable pairwise agreement (LSQ fit) | 0.741 |
| mean achievable top-1 match vs `B*` (LSQ argmax) | 0.312 |
| actor currently selects `B*` | 2/16 |

Even with a free per-state score vector, an additive batch utility cannot
reproduce the exact-Q candidate ordering in **11 of 16** states; the achievable
additive ordering saturates at ~0.74 pairwise agreement. The batch-level value
the FY soft target must emulate is worth more (or less) than the sum of its
parts.

## 4. Encoder context (supporting)

A leave-instance-out sufficiency study (numbers retained from the earlier
`n2_16` dataset; not reproduced by the consolidated diagnostic) showed that
augmenting the aggregate critic input with pick-location structure recovers the
immediate route *service time* (within-state Spearman 0.45 → 0.88) but route
*distance* stays hard (Spearman ≤ 0.19), and that no representation lets an
out-of-instance critic rank exact continuation Q above chance (Spearman ≈ 0.2,
top-1 ≤ 0.19) even with exact labels. The state→continuation-value map embeds a
moving future policy and is not learnable as a static ranking from these
features. This is context for why the critic target is weak, not a separate
intervention.

## 5. Causal conclusion

Ranked by strength of evidence:

1. **Reward/objective identity holds** (§1) — not the confound.
2. **The critic target is not the binding error** (§2): the exact-Q weighted
   soft target — soft, correctly ranked, trustworthy values — still drives
   actions to worse exact Q (1/16 vs 0/16). Replacing the critic with exact
   continuation Q does not rescue the FY update.
3. **The additive action space represents every best batch but not the
   ordering** (§3): 16/16 states decode `B*` with maximal margin, yet 11/16
   states have a non-additive exact-Q ordering. The FY update needs score
   *orderings* that track Q; the additive parameterization cannot represent
   those orderings in most states, so the update has no consistent score
   direction.
4. **The actor retains a large improvement margin** (actor regret fraction
   0.43; only 2/16 states have the actor already at `B*`) while the critic
   value target is learned from the actor's own internal features and cannot
   find that margin.

The remaining structural alternative is **exact-continuation-value actor
updates under a distributional / variance-reduced target shape** — not a
fix to additive representability. The exact-Q audit already showed that a
pointwise soft target over exact Q is insufficient (1/16); the open question is
whether a distributional target that separates the *aggregate* value of a good
batch into a per-action improvement signal can move the additive scores
consistently. The batch-cost formulation (`argmax_B [Σθ_i(s) − c(B)]`,
Heßler–Irnich) addresses the non-additive ordering directly but is a
different action parameterization, not an actor-update target fix.

## 6. What was ruled out

- **TD bootstrap / reward scaling:** the U-shape (improve then forget) and
  train-return degradation reproduce with an exact Monte-Carlo `return_to_go`
  critic (run E1), so TD bootstrap and reward scaling are not root causes.
- **Exploration collapse:** N2 (constant `sigma_B=1.0`) is the best run;
  annealing `sigma_B` 1.0→0.1 (N4) improves critic audits but hurts the
  objective (val 8 490 vs 8 325). Exploration is a symptom, not the cause.
- **Critic representation alone:** augmenting the critic input with spatial
  structure recovers the immediate service-time signal but does not create
  exact-Q ranking signal (§4); and the exact-Q audit (§2) shows ranking is
  moot once the target is exact — the FY update still fails.

## 7. Reproducing the evidence

```
uv run python -m scenarios.scenario_henn_rl.studies.srl_failure_diagnosis \
    --mode reward-identity --instances 4
uv run python -m scenarios.scenario_henn_rl.studies.srl_failure_diagnosis \
    --mode collect --checkpoint outputs/henn_rl/final_diagnosis/checkpoint/best.pt \
    --output outputs/henn_rl/final_diagnosis --split validation \
    --instances 8 --state-quantiles 0.33,0.67 --candidate-count 20 --collect-q
uv run python -m scenarios.scenario_henn_rl.studies.srl_failure_diagnosis \
    --mode actor-update --checkpoint outputs/henn_rl/final_diagnosis/checkpoint/best.pt \
    --output outputs/henn_rl/final_diagnosis --split validation \
    --instances 8 --state-quantiles 0.33,0.67 --exact-q
uv run python -m scenarios.scenario_henn_rl.studies.srl_failure_diagnosis \
    --mode representability --dataset outputs/henn_rl/final_diagnosis/n2/dataset.json \
    --output outputs/henn_rl/final_diagnosis
```

The canonical JSON in `outputs/henn_rl/final_diagnosis/` is the stored result of
these commands against the N2 checkpoint.
