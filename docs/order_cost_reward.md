# Objective-derived order-cost reward

A paper-oriented derivation, exactness proof, assumptions, and reporting
guidance are provided in
[objective_derived_reward_methodology.md](objective_derived_reward_methodology.md).

The implementation lives in
`scenarios/scenario_henn_rl/rewards.py`. It remains scenario-local until a
second scenario needs the same behavior.

For order arrival `A`, completion `C`, weight `w`, SLA threshold `L`, and tail
power `p`, the represented episode cost is:

```text
w * max(0, C - A - L) ** p
```

Summing this over orders gives the complete objective. The environment reward
is the negative normalized increase in the same cost accrued by completed and
currently active orders. No proxy bonuses or heuristic action rewards are
used.

The parameters have direct operational interpretations:

| Parameters | Objective |
|---|---|
| `p=1`, `L=0`, uniform `w` | total/mean flow time |
| `p>1`, `L=0` | convex flow cost with increasing tail emphasis |
| `p=1`, `L>0` | total SLA tardiness |
| per-order `w` | priority-weighted objective |
| per-order `L` | heterogeneous SLA thresholds |
| absolute `due_times` | completion-time tardiness, including overdue arrivals |

`thresholds_s` describes a nonnegative tolerated duration after arrival.
`due_times` describes absolute deadlines; the two inputs are mutually
exclusive. If an order arrives after its due date, its existing tardiness is
accrued at arrival so the final return remains equal to raw tardiness.

This family is objective-exact for separable order completion costs. It does
not claim to represent quantiles such as p95 exactly.

## Validation

`tests/test_order_cost_reward.py` validates the reward without RL:

1. terminal accrued cost equals the direct closed-form KPI for linear,
   squared, tardiness, and squared-tardiness settings;
2. the sum of incremental rewards is invariant to event/step partitioning;
3. seeded randomized trajectories preserve the identity for powers 1, 1.5,
   2, and 3 with heterogeneous weights and thresholds;
4. convex costs prefer balanced completion times over a tail-heavy schedule
   with the same mean flow;
5. SLA and priority parameters affect only the orders they describe;
6. absolute due dates before, at, and after arrival preserve the identity;
7. invalid parameters, incomplete objectives, accrued-cost regressions, and
   impossible completion times fail explicitly.

The existing Henn RL environment now uses the general implementation with
`p=1`, zero thresholds, uniform weights, and its previous normalization. Its
existing episode-level reward/flow identity tests therefore serve as an
integration and backward-compatibility check.

Run the non-RL validation from PowerShell:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_order_cost_reward.py -q
```
