# Objective-Derived Rewards for Online Order-Fulfillment Control

## Methodological objective

We construct reinforcement-learning rewards directly from an operational key
performance indicator (KPI), rather than combining heuristic bonuses for
desirable actions. The construction applies to additive order-level time
objectives, including flow time, weighted flow time, convex tail-sensitive flow
cost, and SLA tardiness. Its central property is **objective exactness**: the
undiscounted episode return equals the negative normalized KPI evaluated after
all relevant orders have completed.

This distinction is important. A shaped reward may be correlated with a KPI
without sharing its optimum. Here, the reward is an additive temporal
decomposition of the KPI itself.

## Problem setting

Let \(\mathcal I\) be the set of orders in an episode. For every order
\(i\in\mathcal I\), define:

- \(A_i\): external arrival time;
- \(C_i\): completion time;
- \(F_i=C_i-A_i\): order flow time;
- \(w_i>0\): exogenous business or priority weight;
- \(L_i\geq0\): tolerated service duration, such as an SLA threshold;
- \(p\geq1\): tail-sensitivity exponent.

Using \((x)_+=\max(0,x)\), define the order-level loss

\[
\ell_i(F_i)
=
w_i\bigl(F_i-L_i\bigr)_+^p.
\]

The episode objective is

\[
J
=
\sum_{i\in\mathcal I}\ell_i(F_i)
=
\sum_{i\in\mathcal I}
w_i\bigl(C_i-A_i-L_i\bigr)_+^p.
\tag{1}
\]

Equation (1) defines a family of operational objectives:

| Parameters | Resulting objective |
|---|---|
| \(p=1, L_i=0, w_i=1\) | total flow time |
| \(p=1, L_i=0\) | weighted flow time |
| \(p>1, L_i=0\) | convex, tail-sensitive flow cost |
| \(p=1, L_i>0\) | total SLA tardiness |
| \(p>1, L_i>0\) | convex SLA-tardiness cost |

For an order whose due date satisfies \(D_i\geq A_i\), the equivalent
service-duration threshold is \(L_i=D_i-A_i\). Absolute due dates are more
general: define \(q_i(t)=w_i(t-D_i)_+^p\) and

\[
G_q(t)=\sum_{i:A_i\leq t}q_i\!\left(\min(t,C_i)\right).
\]

This form also handles \(D_i<A_i\). The unavoidable cost
\(q_i(A_i)=w_i(A_i-D_i)_+^p\) enters when the already-overdue order arrives,
and the terminal value remains exactly \(q_i(C_i)\). More generally, the same
incremental construction applies to any specified nondecreasing order-level
completion loss \(q_i\); Equation (1) is the implemented weighted-power
subfamily. Weights, thresholds, and due dates must be fixed independently of
the agent's actions.

## Incremental reward construction

At simulation time \(t\), define the accrued age of order \(i\) as

\[
a_i(t)
=
\begin{cases}
0, & t<A_i,\\
\min(t,C_i)-A_i, & t\geq A_i.
\end{cases}
\]

Thus, age grows while an arrived order is unfinished and remains fixed after
completion. The cumulative objective accrued by time \(t\) is

\[
G(t)
=
\sum_{i\in\mathcal I}
w_i\bigl(a_i(t)-L_i\bigr)_+^p.
\tag{2}
\]

For consecutive decision times \(\tau_k\) and \(\tau_{k+1}\), the reward is

\[
r_k
=
-\frac{G(\tau_{k+1})-G(\tau_k)}{Z},
\qquad Z>0,
\tag{3}
\]

where \(Z\) is an action-independent normalization constant. A convenient
choice for homogeneous episodes is \(Z=|\mathcal I|H^p\), where \(H\) is a
fixed time scale. Raw KPI values should always be reported alongside normalized
returns.

An equivalent continuous-time interpretation follows from the marginal cost
rate. Almost everywhere,

\[
\frac{dG(t)}{dt}
=
\sum_{i:A_i\leq t<C_i}
w_ip\bigl(t-A_i-L_i\bigr)_+^{p-1}
\mathbf 1\{t-A_i>L_i\}.
\tag{4}
\]

Equation (3) integrates this cost rate exactly over each transition. The
increment form is generally preferable in an event-driven simulator because
it handles variable transition durations without numerical integration.

## Objective-exactness proposition

**Proposition.** Assume that the episode begins before any relevant order has
arrived, ends after every order in \(\mathcal I\) has completed, uses a fixed
positive normalizer \(Z\), and evaluates the undiscounted return
\((\gamma=1)\). Then

\[
\sum_{k=0}^{K-1}r_k=-\frac{J}{Z}.
\tag{5}
\]

**Proof.** Substituting Equation (3) and telescoping gives

\[
\sum_{k=0}^{K-1}r_k
=
-\frac{1}{Z}
\sum_{k=0}^{K-1}
\left[G(\tau_{k+1})-G(\tau_k)\right]
=
-\frac{G(\tau_K)-G(\tau_0)}{Z}.
\]

Before any arrival, \(G(\tau_0)=0\). After every completion,
\(a_i(\tau_K)=C_i-A_i=F_i\), hence \(G(\tau_K)=J\) by Equations (1) and (2).
Therefore Equation (5) follows. \(\square\)

## General properties

Within the stated objective family, the construction has the following
properties.

1. **Objective alignment.** Maximizing undiscounted return is equivalent to
   minimizing the specified KPI, because normalization by a fixed positive
   constant does not change policy ordering.
2. **Step-partition invariance.** Inserting or removing intermediate simulator
   transitions leaves the episode return unchanged. This is essential when
   decision intervals have unequal simulated duration.
3. **Dense temporal credit.** Cost accrues while orders age instead of appearing
   only as a terminal penalty. The density depends on the KPI: SLA tardiness
   correctly accrues no cost before the SLA threshold. An order already overdue
   at external arrival instead contributes its unavoidable initial tardiness at
   arrival.
4. **Interpretable tail sensitivity.** For \(p>1\), the marginal cost in
   Equation (4) increases with order age. Increasing \(p\) places progressively
   more emphasis on long-flow orders without claiming to optimize a quantile.
5. **Priority and SLA compatibility.** Heterogeneous \(w_i\) and \(L_i\) encode
   explicit operational classes and deadlines without action-specific bonuses.
6. **Controller independence.** The identity depends on arrival and completion
   times, not on whether these times result from batching, routing, storage,
   sequencing, or single- versus multi-resource execution.
7. **Auditability.** The return can be checked after every episode against the
   KPI computed independently from recorded arrival and completion timestamps.

These properties concern alignment, not learnability. Long action-to-completion
delays, partial observability, or a large structured action space may still
produce difficult credit assignment even when the reward is exact.

## Tail behavior and SLA interpretation

The exponent \(p\) changes the optimized objective:

\[
\frac{\partial\ell_i}{\partial F_i}
=
w_ip(F_i-L_i)_+^{p-1}\mathbf 1\{F_i>L_i\}.
\]

For \(p=1\), every additional second beyond the threshold has constant
marginal cost. For \(p=2\), the marginal cost grows linearly with excess age.
Consequently, two schedules with equal mean flow can be distinguished by a
convex objective: a balanced schedule is preferred to one containing a short
order and a very long tail order.

The exponent should be specified from the intended risk preference or selected
using validation data. It must not be tuned on the held-out test set. When a
contract specifies a deadline, direct tardiness with the contractual \(L_i\)
is more interpretable than choosing a power solely to improve p95.

## Scope and limitations

The exact formulation covers separable objectives of the form

\[
\sum_i w_i\ell_i(C_i-A_i).
\]

It does **not** exactly represent non-separable statistics such as median, p95,
CVaR, or maximum flow time. A convex power loss may correlate with improved
tail statistics but is not mathematically equivalent to them.

Other cumulative KPIs can follow the same incremental-cost principle but
require their own cumulative functional. Examples include distance, energy,
tour count, and resource occupancy. They should not be forced into the
order-age formulation. Ratio objectives such as throughput also require care,
because their denominators may depend on the policy.

Further conditions are necessary:

- **No discounting:** \(\gamma<1\) changes the relative value of cost incurred
  at different times and breaks Equation (5).
- **Complete episodes:** truncating before completion yields accrued partial
  cost rather than the complete KPI. A justified terminal residual value is
  required for censored orders.
- **External arrival accounting:** orders enter the objective at their external
  arrival, not when a controller admits or releases them.
- **Action-independent normalization:** policy-dependent scaling can change the
  optimization objective.
- **Exogenous parameters:** priorities and SLA thresholds must not be modified
  by the policy being evaluated.

## Validation protocol

The reward should be validated independently of policy learning.

1. **Closed-form identity tests.** Compare final accrued cost with a direct KPI
   calculation for linear flow, convex flow, weighted flow, tardiness, and
   convex tardiness.
2. **Partition tests.** Evaluate one trajectory using multiple event
   partitions and confirm identical total return.
3. **Randomized property tests.** Sample valid arrivals, completions, weights,
   thresholds, and powers; verify Equation (5) numerically.
4. **Behavioral sanity tests.** Confirm that linear loss is indifferent between
   schedules with equal total flow, while convex loss prefers the schedule with
   the smaller tail. Confirm that SLA cost ignores pre-threshold time and that
   weights affect only their assigned orders.
5. **Simulator integration tests.** Reconstruct the KPI independently from
   recorded arrival and completion timestamps and compare it with episode
   return.
6. **Failure tests.** Reject negative service-duration thresholds, conflicting
   threshold/due-date specifications, nonpositive weights or normalizers,
   missing completion records, completion before arrival, and decreasing
   cumulative cost.
7. **Absolute due-date tests.** Check due dates before, at, and after arrival,
   including the arrival-time cost jump for an already-overdue order.

The CASIM implementation and non-learning validation are located in
`scenarios/scenario_henn_rl/rewards.py` and
`tests/test_order_cost_reward.py`, respectively. The current Henn environment
uses the special case \(p=1\), \(L_i=0\), and \(w_i=1\), preserving its original
total-flow objective while obtaining the general identity checks above.

## Reporting recommendations

For each study, report:

- the exact objective \(J\), including units, \(p\), \(w_i\), and either
  \(L_i\) or absolute due dates \(D_i\);
- how parameters were chosen and which split was used for selection;
- the normalization \(Z\) and confirmation that \(\gamma=1\);
- the maximum absolute discrepancy between return and independently computed
  KPI;
- the raw primary KPI and relevant secondary operational metrics;
- treatment of unfinished orders at the episode boundary.

This makes reward design a reproducible statement of the operational objective
rather than an opaque collection of shaping terms.
