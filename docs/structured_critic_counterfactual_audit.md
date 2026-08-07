# Structured critic counterfactual audit

## Question and method

This focused audit asks whether the failed convex-flow study was limited by
the feasible candidate set, critic ranking, or actor extraction. It does not
introduce another batching policy.

The deterministic episode-8 critics for `p=1` and `p=2` were reconstructed
from the original seed and training order. Their actors matched the frozen
study checkpoints exactly by SHA-256 state digest. For each actor, two states
were sampled from four validation instances spanning 40, 60, 80, and 100
orders. At every state, the audit reproduced the existing 40 perturbed SRL
oracle calls, removed duplicate actions, and replayed each distinct batch from
the exact simulation prefix. The remainder of the episode used the same frozen
actor. Complete counterfactual returns were calculated for both objectives.

The audit therefore measures the value of changing the current batch under a
fixed continuation. It is not a globally optimal look-ahead oracle.

## Results

| Metric | `p=1` critic | `p=2` critic |
|---|---:|---:|
| Audited states | 8 | 8 |
| Mean distinct candidates | 34.25 | 34.50 |
| Mean Spearman rank correlation | 0.504 | 0.521 |
| States with positive rank correlation | 100% | 100% |
| Critic top-choice accuracy | 0% | 37.5% |
| Actor top-choice accuracy | 0% | 0% |
| Mean critic regret / available spread | 19.4% | 17.1% |
| Mean actor regret / available spread | 19.6% | 34.3% |
| Mean predicted Q spread | 0.00329 | 0.00382 |
| Mean true Q spread | 0.00195 | 0.00084 |
| Mean softmax entropy | 3.07 | 2.85 |
| States where `p=1` and `p=2` prefer different candidates | 87.5% | 87.5% |

The candidate set is not the immediate limitation. For `p=2`, choosing the
best audited candidate would improve the remaining squared-flow objective by
4.93% on average (median 2.77%, maximum 15.40%) relative to the actor action.
The corresponding values for `p=1` are 1.11%, 0.85%, and 2.72%.

The critic does learn useful ordering information: every audited state has a
positive rank correlation. But calibration is poor and state-dependent. The
ratio of predicted to true candidate spread has a median of 2.32 for `p=1`
and 7.77 for `p=2`, with both under- and over-estimation across states. This
contradicts a simple global-compression explanation.

At temperature `0.001`, the critic-weighted target is also much weaker than
the available counterfactual improvement. Under true returns, its expected
change relative to the actor action is -0.70% for `p=1` and only +0.55% for
`p=2`. Thus the `p=2` candidate set contains a mean 4.93% improvement, but the
critic weighting communicates only a small fraction of it to the structured
actor.

## Interpretation

The convex reward is learnable in the operational sense required here:
feasible candidate batches with materially better squared-flow consequences
exist, and `p=1` and `p=2` usually prefer different batches. The negative
training result is therefore not explained by identical objectives or an
incapable knapsack candidate set.

The immediate bottleneck is the critic-to-actor path:

1. critic rankings are only moderately accurate;
2. predicted Q differences are not calibrated to true local differences;
3. a single global softmax temperature cannot correct state-dependent scale;
4. the soft target loses most of the available `p=2` advantage;
5. the actor's decoded action does not recover the best sampled candidate.

Simply increasing the reward magnitude or training longer is not justified by
these results. A better next experiment is to train or calibrate the critic on
complete return-to-go labels from already collected episodes, record held-out
candidate rank correlation and regret, and normalize candidate advantages
within each state before constructing the soft target. This should be tested
on `p=2` with the existing actor and oracle before expanding the study.

The complete state- and candidate-level results are stored in
`outputs/henn_rl/critic_audit_seed11/result.json`. The audit took 363 seconds;
exact full-continuation labels are practical for diagnostics or a modest
offline calibration set, but too costly to generate for every training update.

## Return-to-go calibration pilot

A gated follow-up trained the `p=2` critic from complete episode return-to-go
labels and standardized its candidate Q-values within each state before the
softmax. It retained seed 11, the first eight training episodes, no-wait
actions, and the existing 40-candidate oracle. The same eight validation states
were then audited. All four predefined gates failed:

| Metric | Previous `p=2` audit | Calibrated pilot | Required |
|---|---:|---:|---:|
| Mean Spearman correlation | 0.521 | -0.079 | at least 0.60 |
| Critic regret / available spread | 17.1% | 57.4% | at most 12% |
| Actor regret / available spread | 34.3% | 55.5% | at most 25% |
| Soft-target capture of available improvement | 11.1% | 10.0% | at least 30% |

The experiment stopped after 130 seconds. It did not launch the 128-episode
arms or evaluate the held-out test split.

Complete return-to-go removes temporal-difference bootstrapping error for the
batch actually taken, but it supplies no direct label for the other candidate
batches. In this pilot the critic fitted observed-action returns without
learning a useful local ordering over untaken actions. State-local score
normalization cannot repair an incorrect ordering. The next justified step is
therefore the predefined fallback: generate a modest exact counterfactual
calibration set from training-instance states, train the critic on those
state--candidate labels, and retain the current validation states exclusively
for auditing.

## Training-only exact-counterfactual fallback

The planned fallback then generated exact complete-continuation labels at two
trajectory positions on four representative training instances. This produced
226 distinct candidate labels across eight states. The validation instances
and audit states were unchanged and were never used for fitting.

The existing critic was fitted to state-standardized exact returns and the
existing actor was updated through the same normalized critic-weighted SRL
target. This was a one-shot offline calibration, not a new policy or a new
training framework.

| Metric | Return-to-go pilot | Exact-label fallback | Required |
|---|---:|---:|---:|
| Mean validation Spearman correlation | -0.079 | 0.371 | at least 0.60 |
| Critic regret / available spread | 57.4% | 46.1% | at most 12% |
| Actor regret / available spread | 55.5% | 55.6% | at most 25% |
| Soft-target capture of available improvement | 10.0% | 46.0% | at least 30% |

The exact labels materially improved validation ranking and made the soft
target useful enough to pass its capture gate. They did not produce reliable
top-candidate selection: on the calibration states, mean Spearman correlation
rose from -0.216 to 0.564, but top-candidate accuracy remained zero. On
validation, three of the four gates still failed and the actor regret was
unchanged. The experiment therefore stopped without 128-episode training or
test evaluation.

This result narrows the failure. Counterfactual information helps, so the
reward signal is not devoid of useful discrimination. However, this small
one-shot set did not make the current mean-pooled critic accurately fit the
best actions, and the partial critic improvement did not transfer to the
actor. More long on-policy training is not justified yet. The next work should
first determine whether the existing critic can deliberately overfit a small
exact-label set to near-perfect ranking; if it cannot, critic representation or
the calibration loss is the immediate limitation. If it can, more diverse
training states and periodic refresh of stale counterfactual labels become the
next hypothesis.

The complete run took 401.8 seconds, including 265.1 seconds for label
generation, fitting, and the second validation audit. Raw output is stored in
`outputs/henn_rl/critic_calibration_seed11/result.json`.

## Critic memorization diagnostic

The existing 226 exact labels were then reused without additional simulation.
Two fresh instances of the unchanged `StructuredCritic` were deliberately
trained on all eight labelled states for 2,000 epochs. This is a training-set
memorization test, not a generalization result. Success was defined in advance
as mean Spearman correlation of at least 0.95 and top-candidate accuracy of at
least 87.5% (seven of eight states).

| Fitting objective | Mean Spearman | Pairwise accuracy | Top-candidate accuracy | Gate |
|---|---:|---:|---:|---:|
| State-standardized Smooth L1 regression | 0.722 | 77.9% | 25.0% | fail |
| Pairwise ranking | 0.867 | 86.1% | 50.0% | fail |

Neither arm memorized the small training set. The direct ranking objective was
substantially better, showing that the regression loss is part of the problem,
but changing the loss alone was insufficient. The strongest current evidence
therefore points to the critic representation or capacity as the primary
bottleneck. In particular, the critic reduces a variable candidate to the mean
of per-order embeddings before predicting value. Candidate quality depends on
interactions among selected locations and their continuation consequences,
which may be lost in that compression.

This does not prove that the architecture can never represent these rankings;
it establishes that it failed a generous controlled memorization test under
both relevant fitting objectives. Generating more expensive labels or resuming
long SRL training is not justified until the critic can pass this test. A next
architecture experiment should remain isolated to this same memorization set
and add the smallest representation of selected-order interactions necessary
to reach the gate before any validation or policy training is repeated.

The two-arm diagnostic took 208.4 seconds. Its raw output is stored in
`outputs/henn_rl/critic_overfit_seed11/result.json`.

## Interaction-aware critic follow-up

The minimal follow-up retained the critic's state--action interface but changed
how it represents a candidate. It separately pools all visible orders,
selected orders, and learned embeddings of every pair of selected orders. Pair
features use the sum and absolute difference of their order embeddings, making
the representation invariant to order permutation while retaining which
locations are selected together.

This matters because batching value is not additive. Two orders can each look
attractive alone but be poor together when their locations create a long tour;
the original mean-pooled critic had no explicit representation of that
co-selection relationship.

With the same pairwise ranking loss, the interaction-aware critic reached
Spearman 0.971, 94.7% pairwise accuracy, and six of eight correct top
candidates. This narrowly failed the predefined seven-of-eight top gate. The
reason is consistent with the loss: one best-candidate comparison contributes
little among hundreds of equally weighted candidate pairs.

A controlled memorization-only follow-up therefore combined the pairwise loss
with top-candidate cross-entropy. It did not change the reward, labels,
architecture, data, or gate.

| Critic and fitting objective | Mean Spearman | Pairwise accuracy | Top accuracy | Gate |
|---|---:|---:|---:|---:|
| Mean-pooled critic, pairwise loss | 0.867 | 86.1% | 50.0% | fail |
| Interaction-aware critic, pairwise loss | 0.971 | 94.7% | 75.0% | fail |
| Interaction-aware critic, pairwise + top loss | 0.964 | 94.2% | 100.0% | **pass** |

The final arm passed both predefined requirements. This establishes that the
small exact-label set is learnable and that selected-order interactions resolve
the main representation bottleneck. It also shows that a purely pairwise loss
is slightly misaligned with a gate that requires selecting the single best
candidate; the top-aware term supplies that missing supervision.

This remains a memorization result. The interaction critic must next be frozen
and evaluated on the unchanged validation counterfactual states. Only if it
improves validation ranking and regret should it be used to construct actor
targets. Long SRL training remains premature.

The implementation was reduced from processing every visible-order pair to
only selected-order pairs, which is equivalent for the discrete candidate
actions. The passing run took 244.6 seconds. Raw output is stored in
`outputs/henn_rl/critic_interaction_top_overfit_seed11/result.json`.

## Frozen validation transfer

The passing interaction critic was frozen and evaluated on the original eight
validation audit states. Candidate generation, quantiles, seed, reward power,
and exact continuation evaluation were unchanged. The actor was not updated,
so actor regret was excluded from this critic-only transfer gate.

| Metric | Mean-pooled exact-label critic | Interaction-aware critic | Required |
|---|---:|---:|---:|
| Mean Spearman correlation | 0.371 | 0.316 | at least 0.60 |
| Critic regret / available spread | 46.1% | 31.5% | at most 12% |
| Soft-target capture of available improvement | 46.0% | 18.2% | at least 30% |
| Critic top-candidate accuracy | 0% | 0% | diagnostic |

The interaction representation reduced critic regret but did not generalize
the training ranking: all three transfer gates failed. This separates capacity
from coverage. The critic can memorize the exact training labels, but eight
states do not adequately cover the state--candidate relationships encountered
on validation. Updating the actor would therefore amplify an unvalidated
signal and was intentionally skipped.

The next justified experiment is a larger but still offline counterfactual
calibration set sampled across more training instances and trajectory stages.
It should retain this frozen validation audit and use the existing passing
interaction critic rather than increasing model capacity again. The validation
run took 128.7 seconds; raw output is stored in
`outputs/henn_rl/critic_interaction_validation_seed11/result.json`.

## Counterfactual coverage study

The coverage experiment expanded the offline training set from 8 to 32 nested
states while retaining the same frozen actor, interaction-aware critic,
pairwise-plus-top loss, initialization, 2,000-epoch budget, and eight-state
validation audit. The final dataset contains 994 exact candidate labels from
eight training instances. It covers all four instance families and order-count
ranges, with early, middle, and late trajectory positions. The test split was
not accessed.

| Training states | Training Spearman | Training top accuracy | Validation Spearman | Validation critic regret | Validation soft-target capture | Gate |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.964 | 100.0% | 0.316 | 31.5% | 18.2% | fail |
| 16 | 0.957 | 100.0% | 0.410 | 26.1% | 15.9% | fail |
| 32 | 0.765 | 96.9% | 0.402 | 35.7% | 20.6% | fail |
| Required | — | — | at least 0.60 | at most 12% | at least 30% | — |

Coverage helped from 8 to 16 states but then plateaued. The 32-state critic no
longer fitted the complete training ranking cleanly under the same budget, and
none of the validation measurements approached its gate monotonically. The
best validation rank correlation was 0.410, the best regret was 26.1%, and the
best soft-target capture was 20.6%; these came from different dataset sizes.

This rejects the simple hypothesis that a modest increase in state coverage is
sufficient. It does not invalidate the squared-flow objective: exact candidate
returns remain distinct and learnable on small sets. It does show that the
current critic inputs, architecture, and supervised objective do not provide a
reliable transferable candidate ranking for that objective. Actor updates and
long SRL training remain unjustified.

The complete study took 1,642.3 seconds (27.4 minutes). The three fits ran in
spawn-safe worker processes with numerical threads limited to one. Raw results
and the reusable dataset are stored in
`outputs/henn_rl/critic_coverage_seed11/`.

## Candidate operational-feature study

The next diagnostic asked whether the critic inputs omitted simple operational
facts that already exist in the deterministic Henn pipeline. No new labels,
heuristic, or simulator logic were introduced. For each of the existing 994
training-only candidate labels, the study reconstructed the state and measured
the candidate with the configured S-shape router. Features included route
distance and service time, route saving relative to separate tours, cart fill,
order and pick counts, selected and unselected order ages, aisle coverage,
spatial span, and visible backlog.

Model selection used leave-one-training-instance-out cross-validation. Thus,
candidates from the same warehouse instance and decision could not appear on
both sides of a fold. Targets were standardized within each decision because
only candidate ranking is relevant and exact squared-flow returns have
state-dependent scales. Three ridge models tested route cost alone, route plus
age, and all operational features. Because the full linear model still
underfit, one 16-unit, one-hidden-layer model was fitted with the same
pairwise-plus-top ranking loss as the interaction critic. The nonlinear model
was selected solely by training CV regret; the test split was never accessed.

| Training-instance CV arm | Spearman | Regret / available spread | Top accuracy | Soft-target capture |
|---|---:|---:|---:|---:|
| Route distance only | 0.131 | 37.3% | 15.6% | 19.2% |
| Route distance + age | 0.554 | 23.2% | 28.1% | 24.2% |
| All operational, linear | **0.632** | 18.9% | 31.3% | **30.8%** |
| All operational, small MLP | 0.583 | **12.1%** | **34.4%** | 30.4% |

Route distance alone is a poor proxy for the squared-flow continuation value.
Adding the ages of selected and left-behind orders supplies substantial signal;
backlog, fill, and spatial features add more. This supports the hypothesis that
the earlier neural critic was missing directly useful operational summaries.
However, the different CV metrics disagree about which formulation is best,
and even the selected MLP remained just outside the 12% regret requirement.

The selected MLP was then frozen and evaluated once on the existing eight-state
validation audit:

| Frozen validation metric | Result | Required |
|---|---:|---:|
| Mean Spearman | 0.523 | at least 0.60 |
| Critic regret / available spread | 31.1% | at most 12% |
| Soft-target capture | 24.5% | at least 30% |
| Top-candidate accuracy | 0/8 | diagnostic |

All gates failed, so the actor was not updated. Performance was weakest on the
two `ran2`/100-order states, where regret reached 66.7% and soft weighting was
worse than the frozen actor. The family and order-count effects cannot be
separated in this eight-state audit because each family occurs at only one
order count. By trajectory stage, validation Spearman was 0.379 early, 0.580
in the middle, and 0.629 late, but each group contains only two or three states.

An earlier staged linear-only audit, performed before the nonlinear arm was
added, reached Spearman 0.553, 15.2% regret, and 29.0% capture on these same
validation states. This was closer to the gate than the CV-selected nonlinear
model, but it is now an exploratory observation rather than a valid basis for
another selection on validation data. The contrast is evidence of overfitting,
not permission to choose the linear arm after seeing its validation result.

The study therefore does not justify another long squared-objective SRL run.
It shows that operational features make the ranking problem easier, while the
available eight training instances are still insufficient for stable model
selection and transfer. Raw results are stored in
`outputs/henn_rl/candidate_features_seed11/result.json`.
