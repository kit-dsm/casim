# Structured RL for online order batching: literature and study proposal

## Finding

As of August 2026, direct feasibility-preserving learning of warehouse order
partitions appears underexplored. The reviewed warehouse RL papers usually
replace the partition with a small menu of categories or heuristics, or learn
routing/movement rather than batching. CO-augmented ML is a credible way to
retain the real combinatorial action. Its structured-RL variant is compatible
with the exact flow-time reward: the optimization layer constructs a feasible
batch, while the critic learns its long-run value from simulator transitions.

This is a research-gap assessment, not a claim that no related unpublished or
unindexed work exists.

## What current work learns

| Work | Learned decision and action space | Reward or objective | Assessment |
|---|---|---|---|
| [Cals et al. (2021)](https://doi.org/10.1016/j.cie.2021.107221) | PPO selects 15 aggregated order categories crossed with pick-by-order/batch, plus wait (31 actions); batch size and within-category selection are fixed | Infeasibility and tardiness penalties plus a squared terminal on-time term | A useful event-driven SMDP, but it does not learn the partition and uses hand-scaled shaping. |
| [Beeks et al. (2022)](https://doi.org/10.1609/icaps.v32i1.19829) | The same category/mode structure, with PPO | Tardiness/terminal terms plus underfilled-batch cost; reward weights tuned by Bayesian optimization | Explicitly studies the service/consolidation tradeoff, but tuned reward weights do not guarantee alignment with one operational KPI. |
| [Cheng et al. (2024)](https://doi.org/10.1007/s10489-024-05532-9) | DQN selects among eight shelf-repetition/deadline batching heuristics | Shelf-move reduction and delay penalty | Feasible and interpretable, but bounded by its handcrafted heuristic menu. |
| [Zhou, Lin and Cao (2023)](https://doi.org/10.1109/MRA.2023.3265515) | RL controls perturbation type and strength inside iterated local search | Integrated batching/job-assignment solution improvement | Learns search control, not an online warehouse batching policy. |
| [Mahmoudinazlou et al. (2025 revision)](https://arxiv.org/abs/2408.01656) | DQN chooses picker movement, stay, and unload primitives in dynamic picking | Pickup bonuses minus movement cost | Rich dynamic simulation, but action granularity is routing and its shaped reward is not the reported completion-time objective. |
| [Kang et al. (2024)](https://doi.org/10.1109/TASE.2024.3428541) | Supervised prediction reserves/releases orders based on expected similarity to future arrivals | Learns a similarity proxy; evaluates turnover time | Strong anticipatory idea and real data, but batch construction remains algorithmic and training is not end-to-end on flow. |
| [Bayram et al. (2022)](https://doi.org/10.1287/ijoo.2021.0066) | ML predicts uncertain processing time; branch-and-price constructs an offline partition | Robust total processing time | Strong ML+OR partitioning, but neither online nor sequential RL. |
| [Suemitsu et al. (2026)](https://doi.org/10.2493/jjspe.92.110) | RL adapts weights over seven online heuristics | Makespan and total shipment delay | Operationally broad and recent, but still learns a heuristic mixture rather than a partition. |

The [van Gils et al. order-batching review](https://doi.org/10.1016/j.ejor.2023.02.019)
is important context: release, batching, picker assignment, sequencing, routing,
capacity, and information assumptions differ enough that headline results are
rarely comparable.

## What is good, and what remains open

Existing work gets several things right: event-driven semi-Markov formulations,
feasibility-preserving actions, full-simulation evaluation, explicit anticipation
of future orders, and—in offline work—real partition optimization. The main open
space is the intersection of these properties.

Four limitations recur:

1. The combinatorial partition is engineered away into 8--31 templates.
2. Rewards mix arbitrary penalties or proxies rather than telescoping to the
   reported flow, tardiness, or energy KPI.
3. Prediction of whether to wait and optimization of what to batch are separate.
4. Aggregate state counts discard pairwise SKU/location compatibility and
   higher-order route synergy.

Benchmark fragmentation and missing end-to-end decision-latency reporting are
additional practical gaps.

## Why COAML/SRL fits

The current [COAML overview](https://arxiv.org/abs/2601.10583) describes a
policy in which a neural model maps state to surrogate optimization parameters,
then a combinatorial oracle maps those parameters to a feasible action. In
[Structured Reinforcement Learning (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/hash/fe100e5e7bca984803f0e4c6e94a6a88-Abstract-Conference.html),
perturbed oracle calls produce feasible candidate actions, a critic evaluates
them, and a Fenchel--Young loss trains the structured actor. The authors provide
an [official implementation](https://github.com/tumBAIS/Structured-RL).

For one picker, the smallest useful oracle selects the next batch rather than
partitioning the complete backlog:

```text
maximize    sum_i score_i x_i
subject to  sum_i demand_i x_i <= cart_capacity
            x_i in {0, 1}
```

This additive model is the necessary Occam baseline. It cannot express that
two spatially compatible orders are valuable specifically together. Only if
that limitation is measured should the surrogate add learned pair scores and
linearized `x_i * x_j` terms. Multiple simultaneous carts would then motivate
candidate-batch columns and set-packing constraints, including an explicit
unreleased-order variable.

The closest architectural precedent is the
[CO-enriched dynamic VRPTW work](https://arxiv.org/abs/2304.00789): learned
request prizes feed a prize-collecting routing oracle that jointly decides what
to dispatch and how to route it. Online batching has the same dispatch-now
versus preserve-for-future-combination tension.

## Compatibility with the flow reward

For decision interval `[tau_k, tau_(k+1)]`, use

```text
r_k = - integral(unfinished_orders(t), dt) / (N * H)
```

With a complete finite episode and `gamma=1`, the return is exactly negative
normalized total order flow time. The combinatorial oracle need not optimize a
different oracle reward. It returns feasible batches from learned scores; the
critic evaluates them with the flow transition reward.

This gives three distinct cases:

- SRL from experience is directly compatible.
- Imitation from a flow-optimal oracle is aligned.
- Imitation from a distance- or tardiness-only oracle is merely initialization
  for a different objective; flow-based fine-tuning may correct it.

A clairvoyant hindsight oracle also sees future arrivals unavailable online.
Use those labels for initialization only, then collect on-policy states or use
limited-lookahead labels. Before adopting the existing SRL implementation,
verify its code and theoretical assumptions for finite-horizon `gamma=1`.

## Recommended CASIM feasibility study

Keep one picker, Henn arrivals, fixed storage and routing, external-arrival flow
clocks, and complete finite episodes. Compare on identical deterministic test
instances:

1. a strong batching heuristic;
2. the current binary release PPO with fixed FIFO batching;
3. structured imitation from small exact or limited-lookahead batch labels;
4. that same actor fine-tuned with SRL and the flow reward;
5. SRL from scratch.

Start with the capacity-constrained additive oracle. For small visible pools,
enumerate feasible subsets or use a small MILP, simulate each under the fixed
continuation, cache results, and parallelize independent states. Add pairwise
compatibility only if the additive ablation fails.

Report mean, median and p95 flow time; travel; utilization; batch-size
distribution; wait frequency; feasibility; policy and solver latency; oracle
calls per update; small-instance exact regret; and robustness to arrival rate,
capacity, order size, and layout changes. Verify the reward/flow identity for
every completed episode.

This sequence answers the feasibility questions separately: whether the
surrogate represents good batches, whether structured imitation can recover
them, and whether online flow-reward SRL improves beyond imitation and strong
heuristics.
