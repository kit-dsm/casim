# Structured online batching feasibility

See also [Structured-RL failure diagnosis](structured_batching_srl_diagnosis.md)
for the post-training causal analysis of why the learned critic target does not
reliably improve the chosen batch.

## Decision and implementation

The experiment uses the existing one-picker Henn online scenario. The learned
action is a capacity-feasible subset of all currently visible orders; an empty
subset waits. Existing FIFO, Clark-and-Wright, local-search batching, S-shape
routing, scheduling, and Henn release logic come from `ware_ops_algos` and the
Henn scenario.

The actor assigns additive scores to visible orders and an exact dynamic-
programming knapsack oracle creates the subset. The SRL learner uses perturbed
feasible candidates, critic weighting, a Fenchel--Young loss, Huber critic
loss, and `gamma=1`. The incremental holding-cost reward sums exactly to
negative normalized total order flow time; the measured identity error remains
below `3e-17`.

The general objective-derived implementation and its non-RL validation are
documented in [order_cost_reward.md](order_cost_reward.md).

CoSy remains the canonical pipeline construction and applicability mechanism.
The learning and evaluation loops use the equivalent fixed pipeline directly
to avoid Luigi overhead. On `H_abc1_40_29`, direct and CoSy executions matched
exactly for FIFO, C&W-like, and LS: decision count and time, candidate
partitions, selected batches, actions, and reasons were identical.

## Corrected comparison design

An earlier report coupled every existing batcher to a scenario-local 75%-fill
or 300-second release helper and incorrectly described it as established. That
comparison is superseded. Release policies now live in the existing Henn
decision path:

- `no_wait`: dispatch whenever work is available;
- `henn_4_1`: the published critical-order threshold and timed wake-up;
- `fill_or_age`: the former 0.75/300 rule, retained only as a fixed diagnostic.

The corrected study evaluated the complete `no_wait` and `henn_4_1` matrix over
FIFO, C&W-like, and LS batching and the existing `first`, `short`, `long`, and
`sav` selectors. All 24 configurations were evaluated on all 16 validation
instances. The best selector for each release/batcher pair was frozen before
one evaluation on all 16 test instances. Validation selected `sav` in every
pair and C&W as the best batcher under both release policies.

The matrix used three Windows spawn workers with one numerical-library thread
per worker. Selected test configurations ran serially because LS is time-
limited to 0.25 seconds per decision. The complete corrected experiment took
468 seconds. The fill/age diagnostic was not tuned and did not participate in
selection. Frozen scratch checkpoints were reused; the invalid C&W-imitation
arm was excluded rather than retrained.

## Results

All figures are means over the 16 held-out test instances. Flow and p95 are
seconds per order. Throughput is completed orders per simulated hour.

| Policy | Mean flow | Mean p95 | Distance | Tours | Fill | Orders/batch | Wait | Throughput | Decision time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No-wait FIFO + SAV | 9,266 | 21,774 | 11,768 | 24.19 | 0.81 | 3.02 | 0% | 8.55 | 0.240 s |
| No-wait C&W + SAV | 8,494 | **20,005** | **10,276** | 22.00 | 0.88 | 3.24 | 0% | **9.21** | 1.812 s |
| No-wait LS + SAV | 8,844 | 20,796 | 10,893 | 23.06 | 0.84 | 3.09 | 0% | 8.87 | 1.569 s |
| Henn FIFO + SAV | 9,340 | 21,853 | 11,772 | 24.19 | 0.81 | 3.02 | 9.4% | 8.53 | 0.137 s |
| Henn C&W + SAV | 8,494 | 20,281 | 10,247 | 22.00 | 0.88 | 3.24 | 9.3% | 9.19 | 2.118 s |
| Henn LS + SAV | 8,928 | 21,347 | 10,821 | 23.06 | 0.84 | 3.08 | 9.5% | 8.88 | 1.370 s |
| SRL scratch | **7,721** | 21,021 | 10,665 | **21.69** | **0.90** | **3.30** | 0.3% | 9.06 | **0.012 s** |
| SRL without locations | 7,731 | 22,234 | 10,719 | 21.75 | 0.89 | 3.30 | 0% | 9.04 | 0.013 s |

The separately stored temperature-selected checkpoint produces exactly the
same operational outcomes as the final scratch checkpoint and is therefore
not shown as a duplicate row.

### Primary comparison: SRL versus no-wait C&W

SRL reduces aggregate mean flow by 773 seconds, or 9.1%. It wins on all 16
paired test instances; the median paired improvement is 809 seconds. The gain
appears in every family and becomes larger with order count:

| Group | Mean flow delta | Mean relative delta | Wins |
|---|---:|---:|---:|
| `abc1` | -755 s | -9.72% | 4/4 |
| `abc2` | -673 s | -8.61% | 4/4 |
| `ran1` | -813 s | -8.57% | 4/4 |
| `ran2` | -854 s | -8.94% | 4/4 |
| 40 orders | -478 s | -6.75% | 4/4 |
| 60 orders | -733 s | -8.78% | 4/4 |
| 80 orders | -854 s | -9.68% | 4/4 |
| 100 orders | -1,028 s | -10.63% | 4/4 |

This is a flow-time improvement, not throughput dominance. Relative to
no-wait C&W, SRL has 5.1% worse mean p95 flow, 3.8% more distance, and 1.6%
lower completion throughput. It uses 1.4% fewer tours and slightly fuller,
larger batches. The learned behavior is therefore flow-oriented service order
and packing, with a measurable tail, travel, and makespan tradeoff.

### Value of waiting

Henn waiting averages about 73 simulated seconds per episode and roughly 9.4%
of decisions. Against matched no-wait batchers it changes mean flow by:

- FIFO: +74 seconds; no-wait wins all 16 instances.
- C&W: effectively zero on average; each policy wins on some instances.
- LS: +84 seconds; no-wait wins 12 of 16 instances.

Waiting is not the source of SRL's advantage. The learned policy waits once in
348 decisions and still beats the strongest no-wait comparator on every test
instance. In this workload, established Henn waiting is neutral for C&W and
slightly harmful for FIFO and LS.

The fixed 0.75/300 diagnostic reproduces the previously reported FIFO and C&W
figures: 12,742 and 10,432 seconds mean flow respectively, with about 18% of
decisions waiting and 221 seconds mean waiting per episode. It is clearly
inferior here and receives no further tuning.

## Interpretation and next step

The flow reward paid off for its stated objective: after controlling release
timing, SRL still finds a consistently lower-mean-flow service order than all
existing batchers. The comparison does not show general superiority over
combinatorial batching. C&W optimizes a routing surrogate and remains better
on distance, tail flow, and completion throughput.

The next useful study is therefore not a larger network or another waiting
grid. Decide which operational objective matters:

1. If mean flow is primary, confirm the frozen policy on additional arrival
   streams and report the observed tail/distance cost.
2. If flow, tail latency, and travel must all improve, add those costs to the
   objective or use constrained/multi-objective selection; the current reward
   should not be expected to optimize them implicitly.
3. Retrain the imitation arm only if a balanced C&W-initialized policy is still
   operationally interesting. Its former result is invalid and is intentionally
   absent from this corrected feasibility conclusion.

Raw results were stored under
`outputs/henn_rl/corrected_comparison_seed11/result.json` (canonical parity
traces under its `cosy_parity` directory). Those historical outputs were
superseded by the final-diagnosis cleanup; the feasibility numbers above are
the documented conclusion. The canonical SRL-failure evidence now lives under
`outputs/henn_rl/final_diagnosis/` (see
`docs/structured_batching_srl_diagnosis.md`).
