# Structured batching objective study

This is an exploratory confirmation on the deterministic Henn test split, which appeared in the earlier feasibility study; it is not a pristine first-use test.

All policies used fixed no-wait actions, the existing additive-score actor and exact knapsack decoder, and S-shape routing. Selection used only the complete validation split.

## Frozen test results

| trained power | selected episode | mean flow (s) | p90 (s) | p95 (s) | max (s) | GM 1.5 (s) | GM 2 (s) | distance | throughput | waits |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8082.2 | 19417.7 | 21676.4 | 29788.3 | 9538.0 | 10813.5 | 10704.4 | 9.040 | 0 |
| 1.5 | 8 | 8330.4 | 19444.4 | 21803.2 | 28248.3 | 9763.4 | 11005.5 | 10685.6 | 9.054 | 0 |
| 2 | 8 | 8363.1 | 19342.5 | 21802.6 | 29364.7 | 9775.6 | 11004.1 | 10719.4 | 9.035 | 0 |

| trained power | median (s) | mean(F^1.5) | mean(F^2) | tours | batch fill | orders / batch | decision latency (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4450.6 | 9.315e+05 | 1.169e+08 | 21.75 | 0.894 | 3.295 | 0.845 |
| 1.5 | 4877.2 | 9.647e+05 | 1.211e+08 | 21.69 | 0.897 | 3.306 | 0.856 |
| 2 | 5101.3 | 9.665e+05 | 1.211e+08 | 21.75 | 0.894 | 3.295 | 0.840 |

| trained power | objective delta vs matching baseline | wins / 16 | objective delta vs p=1 SRL | wins / 16 |
|---:|---:|---:|---:|---:|
| 1 | -7.48% | 16 | +0.00% | 0 |
| 1.5 | -3.37% | 14 | +3.17% | 2 |
| 2 | -0.86% | 8 | +3.27% | 2 |

## Interpretation

The longer p=1 run selected episode 8; its test p95 was 21676.4 s. On validation, checkpoint 8 had p95 41657.6 s and checkpoint 128 had p95 41517.8 s. This 0.3% validation-tail improvement came with a 1.7% worse validation mean-flow objective, so validation still selected episode 8. The selected policy's 21676.4 s test p95 was 3.1% above the earlier eight-episode scratch result (21020.9 s). Longer available training therefore did not resolve the prior tail weakness.

The p=1.5 policy changed its held-out moment objective by +3.17% relative to the p=1 policy.
The p=2 policy changed its held-out moment objective by +3.27% relative to the p=1 policy.
Neither convex arm pays off under the stated criterion: both worsen their own held-out moment relative to p=1 and neither improves p95. The p=1.5 arm lowers maximum flow by 5.2%, but that isolated maximum improvement is not accompanied by better p90, p95, or its aligned moment objective.

Powers above one are smooth tail-sensitive moments, not exact p95 or CVaR objectives. Distance and throughput are reported as external operational consequences, not reward-aligned objectives.

## Method and artifacts

Seed: 11. Training order: four independently shuffled passes over all 32 training instances, reused identically for every power. Checkpoints: 0, 8, 16, 32, 64, and 128. Baselines: FIFO, C&W-like, and LS crossed with first, short, long, and SAV under no-wait; LS used 0.25 s and validation used three workers.

The machine-readable result, per-episode records, paired deltas, timings, diagnostics, and saved checkpoints are in the study output directory.
