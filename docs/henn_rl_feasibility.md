# Henn release-timing RL feasibility study

## Study

This feasibility study asks a one-picker controller to either wait for another
order or dispatch now. Greedy item assignment, FIFO batching, first-route
selection, and S-shape routing are fixed. The scenario directly uses CASIM's
simulation interface and imports neither Luigi, CoSy, nor the decision engine.

The 64 Henn instances are deterministically stratified into 32 training, 16
validation, and 16 test instances. PPO uses three paired seeds, 5,120 simulator
transitions, and `gamma=1.0`. The reward is therefore exactly the negative
normalized increment in total order flow time.

A fill-or-age heuristic was selected on validation data from a 3 by 4 grid.
The selected policy dispatches at 75% cart fill or 300 seconds of oldest-order
age. Counterfactual data uses this heuristic as the continuation policy: for
each state on its training trajectory, one isolated replay takes the opposite
action and then resumes the heuristic.

## Results

All policy figures below are deterministic evaluations on the same 16 held-out
test instances. Flow time is the mean per order; lower is better.

| Policy | Mean flow time (s) | Wait fraction | Distance | Tours |
|---|---:|---:|---:|---:|
| Always dispatch | 10,762.7 | 0.0% | 11,936.4 | 24.56 |
| Tuned fill-or-age heuristic | **10,693.5** | 20.2% | **11,769.6** | **24.00** |
| Supervised only | 10,871.9 | 9.2% | 11,994.6 | 24.44 |
| PPO from scratch, mean of 3 seeds | 10,762.7 | 0.0% | 11,936.4 | 24.56 |
| Supervised then PPO, mean of 3 seeds | 10,825.3 ± 34.0 | 10–17% | — | — |

The tuned heuristic improves flow time by 0.64%, distance by 1.40%, and tour
count by 2.29% relative to always dispatch. The improvements are modest but
show that waiting sometimes has value in this task.

Scratch PPO learned exactly the always-dispatch policy for every seed. The
supervised policies did learn some waiting behavior, but neither supervised
training alone nor 5,120 PPO fine-tuning steps beat the tuned heuristic. The
best fine-tuned seed reached 10,783.3 seconds, still 0.84% behind the heuristic.
Fine-tuned policies improved validation flow time by about 0.3–0.6% relative to
scratch PPO but became worse on test, so the evidence favors limited
generalization rather than a reliable improvement.

## Counterfactual feasibility

Two Windows worker processes generated 342 non-tied labels across all 32
training instances in 86.0 seconds. Reference trajectories contained 9–14
decisions. Replay is therefore practical at this scale and a general simulator
checkpoint is not justified.

The label set is weak as supervision in its current form:

- Dispatch is preferred in 88.9% of labels.
- Supervised training accuracy is 89.8–90.1%, barely above predicting the
  majority action.
- Only one heuristic trajectory is represented per deterministic instance, so
  states reached by different policies are largely absent.
- The mean absolute normalized counterfactual advantage is 0.00141, making many
  decisions low-margin.

These facts explain why supervised initialization mostly learned a softened
always-dispatch policy rather than the reference heuristic's useful waiting
rule.

## PPO collapse diagnosis

A separate seed-11 run kept SB3 PPO unchanged and recorded its native CSV
metrics plus checkpoints every 1,024 transitions for 10,240 transitions. An
invariant test confirms that, for a completed episode,
`sum(reward) == -total_flow_time / reward_normalizer`. The reward implementation
is therefore not losing or double-counting flow time.

The apparent collapse is primarily a deterministic-action calibration failure:

- After only 1,024 transitions, both classes had dispatch probability above
  0.5, so SB3's deterministic argmax dispatched in every state at every later
  checkpoint.
- At 10,240 transitions, PPO still assigned meaningfully different dispatch
  probabilities: 0.66 on heuristic-wait states and 0.78 on
  heuristic-dispatch states. The probability ranges were 0.587--0.688 and
  0.667--0.778 respectively.
- Thus the model learned a useful ranking but placed its global decision
  boundary poorly. Agreement with the reference decisions rose from 83.6% at
  the standard 0.5 cutoff to 90.2% at 0.7.
- Selecting the cutoff on validation only improved validation flow from
  12,433.9 to 12,338.0 seconds and test flow from 10,762.7 to 10,711.4 seconds.
  This recovers most, but not all, of the tuned heuristic's test result of
  10,693.5 seconds.
- Sampling the policy, as PPO does during collection, did not solve the quality
  problem: five test evaluations averaged roughly 10,931--11,160 seconds with
  17.6--24.7% waits, worse than always dispatch.

The policy did not become numerically deterministic. Entropy loss moved from
-0.690 to -0.504. The critic initially failed badly (explained variance
-0.880), then recovered only moderately and unstably (range -0.880 to 0.538,
final 0.481). Mean rollout return peaked at -0.1198 and ended at -0.1383.
Mean KL was 0.0038 and mean clip fraction 2.6%, so clipping is not preventing
large updates.

The underlying learning problem is highly imbalanced and low-margin. Only 38
of 342 counterfactual states prefer waiting. Waiting labels have larger mean
absolute advantage (0.00223 versus 0.00130), but they remain rare. PPO first
learns the dominant global dispatch bias and only then learns state-dependent
separation; 10,240 transitions are insufficient for those rare states to cross
the fixed argmax boundary. The default `gae_lambda=0.95` also means PPO's
advantages do not exactly use the full undiscounted episode return even though
`gamma=1`; with a still-imperfect critic, this adds bias across the 9--14
event-driven decisions. This is a plausible contributor, not yet a causal
finding.

## Runtime and limitations

The four-size pilot completed in 15.9 seconds. The clean full label-generation
phase took 86.0 seconds. A clean isolated supervised PPO run achieved 25.7
timesteps/s; the earlier single 40-order audit result is not representative of
training across 40–100-order instances.

An initial 20,000-step run was stopped after its projected duration exceeded
the feasibility budget. On Windows, stopping its command wrapper initially
left the Python child alive; it was identified by start time and terminated.
This contaminated early wall-time measurements but not transition counts or
policy results. Seed 11 was rerun under isolated load and reproduced its test
and validation flow times exactly. Training throughput should nevertheless be
treated as indicative rather than a controlled benchmark.

## PPO stabilization follow-up

A controlled seed-11 continuation branched both GAE arms from the identical
10,240-step checkpoint and added 20,480 transitions. GAE was changed through
SB3's load configuration so both PPO and its rollout buffer used the requested
value. Checkpoints were recorded every 2,560 added transitions.

Longer plain PPO did not fix default-policy calibration:

| Arm | Best validation threshold | Validation flow | Calibrated test flow | Default behavior |
|---|---:|---:|---:|---|
| GAE 0.95 | 0.700 | 12,338.0 | 10,711.4 | Always dispatch |
| GAE 1.00 | 0.975 | **12,311.6** | 10,715.6 | Always dispatch |

GAE 1 improved calibrated validation by 26 seconds relative to 0.95 but was
four seconds worse on test. This one-seed difference is operationally
negligible. More importantly, both policies drove dispatch probability upward:
at 30,720 steps, GAE 0.95 assigned mean dispatch probabilities 0.93/0.98 to
reference wait/dispatch states; GAE 1 assigned 0.87/0.99. Their deterministic
policies therefore remained always-dispatch at every checkpoint.

This met the preregistered poor-calibration condition, so the experiment used
the existing 342 counterfactuals for advantage-weighted pretraining. Actor
cross-entropy was weighted by absolute counterfactual advantage divided by its
mean; the critic target remained unchanged. The resulting PPO policy no longer
collapsed to one action:

- At the validation-selected 7,680-step checkpoint, default argmax waited in
  15.5% of validation decisions and reached 12,311.6 seconds flow.
- On the held-out test split, the same default policy reached 10,715.6 seconds,
  waited in 18.5% of decisions, travelled 11,804.6 distance units, and used
  24.06 tours.
- This improves always-dispatch flow by 0.44%, but remains 0.21% behind the
  tuned heuristic. It reduces distance by 1.10% and tours by 2.04% relative to
  always dispatch.
- Probability separation is now strong enough for the standard 0.5 boundary:
  at 7,680 steps, mean dispatch probability is 0.28 on reference-wait states
  and 0.95 on reference-dispatch states.

Thus PPO can learn this task, but plain on-policy experience did not overcome
the rare-wait imbalance within 30,720 transitions. Counterfactual
advantage-weighting supplies the missing calibration. Continuing PPO beyond
the selected checkpoint is harmful: later checkpoints drift back toward
dispatch, so validation checkpoint selection remains necessary.

The run used one exploratory seed. SB3 checkpoints do not preserve a bit-exact
simulator/RNG continuation, and other machine work affected throughput. Wall
times and 28.8--33.7 steps/s are therefore not controlled benchmark results.
The initial exhaustive 21-threshold calibration also duplicated thousands of
simulator rollouts and took disproportionately long; a future study should use
a coarse-to-fine validation grid or cache deterministic checkpoint rollouts.

This is a feasibility result, not evidence that PPO or supervised learning is
inferior in general. The budget is small, the simulator and instances are
deterministic, and counterfactual labels estimate one-step improvement relative
to a heuristic continuation rather than globally optimal actions.

## Recommended next study

Keep standard SB3 PPO as the exploratory baseline and address the diagnosed
issue in this order:

1. Retain GAE 1 for the next exploratory run, while recognizing that this
   one-seed comparison found no meaningful generalization advantage over 0.95.
2. Use advantage-weighted pretraining and select checkpoints by default-policy
   validation flow. Do not train blindly to the final budget.
3. Expand counterfactual coverage with a small number of on-policy states if
   the validation-selected policy still fails to beat the existing heuristic.
4. Do not present a tuned cutoff as the learned PPO policy. It is a useful
   diagnostic and a deployable validation-calibrated rule, but it is an extra
   policy-selection parameter.
5. Use structured RL/COAML only when the action becomes an order subset or
   partition. It adds no structure to this binary release action.

## Reproduction

From PowerShell at the repository root:

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
uv sync --extra dev --extra rl
uv run python -m scenarios.scenario_henn_rl.experiment_henn_rl
```

Raw results and models are written below `outputs/henn_rl`. The feasibility run
reported here is in `outputs/henn_rl/feasibility_20260807`; the clean timing
replication is in `outputs/henn_rl/label_timing_clean`; the instrumented PPO
trace is in `outputs/henn_rl/diagnosis_seed11`.
The continuation, GAE, calibration, and advantage-weighted records are in
`outputs/henn_rl/stabilization_seed11`.
