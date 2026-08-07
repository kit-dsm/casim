# CASIM runtime audit

This audit covers the online modes that exercise materially different CASIM
decision paths:

| Family | Mode | Meaning |
|---|---|---|
| Lorenz paper example | `no_wait` | sequential fixed-heuristic dispatch |
| Lorenz paper example | `reopt` | depot-level exact reoptimization |
| Dynamic operations | `periodic_rolling` | rolling commitment of new work |
| Dynamic operations | `unstarted_reopt` | rolling replanning of queued work |
| Dynamic operations | `routing_intervention` | queued-work and active-route replanning |

The two families use different workloads and solvers. Their absolute runtimes
must not be interpreted as a policy-quality or like-for-like speed comparison.

## Reproduce on Windows

Run from the repository root in PowerShell. Keeping the uv cache inside the
workspace avoids Windows sandbox and cross-user cache ownership problems:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache-codex"
uv sync --extra dev --extra rl
uv run python -m scenarios.scenario_runtime_audit.cli benchmark
uv run python -m scenarios.scenario_runtime_audit.cli profile
```

The combined command is:

```powershell
uv run python -m scenarios.scenario_runtime_audit.cli all
```

If a managed Windows environment denies execution of `.venv\Scripts\python.exe`
or copying `venvlauncher.exe`, run the same command with approved native
execution. Do not silently create a second environment: the result records the
exact Python executable and `uv.lock` hash.

Generated JSON and `.prof` files are written below `outputs/runtime_audit`,
which is ignored by Git.

## Measurement protocol

- Each measured workload runs in a fresh Python process.
- Mode order alternates on every repetition to reduce systematic bias from
  changing machine load.
- The default protocol uses one warm-up, seven small repetitions, and three
  three-day dynamic repetitions. Each run has a 15-minute timeout that is
  retained as a result.
- Headline runs disable Dash, KPI logging, and progress rendering. One paired
  small run per mode retains normal progress/KPI behavior to quantify its
  overhead.
- Wall time and process CPU time are both retained. A coefficient of variation
  above 10% is flagged; no slow repetitions are discarded.
- Phase timers cover configuration, simulation setup, pipeline setup, reset,
  state projection, conditions, event-loop work, solver work, event conversion,
  commitment, logging, and output serialization.
- `cProfile` runs are separate from baselines. Their slowdown is reported and
  their durations are never used as headline performance numbers.

Use smaller counts while validating a machine setup:

```powershell
uv run python -m scenarios.scenario_runtime_audit.cli benchmark `
  --warmups 0 --smoke-repetitions 1 --skip-scaling --skip-observed
uv run python -m scenarios.scenario_runtime_audit.cli profile --mode no_wait
```

## Direct-action RL indicator

The RL environment uses Henn instance `H_abc1_40_29`. The action is directly
operational: wait for another natural trigger or dispatch now. FCFS batching,
S-shape routing, and selection of the first returned route are fixed. The agent
cannot select an algorithm or pipeline. A wait after input closure is converted
to a forced dispatch, which prevents deadlock for an always-wait policy.

The observation contains normalized time, visible backlog, oldest-order age,
time to the next arrival, cart fill potential, and accrued unfinished work. The
dense reward is the negative increment in order flow time, normalized by order
count and episode horizon. The historical audit run used `gamma=0.99`; the
maintained feasibility study corrects this to `gamma=1.0` so training remains
aligned with undiscounted total flow time.

The default study reports random-policy environment steps/s and a CPU-only,
single-process PPO run for 10,240 timesteps. `DummyVecEnv` is deliberate:
Windows process spawning and import cost would otherwise contaminate this
small simulator indicator. Pre/post evaluations are a trainability sanity
check, not an optimality claim.

The maintained environment and subsequent learning experiments now live in
`scenarios/scenario_henn_rl`; the completed audit CLI no longer runs them.

## Findings

The default audit was run on 2026-08-07 using Windows 11 build 26200,
Python 3.13.11, a 16-logical-core AMD64 processor, and the Windows balanced
power plan. All 44 measured mode repetitions completed. End-to-end coefficient
of variation was 1--5%, below the 10% warning threshold for every group. These
numbers are machine-specific; unrelated work on the machine can still affect
them.

### Online modes

The table reports medians. End-to-end time includes fresh Python process and
import startup; in-process time starts immediately before Hydra composition.

| Workload | Mode | End-to-end | In-process | Decisions | Events/s | Solver | Snapshot |
|---|---|---:|---:|---:|---:|---:|---:|
| paper example | `no_wait` | 2.93 s | 0.16 s | 2 | 306 | 0.03 s | <0.01 s |
| paper example | `reopt` | 2.82 s | 0.08 s | 2 | 477 | <0.01 s | <0.01 s |
| smoke | `periodic_rolling` | 3.02 s | 0.26 s | 9 | 4,466 | 0.07 s | 0.01 s |
| smoke | `unstarted_reopt` | 3.03 s | 0.27 s | 9 | 4,358 | 0.07 s | 0.01 s |
| smoke | `routing_intervention` | 3.05 s | 0.33 s | 12 | 3,579 | 0.09 s | 0.01 s |
| three-day | `periodic_rolling` | 3.95 s | 0.96 s | 35 | 25,422 | 0.34 s | 0.10 s |
| three-day | `unstarted_reopt` | 3.96 s | 0.94 s | 35 | 26,104 | 0.33 s | 0.10 s |
| three-day | `routing_intervention` | 5.33 s | 2.37 s | 165 | 10,850 | 1.21 s | 0.48 s |

The main findings are:

1. **Fresh-process startup dominates small runs.** Python imports, Hydra
   composition, and pipeline construction account for roughly 2.7--2.9 s
   outside the measured workload. A CLI-level benchmark that reports only
   end-to-end time hides most CASIM differences on small instances.
2. **Active-route intervention is the scaling bottleneck in this matrix.** On
   the three-day workload it is about 2.5 times slower in-process than rolling
   or unstarted replanning. It makes 165 decisions rather than 35; solver time
   is 1.21 s and snapshot time is 0.48 s, versus about 0.33 s and 0.10 s. Event
   throughput falls from roughly 25--26k/s to 10.9k/s. The cost is therefore
   mostly decision frequency plus the more expensive active-tour projection,
   not the base event heap.
3. **Unstarted replanning is not measurably slower here.** It remains within
   noise of periodic rolling on both workloads. The checked-in disruption does
   not create a runtime bottleneck comparable to active-route intervention.
4. **Existing algorithm runtime is not end-to-end decision runtime.** Several
   solutions report near-zero `solution.execution_time` while the measured
   solver/decision phase is nonzero. Performance monitoring should use complete
   decision elapsed time when estimating operational latency.
5. **Normal progress/KPI behavior costs 13--20% in-process on smoke runs** in
   the single paired observation. The end-to-end effect is only 9--16% because
   import/startup time dilutes it. Treat the paired figures as indicative, not
   as statistically stable seven-run estimates.

### Profiling overhead and attribution

Workload-only `cProfile` runs were 2.34--2.95 times slower than paired
unprofiled in-process baselines. This is why no profiled duration appears in
the tables above.

On smoke workloads, cumulative profiles are headed by configuration and
pipeline setup. Hydra/YAML scanning and object instantiation are prominent;
`deepcopy` is the largest named Python self-time hotspot for the dynamic modes,
with roughly 37--39k calls. These profiles support investigating cached or
amortized setup and state-copy volume, but they do not justify a change by
themselves. At three-day scale, the low-overhead phase timers give stronger
evidence that active-route solver and snapshot frequency are the first runtime
targets.

### Direct-action RL indicator

The full study completed 2,000 seeded random-policy transitions and 10,240 PPO
timesteps:

| Measure | Result |
|---|---:|
| Environment construction | 0.266 s |
| Initial reset | 0.050 s |
| Raw environment throughput, excluding reset | 29.17 steps/s |
| Completed random-policy episodes | 1.44 episodes/s |
| PPO training time | 502.35 s |
| PPO throughput | 20.38 timesteps/s |
| Mean flow time before training | 424,702 s |
| Mean flow time after training | 351,948 s |
| Wait fraction before / after | 49.4% / 7.7% |

The learner therefore costs about 30% throughput relative to raw stepping on
this machine. The post-training evaluation is directionally sensible: the
policy waits less and reduces mean flow time by about 17%. It remains only a
trainability sanity check on one deterministic instance, not evidence of a
general or optimal warehouse policy.

### Recommended follow-up order

1. Reuse long-lived processes and already-built pipelines for repeated small
   experiments; do not optimize the event loop based on CLI startup time.
2. For active-route operation, reduce or suppress low-value intervention
   triggers before micro-optimizing algorithms. Decision count is the dominant
   difference in the measured modes.
3. Measure whether active-tour snapshots can avoid broad deep copies while
   preserving their detached-state contract.
4. Use complete decision elapsed time in production-facing monitoring, with
   component algorithm time retained only as a diagnostic submetric.
5. Repeat on an otherwise idle target machine and with representative customer
   streams before setting a performance budget or regression threshold.
