# CASIM configuration compiler

> **Current specification:** `docs/decision_configuration_model.md` is the
> authoritative formal specification for the decision-configuration model.
> This document retains the historical context (previous path, semantic
> holes, equivalence evidence) but the card/mapper descriptions in sections
> 4-6 have been superseded by the simpler formal model.

This document describes the configuration-compilation layer that makes CASIM
decision semantics a first-class part of the system. It records the previous
configuration path, the concrete semantic holes it had, the declarative
`DecisionCard` representation that replaces the procedural mapping, the
`DecisionCompatibilityMapper` that validates configuration against it, the
single `compile_engine()` compilation boundary, and the equivalence evidence
that the runtime behavior is preserved.

The layer is deliberately small. It does not redesign the simulator, introduce
a second configuration language, or add a generic compiler framework. It
makes the semantics that were already implicit in `setup.py`,
`SimulationEngine._conditions_hold()`, `StateAdapter`, `SolutionRanker`, and
`DecisionEngine.commit()` explicit and compile-time safe.

## 1. The previous configuration path

Before this change, `casim.setup.build_runtime(cfg, data_card)` interpreted
raw Hydra configuration directly. The construction path was:

1. `_build_engine_state(cfg)` iterated `cfg.engines.problems`. For each entry:
   - `_build_adapter(problem_cfg)` read `problem_class` and `replanning`,
     rejected unknown replanning values, and called `_derive_exposure(...)`.
   - `_derive_exposure(problem_class, replanning, *, due_horizon_s, limit,
     congestion_penalty)` was a procedural `if/elif` chain that mapped the
     `(problem_class, replanning)` pair to the `StateAdapter` exposure
     vocabulary (`orders`/`batches`/`resources`/`active_tour` source dicts).
   - Triggers were resolved with `_resolve_trigger`, duplicate bindings and
     duplicate triggers were rejected.
   - `_requirements(problem_cfg)` passed the `requires:` block through verbatim
     into `conditions_map`.
   - A `SchedulingCommitmentPolicy` was built whenever a `commit:` block
     existed, with no check against the binding's solution type.
   - `_build_solver` constructed a CoSy or direct solver.
2. `build_decision_engine` copied the data card, added the adapter's
   `projected_features()`, and called `solver.prepare(effective_card)` for
   each binding. Applicability filtering and pipeline synthesis happened
   inside `CoSySolver.prepare` but were not surfaced.
3. `_construct_simulation` built the `SimulationEngine`.

`SUPPORTED_DECISIONS` was a set literal of the seven supported
`(problem_class, replanning)` pairs. The exposure vocabulary was a function
of that pair, but the function's body was the only place that knew which
tuning parameters it consumed, which conditions were meaningful, which
triggers were compatible, which solution type it produced, and which ranking
objectives the ranker honored.

## 2. Where the semantics were distributed

The decision semantics were spread across six unrelated places, each
enforcing only its own concern with no shared definition of the binding:

| Semantic fact | Where it lived | How it failed |
|---|---|---|
| Supported `(problem_class, replanning)` | `setup.SUPPORTED_DECISIONS` set literal | `ValueError` at adapter build |
| Exposure vocabulary + tuning param consumption | `setup._derive_exposure` `if/elif` chain | silently dropped unused params |
| Condition keys | `SimulationEngine._conditions_hold` | unknown keys silently ignored |
| Trigger field requirements | `StateAdapter._project_active_tour` | `AttributeError` at first trigger fire |
| Commitment compatibility | `DecisionEngine.commit` → `SchedulingCommitmentPolicy.apply` | `AttributeError` at first commit |
| Ranking objective semantics | `SolutionRanker.select_best` branches | silently ignored for OBP/ORP; `ValueError` at runtime for OBRSP/ORSP typos |

A reader had to assemble the binding's meaning from all six locations. No
single object could answer "what does an `ORSP / unstarted` decision actually
project, consume, and support?"

## 3. Concrete semantic holes found

Each hole below was verified from the implementation before it was closed.
The compiler now rejects all of them at `compile_engine()` time.

### 3.1 Unknown `requires` keys were silently ignored

`SimulationEngine._conditions_hold()` (`simulation_engine.py:129-168`)
recognized exactly `orders`, `batches`, `pickers`, `not_on_break`,
`dock_capacity`. Any other key was present in the `requires` dict but never
read; the condition passed as if the key were absent. `setup._requirements`
passed the dict through with no schema. A `requires: {tours: 5}` block made a
decision look guarded while being unguarded.

### 3.2 Conditions referred to unavailable projected state

`_conditions_hold` reads `snapshot.orders.orders`, `dynamic.buffered_batches`,
`snapshot.resources.resources`, `dynamic.is_break`, and
`dynamic.n_staged_pallets`. Which of those are populated (and variable) is a
property of the resolved `StateAdapter`:

- `OBP/none`, `OBRSP/none`, `OBRP/none`, `OBRP/active` project a *variable*
  order list; `ORSP/none`, `ORSP/unstarted`, `ORP/active` project an empty
  order list. `requires: {orders: 1}` on an `ORSP/none` decision therefore
  never fires.
- `ORSP/none`, `ORSP/unstarted` project a variable batch list; the order-based
  bindings project an empty batch list; `ORP/active` and `OBRP/active` project
  exactly one residual batch. `requires: {batches: 1}` on `OBP/none` never
  fires; on `ORP/active` it is a tautology (always exactly one).
- Active-tour bindings always project exactly one picker (the active one), so
  `requires: {pickers: 1}` is a tautology there.

These were silent: a never-fires condition made the decision permanently
dormant; a tautology condition looked like a guard but was a no-op.

### 3.3 Irrelevant tuning parameters were silently accepted

`_build_adapter` unconditionally read `due_horizon_s`, `limit`, and
`congestion_penalty` and forwarded them to `_derive_exposure`, which only
threaded a parameter into the exposure dict when the binding consumed it.
Dropped parameters had no effect and produced no warning. For example,
`congestion_penalty: 1000` on an `ORSP/none` decision (no active-tour
projection) and `due_horizon_s: 3600` on an `ORP/active` decision (no batch
projection) were both silently ignored.

### 3.4 Incompatible triggers survived setup and failed at runtime

`StateAdapter._project_active_tour` (`state_adapter.py:356-389`) reads
`trigger.picker_id`, `trigger.tour_id`, `trigger.route_version`, and
`trigger.resumes_execution`. Only `InterventionRequest` supplies all four in
the current event model. Binding `ORP/active` to `PlanningRun` (or `PickerIdle`)
passed setup and crashed with `AttributeError: 'PlanningRun' object has no
attribute 'picker_id'` the first time the trigger fired during simulation.
Non-active bindings were safe: their `trigger_if_present` scope reads
`picker_id` via `getattr(..., None)` with a default.

### 3.5 `commit:` on a non-scheduling decision failed at runtime

`SchedulingCommitmentPolicy.apply` (`decision_engine.py:32-65`) dereferences
`solution.jobs` and is monomorphic. `DecisionEngine.commit` calls
`policy.apply` whenever a policy exists for the binding. Attaching `commit:` to
`OBP/none` (`BatchingSolution`), `OBRP/none`, `ORP/active`, or `OBRP/active`
(`CombinedRoutingSolution`) therefore crashed with
`AttributeError: ... has no attribute 'jobs'` at the first commit, after setup
had already succeeded.

### 3.6 Solver ranking objectives were silently ignored

`SolutionRanker.select_best` (`solution_ranker.py:9-58`) branches by problem
class:

- `OBP`, `OSBP`: take any solution; `self.objective` is never read. Any
  configured objective (including `tardiness`, which the checked-in
  `scenario_dynamic_operations` and `scenario_ijpe` OBP entries used) was
  silently ignored.
- `ORP`, `OBRP`, `BSRP`: always rank by total route distance; `self.objective`
  is never read. `objective: makespan` on an `ORP/active` decision was silently
  ignored.
- `OBRSP`, `ORSP`: honor `distance`/`makespan`/`tardiness`; an unknown
  objective raised `ValueError` only at runtime, only when solutions were
  non-empty.

So a configured `solver.objective` was dead configuration for OBP and ORP/OBRP,
and a typo on OBRSP/ORSP failed late.

## 4. The `DecisionCard` representation

`casim/decision_card.py` introduces a small frozen dataclass that describes
only the configuration semantics that are currently real for one
`(problem_class, replanning)` binding:

```python
@dataclass(frozen=True)
class DecisionCard:
    problem_class: str
    replanning: str
    orders: bool                      # exposes buffered orders
    batches: str | None              # "buffered" | "buffered_and_replannable"
    resources: str | None             # "all"|"nonactive"|"available"|"dispatchable"
    resource_trigger_scoped: bool     # scope == "trigger_if_present"
    active_tour: str | None           # "residual"
    consumes_due_horizon_s: bool
    consumes_limit: bool
    consumes_congestion_penalty: bool
```

The seven supported bindings are registered declaratively in `DECISION_CARDS`.
The card does **not** duplicate the mathematical problem structure stored in
`casim.pipelines.taxonomy.TAXONOMY`. Instead, the compatibility facts are
derived from the card plus the taxonomy:

| Derived fact | Derivation |
|---|---|
| `exposes_orders` | `self.orders` |
| `exposes_active_tour` | `self.active_tour is not None` |
| `requires_intervention_trigger` | `exposes_active_tour` |
| `available_conditions` | `not_on_break`, `dock_capacity` always; `orders` if `self.orders`; `batches` if `self.batches is not None and not exposes_active_tour`; `pickers` if `self.resources is not None and not exposes_active_tour` |
| `solution_type` | `"scheduling" in TAXONOMY[pc].variables` → `SchedulingSolution`; elif `"routing"` → `CombinedRoutingSolution`; else `BatchingSolution` |
| `supports_commit` | `solution_type == "SchedulingSolution"` |
| `ranking_objectives` | `SchedulingSolution` → `{distance, makespan, tardiness}`; `CombinedRoutingSolution` → `{distance}`; `BatchingSolution` → `frozenset()` |

`DecisionCard.exposure_dict(*, due_horizon_s, limit, congestion_penalty)`
reproduces the exact `StateAdapter` exposure vocabulary the old
`_derive_exposure` produced, so the runtime projection is unchanged. The card
is the declarative twin of that procedural mapping.

The whole registry is small enough to understand on paper:

| Binding | orders | batches | resources | scope | active_tour | due_horizon_s | limit | congestion_penalty |
|---|---|---|---|---|---|---|---|---|
| `OBP/none` | yes | – | all | – | – | – | yes | – |
| `ORSP/none` | – | buffered | nonactive | – | – | yes | yes | – |
| `ORSP/unstarted` | – | buffered+replannable | available | – | – | yes | yes | – |
| `OBRSP/none` | yes | – | dispatchable | trigger | – | – | yes | – |
| `OBRP/none` | yes | – | dispatchable | trigger | – | – | yes | – |
| `ORP/active` | – | – | – | – | residual | – | – | yes |
| `OBRP/active` | yes | – | – | – | residual | – | yes | yes |

## 5. `DecisionCompatibilityMapper` behavior

`DecisionCompatibilityMapper` (in `decision_card.py`) validates a configured
decision against its `DecisionCard` and the existing taxonomy. It mirrors the
`AlgorithmCard` + `DataCard` → `DomainAlgorithmMapper` pattern without merging
the two mappers: `DomainAlgorithmMapper` selects algorithms from a warehouse
data card; `DecisionCompatibilityMapper` selects a valid projection from a
user decision binding.

`mapper.validate(problem_class, replanning, problem_cfg, *, trigger_classes)`
returns the resolved `DecisionCard` or raises `ValueError`. It performs, in
order:

1. **Binding support** — `resolve()` raises `Unsupported decision binding`
   for a `(problem_class, replanning)` pair not in `DECISION_CARDS`.
2. **Tuning parameters** — for each of `due_horizon_s`, `limit`,
   `congestion_penalty`: if the value is present and the card does not consume
   it, raise `does not consume tuning parameter ...`.
3. **Conditions** — for each key in `requires`: if it is not in
   `KNOWN_CONDITION_KEYS` (`orders`, `batches`, `pickers`, `not_on_break`,
   `dock_capacity`), raise `unknown requires key`; if it is not in
   `card.available_conditions`, raise `cannot require ... the resolved
   projection does not expose a variable quantity of that type`.
4. **Triggers** — if `card.requires_intervention_trigger`, every trigger class
   must be `InterventionRequest` (or a subclass); otherwise raise `requires a
   trigger supplying picker_id/tour_id/route_version/resumes_execution`.
   Non-active bindings accept any `Event` subclass.
5. **Commitment** — if a `commit:` block is present and
   `card.supports_commit` is false, raise `cannot be committed through a
   SchedulingCommitmentPolicy`.
6. **Ranking objective** — for `cosy` solvers: if `ranking_objectives` is empty
   (OBP), an explicitly configured `objective` is rejected as `silently
   ignored`; if non-empty, the resolved objective (defaulting to `distance`)
   must be in the set, else raise `does not support ranking objective`.
   Direct solvers skip this check (they validate themselves in `prepare()`).

No validation rule was added without a corresponding semantic reason in the
current implementation (see §3).

## 6. Integration with `StateAdapter` and `DomainAlgorithmMapper`

The compiler does not replace `StateAdapter` or `DomainAlgorithmMapper`; it
sits in front of them and feeds them a validated configuration.

```
DecisionCard + user configuration
    → DecisionCompatibilityMapper.validate
    → DecisionCard.exposure_dict(...)
    → StateAdapter (the runtime projection, unchanged)
    → StateAdapter.projected_features()
    → effective DataCard (copy + projected features + problem_class)
    → DomainAlgorithmMapper.filter (unchanged)
    → applicable AlgorithmCards
    → CoSySolver.prepare (now records inspectability)
    → Maestro pipelines
```

`StateAdapter.transform_state()` remains responsible for constructing the
planning snapshot. `StateAdapter.projected_features()` remains the bridge
into the existing algorithm-applicability system. The two mappers solve
different problems and are not merged.

`CoSySolver.prepare()` was made inspectable: it now caches an
`ApplicabilityReport` (alongside the existing `pipelines`). The report records
`considered_components`, `applicable_algorithms`, `excluded_components`,
`retained_components`, and the synthesized pipelines. It surfaces the
single existing applicability evaluation; it does not re-evaluate it. The
report is accessed via `CoSySolver.applicability_report()` and stored on the
compiled decision so a user can determine which configured components were
considered, which algorithms were applicable, which components were excluded
as inapplicable, which were retained, and how many pipelines were generated.
Direct solvers produce no report (they validate themselves in `prepare()`).

## 7. The `compile_engine()` flow

`compile_engine(cfg, data_card=None)` is the single explicit compilation
boundary. It:

1. loads the data card if one was not supplied;
2. calls `_compile_bindings(cfg)`, which for each `engines.problems` entry:
   - validates `replanning` and rejects duplicate bindings;
   - resolves trigger classes;
   - calls `DecisionCompatibilityMapper.validate` (§5);
   - builds the `StateAdapter` from `card.exposure_dict(...)`;
   - rejects duplicate trigger assignments;
   - builds the compiled conditions (`tuple[CompiledCondition, ...]`), the
     (unprepared) solver, and the commitment policy;
3. for each resolved decision, builds the decision-specific effective
   `DataCard` (copy + `projected_features()` + `problem_class`) and calls
   `solver.prepare(effective_card)`, then replaces the decision with one
   carrying the `ApplicabilityReport` from `CoSySolver.applicability_report()`;
4. returns a `CompiledEngine` holding the resolved `CompiledDecision`s, the
   `triggers_map`, and the data card.

`build_runtime(cfg, data_card)` now calls `compile_engine` and instantiates
the `SimulationEngine` and `DecisionEngine` from the compiled result instead
of interpreting raw Hydra configuration again. `build_simulation(cfg)` (the
test helper that does not prepare solvers) calls `_compile_bindings` and
constructs the `SimulationEngine` from the compiled adapters, triggers, and
conditions.

The compiled representation is minimal: it contains the concrete objects CASIM
already needs (adapters, triggers, conditions, prepared solvers, commitment
policies, applicability reports) rather than a large new IR. `CompiledDecision`
and `CompiledEngine` are frozen dataclasses so a compiled configuration cannot
be silently mutated after compilation.

## 8. Henn and intervention handling

### Henn (`OBRP / none` candidate generation)

`scenario_henn` uses `OBRP/none` as a *candidate-generation* path, not a normal
`on_trigger` decision. It calls `DecisionEngine.solve()` directly to obtain a
`CombinedRoutingSolution`, consumes the routing result, later constructs a
`SchedulingSolution` itself (via `FIFOScheduling`), and enters through
`DecisionEngine.commit()` with that scenario-built solution. It does **not**
attach a `commit:` block to the `OBRP/none` entry.

The compiler preserves this exactly. The `OBRP/none` card has
`supports_commit == False` (`CombinedRoutingSolution`), so a `commit:` block
on `OBRP/none` is now rejected — but the absence of `commit:` is fine, and
`DecisionEngine.commit()` remains independently callable with a
scenario-constructed `SchedulingSolution` (the policy is only applied when one
exists for the binding). No compiler metadata is needed: the
`solve()`-then-`commit()` split is a scenario-level use of the existing
`DecisionEngine` API, not a new binding semantics. The scenario also registers
a scenario-local `HennWakeUp` trigger at reset; that trigger is bound to the
non-active `OBRP/none` decision, so the compiler's trigger check (which only
governs config-declared triggers) is not involved.

### Intervention and active insertion

`ORP/active` (routing-only intervention) and `OBRP/active` (active batch
insertion) both require an `InterventionRequest` trigger. The compiler now
enforces that at compile time (§3.4). The active-insertion detection
(`intervention_enabled`, `active_batch_insertion_enabled`) is derived from
the compiled `triggers_map` exactly as before:
`active_batch_insertion_enabled` is true iff the `InterventionRequest` binding
is `("OBRP", "active")`.

## 9. Equivalence and regression-test evidence

For every supported decision binding, the new path produces objects identical
to the old path:

- `DecisionCard.exposure_dict` reproduces the old procedural exposure mapping
  exactly for all seven bindings and for both `None` and concrete tuning
  parameter values (`tests/test_decision_space.py`).
- The `StateAdapter` is constructed from that exposure, so its configuration
  and `projected_features()` are unchanged.
- Trigger mapping, conditions, commitment, the effective `DataCard`, and the
  `DomainAlgorithmMapper` call are unchanged.
- `CoSySolver.prepare` records the same applicability decision; the filtering
  loop and pipeline synthesis are byte-for-byte the same, only augmented with
  inspectability attributes.

The full deterministic scenario suite passes unchanged except for one
configuration correction that is proven semantically invalid:

- `scenario_dynamic_operations` and `scenario_ijpe` OBP entries previously set
  `solver.objective: tardiness` (or `${data_card.objective}`). The
  `SolutionRanker` ignores the objective for `BatchingSolution` (it takes any
  solution), so this was dead configuration (§3.6). The `objective:` line was
  removed from those OBP entries. Ranking output is unchanged — the ranker
  ignored the value before and ignores the default `distance` now. The study
  objective continues to live on the data card's `objective` field, which
  feeds `state.active_objective` and the decision tracker.

The pre-existing `tests/test_intervention.py::test_exact_tsp_residual_distance_matches_exhaustive_enumeration`
failure is a Gurobi numerical-environment issue present before this change
and is unrelated to configuration.

### Test evidence

| Suite | Result |
|---|---|
| `tests/test_decision_space.py` (exposure/adapter derivation) | pass |
| `tests/test_pre_run_applicability.py` (applicability filtering + setup failure) | pass |
| `tests/test_dynamic_operations.py` (4 deterministic policies) | pass |
| `tests/test_henn_offline.py`, `test_henn_wakeup_integration.py` | pass |
| `tests/test_reopt.py` (Lorenz direct solver + CoSy no_wait) | pass |
| `tests/test_intervention_stress.py` (intervention + active insertion) | pass |
| `tests/test_cart_inventory.py` (`SchedulingCommitmentPolicy`, `StateAdapter`) | pass |
| `tests/test_simulation_engine.py` (`_conditions_hold`, step/drain) | pass |
| `tests/test_decision_engine_progress.py` (`on_trigger`/`solve`/`commit`) | pass |
| `tests/test_configuration_compiler.py` (new negative + positive tests) | pass |

The new `tests/test_configuration_compiler.py` adds focused negative tests for
every demonstrated semantic failure:

- unsupported `(problem_class, replanning)`;
- unknown `requires` key;
- condition referring to unavailable projected state (orders/batches/pickers,
  including the active-tour tautology);
- irrelevant tuning parameter (`due_horizon_s`, `limit`, `congestion_penalty`);
- incompatible trigger (`PlanningRun`, `PickerIdle` on an active binding);
- commitment on an incompatible decision type (`OBP`, `OBRP`, `ORP/active`);
- unsupported solver ranking objective (OBP ignored objective; ORP non-distance;
  OBRSP typo now rejected at compile time);
- duplicate binding;
- duplicate trigger assignment;
- no applicable algorithm/pipeline (via `compile_engine`).

It also verifies the declarative registry matches `SUPPORTED_DECISIONS`, the
card's derived semantics match the implementation, the compiled applicability
report is populated, and `CompiledEngine.explain()` renders the resolved
semantics.

## 10. Examples of compiled real configurations

### `ORSP / unstarted` (scenario_dynamic_operations, routing_intervention)

Generated by `CompiledEngine.explain(("ORSP", "unstarted"))`:

```text
problem:
  ORSP / unstarted

state exposure:
  buffered + replannable batches (due_horizon_s=21600) (limit=96)
  available resources (resource ready times)

triggers:
  PlanningRun

conditions:
  batches >= 1
  pickers >= 1
  not_on_break

problem variables:
  routing
  scheduling

solution type:
  SchedulingSolution

projected applicability features:
  buffered_batches
  replannable_unstarted_work
  available_resources
  resource_ready_times

commitment:
  max_jobs_per_picker=1, planning_horizon_s=14400

applicability:
  considered components: InstanceLoader, OrdersProvider, GreedyIA, DueDate, PickListProvider, NearestNeighbourhood, EDDScheduler, ResultAggregationBatching, ResultAggregationRouting, ResultAggregationScheduling
  applicable algorithms: EDDScheduler, LargestGap, Midpoint, NearestNeighbourhood, Return, SShape, ExactSolving
  excluded components: GreedyIA, DueDate
  retained components: InstanceLoader, OrdersProvider, PickListProvider, NearestNeighbourhood, EDDScheduler, ResultAggregationBatching, ResultAggregationRouting, ResultAggregationScheduling
  pipelines: 1
```

The report is generated from the resolved card, adapter, taxonomy, triggers,
conditions, commitment policy, and applicability report — not maintained as a
separate description.

### Rejected configurations (now compile-time errors)

```yaml
# 3.2 — condition on a quantity the projection never exposes
- problem_class: OBP
  replanning: none
  triggers: [WMSRun]
  requires: {batches: 1}        # error: cannot require 'batches'
  solver: {type: cosy, repo: ${cosy_repo}}

# 3.4 — incompatible trigger on an active-tour binding
- problem_class: ORP
  replanning: active
  triggers: [PlanningRun]       # error: requires a trigger supplying ...
  congestion_penalty: 1000
  solver: {type: cosy, objective: distance, repo: ${cosy_repo}}

# 3.5 — commitment on a routing-only decision
- problem_class: OBRP
  replanning: none
  triggers: [OrderArrival, PickerIdle, FlushRemainingOrders]
  requires: {orders: 1, pickers: 1}
  commit: {n_jobs: 1}           # error: cannot be committed through a SchedulingCommitmentPolicy
  solver: {type: cosy, repo: ${cosy_repo}}

# 3.6 — ranking objective the solution type does not support
- problem_class: OBP
  replanning: none
  triggers: [WMSRun]
  requires: {orders: 1}
  solver:
    type: cosy
    objective: tardiness         # error: silently ignored (BatchingSolution)
    repo: ${cosy_repo}
```

## Where to look

- decision cards and compatibility mapper: `src/casim/decision_card.py`
- compilation boundary and compiled engine: `src/casim/setup.py`
- runtime projection (unchanged): `src/casim/simulation_engine/state_adapter.py`
- inspectable solver preparation + `ApplicabilityReport`: `src/casim/solvers/cosy_solver.py`
- ranking objectives (unchanged): `src/casim/pipelines/solution_ranker.py`
- negative and equivalence tests: `tests/test_configuration_compiler.py`
- decision-space tests: `tests/test_decision_space.py`
- pre-run applicability tests: `tests/test_pre_run_applicability.py`
