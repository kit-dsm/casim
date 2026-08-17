# CASIM decision configuration model

This document is the formal specification for the compilation of one CASIM
decision configuration. It is intentionally concise. The code in
`casim.decision_card`, `casim.setup`, and the individual semantic owners
implements exactly this model.

## 1. Problem taxonomy

`TAXONOMY` (in `casim.pipelines.taxonomy`) is the single source of truth for
the mathematical decision structure:

  T(p) = <V_p, S_p>

where `p` is a problem class, `V_p` is the set of decision variables, and
`S_p` is the resulting solution kind. A small helper `solution_kind(p)`
derives the solution kind from the variables:

  "scheduling" in V_p  ->  SchedulingSolution
  "routing"    in V_p  ->  CombinedRoutingSolution
  otherwise            ->  BatchingSolution

No other code copies this fact.

## 2. Decision binding

A decision binding is `b = (p, r)` where `r in {none, unstarted, active}`.
Only explicitly registered bindings are valid:

  b in dom(DELTA)

where `DELTA` is the `DECISION_CARDS` registry. Seven bindings are supported
(see the table at the end). This is a closed registry, not a Cartesian product.

## 3. DecisionCard

A DecisionCard describes **what operational state is projected for a binding**:

  delta_b = <pi_b>

where `pi_b` maps planning quantities to projection sources.

```python
@dataclass(frozen=True)
class DecisionCard:
    orders: str | None = None
    batches: str | None = None
    resources: str | None = None
    resource_scope: str | None = None
    active_tour: str | None = None
```

The card uses the same source vocabulary as `StateAdapter`:
`"buffered"`, `"buffered_and_replannable"`, `"all"`, `"nonactive"`,
`"available"`, `"dispatchable"`, `"residual"`, `"trigger_if_present"`.

The card does not store `problem_class`, `replanning`, `solution_type`,
`ranking_objectives`, `supports_commit`, `available_conditions`, or
parameter-consumption flags. Those facts belong to other owners (taxonomy,
SolutionRanker, ProjectionSpec, ConditionSpec).

## 4. Projection specifications

For each projection primitive `(quantity, source)`, a `ProjectionSpec`
owns:

  Sigma(q) = <Theta_q, D_q, H_q, C_q>

where:
  Theta_q  = configurable parameters consumed by the projection
  D_q      = runtime facts made available for conditions
  H_q      = trigger capabilities required to construct the projection
  C_q      = fixed cardinalities (fact -> exact count)

```python
@dataclass(frozen=True)
class ProjectionSpec:
    parameters: frozenset[str] = frozenset()
    facts: frozenset[str] = frozenset()
    fixed_cardinalities: Mapping[str, int] = field(default_factory=dict)
    trigger_capabilities: frozenset[str] = frozenset()
```

`StateAdapter.projected_features()` remains the authoritative producer of
algorithm-facing features. ProjectionSpec does not duplicate it.

Current parameter ownership (derived from StateAdapter):

  orders / buffered            -> {limit}
  batches / buffered           -> {limit, due_horizon_s}
  batches / buffered_and_replannable -> {limit, due_horizon_s}
  resources / *                -> {}
  active_tour / residual       -> {congestion_penalty}

## 5. Parameter domains

Each parameter has an explicit value domain. A small validator function is
sufficient; there is no generic type system.

  limit              positive integer or omitted
  due_horizon_s      non-negative numeric or omitted
  congestion_penalty non-negative numeric
  planning_horizon_s non-negative numeric or omitted
  n_jobs             positive integer or omitted
  max_jobs_per_picker positive integer or omitted

The invariant: `dom(theta_c) subset Parameters(delta_b)`, and every supplied
value satisfies its declared domain. Unknown or inactive parameters are
errors.

## 6. Conditions

A condition `g` has one declared meaning used by compile-time validation,
runtime evaluation, and `explain()`:

  g = <fact, operator, validate_value, drain_presence>

```python
@dataclass(frozen=True)
class ConditionSpec:
    fact: str
    operator: str           # ">=", "<=", or "false"
    validate_value: Callable
    drain_presence: bool = False
    def render(self, key, value) -> str: ...
```

The five conditions:

  key            fact                      operator  drain_presence
  orders         orders.count               >=        yes
  batches        batches.count              >=        yes
  pickers        resources.count            >=        no
  not_on_break   dynamic.is_break           false     no
  dock_capacity  dynamic.n_staged_pallets   <=        no

Drain behavior: when the input is closed and the binding owns the drain
trigger, `orders` and `batches` reduce to presence checks (any vs none).
This is preserved from `SimulationEngine._conditions_hold`.

## 7. Projected facts

Condition applicability:

  Applicable(g, delta_b)  iff  fact(g) in Facts(delta_b)

  Facts(delta_b) = D_base  union  union D_q  for q in pi_b

Base dynamic facts (every snapshot carries these):

  dynamic.is_break
  dynamic.n_staged_pallets

Count facts and their domains:

  orders.count      >= 0  (variable when orders projected, absent otherwise)
  batches.count     >= 0  (variable for batch projections)
  resources.count   >= 0  (variable for resource projections)

For active residual projection:

  batches.count  = 1  (the residual batch)
  resources.count = 1  (the active picker)

A condition is impossible when the required cardinality exceeds the fixed
cardinality. Impossible conditions are rejected. Redundant-but-true guards
(e.g. `pickers: 1` on an active binding) are accepted.

## 8. Event capabilities

A projection declares required trigger information `H_q`. An event declares
`Cap(e)`. A trigger is valid iff:

  H_q subset Cap(e)

`H_q != empty  =>  |Triggers(c)| > 0` (closes the zero-trigger hole).

Events carry a class-level `decision_capabilities` frozenset. No reflection
or runtime introspection of constructors.

## 9. Solver semantics

`SolutionRanker.supported_objectives(problem_class)` owns and exposes the
ranking objectives:

  OBP / OSBP        -> {}  (no ranking objective)
  ORP / OBRP / BSRP -> {distance}
  OBRSP / ORSP      -> {distance, makespan, tardiness}

Compile-time validation queries this API. If the problem is not ranked by
objective, supplying `solver.objective` is an error.

## 10. Commitment semantics

`SchedulingCommitmentPolicy` owns a closed schema for its fields:

  n_jobs, max_jobs_per_picker, planning_horizon_s

Unknown commit fields are errors. A `commit:` block is valid iff the problem
produces a scheduling solution (derived from the taxonomy). This is the only
configurable commitment policy; there is no hierarchy of policy cards.

## 11. Closed configuration grammar

CASIM rejects unknown configuration fields at every level:

  DecisionConfig ::= Binding Triggers [Requirements]
                       [ProjectionParameters] Solver [Commitment]

Known decision-entry fields: `problem_class`, `replanning`, `triggers`,
`requires`, `due_horizon_s`, `limit`, `congestion_penalty`, `solver`,
`commit`.

Known `requires` keys: the five condition keys.

Known `commit` fields: `n_jobs`, `max_jobs_per_picker`, `planning_horizon_s`.

Known cosy-solver fields: `type`, `objective`, `repo`, `executor`.

Direct-solver constructor arguments are not statically validated beyond what
Hydra already handles.

## 12. Formal validity conditions

A configured decision `c` is valid iff:

  V1  binding(c) in dom(DELTA)
  V2  every supplied field belongs to the declared grammar
  V3  dom(parameters(c)) subset Parameters(delta_b), values satisfy domains
  V4  for every guard g: fact(g) in Facts(delta_b), value satisfies domain;
      reject impossible cardinalities
  V5  for every trigger e: RequiredCapabilities(delta_b) subset Cap(e);
      trigger required when projection needs trigger context
  V6  solver type known; for CoSy: objective in SupportedObjectives(p) if
      ranked; objective on unranked problem is an error
  V7  commit permitted only when problem produces SchedulingSolution
  V8  one config per binding; one binding per trigger event class

## 13. Compilation result

```python
@dataclass(frozen=True)
class CompiledDecision:
    binding: tuple[str, str]
    adapter: StateAdapter
    trigger_classes: tuple[type[Event], ...]
    conditions: tuple[CompiledCondition, ...]
    solver: object
    policy: SchedulingCommitmentPolicy | None
    applicability: ApplicabilityReport | None
```

The compilation flow:

  Hydra config -> binding lookup -> DecisionCard -> projection/parameter
  validation -> condition validation -> trigger capability validation ->
  solver/commitment validation -> StateAdapter -> projected_features() ->
  effective DataCard -> DomainAlgorithmMapper -> CoSy/Maestro prepare ->
  CompiledDecision

## 14. Explainability

`explain()` derives every line from the semantic owners: binding,
DecisionCard projection, ProjectionSpec, compiled conditions (rendered by
ConditionSpec), taxonomy, solver preparation, commitment policy. No
independent semantic knowledge in `explain()`.

## 15. Semantic ownership

  Information                       Owner
  problem variables / result kind    taxonomy + solution_kind()
  valid bindings                     DECISION_CARDS registry
  projection source                  DecisionCard
  projection parameter ownership     ProjectionSpec
  actual state projection            StateAdapter
  condition meaning                  ConditionSpec
  event information                   event decision_capabilities
  ranking support                    SolutionRanker
  commitment fields                  SchedulingCommitmentPolicy
  algorithm requirements              AlgorithmCard
  algorithm applicability            DomainAlgorithmMapper

## 16. Seven supported bindings

| Binding | Projection | Parameters | Guard facts | Fixed cardinalities | Trigger caps |
|---|---|---|---|---|---|
| OBP/none | orders=buffered, resources=all | limit | orders.count, resources.count, is_break, n_staged_pallets | - | - |
| ORSP/none | batches=buffered, resources=nonactive | limit, due_horizon_s | batches.count, resources.count, is_break, n_staged_pallets | - | - |
| ORSP/unstarted | batches=buffered_and_replannable, resources=available | limit, due_horizon_s | batches.count, resources.count, is_break, n_staged_pallets | - | - |
| OBRSP/none | orders=buffered, resources=dispatchable (trigger-scoped) | limit | orders.count, resources.count, is_break, n_staged_pallets | - | - |
| OBRP/none | orders=buffered, resources=dispatchable (trigger-scoped) | limit | orders.count, resources.count, is_break, n_staged_pallets | - | - |
| ORP/active | active_tour=residual | congestion_penalty | batches.count, resources.count, is_break, n_staged_pallets | batches.count=1, resources.count=1 | picker_id, tour_id, route_version, resumes_execution |
| OBRP/active | orders=buffered, active_tour=residual | limit, congestion_penalty | orders.count, batches.count, resources.count, is_break, n_staged_pallets | batches.count=1, resources.count=1 | picker_id, tour_id, route_version, resumes_execution |
