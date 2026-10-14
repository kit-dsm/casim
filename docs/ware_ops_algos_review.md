# ware\_ops\_algos branch review

Oct 13, 2026 · @Janik Bischoff

## Scope

This review compares `origin/codex/revision-scopes` of ware\_ops\_algos with `origin/main` (merge base `b3db54e`), reads the CASOP code paths in ware\_ops\_pipes `main` that consume ware\_ops\_algos, and checks the result against the CASOP manuscript (CASOP\_Manuscript\_3, INFORMS Journal on Computing submission). All statements come from reading diffs and code. No test suite was run, so behavioural consequences that depend on execution are marked as expected rather than observed.

## Size and provenance

The branch has 42 commits over `main`, touching 77 files with 4,816 insertions and 211 deletions. Thirteen commits from 22 September to 10 October come from earlier agent runs; the remaining 29, all from 13 October, come from the correction run.

| Part | Lines | Origin |
| --- | --- | --- |
| `algorithms/waiting/analytic_progress/` | about 2,400 | earlier runs (port of the analytical waiting model) |
| `planning/` | about 670 | correction run |
| waiting, admission, positioning, incumbent batching, order splitting | about 600 | both |
| tests (11 new files) | about 800 | correction run |
| shared interfaces, routing, scheduling, mapper, domain models | about 400 | both |

`main` has two test files, so almost all test coverage of the shared code is new on the branch.

## Algorithm card changes

Four existing cards changed. In `combined_batching_routing_assigning`, `item_assignment_nn` and `item_assignment_single_pos`, only `class_name` was corrected to an existing class; ware\_ops\_pipes imports these classes directly and keys its module map by `algo_name`, so CASOP behaviour does not change. `spt_scheduling.yaml` changed twice in `b77f256` (22 September, earlier run): `component_name` became `SPTScheduler`, which matches the ware\_ops\_pipes component and is a fix, and the `due_date` requirement on orders was removed, which changes CASOP applicability for instance sets without due dates. That second change was never reported.

Thirteen cards are new: `no_waiting`, `order_count_waiting`, `henn_waiting`, `analytic_stochastic_waiting`, `aisle_patrol_positioning`, `center_relocation_positioning`, `center_position_admission`, `remaining_route_admission`, `fifo_scheduler`, `incumbent_start_batching`, `walk_or_wait_sshape_routing`, `reroutable_sshape_routing` and `reroutable_walk_or_wait_sshape_routing`; the four `wait_for_*` cards were replaced by `order_count_waiting`. They live in the same directory that `load_packaged_algo_cards()` reads for CASOP.

Most of them are filtered out in CASOP because their `problem_type` (waiting, positioning, admission, active\_tour\_routing) does not occur in the CASOP taxonomy. Three are not:

| Card | problem\_type | Effect in CASOP |
| --- | --- | --- |
| `walk_or_wait_sshape_routing` | routing | applicable to every conventional layout with human resources (SPRP, OBRP, OBSRP) |
| `fifo_scheduler` | scheduling | `requirements: {}`, so applicable to every OBSRP instance (Kris) |
| `incumbent_start_batching` | batching | rejected, because the default `decision_features.incumbent` of a CASOP data card is false |

The first two enlarge the applicable set A\* in CASOP although ware\_ops\_pipes has no component for them. Whether `_import_models` then skips them silently or fails, and whether the counts of Table 2 in the paper change on a re-run, has not been executed.

## Applicability mapper and feature semantics

On `main`, `DomainAlgorithmMapper.filter` runs the two stages of the paper (Section 4.4.1) in three private methods: `_filter_by_problem` admits a card if its `problem_type` is the problem class or one of its taxonomy variables, and `_is_feasible` with `_check_domain_requirements` checks, per referenced domain, (a) type compatibility with `any` as wildcard, (b) presence of every required feature and (c) every constraint through `ConstraintEvaluator`, which supports `equals`, `not_equals`, the four comparisons, `in`, `not_in`, `and` and `or`. Commits `8405404` (Task 1.4) and `3faf07a` (Task 1.3) replaced the three methods by one method, `rejection_reasons(card, instance)`, which collects every reason instead of returning at the first failure; `filter` keeps its signature and returns the cards without reasons. The decision logic is the same in five respects: the same problem-type stage, the same type check, the same presence check, the requirement that a constrained feature be present, and the same evaluator. Six details differ.

| Aspect | `main` | Branch |
| --- | --- | --- |
| feature with value `None` in a dict domain | present | absent: `{name: value ... if value is not None}` |
| referenced domain missing on the instance | `AttributeError` | reason `missing domain ...` |
| domain access | `get_type_value()` with bare `except` fallback to dict | `isinstance(domain, dict)` |
| operators | as listed above | plus `contains` (`y in x`) |
| output and documentation | verbose prints, detailed docstrings per method | reasons per card; docstrings of the private methods removed |
| accepted instance types | `BaseWarehouseDomain` | `BaseWarehouseDomain` or `DataCard` |

Only the first row can change results, and only through the feature values. In the YAML path that ware\_ops\_pipes uses (`Runner.run_instance` calls `filter` with the flattened data card), a feature without `value` becomes `True`, and no CASOP data card declares `value: null`; this path is therefore unchanged. The instance path is affected: `datacard_from_instance` builds the sections from `get_features()`, and `Resources.get_features()` changed in `90c9364`. On `main` it reports `capacities` and `dimensions_type` as the constant `True`; on the branch it reports the cart's actual values, and `None` if a resource has no cart or the carts differ. Together with the first row, a card that requires `capacities` (FiFo and the other batching cards) is rejected on an instance-derived card for such resources. ware\_ops\_pipes does not call `datacard_from_instance` or `validate_against_card`; casim's use of them was not traced.

`OrdersDomain.get_features()` changed in a smaller way: `article_id` and `amount` are evaluated over all order positions instead of the first one, `order_id` is added as an alias of `order_number`, and an empty order list yields `False` instead of absent keys.

The data card gained two sections that are not among the paper's five domains: `information`, the planner information (forecast or arrival process), and `decision_features`, which casim derives per decision from the state information and scope and which defaults to `{incumbent: false}`. The mapper treats them like domains because it looks every requirement section up by name. Their meaning differs from the domains, which describe the warehouse: `information` describes knowledge about the future and `decision_features` the decision. The mechanism needs no change, but the concepts need a definition.

## Other changes to code shared with CASOP

`earliest_free_assignment` now begins a job at `max(release, available_from)` and adds the setup to the end time instead of to the start (agreed, Task 4.8), takes `available_from` as a parameter, and recomputes the processing time for the picker actually assigned. The last point was not planned; it changes nothing for identical pickers and changes results for heterogeneous ones.

The `Routing` base class takes a `routing_origin` in `__init__`, uses it in `_get_distance` and `_get_route_for_tour`, and `SShapeRouting` inserts the origin node into the route. With `routing_origin=None` the behaviour is unchanged. The workflow `detect-algo-changes` on `main` fingerprints every algorithm class together with its in-package base classes, so after a merge every router counts as changed, although only the new origin path behaves differently.

`algorithm_interfaces.py` gained `RoutingOrigin`, the admission, positioning and waiting inputs and solutions, `ActiveTourBatch` and the active-tour solutions. The routing module gained `WalkOrWaitSShapeRouting`, which reproduces the S-shape traversal of another simulator, and `ReroutableSShapeRouting` and `ReroutableWalkOrWaitSShapeRouting`, both built on a private mixin `_ActiveSShapeRerouting` that overwrites `self.closest_node_to_start` inside `_run`. `DataCard` and `validate_against_card` were extended for `information` and `warehouse_info`. Several CASOP files lost unused variables and imports through the dead-code sweep of Task 3c.2; that is harmless but adds to the fingerprint changes.

## Consistency with the CASOP paper

The paper defines a data card as I = ⟨P, Z, D1, …, D5⟩ over five domains (Section 4.1), an algorithm card as α = ⟨Rα, Pα, Zα⟩ with Zα empty for constructive methods without an explicit objective (Section 4.2), configurations that either reference a card of a subproblem or select one of a card's admissible values and are resolved into a card whose requirements merge those of all selected parts (Section 4.2), a single taxonomy T\_P (Section 4.3, Figure 4), and the two-stage mapping with conditions (a) to (c) (Section 4.4.1). Measured against this, the branch contains one contradiction of mechanism, several undefined extensions, and two places where code and paper already disagreed on `main`.

| The Topic | Paper | `main` | Branch | Assessment |
| --- | --- | --- | --- | --- |
| Mapping stages (a) to (c) | as described | as described | same logic, with reasons | consistent |
| Feature without value | signals availability | `True` | `True`; explicit `null` now counts as absent | consistent extension; the paper does not define `null` |
| Constraint operators | comparisons, `in`, `and`, `or` | same | plus `contains` | extension |
| Domains of a data card | five | five plus `warehouse_info` | plus `information` and `decision_features` | undefined extension: neither describes the warehouse |
| Objective Zα | empty for constructive methods | `Distance` on all 24 cards | invented values (`makespan`, `mean_order_completion_time`, `null`) | disagreement already on `main`; the branch widens it, and `resolve_configuration` compares objectives |
| Parameterized algorithms | configuration options, resolved cards | `configs.yaml` for LSBatching, Clarke-Wright, SeedBatching | `IncumbentStartBatching` sets `implementation.routing_class` by hand; `OrderCountWaiting` uses one subclass per value | contradiction: bypasses the resolution, so the router's requirements are not merged |
| Taxonomy | one T\_P | SPRP, SPRP-SS, OBRP, OBSRP | casim adds online classes and the subproblems waiting, admission, positioning and active\_tour\_routing; their cards sit in ware\_ops\_algos | undefined extension; the taxonomy is split over two repositories |
| Problem class name | OBRSP | OBSRP | OBSRP | naming mismatch already on `main` |

The parameterization is the only real contradiction, and it is partly caused by my instructions: Tasks 4.5 and 4.10 asked for `routing_class` in the card's implementation and for a parameterized component, without pointing to the configuration mechanism of Section 4.2. The paper's mechanism covers both cases exactly. `IncumbentStartBatching` is the component case, like LSBatching with its routing option, and `OrderCountWaiting` is the value case, like SeedBatching with its seed criterion.

For the results of the paper, the effect on Table 2 is limited. SPT is only admitted for OBSRP, and Kris, the only OBSRP set, has due dates, so removing the `due_date` requirement does not change Table 2. `walk_or_wait_sshape_routing` could raise the R column for every conventional layout with human resources, and `fifo_scheduler` the S column for Kris, if A\* is recomputed on the branch. Both must be excluded from CASOP or the paper's counts no longer reproduce.

The alignment I would propose keeps the paper's formalism and extends it at three named points, which a CASIM paper can then state.

1. The data card becomes I = ⟨P, Z, D1, …, D5, B, X⟩, with planner information B (knowledge about the future, the belief of the forecasting document) and decision features X (derived per decision from state information and revision scope). Both are matched by the same conditions (a) to (c).
2. The taxonomy becomes one registry with offline classes (PRP, OBRP, OBRSP) and online classes, whose additional subproblems are waiting, admission, positioning and active-tour routing. Every card's Pα must belong to it.
3. Parameterized algorithms use only the configuration mechanism of Section 4.2, and Zα takes values from the objectives of the taxonomy or is empty, as the paper says. The 24 `Distance` cards on `main` are then either corrected to `distance` or the paper's sentence is changed to match the code; which of the two is your decision.

## Adherence to the repository conventions

The new code follows the conventions that the code makes explicit: algorithms subclass `Algorithm[I, O]`, implement `_run` and set `algo_name`; inputs and solutions are dataclasses in `algorithm_interfaces.py`; packages export through `__init__`; cards carry `implementation.class_name` and `component_name`; the `planning` package is typed throughout.

It does not follow the conventions that exist only by example.

- Module layout. On `main` every subproblem has one module that holds the base class and its implementations (`batching/batching.py`, `routing/routing.py`, `scheduling/scheduling.py`). The new packages split base and implementations over several files (`waiting/waiting.py` and `policies.py`, `positioning/positioning.py` and `empty_cart.py`, `admission/admission.py` with two more files). Task 4.0 of my plan prescribed this split.
- Card form. Cards on `main` spell out the four requirement sections with `type` and `features`, use `objective: Distance`, `implementation.type` from `heuristic`, `metaheuristic` and `exact`, and the `solver` key. Several new cards use `requirements: {}`, new objectives, the new types `policy` and `algorithm`, omit `solver`, and add `routing_class`.
- Configured algorithms. As described above, `configs.yaml` is bypassed.
- Private helper classes. `_ActiveSShapeRerouting` repeats the private-mixin pattern that was rejected for positioning, and it mutates `closest_node_to_start` during `_run`.
- Scope of the library. `WalkOrWaitSShapeRouting` reproduces another simulator inside the shared routing module, and `order_splitting/` (moved from casim) has no card and keeps `OrderSplittingSolution` outside `algorithm_interfaces.py`.
- Documentation. The mapper lost its method docstrings, while the rest of the repository documents public behaviour in docstrings.
- The analytical waiting model (about 2,400 lines) uses single-letter accessors (`g.k`, `g.A`) and `typing.Dict`/`List`, and has one precondition test; its comparison with the closed form is scheduled for verification scenario 9.

## The state model

The new types of the branch follow no stated model, which is the main reason they are hard to read. The following model orders them and states precisely in which sense CASIM generalizes CASOP.

The state of the warehouse at time t has three components.

```latex
s_t = (W_t,\; P_t,\; B_t)
```

The warehouse W\_t consists of the CASOP domain objects in their current condition: layout, articles and storage, the orders revealed up to t, and the resources with position and availability, including physical conditions such as breaks and the dock fill. The plan P\_t consists of the pick lists and tours with their lifecycle status, schedule and execution progress. The belief B\_t is what the decision maker knows about the future: the planner information, and whether further orders can arrive. The first component is deliberately not called instance. The paper uses I for the data card, and the instance in CASOP's sense is what a decision is given, which is derived from all three components.

One boundary needs a fixed rule, because orders and resources appear in both W\_t and P\_t. Entities and their physical condition belong to W\_t, and their assignment to tours belongs to P\_t; the executed part of a tour (walked prefix, picked lines) belongs to P\_t as its irreversible part, and the picker's position belongs to W\_t.

A decision d is defined by a trigger τ\_d, an observation o\_d (the current `StateInformation`), the admissible changes A\_d (the `RevisionScope` together with the physical limits of the lifecycle) and a commitment rule κ\_d. When d is triggered, its instance and data card are derived from the state.

```latex
I_d = o_d(s_t), \qquad \mathrm{card}(I_d) = \big(\mathrm{card}(W_t),\; \mathrm{card}(B_t),\; \phi(o_d, A_d)\big)
```

The data card combines the features of the warehouse, as in CASOP, the features of the belief (the `information` section) and the decision features φ derived from observation and admissible changes. The mapper applies conditions (a) to (c) of the paper to this card unchanged. The pipeline's solution becomes a set of commands x\_t ⊆ A\_d, and the transition applies them together with the exogenous events ω between two decisions, order arrivals and process events.

```latex
s_{t'} = f\big(s_t,\; x_t,\; \omega_{(t, t']}\big)
```

These are the five elements of Powell's framework (state, decision, exogenous information, transition, objective), with the state factored into warehouse, plan and belief. CASOP is the special case t = 0, P\_0 = ∅, B\_0 trivial because all orders are known, o\_d the full state and κ\_d commitment of everything; then I\_d = W\_0, the CASOP instance. This makes precise the sentence of the design document that every decision is a recurring CASOP problem.

The classification of Pardo et al. (2024) fits the model without a new mechanism. Their constraint dimensions need no label of their own: the number of pickers is a feature of W\_t, and whether a setting is offline or online follows from the model, since offline is the case of a trivial belief and a single decision. An algorithm that only makes sense when further orders can arrive, such as a waiting rule, therefore requires that concrete property of the belief, for example a declared arrival process and an open order stream, and a static solver requires nothing and stays applicable in both settings. Their tasks B, S, A, R and W correspond to subproblems of the taxonomy, with item assignment as CASOP's extension and admission and positioning as CASIM's. Their objectives provide the vocabulary for Zα. What their taxonomy does not cover, trigger, observation, admissible changes and commitment, is the decision layer of the model, which classifies policies rather than problems.

Three things that the code currently keeps in decision bookkeeping belong to the state. Breaks and the dock fill are physical conditions of W\_t. `input_closed` is knowledge about the future and belongs to B\_t; the event that sets it is sufficient as a mechanism, but both stochastic scenarios fire it at the last order date, which gives policies oracle knowledge of the stream end (open point O19 of the design document). A wait is probably a plan fact: since Task 4.8 a planned tour with a future start time is queried again at that time, so waiting is either a later planned start or the absence of commitment, and the fields `waiting` and `reconsider_at` of `DecisionMemory` become redundant (open point O17). The last point is a conjecture; it holds only if the Henn and analytical waiting results stay unchanged. Everything else in the decision context and memory (decision name, cause, reconsideration reasons, solve flag, information versions, already considered admission pairs) is bookkeeping of the decision engine. It is part of the simulator's state, but not of the model that algorithms see; of these, only the considered admission pairs influence an algorithm's result.

## Type consolidation under the model

The branch introduces 40 types, 33 in ware\_ops\_algos and 7 in casim. Under the model each of them belongs to one of seven concepts: warehouse, plan, belief, decision specification, commands, violations, and the engine's snapshot. A type that fits none of them, or that duplicates another within its concept, is a candidate for merging or removal.

| Concept | Types today | Under the model | Change |
| --- | --- | --- | --- |
| Plan | `TourView` (18 fields), `PlanView` | the seven execution fields of a started tour (`cursor`, `pinned_prefix`, `remaining_route`, `routing_origin`, `occupied_bins`, `picking_until`, `holding`) form one `TourProgress` | +1 |
| Plan lifecycle | `TourStates`, `OrderStates`, `ResourceStates`, `Accepts`, `TimeAttribute` | unchanged | 0 |
| Decision specification | `Window`, `StateInformation`, `RevisionScope` | unchanged; they are o\_d and A\_d | 0 |
| Commands and violations | six commands, `ViolationCode`, `Violation`, `InvariantViolation` | `InvariantViolation` merges into `Violation` with additional codes | −1 |
| Admission I/O | `AdmissionTour`, `AdmissionInput`, `AdmissionSolution`, `ActiveTourBatch`, `ActiveTourBatchingSolution`, `ActiveTourRoutingSolution` | `AdmissionTour` becomes `TourView` with `TourProgress`; `ActiveTourBatch` is a `BatchObject` whose `batch_id` already equals the tour id; one admission solution and one routing solution remain, and `considered_pairs` moves to the engine instead of appearing in three solutions | −3 |
| Waiting and positioning I/O | `WaitingInput`, `WaitingSolution`, `PositioningInput`, `PositioningSolution` | per-subproblem inputs stay, as is CASOP convention, but are composed from one observation (time, plan, belief) instead of repeating `current_time`, `input_closed` and `information`; `PositioningSolution` stays only if CoSy dispatches on it | 0 or −1 |
| Routing | `RoutingOrigin` | unchanged; also useful offline | 0 |
| Belief | `PlannerInformation`, `ExponentialSingleLineUniformLocationOrderStream`, `InformationType` | unchanged, but the process description is named after the kind of information, not after one algorithm's assumption | 0 |
| Engine snapshot | `DynamicInfo`, `SimWarehouseDomain`, `DecisionContext`, `DecisionMemory`, `ReconsiderReason` | `DynamicInfo` and `SimWarehouseDomain` keep their names, which the CASIM paper uses; `DecisionMemory` shrinks to engine bookkeeping once waiting moves into the plan (Task 6.4); `input_closed` stays as the observed stream closure, whose timing becomes a scenario parameter (Task 6.5) | −1 |
| Warehouse condition | `OperationsView` | dissolves into W\_t (resource availability, dock) | −1 |
| Simulator truth | `TourPlanningState` | unchanged; the mutable state behind `TourView` | 0 |

The result is about 33 types, because the snapshot types keep the names of the CASIM paper. The count matters less than the assignment: most remaining types are small enums and commands, and every one of them now has a place in the model.

The parts are not equally stable. Lifecycle, scope, commands and `RoutingOrigin` are tested and match the design. The admission types are the least stable, because they grew in three layers over three agent runs, and this is where the duplication sits. The removal of the waiting fields is the largest simplification and the riskiest, so it belongs after the Phase 5 verification gate and is checked against the regression values there. The other merges are mechanical and form Task 4.13 of the correction plan.

## Pydantic schemas that enforce the conventions

The conventions above are implicit because cards are loaded as untyped dictionaries: `load_algo_card` reads YAML into `AlgorithmCard.__init__` without validation, `_normalize_requirements` only drops empty list entries, and `resolve_configuration` checks options at run time. A schema turns each convention into a load-time error. ware\_ops\_algos has no pydantic dependency yet; casim already uses pydantic for its configuration schema, so adding it is consistent across the two repositories. A sketch, not checked against the code:

```python
# ware_ops_algos/algorithms/card_schema.py
DomainName = Literal["layout", "articles", "orders", "resources", "storage",
                     "warehouse_info", "information", "decision_features"]
Method = Literal["constructive", "heuristic", "metaheuristic", "exact"]

class Comparison(StrictModel):          # exactly one operator set
    equals: Any = None; not_equals: Any = None
    greater_than: float | None = None   # ... remaining comparisons
    in_: list | None = Field(None, alias="in"); not_in: list | None = None
    contains: Any = None

Constraint = Union[Comparison, "AllOf", "AnyOf"]   # and / or

class Requirement(StrictModel):
    type: list[str] = ["any"]          # validated against the domain's type enum
    features: list[str] = []            # validated against the feature registry
    constraints: dict[str, Constraint] = {}
    # validator: every constrained feature is also listed in features

class ComponentOption(StrictModel):
    problem_type: Subproblem

class ValueOption(StrictModel):
    values: dict[str | int, ValueSpec]   # ValueSpec: optional requirements

class Implementation(StrictModel):     # extra keys forbidden
    class_name: str                     # must resolve in ware_ops_algos
    component_name: str | None = None
    solver: str | None = None
    type: Method
    anytime: bool = False
    budget: str | None = None           # name of the budget parameter, required if anytime

class AlgorithmCard(StrictModel):
    algo_name: str
    problem_type: Subproblem            # union of the registered taxonomies
    description: str
    requirements: dict[DomainName, Requirement]
    objective: Objective | None = None  # taxonomy objectives; None for constructive
    implementation: Implementation
    configuration: dict[str, ComponentOption | ValueOption] = {}
    capabilities: dict[str, Any] = {}

class Configuration(StrictModel):      # one entry of configs.yaml
    name: str
    base: str
    configuration: dict[str, str | int]
```

The schema enforces the form; a second layer of cross-card checks enforces the meaning, and these need the loaded library.

- `algo_name` and `component_name` are unique across cards and resolved configurations.
- `class_name` resolves to a class in ware\_ops\_algos, and that class subclasses the base registered for `problem_type` (batching to `Batching`, routing to `Routing`, waiting to `Waiting`, and so on), which subsumes the test of Task 4.6.
- Every `problem_type` belongs to a registered taxonomy, and every taxonomy subproblem has a registered base class.
- Every entry of `configs.yaml` resolves without `RequirementConflict`, and a key such as `routing_class` may appear in an implementation only on a resolved card, never in a YAML file, because `extra="forbid"` rejects it there.
- Every feature named in a requirement is produced by some data card section or by the `get_features()` of the corresponding domain object, which catches spelling drift between cards, data cards and domain models.

The same applies to data cards: a `DataCardSchema` with the hierarchical YAML form of the paper (objects, features with optional `value`) makes `value: null` and unknown sections explicit, and fixes the meaning of a missing value as availability.

## A check CLI to keep agents grounded

Yes, and it should exist in ware\_ops\_algos, not only in casim. `casim check` already keeps the casim configurations honest, but the deviations found in this review all happened in ware\_ops\_algos and none of them would have failed a test: the SPT requirement, the hand-set `routing_class`, the numbered waiting aliases, the growth of A\* in CASOP. An agent works against what fails. A command that turns the conventions into failures is therefore the most direct way to keep it inside them, and it helps human contributors in the same way.

The command, for example `woa check`, would run four checks and exit non-zero on any finding, with a JSON report for agents and a readable one for people.

1. Cards and configurations: the schema and the cross-card checks of the previous section.
2. Taxonomy coverage: every card's `problem_type` is registered, and every registered subproblem has a base class.
3. Applicability lock: for a fixed set of reference data cards (the CASOP data cards of ware\_ops\_pipes, copied as fixtures or read from a configured path), compute A\* per data card and compare it with a committed `applicability.lock.json`. Any difference fails. `woa check --update-lock --reason "..."` rewrites the lock and records the reason, so every change of CASOP applicability appears as a reviewed diff with a justification, and the counts of Table 2 stay reproducible.
4. Fingerprint report: list the algorithm classes whose fingerprint differs from the manifest on `main`, using the existing `utils/fingerprint.py`. This does not fail, but it shows which CASOP algorithms a branch touched, including indirect changes through a base class.

`casim check` then calls the same library checks for the cards casim uses, so both repositories have one source of truth. Three measures make the command effective: an `AGENTS.md` in each repository that lists the conventions and states that `woa check` and the tests must pass before a commit, a pre-commit hook, and a CI job on every branch rather than only on `main`, where `detect-algo-changes` runs today.

The limit is clear. The check enforces form, vocabulary and the stability of applicability. It cannot detect that an algorithm computes the wrong thing; that remains the task of the verification layers (invariants after every step, KPI recomputation, analytical references) and of review.

## Proposed fixes and merge strategy

Decisions for you, before any code change:

1. Keep or revert the removal of `due_date` from the SPT card.
2. Keep the new `Resources.get_features()` semantics (actual cart values, `None` for missing or heterogeneous carts) or return to the constant `True`.
3. Keep the per-picker processing time in `earliest_free_assignment` or return to the reference picker.
4. Objective vocabulary: correct the 24 `Distance` cards to the taxonomy's `distance`, or adapt the paper's wording.

Fixes, in this order:

1. Exclude `walk_or_wait_sshape_routing` and `fifo_scheduler` from CASOP applicability until they are meant to be part of it, for example through a taxonomy that does not admit them, and confirm that A\* for all CASOP data cards equals that of `main`.
2. Re-express `IncumbentStartBatching` as a card with a component option `routing_class: {problem_type: routing}` and a `configs.yaml` entry with `SShape`, and `OrderCountWaiting` as a card with a value option `min_orders` and four `configs.yaml` entries. This replaces the hand-set `routing_class`, the subclasses `WaitForOne` to `WaitForFour` and the aliases `WaitFor0` to `WaitFor4`. casim then needs the same generation of configured components that `generate_configured_components.py` provides in ware\_ops\_pipes.
3. Replace `_ActiveSShapeRerouting` by a public `ReroutableRouting(Routing)` base without state mutation in `_run`, and move `WalkOrWaitSShapeRouting` into its own module.
4. Return to the repository's module layout (base class and implementations in `<subproblem>/<subproblem>.py`) and correct Task 4.0 of the plan accordingly.
5. Restore the mapper docstrings, move `OrderSplittingSolution` into `algorithm_interfaces.py`, and give order splitting a card or remove it.
6. Add the card schema and `woa check` with the applicability lock, generated once from `main`.
7. Add unit tests of the analytical waiting core against the closed form, ahead of verification scenario 9.

The type consolidation of the previous sections follows the same order: the mechanical merges (`TourProgress`, admission types, `Violation`, `OperationsView`) as Task 4.13 of the correction plan, then the move of waiting into the plan (Task 6.4) and the closure timing (Task 6.5) after the verification gate. The design document now has the state model as its own section.

The branch should reach `main` in three pull requests rather than one: changes that leave CASOP unchanged (class names, the SPT `component_name`, dead code) first; then the changes of shared semantics, each with the applicability lock diff and the decision behind it; then the CASIM additions (`planning`, the new subproblems and their cards).
