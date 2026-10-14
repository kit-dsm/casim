# Revision scopes: correction plan for the implementing agent

This file tells you how to correct the implementation on `codex/revision-scopes`. Work in both repositories: `kit-dsm/casim` at `6cd1705` and `kit-dsm/ware_ops_algos` at `90c9364`. A review against the design and three scenarios of the base commits found 41 problems; the reviews during Phases 3 to 4 added F42 to F59. Each one is listed in [Appendix A](#appendix-a-findings-and-where-they-are-fixed) together with the phase that fixes it.

Read this whole file before you change any code.

## 1. Sources of truth

| Question | Source | On conflict |
|---|---|---|
| What the model is, which names exist, what each type contains | `docs/revision_scopes_design.md` (the design document, exported by the user) | The design document wins over this file for names and semantics |
| Vocabulary of the architecture | The CASIM paper (wsc26): `SimulationEngine`, `DecisionEngine`, `StateAdapter`, adapted state snapshot, `CommitmentPolicy`, `SolutionRanker`, `CoSySolver`, `TourManager`, `TourPlanningState`, `DynamicInfo`, problem class, trigger, condition, state information | A name the paper uses is never replaced by a synonym |
| How to proceed, code rules, phase order, acceptance criteria | This file | This file wins over the design document for process |
| Whether results are correct | The invariants and verification scenarios of Section 7, derived from the model, not from any simulator output | A result that violates an invariant or a hand-derived value is wrong, whatever any earlier version produced |
| Behaviour of the base commits casim `67f9f31`, ware_ops_algos `34bd157` | Section 7.4 | A diagnostic, not a reference: the base contains known bugs, and the walk-or-wait instances were built on a faulty simulator |
| Existing code style and existing names | The base commits (see Section 4) | Follow the existing style and keep existing names, not your own |
| CASOP conventions: algorithm cards, `configs.yaml`, module layout, the mapper | The CASOP paper (Sections 4.1 to 4.4) and ware_ops_algos `main` | A CASIM need never changes a CASOP convention silently; Section 9 applies |
| Review of the ware_ops_algos branch against `main` and the CASOP paper | `docs/ware_ops_algos_review.md` (exported by the user) | Informative; the tasks derived from it are in Phase 4 and 5 of this file |

If the design document, this file and the code disagree in a way these rules do not settle, stop and ask (Section 9). Do not resolve it by picking the reading that is easiest to implement.

Delete `docs/CASIM revision scopes and configuration design.md` from casim. The user replaces it with `docs/revision_scopes_design.md`. Never edit the design document yourself.

## 2. Goal

At the end of this plan:

1. Correctness follows from the design and is checked by machine (Section 7): the lifecycle and the revision scope are finite state machines whose properties are verified exhaustively; every run is checked against plan invariants after every step; KPIs are recomputed independently from the tour records; and small verification scenarios match values derived by hand from the model. Differences from the base commits are explained one by one.
2. `ware_ops_algos.planning` contains the rules of the plan: the lifecycle, `Accepts` and `PHYSICAL`, the plan views, the state information and revision scope, the decision features, the commands with their conversion and validation, and the plan invariants. None of it imports a simulator, CoSy, Luigi, Hydra or pydantic.
3. casim contains the two engines of the paper and everything that decides when something happens or changes live state: events and process events, `State` and its managers, applying commands, the snapshot types, the `StateAdapter`, the `SimulationEngine` with triggers and conditions, the `DecisionEngine` with its solvers, commitment policies and `SolutionRanker`, the online taxonomy, configuration loading, the `CoSySolver` and its components, the CLI, trackers and visualization.
4. All algorithm logic lives in ware_ops_algos algorithms that inherit from the base class of their subproblem (`Batching`, `Routing`, `Scheduler`, `ItemAssignment`, `OrderSplitting`, and the new `Waiting`, `Admission`, `Positioning`), have an algorithm card, and are selected through `DomainAlgorithmMapper`. casim's engine, setup, state, conditions and component templates contain no algorithm-specific knowledge.
5. Every structured value is a typed object: a dataclass, an enum, or a pydantic model at the YAML boundary. No `dict`, `object` or `Any` stands in for a structure, and no string encodes a state.
6. A user cannot write a configuration that loads but means something else. Every mistake the model can detect is rejected by the schema or reported by `casim check`, with the key, the value and what is allowed.

## 3. Ground rules

### 3.1 Never

- Never change what an existing algorithm computes, unless a task in this file says so.
- Never weaken a check, an exception or a guard to make a test pass. Never filter `DomainAlgorithmMapper` rejection reasons. Never skip, `xfail` or delete a failing test without the user's approval.
- Never treat a number as correct because an earlier version produced it, and never treat a difference from the base as acceptable without a cause you have demonstrated (Section 7.4).
- Never change an expected value in a verification scenario to match the code. If code and derivation disagree, find which one is wrong and say so.
- Never add a configuration key, enum value, type, field or command that the design document does not define. If you think one is needed, stop and ask.
- Never rename an existing name that the rename table (Section 6) does not list. Section 6.2 lists names that must stay.
- Never put algorithm-specific names, strings, constants or computations into casim's `simulation_engine`, `state`, `setup.py`, `config` or the component templates. Algorithm-specific CoSy components in `casim/pipelines/subproblems/` may only instantiate the algorithm and pass its inputs.
- Never hard-code a concrete algorithm (for example `SShapeRouting`) inside another algorithm, a component or an engine. Pass the configured instance in.
- Never use `state.__dict__`, `deepcopy` as a transaction, `getattr`/`hasattr` on your own types, access to another class's private attributes (`_order_buffer`, `_picker_tour_queues`), function-local imports used to dodge cycles, or attributes set on an object after construction.
- Never leave dead code: unused classes, unused parameters, commented-out code you added, or "for later" modules.

### 3.2 Always

- Keep tests green after every commit, with the whole suite run in one process: `python -m pytest -q` in each repository.
- Make one commit per task, with a message that names the finding numbers (for example `Restore waiting as a pipeline stage (F4, F10, F11)`).
- When a phase is finished, write the phase report (Section 8) before you start the next phase.
- Move a module with `git mv` and keep its name, so its history stays readable.
- When you are unsure whether existing code is the user's or a previous agent's, check `git log -S` and say what you found in the report.

## 4. Code style

The base code is the reference for style. Imitate it and keep new code at least as plain.

Domain objects are `@dataclass` types, `frozen=True` for every snapshot part, view, input and command. Each domain has its own subpackage, `domain_models/<domain>/models.py`, with an `__init__.py` that re-exports. Data-card domains inherit from `BaseDomainObject` with `tpe` and `get_features()`. The plan types are the exception: they live in `ware_ops_algos/planning/`, because they use solution objects from `algorithms` (Section 5.1). Vocabularies are `class X(str, Enum)`. Where order matters, the order is a module constant next to the enum (`TOUR_ORDER`, and the order `none < insert < rebuild` of `Accepts`), with a comparison helper, not string comparison.

Every subproblem has an abstract base class in its package, which binds the input and solution types: `class Batching(Algorithm[list[WarehouseOrder], BatchingSolution], ABC)` in `batching/batching.py` is the pattern. Every algorithm subclasses the base of its subproblem, never `Algorithm` directly: `IncumbentStartBatching` is a `Batching`, `ReturnToDepotRouting` a `Routing`, `HennWaiting` a `Waiting`. Waiting, admission and positioning lack this base at the base commits; Task 4.0 adds it. Every algorithm implements `_run`, declares `algo_name` as a class attribute, and receives its collaborators (a router, a cart) as typed constructor parameters. It lives in the package of its problem type (`algorithms/waiting/`, `algorithms/positioning/`, ...), is exported from that package's `__init__.py`, and has an algorithm card in `algorithms/algorithm_cards/`.

casim keeps its existing structure. Managers own state through explicit methods. Events implement `handle(state) -> list[Event]`. Hydra `_target_` instantiates configured components. CoSy components subclass the abstract stage and override a `_get_inited_*` hook.

Functions are short and have one level of comprehension. Exit early instead of nesting conditionals. Name values instead of repeating expressions. Put a type hint on every public function and attribute. Use `logging`, not `print`. Docstrings are one line saying what something is, and comments explain why, never what.

Dispatch over solution types with `functools.singledispatch` or a typed registry dict, not `isinstance` chains spread across modules. Compare enums with enums: `tour.accepts is Accepts.REBUILD`, never `tour.accepts.value == "rebuild"`.

Here is the difference in practice:

```python
# Not like this (casim/decision_engine/waiting.py today)
if policy == "henn" and len(jobs) > 1 and index == earliest_by_picker[job.picker_id]:
    planned.append(job); selected.append(job); continue

# Like this: the rule stays in the algorithm, which sees the whole schedule
class HennWaiting(Waiting):
    """Henn's release threshold, applied to the first scheduled batch."""

    algo_name = "HennWaiting"

    def __init__(self, router: Routing):
        super().__init__()
        self.router = router

    def _run(self, data: WaitingInput) -> WaitingSolution:
        if not data.candidates:
            return WaitingSolution(jobs=())
        if data.input_closed or len(data.candidates) > 1:
            return WaitingSolution(jobs=data.candidates)
        threshold = self._release_threshold(data.candidates[0], data)
        if data.current_time < threshold:
            return WaitingSolution(jobs=data.candidates, action="wait", reconsider_at=threshold)
        return WaitingSolution(jobs=data.candidates)
```

## 5. Target architecture

### 5.1 Responsibilities

Each concept goes where its dependencies point. What describes a plan and how it may change needs no simulator and goes to a new package, `ware_ops_algos.planning`. What decides when something happens, and what changes live state, stays in casim. Both engines of the paper stay in casim.

| Package | Owns | May import |
|---|---|---|
| `ware_ops_algos.domain_models` | Static warehouse domain, unchanged from the base | nothing outside the package |
| `ware_ops_algos.algorithms` | Algorithms, subproblem base classes, inputs and solution objects (including `AdmissionTour` and the `ActiveTour*` solutions), algorithm cards | `domain_models` |
| `ware_ops_algos.planning` (new) | Lifecycle, `Accepts`, `PHYSICAL`, `TourView`, `PlanView`, `Window`, `StateInformation`, `RevisionScope`, decision features, commands, `to_commands`, `validation.check`, plan invariants | `domain_models`, `algorithms` |
| `ware_ops_algos.taxonomy`, `ware_ops_algos.domain_algo_mapper` | CASOP taxonomy and applicability, unchanged from the base | `domain_models`, `algorithms.algorithm_cards`, as at the base |
| `casim` | Events and process events, `State` and its managers, `apply`, `TourPlanningState`, `SimWarehouseDomain` with `DynamicInfo`, `DecisionContext`, `ReconsiderReason` and `DecisionMemory`, `StateAdapter`, `SimulationEngine`, triggers, conditions, `DecisionEngine`, `DecisionResult`, `DecisionSolver`, commitment policies, `SolutionRanker`, the online taxonomy with `variant`, configuration, `CoSySolver` and its components, CLI, trackers, visualization, RL environment | `ware_ops_algos`, cosy, luigi, hydra, pydantic |

Nothing in ware_ops_algos imports casim, cosy, luigi, hydra or pydantic. `algorithms` does not import `planning`, so algorithm inputs stay independent of the plan types. Inside casim, `casim.decision_engine` imports nothing from `casim.state`, `casim.simulation_engine` or `casim.events`: the `DecisionEngine` reads the snapshot and returns commands, and only `step` touches live state. Only conditions read `DecisionContext.cause`; CoSy components and algorithms never do. Add one test per repository that walks the imports of every module (with `ast`) and fails on a forbidden import from these rules, and a test that fails if anything under `casim/pipelines/` reads `context.cause`.

### 5.2 Module layout

New, moved or changed modules only. The design document defines the fields of each type.

```text
ware_ops_algos/src/ware_ops_algos/
  algorithms/
    algorithm_interfaces.py   AdmissionInput (tours as TourView, Task 4.13), AdmissionSolution; WaitingInput
                              without single_order_service_times; PositioningInput, PositioningSolution;
                              ActiveTourRoutingSolution; OrderSplittingSolution (Task 4.12)
    waiting/waiting.py        Waiting and NoWaiting, OrderCountWaiting, HennWaiting, AnalyticStochasticWaiting
    admission/admission.py    Admission and RemainingRouteAdmission, CenterPositionAdmission
    positioning/positioning.py  Positioning and AislePatrolPositioning, CenterRelocationPositioning
    routing/routing.py        ReroutableRouting(Routing), the public base for active-tour rerouting
    routing/walk_or_wait.py   WalkOrWaitSShapeRouting and its rerouting variant (Task 4.12)
    batching/incumbent.py     IncumbentStartBatching with a configured routing option (configs.yaml)
  planning/                   new package
    __init__.py
    lifecycle.py              TourStates, ResourceStates, OrderStates, Accepts, TimeAttribute,
                              TOUR_ORDER, TOUR_TRANSITIONS, PHYSICAL (moved from domain_models/lifecycle.py)
    models.py                 TourView, TourProgress, PlanView
    scope.py                  Window, StateInformation, RevisionScope
    decision_features.py      decision_features(state_information, scope)
    commands.py               CreatePickList, CreateTour, CancelTour, ReplaceSuffix, Reschedule,
                              CommitTour, Violation, ViolationCode (git mv from casim/decision_engine)
    conversion.py             to_commands, singledispatch over solution types (git mv from casim/decision_engine)
    validation.py             check(commands, plan) -> tuple[Violation, ...],
                              check_scope(state_information, scope)
    invariants.py             check_plan(plan), check_progress(before, after) -> tuple[Violation, ...]
  taxonomy/taxonomy.py        restored to 34bd157 (F40)

casim/src/casim/
  domain_objects/sim_domain.py   SimWarehouseDomain, DynamicInfo(context, plan, memory), DecisionContext,
                                 ReconsiderReason, DecisionMemory (moved back from ware_ops_algos)
  domain_objects/tour_model.py   TourPlanningState (moved back from ware_ops_algos); imports TourStates
  decision_engine/
    decision_engine.py        DecisionEngine, DecisionResult
    commitment_policies.py    CommitmentPolicy, CommitAllPolicy, SchedulingCommitmentPolicy,
                              WindowCommitmentPolicy
    solution_ranker.py        SolutionRanker (git mv from casim/pipelines)
    interfaces.py             DecisionSolver protocol (moved from casim/interfaces.py)
  events/decision_events.py   PickListCreated; RoutingDone deleted, SequencingDone replaced by Dispatch (Task 4.14)
  state/
    tour_manager.py           advance(tour_id, to) is the only writer of tour status
    apply.py                  check(state, commands) and apply(state, commands) -> list[Event], two phases
    views.py                  OperationsView until Task 4.13 dissolves it into the warehouse
  simulation_engine/
    simulation_engine.py      run() -> (done, state_snapshot), step(result, state_snapshot)
    state_adapter.py          StateAdapter(state_information, scope): State -> SimWarehouseDomain
    conditions.py             Condition.get_decision(state_snapshot, operations)
  config/
    schema.py                 DecisionConfig, DecisionDefinitions (pydantic; embeds the dataclasses)
    data_card.py              data-card loading, moved from scenarios/experiment_commons.py
  setup.py                    resolve(cfg, data_card) -> dict[str, DecisionSetup]; builds both engines
  cli.py                      casim schema | catalog | check
  pipelines/
    pipeline_runner.py        CoSySolver implements DecisionSolver; no ranking inside
    taxonomy.py               online problem classes as at 67f9f31, without ATIP, RORSP, OBRSPW;
                              variant(problem_class, features)
    subproblems/waiting.py    waiting components restored from 67f9f31, adapted
    subproblems/positioning.py
    subproblems/admission.py  builds AdmissionTour from TourView

casim/tests/verification/            verification scenarios, KPI recomputation, invariant runs,
                                     exhaustive checks of variant and write sets (Section 7)
ware_ops_algos/tests/verification/   exhaustive checks of lifecycle, Accepts, PHYSICAL and scope
```

Delete:
- `casim/decision_engine/waiting.py` (and with it `ReleaseDecision`);
- `casim/interfaces.py` once `DecisionSolver` has moved;
- the re-export of the `ActiveTour*` types in the branch's `casim/domain_objects/sim_domain.py`, which the moved-back types replace;
- `ware_ops_algos/domain_models/operational/`;
- `ware_ops_algos/domain_models/planning.py` and `ware_ops_algos/domain_models/lifecycle.py` (their types move to `ware_ops_algos/planning/` or back to casim);
- `SimWarehouseDomain` in `ware_ops_algos/domain_models/base_domain.py` and the branch's changes to `warehouse_info/models.py`;
- `ware_ops_algos/algorithms/pipeline.py`;
- the `ReleaseGates` type and `OperationsView.extra`.

## 6. Names

### 6.1 Rename table

These renames are approved. Make each one completely, everywhere it occurs (code, configs, tests, docs, schema), in a single commit per row. The last column says why the name changes; a name without a reason here stays.

| Name on the branch | New name | Kind | Reason |
|---|---|---|---|
| `selection` (config key), `SelectionConfig` | `state_information`, `StateInformation` | config key, type | Property 4 of the CASIM paper is "state information" |
| `commitment: {rule, within_s}`, `CommitmentConfig` | `commitment_policy: {_target_, ...}`, one `CommitmentPolicy` subclass per rule | config key, types | The base configuration and the paper use `commitment_policy` with `CommitmentPolicy` classes |
| `TourStates.SCHEDULED` as an uncommitted draft | `TourStates.PLANNED` | enum value | `planned` already meant "part of the plan" in the base `TourStates` |
| `TourStates.RELEASED` | `TourStates.SCHEDULED` | enum value | In the base, a scheduled tour is the one the picker executes |
| `ReleaseTour` | `CommitTour` | command | It is what the `CommitmentPolicy` produces |
| `InsertOrders` + `ReplaceSuffix` | `ReplaceSuffix(tour_id, route, added_order_ids)` | command | `InsertOrders` was never valid without a `ReplaceSuffix` |
| `Reassign` | merged into `Reschedule(tour_id, picker_id, start_time, end_time)` | command | Both change who executes a tour and when |
| decision feature `membership_ops` | removed; `incumbent` and `routing_origin` stay | data-card feature | It duplicated `incumbent` |
| `WaitingInput.single_order_service_times` | removed; `HennWaiting` computes them with its router | field | Algorithm logic must not live in casim |
| decision keys `OBRSPW`, `ATIP` in scenario configs | `OBRSP`, `active_tour_admission` | config keys | Waiting is a subproblem of OBRSP in the paper; a second decision of a class is named after what it does |
| Reconsider causes `"Reconsider:wait"`, `"Reconsider:release_window"` | `ReconsiderReason.WAIT`, `ReconsiderReason.COMMITMENT_WINDOW`, `ReconsiderReason.PERIODIC` | enum | A string must not encode a state |
| `casim/pipelines/solution_ranker.py` | `casim/decision_engine/solution_ranker.py` | module move, name kept | The paper configures the ranking rule with the `DecisionEngine`, and validation must run before ranking |
| `casim/decision_engine/commands.py`, `conversion.py` | `ware_ops_algos/planning/commands.py`, `conversion.py` | module move, names kept | They describe plan changes and import nothing from casim |
| `ware_ops_algos/domain_models/lifecycle.py` (branch) | `ware_ops_algos/planning/lifecycle.py` | module move | The plan types need `Route` and `BatchObject`, which `domain_models` cannot import (O12) |
| `ware_ops_algos/domain_models/planning.py` (branch) | `ware_ops_algos/planning/models.py` for `TourView`, `PlanView`; `casim/domain_objects/sim_domain.py` for `DecisionContext`, `ReconsiderReason`, `DecisionMemory` | module split | Context and memory are decision bookkeeping of the simulator, not plan facts |
| `SimWarehouseDomain`, `DynamicInfo`, `TourPlanningState` in ware_ops_algos (branch) | back to `casim/domain_objects/sim_domain.py` and `tour_model.py` | module move | Their base location; they carry the simulator's context, memory and execution record |
| `casim/interfaces.py` (branch) | `casim/decision_engine/interfaces.py` | module move | Its only consumer is the `DecisionEngine` |

### 6.2 Names that stay

Do not rename these, even where another word seems more precise: `DecisionEngine`, `DecisionEngine.on_trigger(state_snapshot, action=None)` with its base signature, `DecisionEngine.solver_map` and `DecisionEngine.commitment_policies`, `DecisionResult`, `SimulationEngine.run()` returning `(done, state_snapshot)`, `SimulationEngine.step`, `StateAdapter`, `SimWarehouseDomain` and `SimWarehouseDomain.dynamic_warehouse_info`, `DynamicInfo`, both in `casim/domain_objects/sim_domain.py`, `AdmissionTour` and `AdmissionInput`, the `_target_` path `casim.decision_engine.commitment_policies`, the process events `SequencingDone` and `PickListDone`, `TourManager`, `TourPlanningState`, `CommitmentPolicy`, `CommitAllPolicy`, `SchedulingCommitmentPolicy(n_jobs)`, `SolutionRanker`, `CoSySolver`, `Condition.get_decision`, `TourStates`, `ResourceStates`, `OrderStates`, `Accepts` with `none`, `insert`, `rebuild`, `PHYSICAL`, `TOUR_ORDER`, `TOUR_TRANSITIONS`, `RevisionScope`, the config key `scope` with `freeze_from` and `frozen_accepts`, `TourView`, `TourView.accepts`, `PlanView`, `DecisionContext`, `DecisionContext.solve`, the decision features `incumbent` and `routing_origin`, the problem classes `OBRSP`, `OBP`, `OSBP`, the existing waiting algorithm names, `WaitingSolution.action` with `release` and `wait`.

### 6.3 Removed without replacement

- `TourStates.ASSIGNED` (a tour is created with picker and times);
- `Accepts.REVISE`;
- `TimeAttribute.ORDER_DATE`;
- `PickListStates`;
- `RevisionRights`, `PlanEntry`, `OperationalPlan`;
- the `on_violation`, `waiting`, `release_gates` and `solver.output` keys;
- `expected_*` fields on commands;
- the `Wait`, `RecordConsidered` and `RecallPositioning` commands;
- `PlanView.active_candidate_ids`;
- the loose fields of `DynamicInfo` (`time`, `active_tours`, `admission_tours`, `current_picker`, `buffered_batches`, `wait_expired`, `is_break`, `n_staged_pallets`, `active_candidate_ids`, ...); their content now lives in `context`, `plan`, `memory` and `OperationsView`;
- the `positioning_mode` and `positioning_hold` fields;
- `casim/decision_engine/waiting.py`;
- `RoutingDone`, because a decision that ends at routing is rejected (F39).

## 7. Verification

No earlier output is a reference. The base commits count the setup time twice (open point O1), and the walk-or-wait instances were built on a simulator with known faults. Correctness is therefore established from the model, in four layers, each a test suite. The first three decide whether the code is correct; the fourth only explains how it differs from the base.

### 7.1 Exhaustive checks over finite domains

The lifecycle, the `Accepts` levels, the revision scopes, the decision features and the variant table are finite. Check every property over the whole domain, so the test is a proof, not a sample. The checks of the lifecycle, `Accepts`, `PHYSICAL` and the scope go to `ware_ops_algos/tests/verification/`; those of `variant` and the write sets go to `casim/tests/verification/`, because both live in casim.

- `TOUR_TRANSITIONS` only moves forward in `TOUR_ORDER`, every status is reachable from `planned`, and `done` and `cancelled` are the only terminal statuses.
- `PHYSICAL` never increases along `TOUR_ORDER`.
- For every `RevisionScope` (each status × `none`, `insert`) and every set of exposed statuses, `check_scope(state_information, scope)` reports no violation exactly when a(s) ≤ `PHYSICAL`(s) holds for every exposed status, as defined in the design document (Task 3.2).
- A tour's `accepts` level never increases as its status advances, under any admissible scope.
- `variant` is defined or rejects with a named reason for every combination of `incumbent` and `routing_origin`, and its result is an ordered subsequence of the class's subproblems.
- For every pair of decisions, setup rejects overlapping write sets without distinct precedences, and accepts all others.

### 7.2 Plan invariants after every step

`ware_ops_algos.planning.invariants` checks any simulator's plan, which also serves a researcher who brings their own simulator. It reads a `PlanView` taken with full state information (all statuses, all resources) and returns `InvariantViolation` objects, never raising.

`check_plan(plan)` verifies one plan view:
- every order is in exactly one place: the order buffer, a pick list, one tour, or delivered;
- the tours of one picker do not overlap in time;
- every `scheduled`, `pending` and `started` tour has a picker and times;
- a started tour's route visits every unpicked position of its batch, and its occupied boxes fit its cart.

`check_progress(before, after)` verifies two consecutive plan views:
- every status change is in `TOUR_TRANSITIONS`;
- the executed prefix of a started tour is a prefix of its later route, and completed picks stay completed;
- `done` and `cancelled` tours never change;
- simulated time never decreases.

In casim, a test fixture runs `check_plan` and `check_progress` after every `step` and every event of every scenario under `scenarios/` and `tests/config/`. Any violation fails the run with the event and the tour.

Add a Hypothesis test that generates small random instances (layout, orders, arrival times, one or two pickers) and random decision configurations that `casim check` accepts, runs them, and asserts no violation. Extend the `RuleBasedStateMachine` of Task 2.2 from `TourManager` to the commands: random sequences of `CreateTour`, `CommitTour`, `ReplaceSuffix`, `Reschedule`, `CancelTour` and process events, where `validation.check` must reject exactly the commands that would violate an invariant.

### 7.3 Independent KPI recomputation and verification scenarios

KPIs are recomputed in `casim/tests/verification/` from the tour records alone, without the trackers: distance as the sum of route edge lengths in the layout graph, tour duration from the time model, order completion as the end time of the tour that delivers the order. The recomputed values must equal the trackers' values in every run of 7.2.

The time model is the one in the design document (Verification): a `ScheduledJob` occupies its picker from `start_time`, when the setup begins, to `end_time = start_time + tour_setup_time + distance / speed + time_per_pick × picks`. The tour becomes `pending` at max(`start_time`, query time) and starts its route `tour_setup_time` later. Write it down once, in the test module's docstring, and derive every expected value from it. If it is not precise enough for a value you need, stop and ask; do not read the model off the code.

Each verification scenario is a configuration under `casim/tests/verification/` with an instance small enough to derive every expected value by hand. The derivation is written in the test's docstring before the test is run. Required scenarios:

1. One picker, one order: tour start, end, distance and completion. This scenario fails on the base because of O1.
2. Two orders, cart capacity two, `NoWaiting` against `OrderCountWaiting(2)`: which batches form, when each tour starts, the completion times.
3. `HennWaiting` with one batch: `reconsider_at` equals Henn's threshold computed by hand, the tour starts there, and the jobs of that picker are delayed accordingly.
4. Planned tours persist: with `SchedulingCommitmentPolicy(n_jobs=1)`, the second tour stays `planned`, a later arrival is merged into it, and the scheduled tour is unchanged.
5. Revision scope: a solver that changes a `scheduled` tour under `scope: {freeze_from: scheduled, frozen_accepts: none}` has its candidate removed with `ViolationCode.ACCEPTS`, and the next candidate is selected.
6. Admission to a started tour: the executed prefix is unchanged, the new suffix visits the added order, and distance is prefix plus suffix.
7. `WindowCommitmentPolicy(within_s)`: which tours are committed at which times, and the requested `reconsider_at`.
8. Periodic trigger: decision times on the grid from the simulation start, none after the order stream closes.
9. `AnalyticStochasticWaiting`: for given inputs, the waiting time equals the closed-form optimum of the model, computed independently in the test.

### 7.4 The base commits as a diagnostic

The base results at casim `67f9f31` and ware_ops_algos `34bd157`, reproduced in a fresh environment on 2026-10-13, are a point of comparison, not a target:

| Scenario command (from the casim root) | Orders | Tours | Decisions | Makespan | Distance | Mean order completion |
|---|---:|---:|---:|---:|---:|---:|
| `python -m scenarios.scenario_henn_online.experiment_henn viz.launch=false` | 100 | 25 | 26 | 34,702 | 11,603 | 14,584.174 |
| `python -m scenarios.scenario_stochastic_waiting.experiment_stochastic_waiting viz.launch=false` | 4 | 2 | | 87.085 | 76 | 54.835 |
| `python -m scenarios.scenario_walk_or_wait.experiment simulation=explicit_example viz.launch=false` | 6 | 3 | | 167 | 160 | 107.167 |

Write `scripts/compare_with_base.py`, which runs a scenario on the base and on the new code and prints, per decision, the input orders, the candidate jobs, the waiting result and the committed tours, and the first decision where they diverge. For every difference in the three scenarios, the phase report names the cause and shows it with this output. Acceptable causes are the O1 fix, a base defect you demonstrate with an invariant or a verification scenario, or a change the design document makes. Anything else is a bug.

Three causes of the branch's Henn deviation are known (26 tours, 12 decisions, distance 11,402 and six single-order tours at `6cd1705`, against 25 tours and no single-order tour in the base):
1. The waiting algorithm receives one candidate at a time, so its own "more than one batch, start now" rule never fires. The engine duplicates it (F4).
2. With `NewInformationCondition` in the Henn decision, a `Reconsider` commits the old planned tour with its stale start time instead of re-solving (F27).
3. The orders in the snapshot are not in the order of the base order buffer. They must be sorted by `(order_date, order_id)` (F26).

The base Henn configuration uses `CommitAllPolicy`, which committed exactly one job because `HennWaiting` returned only that job. Now planned tours persist and the waiting solution contains the whole schedule, so the port uses `SchedulingCommitmentPolicy(n_jobs=1)`.

After the user has accepted the explained differences, pin the new KPIs of the three scenarios in `tests/test_regression_kpis.py`, marked `@pytest.mark.regression`. These values guard against regressions; they do not establish correctness.

## 8. Work plan

Each phase lists tasks, the findings it fixes, and when it is done. Do the phases in order. After each phase, post a phase report in this format:

```markdown
### Phase N report
- Commits: <hash> <message>, ...
- Findings fixed: F..
- Tests: <passed>/<total> in casim, <passed>/<total> in ware_ops_algos, one process each
- Verification: <exhaustive checks, invariant runs, KPI recomputation, verification scenarios: passed/total>
- Differences from the base: <table with cause and evidence, from Phase 5 on>
- Deviations from this plan: <none | what and why>
- Open questions for the user: <none | list>
```

### Phase 0: baseline and a trustworthy test suite

- [ ] **0.1** Reproduce the table of Section 7.4 on the base commits in a separate checkout, so the comparison script has a fixed point. If any number differs, stop and report.
- [ ] **0.2** Make the casim tests independent of their order (F25). casim depends on the unpinned cosy-luigi nightly (`pyproject.toml`), and the behaviour differs between nightlies. The build installed in the review environment on 2026-10-13 (`0.0.1.dev18`) kept Luigi's task instance cache, so 12 of 41 tests failed in one process, and it called `output()` inside `CoSyLuigiTask.__init__` before `BaseComponent` set `pipeline_params`. The build the agent installed (`0.0.1.dev7`, from April) calls `Register.disable_instance_cache()` in `CoSyLuigiRepo.__init__`. Which build casim gets depends on when it is installed, which is why the dependency is to be pinned.
    - Add a regression test that builds two `CoSySolver`s with different cache directories in one process and checks that each runs its own pipelines. This test is required whatever the installed version does.
    - Add `clear_instance_cache()` or a `BaseComponent` fix only if that test fails with the installed version. Do not add defensive code for a problem the dependency has already solved.
    - Report the installed cosy-luigi version and, if you can find it, its commit. Pinning the dependency is the user's decision; propose it in the report.

Done when `python -m pytest -q` passes in one process in both repositories.

### Phase 1: remove dead and duplicate code

- [ ] **1.1** Delete `domain_models/operational/` (`RevisionRights`, `PlanEntry`, `OperationalPlan`) and its test (F8). `TourPlanningState` moves in Phase 2; until then, keep it importable from its current place.
- [ ] **1.2** Delete `algorithms/pipeline.py` (unused DAG executor) (F8).
- [ ] **1.3** Remove `required_capabilities` and `_capability_rejections` from `DomainAlgorithmMapper`. Also remove the `capabilities:` entries added in `90c9364` to `remaining_route_admission.yaml`, `reroutable_sshape_routing.yaml` and `reroutable_walk_or_wait_sshape_routing.yaml` (F8). Keep the `capabilities` field of `AlgorithmCard` itself: it comes from the user's commit `badfbb3`.
- [ ] **1.4** Make `DomainAlgorithmMapper.filter` use `rejection_reasons` as its only implementation of the requirement checks, so there is one implementation, not two (F30). The existing mapper tests and CASOP behaviour must not change.
- [ ] **1.5** Delete the copied design document in `casim/docs` (F29).

Done when nothing references the deleted code and the tests pass.

### Phase 2: lifecycle and typed domain models

- [ ] **2.1** Create the package `ware_ops_algos/planning/` and move the branch's `domain_models/lifecycle.py` to `planning/lifecycle.py` (F14, F23):
    - `TourStates` = `planned, scheduled, pending, started, done, cancelled`. `planned` is uncommitted and revisable, `scheduled` is committed.
    - `Accepts` = `none, insert, rebuild`, ordered in that sequence, with a helper that compares levels.
    - `PHYSICAL` = planned/scheduled → `rebuild`, pending/started → `insert`, done/cancelled → `none`.
    - `TOUR_ORDER` and `TOUR_TRANSITIONS` as in the design document.
    - `TimeAttribute` = `due_date, start_time`.
    - Delete `PickListStates`. Keep `OrderStates` only if it is used to derive the order status from the buffers and tours, and never store it.
- [ ] **2.2** `TourManager.advance(tour_id, to)` is the only code that writes `TourPlanningState.status`. It raises `ValueError` for a transition not in `TOUR_TRANSITIONS`.
    - `CreateTour` yields a `planned` tour with picker and times; `CommitTour` advances it to `scheduled`. Remove `assign_tour`/`schedule_tour` as status writers and keep them as plain setters if still needed.
    - Add an `ast` test that fails on any `.status =` assignment outside `tour_manager.py`.
    - Add a Hypothesis `RuleBasedStateMachine` test over `TourManager` that checks statuses never move backward. Add `hypothesis` to the dev dependencies.
- [ ] **2.3** Move `TourPlanningState` back to `casim/domain_objects/tour_model.py` (its location at `67f9f31`) without the positioning fields (F15). It is simulator execution state. No ware_ops_algos module may import it. `TourStates` is now imported from `ware_ops_algos.planning.lifecycle`.
- [ ] **2.4** Split the branch's `domain_models/planning.py` (F16, F17, F18):
    - `ware_ops_algos/planning/models.py` gets `TourView` (flat, with `accepts`, the execution fields of started tours as optional fields, and `holding`; no positioning fields) and `PlanView` with exactly `time`, `order_buffer` (sorted by `(order_date, order_id)`), `pick_list_buffer`, `tours` and `available_from`: no `active_candidate_ids`, and no `resources`, because the snapshot carries them. The names follow the paper's order buffer and pick-list buffer, so they no longer clash with `SimWarehouseDomain.orders`, which holds every order the pipelines may place;
    - `casim/domain_objects/sim_domain.py` gets `DecisionContext` (with `ReconsiderReason` and `solve`, without `time` and `version`: the time is `plan.time`, and stale `Reconsider` events are checked against casim's state) and `DecisionMemory`, because they are the simulator's decision bookkeeping;
    - delete `domain_models/planning.py`.
- [ ] **2.5** Move `SimWarehouseDomain` and `DynamicInfo` back to `casim/domain_objects/sim_domain.py`, their location at `67f9f31`, and restore `base_domain.py` and `warehouse_info/models.py` in ware_ops_algos to `34bd157`. `DynamicInfo` gets exactly the fields `context: DecisionContext`, `plan: PlanView`, `memory: DecisionMemory`, all typed and required (F17). Remove `SimWarehouseDomain.operations` and `SimWarehouseDomain.memory`, and update the scenario loaders that construct `DynamicInfo`. Delete the branch's re-export of the `ActiveTour*` types from `sim_domain.py`; import them from `ware_ops_algos.algorithms`.
- [ ] **2.6** Keep `AdmissionTour` and `AdmissionInput` as at `34bd157`, and remove `TourView.as_admission_tour` (F16). The admission component in `casim/pipelines/subproblems/admission.py` builds each `AdmissionTour` from a `TourView`. Algorithm inputs must not depend on the plan types; that is why the two types stay separate. Verification scenario 6 covers admission.
- [ ] **2.7** The generic `StateAdapter(state_information, scope)` builds the typed snapshot: the plan view contains only what the state information exposes, and every `TourView.accepts` is the level the scope gives that status. It may read `State` only through manager methods, so add the methods you need to the managers instead of reaching into private attributes (F22). It replaces the per-class adapters of the base (`HennWaitingAdapter`, ...).
- [ ] **2.8** `OperationsView` in `casim/state/views.py` keeps `is_break` and `n_staged_pallets` and loses `extra` (F17).

- [ ] **2.9** Write the exhaustive checks of Section 7.1 for the lifecycle, `Accepts` and `PHYSICAL`, and `planning/invariants.py` with its tests on hand-built plan views (Section 7.2).

Done when the tests pass and the AST, state-machine and exhaustive tests exist.

### Phase 3: the planning package and the decision engine

- [ ] **3.1** Moves, each with `git mv` and the module name kept:
    - `casim/decision_engine/commands.py` and `conversion.py` to `ware_ops_algos/planning/`;
    - `casim/pipelines/solution_ranker.py` to `casim/decision_engine/solution_ranker.py`;
    - `casim/interfaces.py` to `casim/decision_engine/interfaces.py`.

    The `DecisionEngine` and the commitment policies stay in `casim/decision_engine/`. The tasks below change the content.
- [ ] **3.2** `planning/scope.py` (F2, F10, F17):
    - frozen dataclasses `Window`, `StateInformation` and `RevisionScope` with their own invariants in `__post_init__` (`RevisionScope.frozen_accepts` is not `rebuild`, a window has a positive horizon);
    - `RevisionScope.accepts(status) -> Accepts` computes a(s) and is its only implementation: the `StateAdapter` uses it for `TourView.accepts`, `check_scope` for admissibility and `decision_features` for the features. No other module computes a level from a scope;
    - `planning/validation.py` has `check_scope(state_information, scope)`, which returns the exposed statuses that would get more than `PHYSICAL` allows. It needs both objects, so the schema's `DecisionConfig` validator and `StateAdapter.__init__` call it and raise with the status, the level and the bound;
    - set `__pydantic_config__ = {"extra": "forbid"}` on each dataclass, as a plain dict with no pydantic import, so casim's schema rejects unknown keys;
    - there is no object that bundles a whole decision. Each engine receives its own part, as in the paper: the `SimulationEngine` gets triggers, conditions, precedence and a `StateAdapter(state_information, scope)`; the `DecisionEngine` gets the solver, the `SolutionRanker` and the commitment policy. Only the configuration file and `DecisionSetup` hold a decision as a whole.
- [ ] **3.3** `casim/decision_engine/commitment_policies.py` (F10, F27):
    - `CommitmentPolicy` is abstract with `apply(commands, state_snapshot) -> tuple[tuple[CommitTour, ...], float | None]`: the commits for the planned tours that exist after `commands`, and an optional reconsider time. Keep the base class names `CommitAllPolicy` and `SchedulingCommitmentPolicy(n_jobs)`, and add `WindowCommitmentPolicy(within_s)`;
    - every policy considers only planned tours of pickers in the state information, so a decision that sees only idle pickers never commits a busy picker's next tour (this is what state information means, and how the base Henn decision behaved);
    - no policy commits anything while the decision waits;
    - `WindowCommitmentPolicy` compares a tour's ready time, its `start_time` (the begin of its setup, Task 4.8), with `now + within_s`, and returns the time at which the next tour becomes ready, minus `within_s`.
- [ ] **3.4** `planning/decision_features.py`: `decision_features(state_information, scope)` returns `incumbent` and `routing_origin` as the design document defines (F2). Setup writes them into the data card's `decision_features` section. Cards require them like any other feature.
- [ ] **3.5** Taxonomies (F2, F33, F40):
    - restore `ware_ops_algos/taxonomy/taxonomy.py` to `34bd157` exactly; the branch changed the CASOP entries for `SPRP` and `OBRP` (F40);
    - `casim/pipelines/taxonomy.py` keeps the online problem classes of `67f9f31` without `ATIP`, `RORSP` and `OBRSPW`, with OBRSP's subproblems as in the design document, in pipeline order, and gets `variant(problem_class, features) -> tuple[str, ...]` with exactly the rules of the design document. For `routing_origin: live` with `incumbent: true`, `variant` raises a `ValueError` that names open point O2 (F33);
    - unifying the two taxonomies is open point O14; do not attempt it.
- [ ] **3.6** `planning/commands.py`: the six commands of the design document, without `expected_*` fields (F24, F35, F36), plus `Violation` and `ViolationCode` (`state_information`, `accepts`, `physical`, `executed_prefix`, `membership`). `CommandViolation` is raised by casim's `apply` and stays in casim.
- [ ] **3.7** `planning/conversion.py`: `to_commands(solution, plan)` with `functools.singledispatch` over the solution types. Matching a candidate tour to an existing planned tour keeps its rule (largest order overlap, then smallest tour id, then candidate index), written as a named helper, not one long sort key.
- [ ] **3.8** `planning/validation.py`: a pure `check(commands, plan)`. The plan view already contains only what the state information exposes, and each `TourView.accepts` carries the scope, so nothing else is needed. It verifies:
    - that commands touch only tours in the state information;
    - each tour's `accepts` level;
    - `PHYSICAL`;
    - the executed prefix and occupied boxes of started tours;
    - that `ReplaceSuffix` only adds orders and that `added_order_ids` match the route.

    It returns typed violations and never raises.
- [ ] **3.9** `casim/decision_engine/solution_ranker.py` (F6):
    - `SolutionRanker(objective, change_penalty)` scores feasible candidates only, as `objective + change_penalty * changed(commands)`;
    - `changed` counts distinct existing tours that are cancelled, revised or rescheduled. New tours do not count;
    - delete `CHANGE_WEIGHTS` and the problem-class branches of `select_best`;
    - `CoSySolver` no longer receives a ranker; it returns all candidates;
    - when a decision whose variant ends at batching has more than one candidate, `casim check` reports an error, because there is no objective to rank batches by. That replaces the silent "last key wins", which is from the user's base code: say so in the report.
- [ ] **3.10** `casim/decision_engine/interfaces.py`: `class DecisionSolver(Protocol): def solve(self, state_snapshot: SimWarehouseDomain) -> dict[str, AlgorithmSolution]` (F19). Candidate names must be stable across calls (an RL action indexes them).
- [ ] **3.11** `casim/decision_engine/decision_engine.py`: `DecisionEngine(solver_map: dict[str, DecisionSolver], solution_rankers: dict[str, SolutionRanker], commitment_policies: dict[str, CommitmentPolicy])`, keyed by decision name, as the base keys `solver_map` and `commitment_policies` by problem class. The constructor raises if the keys differ, except that a decision without planned tours has no commitment policy. `on_trigger(state_snapshot, action=None) -> DecisionResult` keeps the base signature and reads the decision's name from `state_snapshot.dynamic_warehouse_info.context.decision`. Follow the steps of the design document: solve if `context.solve`, convert with `to_commands`, filter with `validation.check`, rank or take the action, waiting from a `WaitingSolution`, commitment policy, result. `DecisionResult` has exactly the fields of the design document. The engine imports nothing from `casim.state`, `casim.simulation_engine` or `casim.events`, and contains no `isinstance` on solution types; those live in conversion.
- [ ] **3.12** casim configuration (F2, F21):
    - `config/schema.py` defines `DecisionConfig` with `problem_class` (must be a key of casim's `TAXONOMY`), `state_information`, `scope`, `commitment_policy` (a `_target_` naming a `CommitmentPolicy` subclass, with its parameters; required when the decision may create or see planned tours, omitted otherwise), `change_penalty`, `precedence: int | None`, `solver` (CoSy configuration without `output` and without a ranker), `triggers` and `conditions`, and `DecisionDefinitions` for the whole file;
    - the dataclasses of `ware_ops_algos.planning.scope` and `casim.decision_engine` are used as field types, so no field is declared twice;
    - move `load_and_flatten_data_card` into `casim/config/data_card.py`; `casim check` must not import `scenarios`.
- [ ] **3.13** `casim/setup.py` (F2, F3):
    - delete `_derive_problem_class`, the waiting-name table, the condition-name rule and the filtered mapper reasons;
    - `resolve` builds, per decision, a `DecisionSetup` with the decision features, the decision's data card (`problem_class`, `decision_features`), the solver, the `StateAdapter`, the conditions and the triggers;
    - it rejects overlapping write sets that lack distinct precedences;
    - it rejects a decision whose variant ends at `routing`, because pickers execute only scheduled tours and such a decision creates none. Delete the `RoutingDone` emission in the engine (F39);
    - it reports card exclusions with their reasons;
    - `DecisionSetup` keeps only what `casim check` reports and what building the two engines needs; the engines receive their parts, never a `DecisionSetup`.
- [ ] **3.14** `SimulationEngine` (F12, F13, F18, F20):
    - all collaborators are constructor parameters;
    - `triggers_map: dict[type[Event], tuple[str, ...]]`, so one event can make several decisions due, ordered by precedence (lower first, unset after, ties by name);
    - each decision runs at most once per timestamp;
    - `run()` returns `(done, state_snapshot)` as in the base, with the decision name in `dynamic_warehouse_info.context.decision` and `solve` set by the conditions; `step(result, state_snapshot)` applies;
    - `Reconsider` carries a `ReconsiderReason` and the decision's version, not a string;
    - remove `on_violation`, the `repair` path and every call from the `SimulationEngine` into the `DecisionEngine`.
- [ ] **3.15** `casim/state/apply.py` (F22):
    - `check(state, commands)` verifies what only live state knows (order still in the buffer, tour exists, picker exists) and raises `CommandViolation` on the first violation, because it is a programming error once `validation.check` has removed infeasible candidates;
    - `apply(state, commands)` then applies through manager methods and returns the process events: `SequencingDone` for a committed tour (its handler sends `PickerTourQuery` to an idle picker, as at `67f9f31`), `PickListDone` for a created pick list, and `TravelEvent` for a running tour whose suffix was replaced;
    - process event handlers create no tours, pick lists or orders and clear no buffers; they only continue the operational process, as `SequencingDone` marks the picker occupied and sends `PickerTourQuery` today. Delete `RoutingDone` (F39);
    - no `deepcopy`, no `__dict__`, no function-local imports.
- [ ] **3.16** `DecisionResult.waiting`, `reconsider_at` and `considered_active_orders` are stored per decision in `State` and come back in `DecisionMemory` (F35).
- [ ] **3.17** `BaseEnv` and the experiment scripts use the corrected engine, and nothing imports the moved modules from their old paths.

Done when:
- the tests pass;
- `casim schema` is regenerated and committed;
- `casim check` runs on every scenario configuration without importing `scenarios`;
- the import-boundary test (5.1) passes.

### Phase 3b: OOBP baseline, feasibility (after Phase 3, before Phase 4)

The external reference for the Henn instances is the OOBP benchmark of Gil-Borrás, Pardo, Alonso-Ayuso and Duarte (2020), from which `scenario_henn_online/data` was taken. This phase only investigates; it changes no engine or algorithm code.

- [ ] **3b.1** Download `https://grafo.etsii.urjc.es/optsicom/oobp/oobp-files/experiments.xlsx` and the instance archive `oobp_instances.zip` from `https://grafo.etsii.urjc.es/optsicom/oobp`. Commit a script `scenarios/scenario_henn_online/scripts/oobp_reference.py` that parses the spreadsheet into `data/oobp_reference.csv` (instance, objective, GRASP+VND, ILS, best known), and commit that CSV. Check that the archive's instance and arrival files are byte-identical to the ones in `data/`, and report any that differ.
- [ ] **3b.2** Write `docs/oobp_assumptions.md`: one table row per assumption of the paper (Table 4 and Sections 2 and 4: arrival horizon, speed 48 LU/min, 6 items/min, setup 3 min, S-shape, start as soon as the picker is available, batch selection by oldest order, objectives maximum completion time and maximum turnover time, real-time solving, depot position, capacity unit), with the casim configuration or code that maps it, and a verdict: mapped, mappable with a configuration change, needs code, or not reproducible. Known facts to confirm: the Henn loader sets `speed=1` (1 LU/s) where the paper gives 0.8 LU/s; the setup time is counted twice until Task 4.8; the paper's Table 3 says "Center" for the depot while the instance header says `DEPOT_AISLE : 0`, `DEPOT_LOCATION : bottom`. Do not resolve a conflict between paper, website and instance files yourself; list it.
- [ ] **3b.3** State in the same document what can be compared, and how. ILS and GRASP+VND ran in real time with a time budget, and casim solves in zero simulated time, so equality with their columns is not expected. Propose the comparison: for example, a deterministic batching rule under the paper's protocol must not fall below the best known value, and an available ILS in ware_ops_algos should land near the ILS column. Say which ware_ops_algos algorithms correspond to the paper's batching, selection and routing, and which are missing.

Report the result with the Phase 3 report and wait. The runs themselves come in Task 6.0, after the verification gate.

### Phase 3c: consolidation (after Phase 3b, before Phase 4)

This phase changes no behaviour. It checks that Phases 0 to 3b are complete against the current plan and that the moves of Phases 2 and 3 left nothing behind.

- [ ] **3c.1** Re-read this file as it stands now; it was edited while you worked. For every task of Phases 0 to 3b, every Appendix A row whose phase column contains 0 to 3, and every rule in Sections 5, 6 and 7.1 to 7.2, write one row in `docs/revision_scopes_status.md`: task, status (done, partial, not done, moved to a later task), and the evidence (commit, test name or file). Fix what is partial or not done, unless it belongs to Phase 4 by the plan.
- [ ] **3c.2** Search both repositories for leftovers: every name in Sections 6.1 (old names) and 6.3, old import paths of the modules moved in Phases 2 and 3, and the modules those phases replaced. This covers code, tests, configuration YAML, scripts, notebooks, docs, the CLI and the generated schema. Run `ruff check --select F401,F811,F841` and `vulture` (confidence 80) on the changed packages and remove what is dead. List every file under `scripts/`, `scenarios/*/scripts/` and `docs/` that you did not touch in Phases 2 to 3b and say whether it still runs or still holds; delete or fix it. Report untracked files instead of deleting them.
- [ ] **3c.3** The local instance files come from the user's adoption of the Hessler instances, which may have changed the originals; the archive files are what Gil-Borrás et al. used. Store the archive instance and arrival files unchanged in `scenarios/scenario_henn_online/data/oobp/`, with a short `SOURCE.md` (URL, download date, checksums), and keep the local files where they are. Compare both versions semantically: parse them and report per file which fields differ (header, depot, capacities, item positions, order lines). Make sure the loader reads the archive format; if it cannot, say what is missing.

Report with a short Phase 3c report and continue with Phase 4. The rule for every reproduction is that it uses the data of the source it reproduces: the OOBP comparison uses `data/oobp/` and the depot those files and the benchmark page give, not the local files.

### Phase 4: algorithms own their logic

- [ ] **4.0** Add the missing subproblem base classes, following `Batching` and `Routing`:
    - `Waiting(Algorithm[WaitingInput, WaitingSolution], ABC)` in `waiting/waiting.py`; `NoWaiting`, `OrderCountWaiting`, `HennWaiting` and `AnalyticStochasticWaiting` subclass it;
    - `Admission(Algorithm[AdmissionInput, AdmissionSolution], ABC)` in `admission/admission.py`; `RemainingRouteAdmission` and `CenterPositionAdmission` subclass it;
    - `Positioning(Algorithm[PositioningInput, PositioningSolution], ABC)` in `positioning/positioning.py`. It takes over the shared empty-cart mechanics of the branch's private `_EmptyCartPositioning` in its `_run` and declares `@abstractmethod _route(self, layout: LayoutData) -> Route`; delete `_EmptyCartPositioning`. `AislePatrolPositioning` and `CenterRelocationPositioning` subclass `Positioning` directly in `positioning/empty_cart.py` and only implement `_route`;
    - remove the `mode` class attribute and `PositioningSolution.modes` (F9): a positioning tour is identified by its empty batch, not by a string;
    - export each base from its package's `__init__.py` and type the CoSy components against the base, not against a concrete algorithm.
    - Correction after review (F48): the split into several modules per subproblem (`waiting.py` and `policies.py`, `positioning.py` and `empty_cart.py`, three admission modules) contradicts the convention of `main`, where each subproblem has one module with the base class and its implementations (`batching/batching.py`, `routing/routing.py`). Task 4.12 merges them.
- [ ] **4.1** Restore waiting as a pipeline stage (F4, F10, F11):
    - the waiting algorithms keep their `_run` and become `Waiting` subclasses (Task 4.0); they receive the whole schedule in `WaitingInput.candidates` and return all of it in `WaitingSolution.jobs`;
    - when an algorithm waits until a known time, it delays every job of that picker, not only the first, so the schedule stays consistent;
    - in casim, restore `pipelines/subproblems/waiting.py` from `67f9f31` and the waiting stage in the template, adapted to the typed snapshot;
    - waiting is selected through the cosy repository and the cards, as before.
- [ ] **4.2** `HennWaiting` receives the configured `Routing` instance in its constructor and computes the single-order service times itself. `HennWaitingNode` passes the router of its upstream routing stage, as `HennWaitingNode._single_order_service_times` did at `67f9f31`. Delete `_single_order_service_times` from casim (F7).
- [ ] **4.3** `AnalyticStochasticWaiting` (F5):
    - restore the original `ValueError` for anything other than exactly q−1 known orders;
    - take q from the picker's cart (`n_boxes`) instead of `target_batch_size_orders`;
    - the card already encodes the static preconditions (exponential single-line order stream, one line per order, amount 1); add the cart constraints the model needs (`box_can_mix_orders: false`, `dimensions_type: [orders]`, `capacities: [1]`);
    - the dynamic precondition (q−1 orders in the order buffer) is ensured by the decision's `NbrOrdersCondition(threshold=3, allow_when_done=true)`, as at `67f9f31` (open point O8);
    - verification scenario 9 checks the computed waiting time against the closed form.
- [ ] **4.4** Positioning (F9, F32):
    - move `algorithms/positioning.py` to `algorithms/positioning/`;
    - a positioning tour is an ordinary tour with an empty batch whose route ends at a hold node. `TourView.holding` says that the picker has reached the end and holds;
    - delete every `"center_relocation"` string in casim and `RecallPositioning`;
    - the return at input closure is a `ReplaceSuffix` that the `active_tour_admission` decision's active-tour router produces when triggered by `OrderStreamClosed`;
    - first check whether `ReroutableSShapeRouting` routes a holding empty tour to the depot. If it does not, move the shortest-path code from `apply.py` into a `ReturnToDepotRouting(Routing)` algorithm with a card, `problem_type: active_tour_routing`;
    - run the walk-or-wait center example under the invariant fixture; compare it with its current result (3 orders, 2 tours, makespan 91, distance 88) through the comparison script and explain every difference.
- [ ] **4.5** `IncumbentStartBatching` receives a `Routing` instance instead of a `route_cost` closure. `IncumbentBatchingNode` passes the configured router. Remove the hard-coded `SShapeRouting` and `resources[0].pick_cart`: take the cart from the resources and reject heterogeneous carts through the card (F7, open point O11). Correction after review (F48): the hand-written `routing_class` in the card's `implementation` bypasses the configuration mechanism of the CASOP paper (Section 4.2), so the router's requirements are not merged; Task 4.12 replaces it.
- [ ] **4.6** Add a test that every class with an algorithm card subclasses the base class of its card's `problem_type`, and fails with the class and the expected base otherwise.
- [ ] **4.7** Search casim (outside `pipelines/subproblems/`) for algorithm names, policy names and positioning mode strings. The result must be empty. Add the search as a test (F9, F3).

- [ ] **4.8** Count the setup time once (F28, O1), as the design document's time model defines:
    - in ware_ops_algos, `earliest_free_assignment` sets `start_time = max(release_time, available_from)` and `end_time = start_time + tour_setup_time + processing_time`; planned end times stay exactly as before, only `start_time` moves earlier by the setup. Add a test that the end times of every scheduler on the CASOP test instances are unchanged, and check whether anything in CASOP (ware_ops_pipes) reads `start_time`; if it does, stop and ask;
    - pass `available_from` to the scheduler as a constructor parameter; the branch sets `scheduler.available_from` after construction in `problem_based_template.py`, which Section 3.1 forbids;
    - in casim (superseded by `Dispatch` in Task 4.14, which keeps this timing), `PickerTourQuery` keeps adding the setup once: the tour becomes `pending` at max(`start_time`, t) and `TourStart` follows `tour_setup_time` later. If `start_time` lies after t, the picker stays reserved and `PickerTourQuery` is repeated at `start_time` instead of setting the tour pending early;
    - `HennWaiting` and the single-order service times keep adding the setup, as in Henn's model;
    - verification scenario 1 shows the fix; report its effect on the three scenarios of Section 7.4 with the comparison script.
- [ ] **4.9** Make the dock a domain object instead of an attribute that a scenario hook attaches to `State` (F42). Today `make_dock_manager_hook` in `scenario_grocery_retailer` and `scenario_ijpe` sets `sim.state.dock_manager`, and core code branches on `hasattr(state, "dock_manager")` in `TourEnd`, `TruckDeparture`, `TruckArrival`, the `StateAdapter` and the logger. The core already depends on the concept, so it must own it. Keep the existing names.
    - In ware_ops_algos, add `Dock(capacity: int | None = None, pallets_per_tour: int = 1)` as an optional, typed part of the warehouse in `BaseWarehouseDomain` (default `None`), so that CASOP and its data cards are unaffected. The loaders of both scenarios build it from `dock.K_dock`.
    - In casim, move `DockManager` from the two hook files to `casim/state/` and create it in `State` from the domain, always: without a `Dock`, the capacity is unbounded and staging is only counted. `TourEnd` stages `pallets_per_tour` instead of the literal 1. Delete both `make_dock_manager_hook` functions and every `hasattr(state, "dock_manager")`.
    - `OperationsView` gets `dock_capacity: int | None` beside `n_staged_pallets`. `DockCapacityCondition` reads the capacity from the view and loses its own `threshold`, since no configuration sets it; `casim check` rejects the condition when the warehouse has no `Dock`.
    - `TruckDeparture` releases pallets and wakes idle pickers whether or not a dock exists, because only truck scenarios have this event; today the wake-up depends on the dock by accident. Keep the wake-up as it is (it emits `PickerArrival` for idle pickers) so that KPIs stay equal, and report which decisions of the two scenarios it triggers through `triggers_map`.
    - Behaviour must not change. Neither scenario can run today: the grocery loader does not exist (F58) and the IJPE data is not in the repository until Task 6b.6. Compare instead on a synthetic dock fixture: a small warehouse with a `Dock`, two pickers, a truck schedule with and without capacity, run with the same seeds before and after the change; compare the event trace, KPIs, dock fill over time and decision counts. Every difference is a defect of this task. Do not add or modify scenario data.
    - Do not change two things that look wrong; list them for the user instead: `TruckDeparture` releases nothing when `capacity` is `None`, which the IJPE configurations use, so the dock fill only grows; and the grocery truck capacity counts orders while staging counts pallets per tour.
- [ ] **4.10** One algorithm, one card (F43). `OrderCountWaiting` has four cards (`wait_for_one.yaml` to `wait_for_four.yaml`) and four CoSy components (`WaitForOne` to `WaitForFour`) that differ only in `min_orders`. A card describes an algorithm's problem type and requirements; a parameter value is configuration, and k is one of the policy parameters θ that are tuned.
    - Keep one card, `order_count_waiting.yaml`, and one component, `OrderCountWaitingNode`, whose `min_orders` is a `luigi.IntParameter` set from the configuration. First check that cosy-luigi passes a component parameter from `luigi_cfg` (or the repo configuration) into the synthesized task; if it does not, keep the four subclasses of `OrderCountWaitingNode`, delete their cards, and make card matching in `algorithm_applicability_constraint` and `casim/setup.py` use the class hierarchy (a card for a component class applies to its subclasses) instead of `component_name == __name__`.
    - `scenario_walk_or_wait/config/policy/wait_1.yaml` to `wait_4.yaml` set `min_orders` instead of naming a class. The KPIs of `wait_1` to `wait_4` must not change.
    - Check every other card in ware_ops_algos for the same pattern (cards that differ only in a constructor argument) and report them.
    - Correction after review (F48): the CASOP paper already defines this case. A value option on the card, resolved through `configs.yaml`, is the mechanism of `SeedBatching`. Task 4.12 replaces the subclasses `WaitForOne` to `WaitForFour` and the aliases with it.
- [ ] **4.11** Phase 4 follow-ups from the review of `9458945`..`640b5ce` (F45):
    - the waiting aliases and the incumbent card lookup are replaced by Task 4.12;
    - the test of re-queued picker queries is replaced by the dispatch tests of Task 4.14;
    - `AdmittedTourBatch` decides the return at closure itself (`context.input_closed and view.holding`). Move that rule into the admission algorithm, so that the batch provider only assembles batches from `AdmissionSolution` and makes no choice (F52);
    - fix the defects of the CoSy template found in the review of `640b5ce` (F54): remove sequencing as a separate stage: `AbstractSequencing`, `SequencingPlanCandidate`, `subproblems/sequencing.py`, the `sequencing` entry of `TAXONOMY["OBRSP"]` and its entries in the runner's maps. It is dead code: ware_ops_algos has had no `SPTSequencer` or `EDDSequencer` since `6a9a3d4` (11 June), so `subproblems/sequencing.py` cannot even be imported, `AbstractSequencing` uses an unimported `SequencingInput`, no card has `problem_type: sequencing`, and no repository file names a sequencing component since the Henn scenario switched to `SPTScheduler` in `7e47bbe`. Sequencing in the sense of OBRSP is the scheduling subproblem, as ware_ops_pipes already maps `sequencing_sol` to scheduling;
    - rename the process event `PickListDone`, emitted for every `CreatePickList`, to `PickListCreated`: it reports the effect of a command, not the end of a pipeline stage, like the other event names (`OrderArrival`, `TourEnd`, `PickerIdle`). A pure rename; no configuration names it. `SequencingDone` is not renamed, because Task 4.14 replaces it;
    - `decisions/rolling_horizon.yaml` names `problem_class: rolling`, which `TAXONOMY` does not define, and neither it nor `cosy_repo/rolling_horizon.yaml` is referenced by any scenario configuration. Do not delete them: Task 6.1 ports `rolling_horizon.yaml`, and the undefined problem class is part of that port. Until then `casim check` must report the file as invalid rather than ignore it;
    - restore the three Hydra override files in `scenarios/scenario_henn_online/config/scenario/` (F56). At `6cd1705` they were `# @package _global_` overrides of the decision `OBRSPW` or `OBRSP` for the waiting-strategy sweep (`sweep/waiting_strategy_sweep.yaml`, plots in `scripts/plots/waiting_strategy/`): Henn waiting, an order window with k = 5, and both combined. The port in `d1f1447` dropped the package directive, read the group key `decisions` as a decision name and replaced the content by an empty decision with `problem_class: decisions`, so the sweep no longer runs. Translate each file from `6cd1705` into the new keys: the `waiting:` key becomes a component of the cosy repository (Task 4.12), `commitment: {rule: release_all}` becomes `CommitAllPolicy`, `release_next_per_picker` becomes `SchedulingCommitmentPolicy` with `n_jobs: 1`, `selection` becomes `state_information`; keep `# @package _global_`. Run the sweep for one instance and compare with `scripts/plots/waiting_strategy/summary.csv`; explain every difference;
    - fix the two other stale references to the old schema (F56): `scenario_grocery_retailer/config/sweep/cosy_repo_sweep.yaml` overrides `scenario.simulation_engine.problems.ORSP.conditions.2.threshold`, a path that no longer exists, and the Henn README tells the user to override `decisions.OBRSPW.solver.solution_ranker.change_penalty`, which the schema rejects (`change_penalty` is a decision key now). The stochastic waiting README claims that the loader checks the three-order gate; it does not. Correct all three;
    - remove the committed `scenarios/scenario_henn_online/outputs/` and add it to `.gitignore`;
    - update `docs/revision_scopes_status.md` for every task of Phase 4.

- [ ] **4.12** Align ware_ops_algos with the CASOP conventions (F46 to F49). Read `docs/ware_ops_algos_review.md` first.
    - Parameterized algorithms use the configuration mechanism of `algorithm_cards.py` and `configs.yaml`. `IncumbentStartBatching` declares a component option `routing_class: {problem_type: routing}`, and `configs.yaml` gets the entry `IncumbentStartBatchingSShape` with `routing_class: SShape`, so that the SShape requirements are merged. `OrderCountWaiting` declares a value option `min_orders` with the values 1 to 5 (1 to 4 for walk-or-wait, 5 for the order-window variants of F56), and `configs.yaml` gets one entry per value. The hand-written `routing_class` in the card, the subclasses `WaitForOne` to `WaitForFour`, the aliases `WaitFor0` to `WaitFor4` and `WaitFor${policy.min_orders}` disappear. casim generates components for resolved cards, as `generate_configured_components.py` does in ware_ops_pipes; reuse its approach rather than inventing one. `wait_0` stays `NoWaitingNode`. The KPIs of `wait_0` to `wait_4` and of the Henn scenarios must not change.
    - CASOP applicability must equal that of `main`. `walk_or_wait_sshape_routing` (routing) and `fifo_scheduler` (scheduling) are currently applicable in CASOP problem classes although ware_ops_pipes has no component for them. Make them inapplicable there, for example by a requirement CASOP data cards never satisfy or by a CASIM-only problem type, and say which you chose. Prove equality with the applicability lock of Task 5.0, generated from `main`.
    - Module layout as on `main`: `waiting/waiting.py` holds `Waiting` and its four implementations, `admission/admission.py` holds `Admission` and both implementations, `positioning/positioning.py` holds `Positioning` and both implementations; the extra modules go.
    - Replace the private mixin `_ActiveSShapeRerouting` by a public base `ReroutableRouting(Routing)` that does not assign attributes inside `_run`; move `WalkOrWaitSShapeRouting` and its rerouting variant into `routing/walk_or_wait.py`.
    - Restore the docstrings of `DomainAlgorithmMapper` in the style of `main`, now documenting `rejection_reasons` and the rule that a feature whose value is `None` counts as absent.
    - Move `OrderSplittingSolution` into `algorithm_interfaces.py`. Report whether CASOP uses order splitting; give it a card or propose its removal.
    - Do not change, but list in the report, the four points that wait for the user's decision (Section 9): the removed `due_date` requirement of `spt_scheduling.yaml`, the new semantics of `Resources.get_features()`, the per-picker processing time in `earliest_free_assignment`, and the objective vocabulary of the cards.
- [ ] **4.13** Consolidate the types under the state model of the design document (F50). Mechanical merges only; behaviour must not change.
    - `planning/models.py`: the execution fields of a started tour (`cursor`, `pinned_prefix`, `remaining_route`, `routing_origin`, `occupied_bins`, `picking_until`, `holding`) form one frozen `TourProgress`; `TourView.progress: TourProgress | None`.
    - Admission: delete `AdmissionTour` and `to_admission_tour`; `AdmissionInput.tours` holds `TourView`s with `progress`. Delete `ActiveTourBatch` (its `batch_id` already equals the tour id) and merge `ActiveTourBatchingSolution` into the routing stage, so that one admission solution and one routing solution remain. `considered_pairs` appears once, in `AdmissionSolution`.
    - `InvariantViolation` merges into `Violation`; `ViolationCode` gets the invariant codes.
    - After Task 4.9: `OperationsView` dissolves. The dock fill and capacity are read from the warehouse's `Dock`; breaks leave it in Task 4.14; conditions read the snapshot only.
    - Keep `DynamicInfo` and `SimWarehouseDomain`; they are names of the CASIM paper. `DecisionContext` stays unchanged; `DecisionMemory` stays unchanged until Task 6.4.
    - Type the CoSy stages so that synthesis, not repository discipline, excludes invalid pipelines (F52). Today `AbstractBatchProvider` admits both `AbstractBatching` and `AdmittedTourBatch`, and `ReroutableSShape` and `ReroutableWalkOrWaitSShape` are ordinary `PickerRouting` subclasses with the private mixin `_ActiveTourRoutingNode`, so by parameter type CoSy may combine FiFo with a reroutable router or `AdmittedTourBatch` with a depot router. Introduce `AbstractActiveTourRouting(BaseComponent)` with `batching_sol = CoSyLuigiTaskParameter(AdmittedTourBatch)` and its own output, make the reroutable nodes subclass it, and let depot routing depend on `AbstractBatching` only; remove `AbstractBatchProvider` if nothing else needs it. Add a test that a repository containing FiFo, `AdmittedTourBatch`, `SShape` and `ReroutableSShape` synthesizes exactly the two valid pipeline families.
    - Compare lifecycle values with the enums (`TourStates`, `Accepts`) instead of strings (`view.accepts.value == "rebuild"`, `status.value != "started"`) in all CoSy nodes and in `IncumbentProvider`; filter the tours of a decision once, through one function on `PlanView`, instead of repeating the filter in `AdmittedTourBatch`, the admission nodes and `_ActiveTourRoutingNode`.
    - `AbstractWaiting` takes the picker of the first scheduled job. Make the single-picker assumption explicit, as a requirement of the waiting cards that need it (Henn, analytical), and fail with a clear message otherwise.
    - Build routers in one place: a helper that constructs any `Routing` subclass from layout and resources, used by the routing nodes, `IncumbentBatchingNode` and the local-search and Clarke-Wright nodes, instead of repeating the constructor arguments.
    - Count the types before and after and report both numbers.

- [ ] **4.14** One dispatch mechanism for resources (F55). Read the section "Resources and dispatch" of the design document first. Defects at `640b5ce`, found by reading the code: `Reschedule` emits no event, so a tour moved to another idle picker never starts and a tour moved earlier starts at the old time; committing an earlier tour for a picker that waits for a later one sends no query, because `occupied` is already set; the break is one global flag that the first `BreakEnd` clears for every picker, and idle pickers never take it; neither `SequencingDone` nor `PickerTourQuery` checks availability, and `PickerArrival(picker_available=True)` wakes no picker; `occupied` is set at commit, so a picker waiting for a later start counts as `BUSY` and its waiting time as neither idle nor busy; the grocery hook creates `PickerArrival` without `picker_available`, whose default `False` takes every picker off shift at shift start (also on `main`; confirm with one run).
    - A resource's queue is the ordered list of its committed (`scheduled`) tours. A `planned` tour may carry a picker as part of a draft, but it is not in the queue and is never dispatched.
    - Resource state is derived, never stored: `BUSY` if the resource has a `pending` or `started` tour; otherwise `OFF_SHIFT` if it is absent; otherwise `RESERVED` if its queue is not empty; otherwise `IDLE`. Today a `planned` tour with a picker already makes the picker `RESERVED` (`has_future_tours` counts every queued id); under the new rule it does not, because a draft reserves nothing. Report for every scenario whether a decision's `state_information.resources` selects a different set of resources because of this. Keep `ResourceStates` and `ResourceManager.resource_status`. casim stops reading and writing `Resource.occupied`; the field belongs to the shared domain model of ware_ops_algos, so leave it there and list it for removal in the report. Delete `mark_picker_occupied`, `mark_picker_free`, `_busy_until` and its accessors, `state.is_break` and `state.break_duration`.
    - Availability is one per-resource fact changed by two events. A resource is absent until its first `PickerArrival`, as today, where `Resource.available` starts as `None`. `PickerArrival(picker_id)` makes the resource present; the `picker_available` argument goes. `PickerDeparture(picker_id, duration: float | None)` makes it absent from its next free point on: immediately if it is not `BUSY`, otherwise at the end of its current tour; with a duration it returns by itself through `PickerArrival` after that duration, which is a break; without one it stays absent until the next `PickerArrival`, which is a shift end. `BreakStart`, `BreakEnd`, `ShiftStart` and `BreakCondition` are deleted; scenario hooks emit one `PickerDeparture` per picker instead of one `BreakStart`, and the IJPE hook emits, at each day's arrival time, `PickerArrival` for the pickers working that day and `PickerDeparture` without duration for the others. A tour is never interrupted.
    - One internal event `Dispatch(picker_id, version)` replaces `PickerTourQuery` and the wake-up of `SequencingDone`. Each resource has a queue version, increased by every change to its queue (commit, reschedule of a committed tour for the old and the new picker, cancel of a committed tour) and by every change of its availability; `Dispatch.is_stale` compares the version. `apply` emits one `Dispatch` per affected resource at the current time; `TourEnd`, `PickerArrival` and the effective start and end of a departure emit one as well. The handler does nothing for a `BUSY` resource; makes a pending departure effective; does nothing for an absent resource; otherwise takes the first tour of its queue: if there is none, the resource becomes `IDLE`; if the tour's `earliest_start` lies in the future, it queues `Dispatch` at that time with the same version; otherwise the tour becomes `pending` and `TourStart` follows after the setup time. `Dispatch` cannot be configured as a trigger. It takes the priority of the event it replaces, `0` when `apply` emits it (as `SequencingDone`) and `1` otherwise (as `PickerTourQuery`), so that the order of events at equal timestamps is unchanged.
    - Separate the times of a committed tour (the section "Planned and executed times" of the design document). `TourPlanningState` gets `earliest_start: float | None` (binding), `planned_start` and `planned_end` (the forecast) and `actual_start` and `actual_end` (execution); `start_time` and `end_time_planned` go, and `start_tour` no longer overwrites the plan with the actual start (its comment "todo seperate between planned and actual start time" marks this). `TourView` keeps `start_time` and `end_time`, now meaning the planned (projected) times, and gains `earliest_start`; algorithms read the same fields as before. `CreateTour` and `Reschedule` get a field `earliest_start: float | None`, which the decision engine sets after `to_commands` from the decision's new key `start_times: binding | forecast` in `DecisionConfig` (a `Literal` with default `binding`, which keeps today's behaviour): `binding` sets it to the scheduled start, `forecast` leaves it empty. The queue order of a resource is the order of planned starts at its last commit or reschedule; nothing reorders it except a `Reschedule` or `CancelTour` command, and no tour overtakes the first one.
    - After every `TourStart`, `TourEnd` and change of a resource's queue or availability, the engine projects the resource's queue: the first queued tour gets `planned_start = max(earliest_start, t_free)`, where t_free is now for a free present resource, the current tour's `planned_end` (its `actual_start` plus its current processing time, which a `ReplaceSuffix` may change) for a busy one, and the return time for a resource on a break; for a resource absent without known return the planned times stay as they are, every later tour `planned_start = max(earliest_start, planned_end of its predecessor)`, and `planned_end = planned_start + processing time`. The projection is arithmetic on the plan, not a decision. `StateAdapter` takes `available_from` and the tour views' times from it. The tracker records the planned start and end of every tour at its commit, so that the deviation of executed from committed times can be reported.
    - `PickerIdle` is emitted once per transition into `IDLE`, from the derived state, and stays a trigger. `TruckDeparture` no longer emits `PickerArrival` for idle pickers; configurations that need a decision at a truck departure name `TruckDeparture` as trigger. `ShiftStart` and `BreakEnd` in triggers become `PickerArrival`.
    - The tracker records every change of derived resource state with its time and computes idle, busy and absent time per resource from that log; `on_idle_start` and `on_idle_end` go. Idle time is the time a resource is present and not `BUSY`, that is in `IDLE` or `RESERVED`, so waiting for a later scheduled start counts as idle.
    - Checks, in casim's verification layer, after the last event of every timestamp (Section 8): (1) at most one `pending` or `started` tour per resource; (2) no tour became `pending` before its `earliest_start` or while its resource was absent; (3) no lost wake-up: no present, non-busy resource has a first queued tour without `earliest_start` or with `earliest_start` reached; (4) per resource, idle, busy and absent time sum to the simulated horizon; (5) the planned times of every queue equal its projection, except for resources absent without known return. Check (3) is the one that fails on the defects above.
    - Tests, each a small fixture with two pickers: a tour rescheduled earlier; a tour moved to another idle picker; an earlier tour committed while the picker waits for a later one; a break while one picker is busy and one idle; a shift end with queued tours; a departure and arrival at the same timestamp; a tour that ends earlier than planned, once with `start_times: binding` (the next tour waits for its start) and once with `forecast` (it starts at once); a tour that ends later than planned (the next tour starts late in both modes, and the projection moves every later planned time). Each asserts start times and the checks.
    - KPIs of the Henn, walk-or-wait and stochastic waiting scenarios must stay identical except idle time, whose definition changes; report idle time before and after. Grocery and IJPE may change; explain every difference by one of the defects. If a difference has no such cause, stop.

- [ ] **4.15** Remove hidden rules that bypass the decision model (F57). Each rule below decides something that the configuration of the decision should state, or states one thing in two places. Apply them in this order and report the KPIs after each.
    - `FlushRemainingOrders` is listed as trigger in five decisions, but no code emits the event, and `SimulationEngine.run` lets it bypass all conditions (`"FlushRemainingOrders" in causes`). Delete the event class, the engine clause and the trigger entries. No KPI may change.
    - The engine also lets `ReconsiderReason.WAIT` bypass the conditions. Keep it until Task 6.4, but name it in the design document as the one remaining engine rule of waiting.
    - `StateAdapter` switches the due-date window off once the stream is closed (`not state.done_flag`), so the observation depends on a fact the configuration does not mention. Remove the exception; a decision that must flush remaining orders at closure says so with `OrderStreamClosed` as trigger and no window, or with a second decision. The due-date window also drops every order without due date silently; `casim check` rejects a window `by: due_date` when the data card has no `due_date` order feature.
    - `StateAdapter` adds the picker of every exposed started tour to the snapshot, even when `state_information.resources` excludes it (`ATIP` has `resources: []`). Make it explicit: such pickers are reachable through the tour view (`picker_id`, `progress`) and are not part of `snapshot.resources`, or the decision lists `busy`; choose one, and state it in the design document.
    - `StateAdapter` builds the snapshot with `ResourceType.HUMAN` and `OrderType.STANDARD` regardless of the warehouse; the IJPE data card declares `splittable` orders. Take both types from the state.
    - In the stochastic waiting scenario, `NbrOrdersCondition(threshold=3, allow_when_done=true)` encodes the precondition of `AnalyticStochasticWaiting` (q − 1 known orders for a cart with q = 4 bins) as a condition of the decision, so the `no_waiting` and `henn_waiting` variants, which share the decision file, also wait for three orders. The precondition belongs to the algorithm: the analytic waiting returns `wait` until it holds, as it already rejects other batch sizes. Remove the condition and report the KPIs of all three variants before and after; the two non-analytic variants are expected to change.
    - Events with empty handlers that only serve as timers live in the core (`WMSRun`, `OrderIngestion`, `VolumeShiftAcrossDay`, `TruckArrival`). Replace them by one event `Timer(name)` and one trigger form `{timer: <name>}`. A scenario declares its fixed timers once, outside the decisions, as `timers: {<name>: {every_s, offset_s, until_s} | {at_s: [...]}}`, and the engine schedules them; a hook may add `Timer(name)` at a time computed from data (the IJPE order injection and volume shift); in Phase 6b this moves into the exogenous stream. Several decisions may share one timer. The IJPE hooks currently emit `WMSRun` and a second marker event at `t + 1e-6` so that `RORSP` (precedence 20) runs after `OBP` (no precedence, so it runs after every numbered decision); give `OBP` a precedence below 20 and let both decisions use one timer at t. A decision's own `every_s` stays, because it is anchored to the decision and stops at stream closure. Report the KPIs; only times shifted by 1e-6 may differ.
    - Instance data is split between the data card and a `simulation` config group (walk-or-wait: `capacity_orders`, `walking_speed`, `pick_time_s`; stochastic waiting: realized arrival times and aisles), which the data card reads by interpolation. Move these values into the data card's `source` section, which the loaders already read, and delete the `simulation` groups. Keep the stochastic scenario's realized stream separate from its `information` section, and say in its data card description that the belief (exponential stream) and the realized stream (four fixed arrivals) differ on purpose.
    - Walk-or-wait selects one component through two extra groups, `routing` (`${routing.component}` inside the cosy repository) and `policy` (`WaitFor${policy.min_orders}`). After Task 4.12, a repository file lists its components and configured cards by name; a sweep varies the repository file. Delete both groups.
    - `warehouse_info: {type: online}` in two data cards is an online label that no card reads, which open point O14 rules out. Delete it.
    - `setup.py` and the schema name classes directly (`problem_class in {"ORSP", "OBRSP"}` as "scheduling decision"); derive the rule from `variant()` (the variant contains `scheduling`). `DecisionTracker` records the decision name as `problem_class`; record both under their own names.
    - Delete the unused duplicate `BaseTourEvent` in `events/base_events.py`.

Done when the tests pass and the search test passes.

### Phase 5: verification (gate)

- [ ] **5.0** Enforce the conventions mechanically before the gate:
    - add pydantic to ware_ops_algos and a schema for algorithm cards and `configs.yaml` as sketched in the review document: closed vocabularies for `problem_type`, `implementation.type` and `objective`, `extra="forbid"` on the implementation, every constrained feature also listed in `features`;
    - cross-card checks: unique `algo_name` and `component_name`; `class_name` resolves to a class that subclasses the base registered for the card's `problem_type` (this subsumes Task 4.6); every `configs.yaml` entry resolves;
    - a command `woa check` that runs these checks and an applicability lock: for every data card of ware_ops_pipes (`data/data_cards/*.yaml`, copied as test fixtures), the set of applicable cards per problem class, committed as `applicability.lock.json` and generated once from `main`. A difference fails; `woa check --update-lock --reason "..."` records the reason;
    - `casim check` calls the same checks for the cards casim uses, and `casim check --all <scenario>` composes every file of every config group and every sweep of a scenario, so that a group file nobody selects by default cannot break unnoticed (F56); a `problem_class` outside `TAXONOMY` is an error;
    - one registry in `casim/pipelines/taxonomy.py` that maps each subproblem to its abstract CoSy stage and its plan-candidate bridge (F53). `pipeline_runner.py` derives `stage_classes` and `PLAN_SOURCES` from it instead of defining them; the stale `endpoint` entries (`ResultAggregation*`) leave `TAXONOMY`; `variant()` and the registry are checked against each other by `casim check`, which also reports a configured component whose stage the decision's variant does not contain. Find out why `build_pipelines` filters synthesized pipelines to configured classes; if CoSy inhabits with abstract classes because they lack an `abstract` flag (ware_ops_pipes sets `abstract = False` on generated classes), fix the class definitions and remove the filter, otherwise document the reason next to it;
    - an `AGENTS.md` in both repositories listing the conventions of Section 4, the CASOP conventions and the rule that `woa check`, `casim check` and the tests pass before every commit; a CI job on every branch.
- [ ] **5.1** Port the base scenario configurations to the new keys:
    - Henn: key and `problem_class` `OBRSP`, `HennWaitingNode` in the cosy repository, `state_information` and `scope` as in the design document's Henn example, `commitment_policy` `SchedulingCommitmentPolicy` with `n_jobs: 1`, conditions `NbrPickersCondition(1)` and `NbrOrdersCondition(1)`, no `NewInformationCondition`;
    - stochastic waiting and walk-or-wait: likewise, with their base conditions;
    - the walk-or-wait intervention is a second decision, `active_tour_admission`, declared OBRSP.
- [ ] **5.2** Write the remaining checks of Section 7.1 (scopes, variant, write sets), the invariant fixture and the Hypothesis runs of 7.2, the KPI recomputation and the nine verification scenarios of 7.3.
- [ ] **5.3** Run `scripts/compare_with_base.py` on the three scenarios of Section 7.4 and explain every difference as Section 7.4 requires.

Do not continue to Phase 6 while a verification test fails or a difference from the base has no demonstrated cause. Never adjust an expected value to the code. When the user has accepted the differences, pin the regression KPIs (Section 7.4).

### Phase 6: extensions on the corrected core

- [ ] **6.0** Reproduce the OOBP baseline as Task 3b.3 proposed and the user approved: add the configurations the assumption table requires (no new engine mechanisms), run all 64 Henn instances from `data/oobp/` (Task 3c.3) under the invariant fixture, and report per instance and in aggregate against `data/oobp_reference.csv`, for both objectives. A result below the best known value is a modelling error until shown otherwise; stop and report it.

- [ ] **6.1** Port `rolling_horizon.yaml`, `release_only_example.yaml` (rename it `commit_only_example.yaml`), wait-0 patrol and center relocation to the new keys (F31).
    - `rolling_horizon.yaml` follows the row "Rolling horizon" of the strategy table in the design document: `problem_class: OBRSP`. Rolling horizon is a decision policy (trigger, observation window, scope, commitment), not a problem; `problem_class: rolling` came from the base schema, where the decision key doubled as the problem class and `output: scheduling` fixed the terminal stage, and the port in `d1f1447` copied the key into `problem_class`. Use the triggers of the strategy table or justify each additional one (`TourEnd`, `PickerArrival`, `OrderStreamClosed`). State its `start_times` mode with a one-line reason in its README; do not choose `forecast` silently, because it changes results only when tours end earlier than planned.
    - `henn_rolling` synthesizes due dates (`due_offset_s: 5400`). Keep it, and state in its data card description and the Henn README that its due dates are synthetic and its results are not comparable with Henn (open point O10).
- [ ] **6.2** Run them under the invariant fixture and report their KPIs in the phase report. Explain every change compared with the numbers in `docs/revision_scopes_baseline.md` with the comparison script.
- [ ] **6.3** The IJPE grocery scenario is out of scope until Task 6b.5.
    - Its configurations must pass `casim check`, using `DockCapacityCondition` with the warehouse's `Dock` and the resource events of Task 4.14.
    - If its behaviour changes because release gates no longer exist, say so in its README. Do not add a mechanism for it.

- [ ] **6.4** Waiting as a plan fact (O17), only after the user has accepted the Phase 5 results. A wait is a later binding start (`earliest_start`, Task 4.14) or the absence of commitment; a committed tour with a future binding start is dispatched again at that time. Remove `DecisionMemory.waiting` and `reconsider_at` if, and only if, the Henn, analytic waiting and walk-or-wait regression values stay identical. Report the comparison; if a value changes, stop.
- [ ] **6.5** Timing of stream closure (O19), only after Task 6.4. No belief infrastructure is introduced; `OrderStreamClosed` and `input_closed` stay. Both `experiment_henn.py` and the stochastic waiting hook currently emit `OrderStreamClosed` at the last order date, so policies learn at the last arrival that nothing follows, which is oracle information. Replace the hard-coded emission by one scenario parameter `stream_closure` with the values `at_last_arrival` (labelled as oracle in results), `announced` (a cutoff time known from t=0) and `never`. Add a card feature for observable closure; policies that only terminate with closure knowledge, such as `OrderCountWaiting`, require it unless they have a timeout. Whether closure is observable is a static card feature; the current value of `input_closed` is read at decision time and never changes applicability. Keep `at_last_arrival` as the default so regression values stay identical, and record the variant in every result. Before accepting the Henn and Gil-Borrás reproductions, report which closure assumption each source makes; do not guess it.

### Phase 6b: one scenario structure (after Phase 6)

Read the section "Scenarios and exogenous information" of the design document first. The five scenarios share the engine but nothing else: five entry scripts with copies of the main loop, three hook styles, loaders that read different config groups, scenario-specific output files, plots that partly run the simulation. The grocery scenario names a loader (`scenarios.scenario_grocery_retailer.grocery_retailer_loader`) that exists neither on the branch nor on `main`, and its configuration duplicates IJPE's. The IJPE hooks rewrite the event heap (`sim.events[:] = ...`), mutate `domain.orders` "for tracking", attach a `DockManager` to the state, and the event `TruckDisruption` cancels tours in its handler, so an exogenous event changes the plan without a decision (F58).

- [ ] **6b.1** Exogenous information as one typed stream. A loader returns the warehouse W_0, the belief B_0 and an `ExogenousStream`: a time-ordered tuple of exogenous events from a closed set: `OrderArrival(order)`, `OrderUpdate(order_id, due_date=None, cancelled=False)`, `PickerArrival(picker_id)`, `PickerDeparture(picker_id, duration)`, `DepartureUpdate(departure_id, time=None, capacity=None)`, `Timer(name)` and `OrderStreamClosed`. Their handlers change W_t or B_t only, never P_t; a plan's reaction to new information is a decision triggered by the event, expressed in commands within its scope. Truck departures are part of W (the `Dock` of Task 4.9 holds the departure schedule), and the engine emits `TruckDeparture` from it. `sim.reset` takes the stream; scenario hooks are deleted.
- [ ] **6b.2** Perturbations as declared transformations of the stream. A scenario config lists `perturbations:`, each a typed, seeded function from stream to stream, applied by the common code before the run, in the listed order:
    - `inject_orders(at, due, generator, n | volume)`: new orders known at `at` with due date `due` (emergency orders; IJPE's order injection);
    - `reveal_early(select, at, due)`: selected future arrivals become known at `at`, optionally with a new due date (IJPE's cross-day volume shift);
    - `update_due_dates(select, at, due)`: an `OrderUpdate` at `at` for selected known orders (IJPE's truck disruption, without the cancellation);
    - `move_departure(departure, at, time | capacity)`: a `DepartureUpdate` at `at`, plus `OrderUpdate`s for the orders due at that departure;
    - `absence(pickers, at, duration)`: `PickerDeparture` events (breaks, sickness, a lost shift).

    A selector names orders by due date, arrival window, customer or a volume limit in transport units, so that "move 100 TE due at departure d to departure d'" is one entry. Each perturbation records what it changed in the trace. The model is: exogenous information ω arrives as events, changes W or B, and decisions react; nothing else.
- [ ] **6b.3** One layout and a generator. Every scenario has `README.md` (question, sources, how to reproduce, which results it backs), `config/<name>.yaml` with exactly the groups `data_card`, `decisions`, `cosy_repo`, `sweep`, a `loader.py` with one `DataLoader` subclass that reads only `data_card.source`, an optional `metrics.py` (extra KPIs as functions of the tracker, never redefining the common ones), an optional `plots.py` that reads output files only, and `data/` with small data or a README that names the external source with a checksum. `casim new <name>` creates this skeleton. Shared code moves into `casim.scenarios`: the run loop, stream builders (order streams from tables, shift calendars, departure schedules, timers), the perturbation library, layout and order readers used by more than one loader, and the common plots (Gantt, KPI tables).
- [ ] **6b.4** One entry point and one output format. `casim run <scenario> [overrides]` replaces the five entry scripts; it writes `kpis.json`, `decisions.json`, the trace, the output of `metrics.py` and the Gantt chart into one directory layout for single runs and multiruns; `--dash` starts the dashboard. `casim check --all <scenario>` checks the layout: the loader class resolves, every group file composes, the stream contains only the closed set of event types, `plots.py` does not import the engine.
- [ ] **6b.5** Port the scenarios one at a time, each in its own commit, with the regression values of Phase 5 unchanged: Henn online (including the waiting-strategy sweep of F56), walk-or-wait, stochastic waiting. IJPE is ported only through Task 6b.6. Delete the grocery scenario unless the user decides otherwise; IJPE covers its configuration.

- [ ] **6b.6** Reintegrate the IJPE study (F59). The user adds the original WMS data, the original loaders and the IJPE paper to `docs/`. Read the paper first and list every number it reports that a simulation produces (tables and figures), with its section. Work in three steps and report after each; stop after each report.
    1. Reproduction. Run the original code path on the original data, at the commit the paper used (ask the user which, if the paper or the history does not say), and compare with the paper's numbers. Then port the scenario to the structure of Phase 6b, with the original loaders as its `loader.py` and the hooks as perturbations, run the same study on the corrected core and compare again. Explain every difference by a cause you can show: a fix of this plan (F-number), a changed semantics (the cancellation in `TruckDisruption` is now a decision, the setup time F28, the dispatch of F55), or a defect of the original. Where the corrected core cannot reproduce a published number without restoring a defect, present the options and wait.
    2. Data audit. Check the original data and its handling for errors: units (transport units, pallets, the factor 1.6 TE per order used by the hooks), time zones and day boundaries, duplicate or missing order lines, orders whose due date precedes their arrival, articles without storage location, pickers per day versus the records, and the configuration values the scenario relies on. Known suspects: `truck_schedule.capacity: null` makes `TruckDeparture` release no pallets (`if self.capacity:`), so the dock only fills; the grocery configuration has a unit mismatch that was never resolved. Report each finding with the evidence (query, count, example rows) and its effect on the published numbers; do not change the data.
    3. Closeness to the WMS records. Define, before any calibration, the measures that compare simulation and records: per day the number of tours, picks per tour, tour durations, picker utilization, the time profile of completed picks, and lateness against departures. Calibrate parameters (speed, pick time, setup time, picker arrival times) only on a subset of days and report the measures on the remaining days, so the fit is not tuned to the evaluation. State which differences the model cannot represent (for example picker behaviour the records show but the WMS does not plan).
    - The original data may be confidential: do not commit or push it unless the user says so; keep it out of the repository through `.gitignore`, and let `data/README.md` describe where it lives.

### Phase 7: documentation and user guidance

- [ ] **7.1** Rewrite `docs/revision_scopes_baseline.md` as `docs/verification.md`: what each verification layer checks, the time model, the derivations of the verification scenarios, the base comparison with its explained differences, the extension KPIs of Phase 6, and the open points the implementation touches (O2, O3).
- [ ] **7.2** In casim's README, add a section "Configuring a decision" with one annotated YAML per strategy from the design document's strategy table. Each example lives under `tests/config/strategies/`, and a test runs `casim check` on each one.
- [ ] **7.3** `casim catalog` lists:
    - the tour statuses and their transitions;
    - the `Accepts` levels and `PHYSICAL`;
    - the commitment policies;
    - the problem classes with their subproblems;
    - the decision features;
    - the conditions;
    - the algorithm cards with their requirements.

    `casim check` prints the derived variant and the subproblems of each decision. Every error message names the key, the value and what is allowed.

## 9. Stop and ask when

- a verification scenario and the code disagree and you cannot show which one is wrong;
- a difference from the base has no cause you can demonstrate;
- the time model of Section 7.3 is not stated precisely enough to derive a value;
- a task requires a type, field, key, command or enum value that the design document does not define;
- a name seems wrong to you, but neither Section 6.1 nor the design document changes it;
- the design document, this file and the code disagree, or a statement here turns out to be false (say which and how you checked);
- keeping an existing algorithm's result requires changing it;
- a task would touch CASOP behaviour (`DomainAlgorithmMapper.filter`, the ware_ops_algos taxonomy, cards used by benchmarks, scheduling results with homogeneous pickers), or the applicability lock changes;
- one of the four pending user decisions of Task 4.12 is affected. They stay as they are on the branch until the user decides.

Ask in a short message: what you found, the evidence (command and output), and the options you see. Then wait.

Do not stop for a factual slip in this file that changes no task (a version number, a date, a count): note it under "Deviations" in the phase report and continue.

## 10. Done criteria

- [ ] All phases are reported, and all findings in Appendix A are marked fixed or explicitly deferred by the user.
- [ ] `python -m pytest -q` passes in one process in both repositories, including the exhaustive, invariant, KPI recomputation, verification, regression, import-boundary, AST and state-machine tests.
- [ ] `casim check` passes on every configuration under `scenarios/` and `tests/config/`, and `woa check` passes with an applicability lock equal to that of `main` or changed only with the user's recorded approval.
- [ ] No module listed for deletion exists, nothing in Section 6.3 is still referenced, and every old name in Section 6.1 is gone.

## Appendix A: findings and where they are fixed

| # | Finding (verified on `6cd1705` / `90c9364`) | Fix | Phase |
|---|---|---|---|
| F1 | Henn: 26 tours, 12 decisions, distance 11,402, six single-order tours; base 25, 26, 11,603, none; causes F4, F26, F27 | Fixes of F4, F26, F27; comparison with explained differences | 5 |
| F2 | Problem class "derived" by `_derive_problem_class` from `solver.output`, the selection and component names (`"OrderSplitter" in components`); ATIP and RORSP re-created | Declared `problem_class`, `variant()` in `casim/pipelines/taxonomy.py` | 3 |
| F3 | `setup.py` filters mapper reasons starting with `"problem type "`, maps waiting names to `WaitForOne`..`WaitForFour`, and special-cases `NbrOrdersCondition`/`NbrBatchesCondition` | Removed; waiting via cards and cosy repository | 3, 4 |
| F4 | Waiting algorithm receives one candidate; Henn's own rule never fires; engine duplicates it (`waiting.py:92`) | Waiting stage gets the whole schedule | 4 |
| F5 | `AnalyticStochasticWaiting` turns its q−1 precondition into a silent `wait` | Restore error; card and condition | 4 |
| F6 | Ranker: invented `CHANGE_WEIGHTS`; `CreateTour` counted as change; no feasibility filter; batching takes the last key (from the user's base code) | `changed`, feasibility, check error | 3 |
| F7 | `SShapeRouting` hard-coded in `_single_order_service_times` and `IncumbentBatchingNode`; `resources[0].pick_cart` | Router injection | 4 |
| F8 | Dead: `operational/models.py`, `algorithms/pipeline.py`, the loose `DynamicInfo` fields, `required_capabilities`, new card `capabilities` entries | Delete | 1, 2 |
| F9 | `"center_relocation"` in five casim modules; route computation in `apply.py` | Generic hold; router | 4 |
| F10 | `decision_engine/waiting.py` mixes gates, waiting dispatch by string and commitment rules; applies only to `SchedulingSolution` | Waiting stage + `commitment_policies.py` | 3, 4 |
| F11 | Waiting delay retimes only the picker's first job | Waiting algorithms retime all of the picker's jobs | 4 |
| F12 | String-encoded causes (`"Reconsider:wait"`, `"Reconsider:release_window"`) | `ReconsiderReason` | 3 |
| F13 | An event type may trigger only one decision | Tuple of decisions | 3 |
| F14 | `planned`/`assigned` never observable; `revise` unused; `order_date` window rejected; status lists that allow one value | Trim vocabulary; `planned` is the uncommitted status | 2 |
| F15 | `TourPlanningState` moved into ware_ops_algos | Back to `casim/domain_objects/tour_model.py` | 2 |
| F16 | `TourView` with positioning fields; `AdmissionTour` plus the bridge `TourView.as_admission_tour` | Flat `TourView` in `planning`; `AdmissionTour` stays the algorithm input, built by the admission component | 2 |
| F17 | `SimWarehouseDomain.memory/operations: object`; `DynamicInfo` with loose fields; `OperationsView.extra: dict`; `ReleaseGates` | Typed `DynamicInfo(context, plan, memory)`; `OperationsView` without `extra`; removal | 2, 3 |
| F18 | `PlanView.active_candidate_ids` duplicates `orders` | Removed | 2 |
| F19 | `DecisionSolver` declared but setup always builds `CoSySolver` with CoSy arguments; engine reads `runner.last_conversion_seconds` | Protocol `solve(state_snapshot)` in `interfaces.py` | 3 |
| F20 | `SimulationEngine` gets `precedence`, `periods`, `on_violation`, `decision_engine` assigned after construction; `repair` calls the `DecisionEngine` from `step` | Constructor; removal | 3 |
| F21 | `casim check` imports `scenarios.experiment_commons` | `casim/config/data_card.py` | 3 |
| F22 | `apply_all` deep-copies `State` and overwrites `state.__dict__`; private attribute access; function-local imports | Two-phase check/apply | 2, 3 |
| F23 | Tour status written in nine places | `TourManager.advance` + tests | 2 |
| F24 | `expected_*` fields can never fail: snapshot and apply happen without an event in between | Removed | 3 |
| F25 | 12 of 41 casim tests fail in one process with cosy-luigi `0.0.1.dev18` (Luigi task instance cache); `0.0.1.dev7` disables the cache | Regression test; a fix only if it fails; version reported | 0 |
| F26 | Snapshot orders are buffered orders followed by the orders of planned tours, which changes FIFO input | Sort by `(order_date, order_id)` | 2 |
| F27 | The commit-only path commits a planned tour with its stale start time; Henn config has `NewInformationCondition` | Waiting in `DecisionMemory`, commitment policies; config | 3, 5 |
| F28 | Setup time counted twice (pre-existing at `67f9f31`): `earliest_free_assignment` adds it to `start_time`, `PickerTourQuery` adds it again | `start_time` is the begin of the setup, `end_time` includes it, `PickerTourQuery` adds it once; verification scenario 1 | 4 |
| F29 | Design document copied into the repository; baseline doc presents the Henn deviation as intended | Delete; rewrite | 1, 7 |
| F30 | `rejection_reasons` re-implements `_filter_by_requirements` | One implementation | 1 |
| F31 | `henn_rolling` synthesizes due dates | Label as synthetic | 6 |
| F32 | `algorithms/positioning.py` is a flat module, unlike every other problem type | Package | 4 |
| F33 | No algorithm both rebuilds planned tours and inserts into started tours in one decision | `variant` raises; open point O2 | 3 |
| F34 | `earliest_free_assignment` now uses each picker's processing time | Keep; test that homogeneous results are unchanged | 3 |
| F35 | `Wait` and `RecordConsidered` are commands although they change no plan object | `DecisionResult` fields | 3 |
| F36 | `InsertOrders` must always be paired with `ReplaceSuffix`; `Reassign` separate from `Reschedule` | One `ReplaceSuffix`; merged `Reschedule` | 3 |
| F37 | `ReleaseGates` exist only for the IJPE scenario | Removed; README note | 3, 6 |
| F38 | Enum comparisons through `.value` strings | Enum comparisons | 2–4 |
| F39 | The engine emits `RoutingDone` for routing-only solutions, so the outcome of such a decision depends on a simulator event, not on commands | Rejected at setup; emission and `RoutingDone` deleted | 3 |
| F40 | The branch merged casim's online taxonomy into ware_ops_algos and changed the CASOP entries: `SPRP` lost `item_assignment` and `order_selection`, `OBRP` lost `batching_routing` | Restore `34bd157`; online classes in `casim/pipelines/taxonomy.py`; open point O14 | 3 |
| F41 | `SimWarehouseDomain` and `DynamicInfo` moved into ware_ops_algos, with a re-export of the `ActiveTour*` types left in `casim/domain_objects/sim_domain.py` | Back to `sim_domain.py`; re-export deleted | 2 |
| F42 | Scenario hooks attach `dock_manager` to `State`; core events, `StateAdapter` and logger branch on `hasattr`; `TourEnd` stages a literal 1; the truck wake-up depends on the dock | `Dock` in the warehouse domain, `DockManager` in `casim/state`, typed `OperationsView` | 4 |
| F43 | Four algorithm cards and four CoSy components for `OrderCountWaiting`, one per `min_orders`, identical requirements (from `8d7815d`, `a562716`) | One card, one parameterized component | 4 |
| F44 | `6cd1705` required picked orders of a started tour in the replacement route, so admission after a pick failed (`wait_2`, `wait_3` on `explicit_example`) | Validation requires the unpicked orders; tour membership kept in `commit_active_route` | 4 |
| F45 | Review of Phase 4: numeric class aliases for waiting, inconsistent card lookup, untested re-queued `PickerTourQuery` | Task 4.11 | 4 |
| F46 | `spt_scheduling.yaml` lost its `due_date` requirement in `b77f256`; changes CASOP applicability for OBSRP sets without due dates | User decision; Task 4.12 lists it | 4 |
| F47 | `walk_or_wait_sshape_routing` and `fifo_scheduler` are applicable in CASOP problem classes without a ware_ops_pipes component | Task 4.12, applicability lock of Task 5.0 | 4, 5 |
| F48 | Parameterized algorithms bypass the CASOP configuration mechanism (`routing_class` by hand, one subclass per `min_orders`); module layout and private mixin deviate from `main` | Task 4.12 | 4 |
| F49 | Mapper treats `None` feature values as absent; `Resources.get_features()` reports cart values or `None` instead of `True`; mapper docstrings removed | Docstrings in Task 4.12; semantics by user decision | 4 |
| F50 | 40 new types without a stated model; duplicated admission types and fields | Task 4.13 under the state model of the design document | 4 |
| F51 | `OrderStreamClosed` emitted at the last order date gives policies oracle knowledge of stream end | Task 6.5, scenario parameter `stream_closure` and card feature for observable closure | 6 |
| F52 | CoSy stages typed too loosely: `AbstractBatchProvider` and the reroutable routers admit invalid combinations; `AdmittedTourBatch` contains the closure rule; string comparisons of lifecycle values | Tasks 4.11 and 4.13: `AbstractActiveTourRouting`, closure rule in admission, enum comparisons | 4 |
| F53 | `pipeline_runner.py` hard-codes the subproblem-to-stage and terminal-bridge maps; `TAXONOMY` names removed `ResultAggregation*` endpoints; unexplained post-synthesis filter | Task 5.0, one registry checked by `casim check` | 5 |
| F54 | Dead sequencing stage (algorithms removed in `6a9a3d4`, unimportable node module, unimported `SequencingInput`); unused `rolling_horizon` configurations with an undefined problem class; committed scenario outputs | Task 4.11 | 4 |
| F55 | Picker dispatch: no wake-up on reschedule or earlier commit, global break cleared by the first `BreakEnd`, no availability check, `occupied` set at commit, grocery pickers taken off shift by a boolean default | Task 4.14, derived resource state, per-resource availability, versioned `Dispatch`, four checks | 4 |
| F56 | The port in `d1f1447` emptied the three waiting-strategy override files (`config/scenario/*.yaml`) and copied the decision key `rolling` into `problem_class`; no check composes group files that are not selected by default; a stale sweep path and a stale README override | Task 4.11 (restore), Task 6.1 (`rolling_horizon.yaml` as `OBRSP`), Task 5.0 (`casim check --all`) | 4 |
| F57 | Hidden rules outside the decision model: an unemitted trigger that bypasses conditions, a stream-closure exception in the window, pickers added outside `state_information`, hard-coded snapshot types, a waiting precondition encoded as condition, timer events in the core, instance data in a `simulation` group, component selection through extra groups, an online label in data cards, class names in setup rules | Task 4.15 | 4 |
| F58 | Scenarios without a common structure: five entry scripts, three hook styles, a missing grocery loader, hooks that rewrite the event heap and the domain, a dock manager attached by a hook, an exogenous event (`TruckDisruption`) that cancels tours | Phase 6b | 6b |
| F59 | IJPE results, data handling and closeness to the WMS records never checked against the paper and the original data | Task 6b.6 | 6b |
