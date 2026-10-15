# CASIM revision scopes and configuration design

Oct 13, 2026 · @Janik Bischoff

## Summary

The CASIM paper describes a decision by five properties: the problem class and its algorithms, the trigger events, the conditions, the state information available to the solver, and the components of the solution that are committed to the operational state. This design keeps all five and adds a sixth, the revision scope, which states how far a decision may change what earlier decisions planned.

What changes is how the parts meet. Uncommitted parts of a solution now stay in the plan as `planned` tours, which a later decision may revise. The rules of the plan, that is, its lifecycle, the revision scope, the state information and the commands with their validation, move to a new package `ware_ops_algos.planning`, because they need no simulator. Both engines stay in casim.

This version replaces the drafts of 2026-10-12 and of this morning. The corrections to the implementation on `codex/revision-scopes` are planned in `docs/revision_scopes_corrections.md`. Statements about code refer to casim `6cd1705` and ware\_ops\_algos `90c9364`, unless they name the base commits `67f9f31` and `34bd157`.

Three companion documents hold what this document only refers to: [Forecasting and uncertainty in CASIM](https://claude.ai/code/artifact/fd302ba1-31bd-4b66-87e1-31efcfd75485) (belief, forecasting, productive use), [Benchmarking and showcasing CASIM](https://claude.ai/code/artifact/7219a7de-d53a-4189-8c0d-92e718eba686) (benchmarks, decision latency) and [ware\_ops\_algos branch review](https://claude.ai/code/artifact/35000c9a-c86a-4f97-8109-ed0eb69d16ad) (CASOP consistency, card conventions, type consolidation).

| Property in the CASIM paper | Configuration key | Defined by |
| --- | --- | --- |
| Problem class and algorithms | `problem_class`, `solver` | `TAXONOMY`, algorithm cards, `CoSySolver` |
| Trigger events | `triggers` | casim events |
| Conditions | `conditions` | `Condition` subclasses (ConditionPolicy in the paper) |
| State information | `state_information` | `StateInformation`, read by the generic `StateAdapter` |
| Committed components | `commitment_policy` | `CommitmentPolicy` subclasses |
| Revision scope (new) | `scope` | `RevisionScope` |

## Layers

Each decision is a CASOP problem whose instance is derived from a persistent plan. CASOP has an instance with its data card, a problem class broken down by the taxonomy, algorithm cards, the mapper and CoSy; a pipeline maps the instance to solution objects, and nothing exists before or after it. CASIM adds time and process, the plan as an object that persists between decisions and is partly irreversible, the decision as a recurring and configured CASOP problem (when it runs, which part of the plan becomes its instance, how far it may change the plan, what becomes binding), admissibility and change cost relative to the plan, and applicability that depends on the decision's relation to the plan.

| Layer | Concern | Location |
| --- | --- | --- |
| Warehouse and workload | Static facts: layout, storage, articles, resources, orders | `domain_models` |
| Algorithms | Instance to solution objects; no knowledge of the plan or of time | `algorithms` |
| Synthesis | Which algorithms combine for a decision | taxonomy, cards, mapper; CoSy components and the online taxonomy in casim |
| Plan | What exists, what may still change, what a change is, whether it is admissible | `planning` |
| Decision | When it runs, what becomes binding, which candidate wins | `casim.decision_engine`, triggers, conditions |
| Process | Time, events, live state, execution record | casim `state`, `events`, `simulation_engine` |

Each crossing between layers has one place. The `StateAdapter` projects the plan into the decision's instance and is the only reader of live `State`. `to_commands` turns solution objects into plan changes and is the only code that knows both. `apply` turns plan changes into state and process events and is the only writer of live `State`. `decision_features` turns the state information and the revision scope into data-card features.

## State model

The layers describe who does what; the state model describes what a decision sees and changes. The state of the warehouse at time t has three components.

```latex
s_t = (W_t,\; P_t,\; B_t)
```

The warehouse W\_t consists of the CASOP domain objects in their current condition: layout, articles and storage, the orders revealed up to t, and the resources with position and availability, including physical conditions such as breaks and the dock. The plan P\_t consists of pick lists and tours with lifecycle status, schedule and execution progress. The belief B\_t is what the decision maker knows about the future: the planner information, including whether further orders can arrive and, where known, when the stream ends or how many orders remain. The first component is not called instance, because the paper uses I for the data card and the instance in CASOP's sense is what a decision is given. Entities and their physical condition belong to W\_t, their assignment to tours to P\_t; the executed part of a tour belongs to P\_t as its irreversible part, and the picker's position to W\_t.

A decision d is defined by a trigger, an observation o\_d (`StateInformation`), the admissible changes A\_d (`RevisionScope` with the physical limits of the lifecycle) and a commitment rule. Its instance and data card are derived from the state, and its solution becomes commands that the transition applies together with the exogenous events ω until the next decision.

```latex
I_d = o_d(s_t), \qquad \mathrm{card}(I_d) = \big(\mathrm{card}(W_t),\; \mathrm{card}(B_t),\; \phi(o_d, A_d)\big), \qquad s_{t'} = f\big(s_t,\, x_t,\, \omega_{(t,t']}\big)
```

These are the elements of Powell's framework with the state factored into warehouse, plan and belief. CASOP is the special case t = 0, P\_0 = ∅, B\_0 trivial because no further orders arrive, o\_d the full state and commitment of everything; then I\_d = W\_0. Whether a setting is offline or online therefore needs no label: it follows from the belief and the number of decisions. An algorithm that needs further arrivals, such as a waiting rule, requires that property of the belief; a static solver requires nothing and is applicable in both settings. The classification of Pardo et al. (2024) describes settings in the same terms (information availability, pickers, objective, tasks including waiting) and needs no mechanism of its own.

Whether some knowledge about the future exists (an arrival process, an observable end of the stream) is a static property of the setting and appears on the data card, where the mapper matches it. Its current value (no orders remain, the stream closed at t) changes during a run and is read by the algorithm at decision time; it never changes applicability (O16). Decision name, cause, reconsideration reasons, solve flag and information versions are bookkeeping of the decision engine, not part of the state an algorithm sees. Knowledge of the stream end needs no machinery of its own: it is a component of B\_t that arrives as an event (`OrderStreamClosed`) and is read as `input_closed`. What matters is when the event fires, which the branch fixes at the last order date (O19). Waiting still lives in `DecisionMemory`; Task 6.4 of the correction plan moves it into the plan after the verification gate (O17). The assignment of every type of the branch to this model, and the merges it allows, are in the review document of ware\_ops\_algos.

## Model

As in the paper, a decision is taken when a trigger fires and its conditions hold. The `StateAdapter` exposes the decision's state information, the `DecisionEngine` lets the solver produce candidate solutions and ranks them, and the `CommitmentPolicy` commits part of the selected solution. Two things are new. Candidates that would change the plan beyond the revision scope are removed before ranking. And the uncommitted part of a solution is kept as planned tours.

&#91;embedded content: decision flow · setup and runtime\]

### Plan and tour lifecycle

The `TourManager` holds the tours that are planned, scheduled or being executed, as in the paper. A tour's status only moves forward, along the transitions in the table. `TourManager.advance` is the only code that writes a status, and it rejects any other transition.

| Status | Meaning | Entered through | Next statuses | Physical bound |
| --- | --- | --- | --- | --- |
| `planned` | Part of the plan, not yet committed; a later decision may revise it | `CreateTour` | `scheduled`, `cancelled` | `rebuild` |
| `scheduled` | Committed; the picker will execute it | `CommitTour` | `pending`, `cancelled` | `rebuild` |
| `pending` | Picker is setting up at the depot | `Dispatch` (until Task 4.14: `PickerTourQuery`) | `started` | `insert` |
| `started` | Picker is underway | `TourStart` | `done` | `insert` |
| `done` | Delivered at the depot | `TourEnd` | none | `none` |
| `cancelled` | Removed before it started | `CancelTour` | none | `none` |

In the base code, `SchedulingCommitmentPolicy` committed the first scheduled job and dropped the rest, which were reconsidered at the next trigger. Now the rest stays in the plan as `planned` tours with picker and tentative times. The next decision sees them and may keep or revise them. `assigned` disappears: a tour is created with its picker and times.

Pickers execute only scheduled tours. Pick lists are committed when they are created, as in the paper's batching example. A decision whose solution ends at routing produces neither tours nor pick lists, and `casim check` rejects it.

Order status is derived from the order buffer, the pick-list buffer and the tours, not stored. `TruckDisruption` is the one event that still changes tours outside a decision; it is an open point.

The statuses, `TOUR_ORDER`, the transition table `TOUR_TRANSITIONS` and the physical bound `PHYSICAL` live in `ware_ops_algos/planning/lifecycle.py`. `TourStates` moves there from `casim/domain_objects/tour_model.py`, because `TourView` needs it.

### Revision scope and the physical bound

What a tour still accepts is a level of the ordered enum `Accepts`: `none < insert < rebuild`.

- `insert` means orders may be added and the unexecuted suffix of the route may be replaced. Completed picks, the edge being traveled, orders with a completed pick and occupied boxes stay fixed.
- `rebuild` allows any change: cancelling, changing membership, route, picker or times.

Removing orders from a started tour is not supported.

`PHYSICAL[status]` is the highest level the process allows: `rebuild` for planned and scheduled tours, `insert` while the picker sets up or is underway, and `none` for done or cancelled tours. It is a property of the warehouse, not of a configuration.

A decision's revision scope `RevisionScope(freeze_from, frozen_accepts)` cuts the lifecycle. Tours before `freeze_from` accept `rebuild`; tours from `freeze_from` on accept `frozen_accepts`, which is `none` or `insert`. The scope is admissible if no status in the decision's state information gets more than `PHYSICAL` allows:

```latex
a(s) = \begin{cases} \text{rebuild} & s \prec \text{freeze\_from} \\ \text{frozen\_accepts} & \text{otherwise} \end{cases} \qquad a(s) \le \text{PHYSICAL}(s)
```

`RevisionScope.accepts(status)` computes a(s) and is its only implementation: the `StateAdapter` uses it for `TourView.accepts`, `check_scope` for admissibility, and `decision_features` for the features.

The check runs when the configuration loads. For example, a decision that sees `pending` tours cannot freeze from `started`, because that would let `pending` tours accept `rebuild`. The error names the status, the level the scope gives it, and the bound.

```yaml
scope: {freeze_from: scheduled, frozen_accepts: none}   # planned tours may be rebuilt, committed tours are fixed
scope: {freeze_from: pending, frozen_accepts: insert}   # planned and scheduled tours may be rebuilt, running tours accept orders
```

### Decisions and state information

The base configuration is keyed by problem class. Now a scenario may contain two decisions of the same class, so a decision has a name and declares its class. By convention, the name is the problem class when a scenario has one decision of that class, for example `OBRSP`. A second decision is named after what it does, for example `active_tour_admission`.

As in the paper, each engine gets its own part of a decision's configuration, keyed by the decision's name. The configuration file keeps one entry per decision, so that rules across the parts, such as "a decision that sees planned tours needs a commitment policy", are checked in one place. Setup then hands each engine its part.

| Engine | Gets per decision | At the base, per problem class |
| --- | --- | --- |
| `SimulationEngine` | triggers, conditions, precedence, a `StateAdapter` configured with the decision's `StateInformation` and `RevisionScope` | triggers, conditions, a `StateAdapter` subclass |
| `DecisionEngine` | solver, `SolutionRanker(objective, change_penalty)`, `CommitmentPolicy` or none | `solver_map`, `commitment_policies`, the ranker inside `CoSySolver` |

The `StateAdapter` writes the state information and the scope into the snapshot. The plan view contains only what the state information exposes, and every exposed tour carries its level in `TourView.accepts`. Validation therefore needs nothing but the snapshot, and `DecisionEngine.on_trigger(state_snapshot, action=None)` keeps its base signature: it finds the decision's solver, ranker and commitment policy by the name in the snapshot's context. `StateInformation` and `RevisionScope` come from `ware_ops_algos.planning.scope`, because they describe which part of the plan a decision sees and may change. Triggers, conditions and precedence belong to the `SimulationEngine`, because only a simulator has events and a clock.

`StateInformation` has five fields.

- `orders: bool` exposes the order buffer.
- `pick_lists: bool` exposes the pick-list buffer.
- `tours` is a set of tour statuses, and `resources` a set of `ResourceStates` (`off_shift`, `idle`, `reserved`, `busy`).
- `window: Window | None` limits what is exposed to `horizon_s` ahead of now. With `by: due_date`, it keeps orders and pick lists due within the horizon and requires the data card to declare `due_date`; with `by: start_time`, it keeps tours that start within it. The window applies regardless of other state such as the closure of the order stream.

The observation of a decision is defined by `state_information` alone, and its timing by `triggers` and `conditions` alone. The engine adds no rule of its own: no event bypasses the conditions, no fact of the simulator switches part of the observation on or off, and no resource enters the snapshot that `state_information.resources` does not select. An algorithm's precondition belongs to its card or to the algorithm, never to a condition of the decision, because other algorithms in the same repository would inherit it. The snapshot reports the warehouse as it is, including the types of its resources and orders. Task 4.15 of the correction plan removes the rules on the branch that break this (F57); the one that stays until Task 6.4 is that a due `Reconsider` of a waiting decision solves regardless of conditions.

Three rules complete the state information. The pickers of exposed started tours are always exposed, because admission needs their cart and speed. Once the order stream is closed, a due-date window no longer applies. A due-date window requires the data card to declare `due_date` on orders, and `casim check` rejects it otherwise.

A decision without a solver only commits. Its state information must expose `planned` tours, and it never changes a plan object.

### Selecting and committing a solution

`DecisionEngine.on_trigger(state_snapshot, action=None)` keeps its name, its signature and its module, `casim/decision_engine/decision_engine.py`. It runs these steps.

1. If the snapshot's `solve` flag is set, the solver returns its candidate solutions, keyed by stable names.
2. `to_commands` converts each candidate into the commands it implies against the snapshot.
3. `validation.check` removes every candidate whose commands touch tours outside the state information, exceed a tour's `accepts` level or `PHYSICAL`, or change what a started tour has executed.
4. The `SolutionRanker` selects among the remaining candidates, unless an RL action names one.
5. If the selected solution is a `WaitingSolution` with `action: wait`, the decision waits.
6. Otherwise, the `CommitmentPolicy` selects the planned tours to commit.
7. The `DecisionResult` carries the commands, the waiting state, the time the decision wants to run again, and the timings.

The ranking is:

```latex
\lambda^* = \operatorname*{arg\,min}_{\lambda \text{ feasible}} \; z_\lambda + \mu \cdot \text{changed}(\Delta_\lambda)
```

Here z is the objective of the candidate's schedule, μ the decision's `change_penalty`, and changed(Δ) the number of existing tours the candidate cancels, revises or reschedules; new tours do not count. A batching solution has no objective, so if a decision ends at batching and its pipelines produce more than one candidate, `casim check` rejects it. The base code silently took the last candidate.

Waiting is a pipeline stage after scheduling, as in the base code. The existing algorithms (`NoWaiting`, `OrderCountWaiting`, `HennWaiting`, `AnalyticStochasticWaiting`) receive the whole schedule in `WaitingInput.candidates` and return a `WaitingSolution` with `action` (`release` or `wait`) and an optional `reconsider_at`. When a policy waits until a known time, it delays the jobs of that picker accordingly, so the schedule stays consistent. The decision memory keeps `waiting` and `reconsider_at`, so a run without solving does not commit while the decision waits.

| Commitment policy | Commits | Asks to run again |
| --- | --- | --- |
| `CommitAllPolicy` | Every planned tour | never |
| `SchedulingCommitmentPolicy(n_jobs)` | The first `n_jobs` planned tours by start time | never |
| `WindowCommitmentPolicy(within_s)` | Planned tours whose ready time is at most now + `within_s` | when the next tour becomes ready, minus `within_s` |

The first two are the policies of the base code. `WindowCommitmentPolicy` is new and carries the rolling horizon. Every policy considers only planned tours of pickers in the state information, and commits nothing while the decision waits. A tour's ready time is its `start_time`, the moment its setup begins (Verification, time model).

If the conditions do not hold, `solve` is false. The engine then computes no candidates and applies only the commitment policy to the existing planned tours. In the base code, the trigger was ignored in that case.

CASIM applies the commands of one result together or not at all. Each decision has a version, and a `Reconsider` event that carries an older version is stale and skipped.

## Synthesis per decision

As in the paper, applicability is determined from the problem class and the warehouse context on the data card. What is new is that each decision gets its own data card. Its `decision_features` section is derived from the decision's state information and revision scope. The problem class is declared, because the scope says what a decision may change, not which decision variables it sets.

| Decision feature | Value |
| --- | --- |
| `incumbent` | `true` if the state information exposes planned or scheduled tours that accept `rebuild` |
| `routing_origin` | `live` if it exposes started tours that accept `insert` together with the order buffer; otherwise `depot` |

`ware_ops_algos/planning/decision_features.py` derives the features, next to the scope types they come from and in the same repository as the cards that require them. Cards require them like any other feature; for example, `IncumbentStartBatching` requires `incumbent: true`.

casim's `TAXONOMY` in `casim/pipelines/taxonomy.py` lists the online problem classes with their subproblems in pipeline order, as at the base commit. ware\_ops\_algos keeps its CASOP taxonomy unchanged (O14). `ATIP`, `RORSP` and `OBRSPW` are removed, because each was a class combined with a scope or a waiting stage. OBRSP has the subproblems `item_assignment`, `batching`, `routing`, `scheduling`, `waiting`, `positioning`, `admission` and `active_tour_routing`. A separate `sequencing` stage no longer exists: the sequencer algorithms left ware\_ops\_algos in `6a9a3d4`, and sequencing in the sense of OBRSP is the scheduling subproblem, as in ware\_ops\_pipes (F54).

`variant(problem_class, features)` derives which subproblems are synthesized:

| Decision features | Subproblems synthesized |
| --- | --- |
| `routing_origin: live`, `incumbent: false` | `item_assignment`, `admission`, `active_tour_routing` |
| `routing_origin: depot` | The class's subproblems without `admission` and `active_tour_routing` |
| `routing_origin: live`, `incumbent: true` | Rejected: no algorithm rebuilds planned tours and admits orders to started tours at once (open point O2) |

Each subproblem has one abstract CoSy stage, and a stage's parameters name exactly the stages whose output it can consume. Synthesis, not the composition of a repository file, then excludes invalid pipelines: depot routing consumes `AbstractBatching`, active-tour routing consumes `AdmittedTourBatch`, and a node that only assembles input from an earlier decision makes no choice of its own. One registry in `taxonomy.py` maps each subproblem to its abstract stage and its plan-candidate bridge; the solver derives both from it and `casim check` compares it with `variant()`. Repair and rebuild are not engine branches but synthesized alternatives: with an incumbent, the repository may hold both `FiFo` and `IncumbentStartBatching`, the incumbent card's requirement `decision_features.incumbent` admits the second, and the ranker chooses between the candidates with the change penalty (Tasks 4.13 and 5.0 of the correction plan, F52 and F53).

A decision that exposes neither orders, pick lists nor revisable tours is rejected, unless it has no solver.

`CoSySolver` configures the terminal task for the last subproblem of the variant that the cosy repository provides; `waiting` and `positioning` are optional. A card is applicable if its `problem_type` is one of the variant's subproblems and `DomainAlgorithmMapper` accepts its requirements. `casim check` reports every configured card the mapper excludes, together with its reasons.

An admission decision is therefore declared OBRSP, like the base decision. It exposes only started tours, so its variant has no batching or scheduling. This matches Lorenz et al., who treat intervention within the same OBRSP.

Algorithm cards follow the conventions of CASOP. A card describes an algorithm's subproblem, objective and requirements; a parameter value is configuration. Parameterized algorithms therefore use the configuration mechanism of the CASOP paper (Section 4.2) and `configs.yaml`: `IncumbentStartBatching` declares a component option for its router, and `OrderCountWaiting` a value option `min_orders`, each resolved into configured cards whose requirements merge those of the selected parts. Cards of CASIM-only subproblems live in the same directory as the CASOP cards, so CASOP applicability is pinned by an applicability lock that `woa check` compares against `main` (Tasks 4.12 and 5.0 of the correction plan).

## Packages and responsibilities

Each concept goes where its dependencies point. What describes a plan and how it may change depends on no simulator and goes to a new package, `ware_ops_algos.planning`; on the branch, `commands.py` and `conversion.py` already import nothing from casim. What decides when something happens, and what changes live state, stays in casim. That covers the `DecisionEngine`, the commitment policies (which schedule their own reconsideration), the ranking, the `SimulationEngine`, triggers and conditions, and the snapshot with its decision context and memory. Both engines of the paper therefore stay in casim, as at the base commit.

| Package | Owns | May import |
| --- | --- | --- |
| `ware_ops_algos.domain_models` | Static warehouse domain, unchanged | nothing outside the package |
| `ware_ops_algos.algorithms` | Algorithms, subproblem base classes, inputs and solution objects (including `AdmissionTour` and the `ActiveTour*` solutions), cards | `domain_models` |
| `ware_ops_algos.planning` (new) | Lifecycle, `Accepts`, `PHYSICAL`, `TourView`, `PlanView`, `StateInformation`, `RevisionScope`, decision features, commands, `to_commands`, `validation.check`, plan invariants | `domain_models`, `algorithms` |
| `ware_ops_algos.taxonomy`, `domain_algo_mapper` | CASOP taxonomy and applicability, unchanged | `domain_models`, `algorithms.algorithm_cards`, as at the base |
| `casim` | Events and process events, `State` and its managers, `apply`, `TourPlanningState`, `SimWarehouseDomain` with `DynamicInfo`, `DecisionContext` and `DecisionMemory`, `StateAdapter`, `SimulationEngine`, triggers, conditions, `DecisionEngine`, `DecisionSolver`, commitment policies, `SolutionRanker`, the online taxonomy with `variant`, configuration, `CoSySolver` and its components, CLI, trackers, visualization, RL environment | `ware_ops_algos`, cosy, luigi, hydra, pydantic |

Nothing in `ware_ops_algos` imports casim, cosy, luigi, hydra or pydantic, and `algorithms` does not import `planning`, so algorithm inputs stay independent of the plan types. A test in each repository checks these rules. Because `planning` sits above `domain_models` and `algorithms`, the plan types can use `Route`, `BatchObject` and `ScheduledJob` without the import cycle of O12. The name continues `TourPlanningState` and the planning-state methods of the base `State`.

For a CASIM user, the `_target_` paths of solvers, commitment policies and conditions stay as they are. The two blocks per problem class become one entry per decision, `state_adapter` gives way to `state_information` and `scope`, and the ranker leaves the solver block. An algorithm developer types dynamic inputs without importing casim. A researcher with their own simulator builds a `PlanView`, runs the algorithms, converts the result with `to_commands`, checks it with `validation.check` and the invariants, and applies the commands to their own state. CASOP is unaffected, because `planning` is new and its taxonomy stays as it is.

`DecisionSolver` is a protocol in `casim/decision_engine/interfaces.py` with one method, `solve(state_snapshot) -> dict[str, AlgorithmSolution]`. `CoSySolver` is one implementation; a fixed composition of `ware_ops_algos` algorithms is another. Validation, ranking and commitment happen in the `DecisionEngine`, so they apply to every solver alike. `SolutionRanker` therefore moves from `casim/pipelines` to `casim/decision_engine`: the paper configures the ranking rule with the `DecisionEngine` (Section 3.3), and validation has to run before ranking, which `CoSySolver` cannot do.

```text
ware_ops_algos/src/ware_ops_algos/
  algorithms/
    algorithm_interfaces.py   inputs and solution objects, incl. AdmissionTour, AdmissionInput,
                              WaitingInput, WaitingSolution, PositioningInput, PositioningSolution
    waiting/waiting.py        Waiting and its implementations, one module per subproblem as on main:
                              NoWaiting, OrderCountWaiting, HennWaiting, AnalyticStochasticWaiting
    admission/admission.py    Admission, RemainingRouteAdmission, CenterPositionAdmission
    positioning/positioning.py  Positioning, AislePatrolPositioning, CenterRelocationPositioning
    batching/incumbent.py     IncumbentStartBatching, router as a configuration option
  planning/                   new
    lifecycle.py              TourStates, ResourceStates, OrderStates, Accepts, TimeAttribute,
                              TOUR_ORDER, TOUR_TRANSITIONS, PHYSICAL
    models.py                 TourView, PlanView
    scope.py                  Window, StateInformation, RevisionScope
    decision_features.py      decision_features
    commands.py               commands, Violation, ViolationCode (moved from casim/decision_engine)
    conversion.py             to_commands (moved from casim/decision_engine)
    validation.py             check, check_scope (the admissibility of a scope)
    invariants.py             check_plan, check_progress, InvariantViolation
  taxonomy/taxonomy.py        CASOP taxonomy, as at the base commit

casim/src/casim/
  domain_objects/sim_domain.py  SimWarehouseDomain, DynamicInfo, DecisionContext, ReconsiderReason,
                                DecisionMemory
  domain_objects/tour_model.py  TourPlanningState
  decision_engine/
    decision_engine.py        DecisionEngine, DecisionResult
    commitment_policies.py    CommitmentPolicy, CommitAllPolicy, SchedulingCommitmentPolicy,
                              WindowCommitmentPolicy
    solution_ranker.py        SolutionRanker (moved from casim/pipelines)
    interfaces.py             DecisionSolver
  events/decision_events.py   PickListCreated (Dispatch and the resource events in operational_events.py)
  state/                      State, managers, apply.py (check, apply), views.py (OperationsView)
  simulation_engine/          SimulationEngine, StateAdapter, conditions
  config/                     schema.py (DecisionConfig, DecisionDefinitions), data_card.py
  setup.py                    resolve
  cli.py                      casim schema | catalog | check
  pipelines/                  CoSySolver, component templates, subproblems/, taxonomy.py (online
                              problem classes, variant)
```

Every subproblem has an abstract base class in its package that binds its input and solution types, as `Batching` and `Routing` do, and every algorithm subclasses the base of its subproblem. `Waiting`, `Admission` and `Positioning` are added for the subproblems that lacked one.

The pydantic schema in casim uses the dataclasses of `ware_ops_algos.planning.scope` and `casim.decision_engine.commitment_policies` as field types. Their invariants live in `__post_init__`, and `StateAdapter.__init__` calls `check_scope`, so they hold for a user who builds the objects in Python just as for one who writes YAML.

## Domain objects and decision types

Every structured value is a frozen dataclass or an enum. As in the paper, the snapshot extends the static warehouse domain with a `DynamicInfo` object. `DynamicInfo` keeps its name and its module, `casim/domain_objects/sim_domain.py`, but its loose fields become three typed parts. `PlanView` and `TourView` come from `ware_ops_algos.planning`; the other types in the first table are casim types.

| Type | Fields |
| --- | --- |
| `SimWarehouseDomain` | the `BaseWarehouseDomain` fields, plus `dynamic_warehouse_info: DynamicInfo` |
| `DynamicInfo` | `context: DecisionContext`, `plan: PlanView`, `memory: DecisionMemory` |
| `ReconsiderReason` (enum) | `wait`, `commitment_window`, `periodic` |
| `DecisionContext` | `decision: str`, `cause: tuple[str, ...]` (event types, read only by conditions), `reconsider: tuple[ReconsiderReason, ...]`, `input_closed: bool`, `solve: bool`; the decision time is `plan.time` |
| `PlanView` | `time: float` (the moment the plan view describes), `order_buffer: tuple[WarehouseOrder, ...]` (exposed, sorted by `(order_date, order_id)`), `pick_list_buffer: tuple[BatchObject, ...]`, `tours: tuple[TourView, ...]`, `available_from: Mapping[int, float]`; the resources are those of the snapshot |
| `TourView` | `tour_id`, `picker_id`, `status: TourStates`, `accepts: Accepts`, `batch`, `route`, `start_time`, `end_time` (the planned times, projected for committed tours), `earliest_start: float \| None` (Task 4.14), `processing_time`, `pick_cart`; for started tours also `cursor`, `pinned_prefix`, `remaining_route`, `routing_origin`, `occupied_bins`, `picking_until`, `holding: bool` |
| `DecisionMemory` | `considered_active_orders: frozenset[tuple[int, int]]`, `last_information_version: int`, `current_information_version: int`, `waiting: bool`, `reconsider_at: float \| None` |

Admission, after Task 4.13 of the correction plan: `AdmissionInput` holds `orders`, `tours` (the started `TourView`s with `progress`), `candidate_order_ids`, `considered_pairs` and `input_closed: bool`, as the waiting and positioning inputs do. `AdmissionSolution` holds `assignments` (tour, order), `considered_pairs` and `return_tour_ids: tuple[int, ...]`, the started tours that hold away from the depot and are sent back without added orders. Returning an empty holding tour once the stream is closed is a choice of the admission algorithm; the pipeline component that assembles the routing input only reads `assignments` and `return_tour_ids`.

`SimWarehouseDomain.orders` holds every order the pipelines may place: the exposed order buffer, the orders of planned tours the decision may rebuild, and the unpicked part of started tours it may admit orders to. `PlanView.order_buffer` holds only the exposed order buffer.

A tour whose route does not end at the depot holds at its last node until a later decision replaces its suffix, and `TourView.holding` reports that the picker has arrived there. Positioning uses this, so the snapshot carries no positioning mode. On the branch, `AdmissionTour` is the input type of admission algorithms, built by the admission component from a `TourView`. Task 4.13 replaces it by `TourView` with a `TourProgress`, which groups the execution fields of a started tour; the tables here describe the state before that task.

| Type | Fields |
| --- | --- |
| `Window` | `by: TimeAttribute` (`due_date` or `start_time`), `horizon_s: float > 0` |
| `StateInformation` | `orders: bool`, `pick_lists: bool`, `tours: frozenset[TourStates]`, `resources: frozenset[ResourceStates]`, `window: Window \| None` |
| `RevisionScope` | `freeze_from: TourStates`, `frozen_accepts: Accepts` (not `rebuild`) |
| `CommitAllPolicy` | no parameters |
| `SchedulingCommitmentPolicy` | `n_jobs: int >= 1` |
| `WindowCommitmentPolicy` | `within_s: float >= 0` |
| `WaitingInput` | `candidates: tuple[ScheduledJob, ...]` (the whole schedule), `current_time`, `input_closed`, `picker`, `layout`, `information`, `deadline_reached`; `single_order_service_times` is removed, because `HennWaiting` computes them with its router |
| `WaitingSolution` | `jobs` (possibly delayed), `action: Literal["release", "wait"]`, `reconsider_at: float \| None` |

| Command | Fields | Allowed when the tour accepts |
| --- | --- | --- |
| `CreatePickList` | `batch: BatchObject` | the order buffer is exposed |
| `CreateTour` | `job: ScheduledJob`, `earliest_start: float \| None` (Task 4.14, set from `start_times`) | the order or pick-list buffer is exposed |
| `CancelTour` | `tour_id: int` | `rebuild` |
| `ReplaceSuffix` | `tour_id: int`, `route: Route`, `added_order_ids: tuple[int, ...]` | `insert` or `rebuild`; it only adds orders and keeps the executed prefix |
| `Reschedule` | `tour_id: int`, `picker_id: int`, `start_time: float`, `end_time: float`, `earliest_start: float \| None` (Task 4.14, set from `start_times`) | `rebuild` |
| `CommitTour` | `tour_id: int \| None`, `creation_index: int \| None` (exactly one set) | a `planned` tour |

`InsertOrders` merges into `ReplaceSuffix`, and `Reassign` into `Reschedule`. `Violation` has the fields `code: ViolationCode` (`state_information`, `accepts`, `physical`, `executed_prefix`, `membership`), `tour_id: int | None` and `message: str`.

`DecisionResult` keeps its name and has the fields:

- `commands: tuple[Command, ...]`
- `waiting: bool` and `reconsider_at: float | None`
- `considered_active_orders: frozenset[tuple[int, int]]`
- `solver_name: str | None` and `objective_value: float | None`
- `solve_seconds: float` and `conversion_seconds: float`

Two further types stay in casim:

- `TourPlanningState` is the mutable execution record of a tour, as in the paper: its route field holds the latest committed route, original\_route the initial one, and executed\_prefix the nodes walked before the last route replacement; only `TourManager.advance` writes its status.
- `OperationsView` (`n_staged_pallets: int`) is what conditions may read beyond the snapshot. It stays until Phase 6b, where the warehouse's `Dock` holds the dock schedule and replaces it; breaks are resource availability (see Resources and dispatch).

## Configuration

The base configuration split each problem class into a `simulation_engine` block (state adapter, triggers, conditions) and a `decision_engine` block (solver, commitment policy). Now the generic `StateAdapter` needs no class of its own, so both blocks merge into one entry per decision, a `DecisionConfig` from `casim/config/schema.py`. The Henn reference in the new form:

```yaml
# scenarios/scenario_henn_online/config/decisions/henn_waiting.yaml
OBRSP:                            # decision name; OBRSPW on the branch until Task 4.11 renames it
  problem_class: OBRSP
  state_information:
    orders: true
    tours: [planned]
    resources: [idle, reserved]
  scope: {freeze_from: pending, frozen_accepts: none}
  solver:
    _target_: casim.pipelines.pipeline_runner.CoSySolver
    repo: ${cosy_repo}            # its components include HennWaitingNode
  commitment_policy:
    _target_: casim.decision_engine.commitment_policies.SchedulingCommitmentPolicy
    n_jobs: 1
  triggers: [OrderArrival, OrderStreamClosed, PickerIdle]   # FlushRemainingOrders goes in Task 4.15
  conditions:
    - {_target_: casim.simulation_engine.conditions.NbrPickersCondition, threshold: 1}
    - {_target_: casim.simulation_engine.conditions.NbrOrdersCondition, threshold: 1}
```

| Key | Question it answers | Values |
| --- | --- | --- |
| `problem_class` | Which problem do the pipelines solve? | a key of casim's `TAXONOMY` |
| `state_information` | Which state information does the decision get? | `orders`, `pick_lists`, `tours`, `resources`, `window` |
| `scope` | How far may the tours it sees change? | `freeze_from`, `frozen_accepts` |
| `solver` | Which pipelines produce candidates? | a `DecisionSolver`; omitted for a decision that only commits |
| `commitment_policy` | Which planned tours are committed? | `CommitAllPolicy`, `SchedulingCommitmentPolicy` with `n_jobs`, `WindowCommitmentPolicy` with `within_s`; omitted when the decision sees no planned tours |
| `change_penalty` | How much does changing a tour cost in ranking? | number at least 0, default 0 |
| `precedence` | Which decision runs first at one timestamp? | integer at least 0, or omitted |
| `triggers` | Which events make the decision due? | event type names, and at most one `{every_s: n}` |
| `conditions` | When is solving worthwhile? | `Condition` configurations |

The objective comes from the data card, as before. Task 4.14 adds one key, start\_times with the values binding (default) and forecast, which says whether the scheduled starts of the decision's committed tours are binding earliest starts or only forecasts (see Planned and executed times); omitting it keeps today's behaviour. Waiting is chosen by the cosy repository's components, not by a decision key, and it applies only where its card accepts the data card.

The schema rejects:

- unknown keys, unknown `_target_` classes, and values outside an enum;
- `frozen_accepts: rebuild`, and a scope that gives an exposed status more than `PHYSICAL`;
- a window without a horizon.

Setup rejects:

- a `problem_class` that is not in `TAXONOMY`;
- a variant that the table in Synthesis per decision rejects;
- two decisions whose write sets overlap but whose precedences are not distinct;
- a configured component whose card the mapper excludes, together with the reasons.

The write set of a decision consists of `orders` if the order buffer is exposed, `pick_lists` if the pick-list buffer is, and `tours[s]` for every exposed status s whose level is not `none`.

Three commands support the user.

- `casim schema` prints the JSON Schema of the decisions file, generated from the models that validate it.
- `casim catalog` lists:
  - the tour statuses and their transitions;
  - the `Accepts` levels and `PHYSICAL`;
  - the commitment policies;
  - the problem classes with their subproblems;
  - the decision features;
  - the conditions;
  - the algorithm cards with their requirements.
- `casim check <config_dir> <config_name>` resolves the configuration exactly as an experiment would. For each decision it prints the declared class, the decision features, the derived subproblems, the number of pipelines, the excluded cards with their reasons, the write set and the precedence. Each error names the key, the value and what is allowed; `--json` prints the same as structured findings.

## Strategies as configurations

Each strategy below consists of one or two OBRSP decisions, which differ in triggers, state information, revision scope and commitment policy. Rolling horizon and admission to running tours combine freely.

| Strategy | Decision | Triggers | `state_information.tours` | `scope` | `commitment_policy` |
| --- | --- | --- | --- | --- | --- |
| Depot reoptimization | `OBRSP` | `OrderArrival`, `PickerIdle` | planned, scheduled | pending, none | `SchedulingCommitmentPolicy(n_jobs=1)` |
|  with admission to running tours | `OBRSP` | as above | as above | as above | as above |
|  | `active_tour_admission` | `OrderArrival`, `OrderStreamClosed` | started | started, insert | none |
| Rolling horizon | `OBRSP` | `OrderArrival`, `PickerIdle`, `every_s: 600` | planned, scheduled, pending, started | scheduled, none | `WindowCommitmentPolicy(within_s=300)` |
|  with admission at pick completion | `OBRSP` | as above | as above | as above | as above |
|  | `active_tour_admission` | `ActiveTourOpportunity` | started | started, insert | none |

In the rolling horizon, the state information also carries `window: {by: due_date, horizon_s: 3600}`, and ranking uses a `change_penalty`.

Admission to running tours needs a second decision, as long as no algorithm rebuilds planned tours and admits orders to started tours at once (open point O2). The admission decision sees no planned tours, so it has no commitment policy.

Both decisions write `orders`, so they need distinct precedences. `active_tour_admission` runs first, so that orders it admits are no longer in the order buffer when `OBRSP` runs at the same timestamp.

## Execution in CASIM

The `SimulationEngine` advances the event heap as described in the paper. After all events at a timestamp are processed, it runs the decisions due at that timestamp. A decision is due:

- when one of its trigger events occurred;
- when its own `Reconsider` event fires, which needs no trigger entry;
- when its periodic tick is reached. The periodic grid is anchored at the simulation start and stops once the order stream is closed.

Due decisions run in precedence order: lower values first, unset precedences after, ties by name. One event type may make several decisions due. Each decision runs at most once per timestamp (open point O3).

For each due decision, the `StateAdapter` builds the snapshot from the decision's state information, and the conditions set `solve`. The interface between the engines stays as in the base code: `run()` returns `(done, state_snapshot)`, the caller passes the snapshot to the `DecisionEngine`, and `step` commits the result.

```python
done, snapshot = simulation.run()
while not done:
    result = decision_engine.on_trigger(snapshot)
    simulation.step(result, snapshot)
    done, snapshot = simulation.run()
```

`step` does five things in order.

1. It applies the commands in two phases. `apply.check` verifies what only live state knows (the order is still in the buffer, the tour and picker exist) and raises `CommandViolation` on the first problem. Because the `DecisionEngine` has already removed infeasible candidates, such an exception is a programming error. `apply.apply` then changes state through manager methods only.
2. It stores the decision memory: considered active orders, waiting state and `reconsider_at`.
3. It increments the decision's version.
4. It schedules `Reconsider` at `reconsider_at` and the next periodic tick.
5. It inserts the process events that applying produced: one `Dispatch` for every resource whose queue changed (see Resources and dispatch), `PickListCreated` for a created pick list, and `TravelEvent` for a running tour whose suffix was replaced. `RoutingDone` is removed, because a decision that ends at routing is rejected. `SequencingDone`, whose only effect was to wake an idle picker, is replaced by `Dispatch`; `PickListDone` is renamed, because it reports the effect of a command, not the end of a pipeline stage.

The state changes in `apply`, not in the handlers of process events. That continues the code. At the paper commit `b8e8279`, `SequencingDone.handle` and `RoutingDone.handle` created no tours and cleared no buffers, and `StateAdapter.cleanup_state` applied the solution in `step`; at `67f9f31`, `State.commit_solution` took over. Commands make this step typed and generic, and they keep one decision's changes atomic: they are checked and applied together on the state the snapshot came from, so the next decision at the same timestamp sees them. Process events only continue the operational process, and they can trigger later decisions. A decision delay (O6) can later be modeled by one process event that carries a decision's commands and applies them at t + Δ. The paper's descriptions of `SchedulingDone` creating the `TourPlanningState` (Section 3.1) and of `cleanup_state` (Section 3.2) should be updated accordingly.

The `SimulationEngine` never calls the `DecisionEngine`, and nothing is assigned to it after construction.

A condition implements `get_decision(state_snapshot, operations) -> bool`. It reads the snapshot and the `OperationsView` and nothing else. The existing conditions are:

- `NbrPickersCondition`, `NbrOrdersCondition` and `NbrBatchesCondition`;
- `NewInformationCondition` and `VisiblePlanningWorkCondition`;
- `ActiveTourReadyCondition`;
- `DockCapacityCondition`; `BreakCondition` is removed, because a resource on a break is absent and no decision sees it as idle.

Solving takes no simulated time. Decisions due at one timestamp run one after another, each on a snapshot that already contains the commits of the decisions before it, so a command cannot become stale between snapshot and apply.

## Verification

Correctness is established from the model, not from the output of an earlier version. The base commits count the setup time twice (O1), and the walk-or-wait instances were built on a simulator with known faults, so their results are no reference. Four layers of checks replace the reference numbers.

| Layer | What it checks | Where |
| --- | --- | --- |
| Exhaustive checks | Over the whole finite domain: `TOUR_TRANSITIONS` moves forward in `TOUR_ORDER`; `PHYSICAL` never increases along it; every `RevisionScope` is accepted exactly when a(s) ≤ `PHYSICAL`(s) for all exposed statuses; `variant` is total; the write-set rule holds for every pair of decisions | `ware_ops_algos/tests/verification/` for the lifecycle and the scope, `casim/tests/verification/` for `variant` and the write sets |
| Plan invariants | After every step of every run: each order is in exactly one place; a picker's tours do not overlap; status changes follow `TOUR_TRANSITIONS`; executed prefixes and done tours never change; time never decreases. Also on random small instances (Hypothesis) | `ware_ops_algos.planning.invariants`, run by a casim test fixture |
| KPI recomputation and verification scenarios | KPIs recomputed from the tour records and the time model, independently of the trackers; nine scenarios small enough to derive every expected value by hand | `casim/tests/verification/` |
| Comparison with the base | Per decision, where the new code and the base diverge, and why | `scripts/compare_with_base.py` |

`invariants.check_plan(plan)` and `invariants.check_progress(before, after)` read plan views taken with full state information and return `InvariantViolation` objects. They need no simulator, so a researcher who brings their own can run them too.

The time model follows the resource parameters of the paper (speed, pick time, setup time), and plan and execution each count the setup exactly once:

```latex
\text{end} = \text{start} + s + \frac{d}{v} + n \cdot t_{\text{pick}}
```

A `ScheduledJob` occupies its picker from `start_time` to `end_time`. `start_time` is when the picker begins the setup at the depot, s is the picker's `tour_setup_time`, d the route distance, v the speed and n the number of picks. `earliest_free_assignment` sets `start_time = max(release_time, available_from)` and `end_time` as above. A committed tour takes these values as its planned times and, with `start_times: binding`, the start as its binding earliest start (see Planned and executed times). In the simulation, `Dispatch` makes the tour `pending` once its resource is free and the earliest start, if any, is reached, and `TourStart` follows s later. Without disturbances, the executed times equal the planned ones, which the KPI recomputation checks.

At the base, `earliest_free_assignment` already added s to `start_time`, and `PickerTourQuery` added it again (O1). The fix removes s from `start_time` and keeps it in `end_time`. Planned end times, and with them every CASOP objective computed from end times, stay the same; executed tours become one setup shorter. `HennWaiting` keeps adding s to the batch service time, as in Henn's model.

### Resources and dispatch

A resource's queue is the ordered list of its committed (`scheduled`) tours; a `planned` tour may carry a resource as part of a draft, but it is not in the queue and is never dispatched. A resource's state is derived from its tours and its availability and is never stored. It is `BUSY` if the resource has a `pending` or `started` tour, otherwise `OFF_SHIFT` if it is absent, otherwise `RESERVED` if its queue is not empty, and otherwise `IDLE`. These are the existing `ResourceStates`, which decisions use to select the resources they see, so the configuration keys do not change. One meaning does: on the branch a draft with a resource already makes it `RESERVED`, under this rule it does not, because a draft reserves nothing. Availability is one per-resource fact of W\_t, changed by two events; a resource is absent until its first arrival, as today. `PickerArrival(picker_id)` makes the resource present. `PickerDeparture(picker_id, duration)` makes it absent from its next free point on, that is immediately or at the end of its current tour; with a duration it returns by itself after that time, which models a break, and without one it stays absent until the next arrival, which models a shift end. A tour is never interrupted.

One mechanism hands work to a resource. Every change of a resource's queue (a committed tour, a reschedule of a committed tour for its old and its new resource, a cancelled committed tour) and every change of its availability increases its queue version and emits `Dispatch(picker_id, version)`; `TourEnd` emits one as well. An older `Dispatch` is stale. The handler leaves a busy or absent resource alone, makes a pending departure effective, and otherwise takes the first tour of the queue: it starts it if it has no earliest start or the earliest start is reached, queues `Dispatch` at the earliest start if not, and leaves the resource `IDLE` if the queue is empty. Push and pull are therefore not two mechanisms but two trigger configurations. Pull is a decision triggered by `PickerIdle`, which is emitted once per transition into `IDLE`, as in Henn's model where a batch is formed when a picker becomes free. Push is a decision triggered by arrivals or periodically that commits tours into the queues, as in rolling horizon planning. Within one timestamp, a decision's `Dispatch` is processed before the next decision at that timestamp, so later decisions see the started tours; `Dispatch` keeps the priority of the event it replaces, so the order of events at equal timestamps does not change.

Five checks hold after the last event of every timestamp: at most one `pending` or `started` tour per resource; no tour became `pending` before its earliest start or while its resource was absent; no present, non-busy resource has a first queued tour without earliest start or with its earliest start reached; per resource, idle, busy and absent time, computed from the log of state changes, sum to the simulated horizon; and the planned times of every queue equal its projection, except for a resource that is absent without a known return. Idle time means present and not busy, so it includes waiting for a later scheduled start. A transition table for `ResourceStates`, analogous to `TOUR_TRANSITIONS`, would add little: since states are derived and several changes can happen at one timestamp, almost every transition between snapshots is legitimate. The checks constrain the coupling of resources and tours instead, where the defects were. The model does not cover preemption of a running tour, tours shared by several resources, or a resource executing two tours at once; each would need a new rule in the handler, not a new mechanism (Task 4.14 of the correction plan, F55).

### Planned and executed times

A committed tour carries three kinds of time, which the branch stores in one field (`start_tour` overwrites the planned start with the actual one). The earliest start is binding and optional: the resource may not begin the tour before it. The planned start and end are the forecast that later decisions read (`available_from`) and that the overlap check and the KPIs use. The actual start and end are recorded by execution. Whether a decision's start times are binding is one key of the decision, `start_times: binding | forecast`, with `binding` as default because it is today's behaviour. With `binding`, the scheduled start becomes the earliest start, so the plan is a timetable: a tour never starts earlier and may start later, the rule that project scheduling under uncertainty calls railway scheduling. With `forecast`, no earliest start is set, so the plan is a sequence per resource: the next tour starts as soon as the resource is free, which corresponds to a priority-list policy in stochastic scheduling and to the per-operator work queues of task-feeding systems. Once Task 6.4 moves waiting into the plan, a wait in Henn's sense would be a binding start; whether rolling horizon and IJPE should use forecasts is decided when their configurations are ported, not assumed here.

After every start, end and queue or availability change, the engine projects each affected queue: the first tour is planned at the later of its earliest start and the time the resource becomes free (now, the planned end of its current tour, or the end of its break), every later tour at the later of its earliest start and its predecessor's planned end. For a resource absent without known return the planned times stay as they are. Tour views keep the fields start\_time and end\_time, which now hold the projected times, and gain earliest\_start. The projection is arithmetic on the committed sequence and never reorders it; reordering, reassigning or retiming is a decision, expressed as `Reschedule`. The tracker keeps the planned times at commit, so the deviation of executed from committed times can be reported. In deterministic scenarios both modes give the same result unless a tour ends earlier than planned; with stochastic durations they differ in the expected way, and the projection keeps the forecast that decisions see consistent with execution.

Two cases need no engine mechanism. A shared pool, where the next free resource takes the next piece of work, is the pick-list buffer together with a decision triggered by `PickerIdle` that routes and assigns one pick list to that resource; the assignment rule stays an algorithm, which is also how task-feeding products describe it (work fed to operators by equipment, location and priority). A wave release at a fixed time without a fixed resource is a periodic or timed trigger on such a decision. The two binding dimensions of a commitment, to a resource and to a time, are thus expressed by when the decision runs and by `start_times`, not by new tour states.

### Scenarios and exogenous information

In the state model, the exogenous information ω between two decisions changes the warehouse W\_t or the belief B\_t, and only decisions change the plan P\_t. A scenario is therefore fully described by its initial state (W\_0, B\_0), the stream of exogenous events, and its decisions. The loader produces the first two parts from the data, and the configuration declares the third.

The stream consists of a closed set of event types: order arrivals, updates of known orders (a new due date, a cancellation), picker arrivals and departures, updates of the departure schedule, named timers, and the closure of the order stream. Their handlers change W\_t or B\_t and nothing else. A disruption is thus never a mechanism of its own. A truck that leaves earlier is a `DepartureUpdate` together with `OrderUpdate`s for the orders due at it, emergency orders are arrivals with a near due date, and a lost shift is a picker departure. How the plan reacts is decided by the decisions triggered by these events, within their scope, for example a rescheduling decision that may cancel scheduled tours. On the branch, `TruckDisruption` cancels tours in its handler, so an exogenous event changes the plan without a decision; that is the case this rule excludes.

Perturbations are declared transformations of the stream, applied before the run: inject orders, reveal future orders earlier, update due dates, move a departure, remove pickers for a time. Each is typed and seeded, selects the orders it affects by due date, arrival window, customer or volume, and records its changes in the trace. Every study can use them, so a disruption study needs configuration, not code.

A scenario is one warehouse case: a literature benchmark, a study on company data, or later a shadow of a running WMS. It has three parts that follow the state model. Its source gives (W\_0, B\_0) and the exogenous events as an iterator in time order, so that a replayed file, a generator and a live feed share one interface. Its decisions are configured as above. Its runs are sweeps, variants of the exogenous events (for example disruptions), and for a benchmark the published reference values. Readers for public formats are sources in the library; a scenario on new data adds its own source. One runner, `casim run`, validates and executes every scenario and writes the same outputs, and `casim check --all` rejects any other structure. Phase S of the correction plan specifies them (F58, F66); how users create new scenarios from the command line is still open.

The base results at casim `67f9f31` and ware\_ops\_algos `34bd157` remain a point of comparison:

| Scenario | Orders | Tours | Decisions | Makespan | Distance | Mean order completion |
| --- | --: | --: | --: | --: | --: | --: |
| Henn online (`72s-100-75-0`) | 100 | 25 | 26 | 34,702 | 11,603 | 14,584.174 |
| Stochastic waiting | 4 | 2 |  | 87.085 | 76 | 54.835 |
| Walk-or-wait, explicit example | 6 | 3 |  | 167 | 160 | 107.167 |

Every difference needs a demonstrated cause: the O1 fix, a base defect shown by an invariant or a verification scenario, or a change this design makes. The branch at `6cd1705` gives 26 tours, 12 decisions and distance 11,402 for Henn, with six single-order tours where the base has none. Three causes are known: the waiting algorithm saw one candidate at a time, a `Reconsider` committed an old planned tour instead of re-solving, and the snapshot ordered orders differently from the base order buffer.

The base Henn configuration used `CommitAllPolicy`, which committed exactly one job because the waiting solution contained only that job. Now planned tours persist, so the port uses `SchedulingCommitmentPolicy(n_jobs=1)`. Once the user accepts the explained differences, the new KPIs are pinned as regression values; they guard against regressions and do not establish correctness.

## Uncertainty and forecasting

Forecasts and uncertainty are not part of this plan. Their design, from the separation of truth and belief over the forecasting subproblem and the scenario objective of the `SolutionRanker` to productive use and the literature, is in a separate document: [Forecasting and uncertainty in CASIM](https://claude.ai/code/artifact/fd302ba1-31bd-4b66-87e1-31efcfd75485). The only fact this document relies on is that a forecast is belief and belongs in `PlannerInformation`, not in `PlanView`.

## Open points

| # | Open point | Current handling |
| --- | --- | --- |
| O1 | The setup time is counted twice: `earliest_free_assignment` adds it to `start_time`, and `PickerTourQuery` adds it again. This already holds at `67f9f31`. | Fixed: `start_time` is the begin of the setup, `end_time` includes it, and `PickerTourQuery` adds it once; verification scenario 1 checks it |
| O2 | No algorithm rebuilds planned tours and admits orders to started tours in one decision | `variant` rejects the combination; admission is a second decision |
| O3 | Each decision runs at most once per timestamp, so events its own commit produces at t do not make it due again at t | Documented limitation |
| O4 | Removing orders from a started tour is not supported | No `Accepts` level for it |
| O5 | There is no "keep the plan unchanged" candidate; the incumbent enters only through repair pipelines such as `IncumbentStartBatching` | Repair pipelines |
| O6 | Solving takes no simulated time, so computation delay is not modeled | By design for now; budgets in iterations, recorded solve times and a configured latency are analysed in the benchmarking document |
| O7 | `AnalyticStochasticWaiting` assumes the walk-or-wait S-shape route, and a card cannot require a component of another subproblem | Stated in the card description |
| O8 | The analytical policy needs exactly q−1 orders in the order buffer. That is a state at decision time, which the mapping cannot check | `NbrOrdersCondition` ensures it; the algorithm raises otherwise |
| O9 | `TruckDisruption` cancels tours outside a decision | To become an exogenous event plus a decision |
| O10 | `henn_rolling` synthesizes due dates (arrival + 5,400 s) | Labelled synthetic; not comparable with Henn |
| O11 | `IncumbentStartBatching` assumes one cart type for all pickers | IncumbentBatchingNode rejects heterogeneous carts at run time; the card requires the cart features |
| O12 | The solution objects `Route`, `BatchObject`, `RouteNode` and `RoutingOrigin` live in `algorithms/algorithm_interfaces.py`, which imports domain objects, so plan types in `domain_models` would form an import cycle | Resolved by placing the plan types in `ware_ops_algos.planning`, above `domain_models` and `algorithms` |
| O13 | `SchedulingCommitmentPolicy(n_jobs=1)` commits one tour per decision, not one per picker; with several idle pickers, the others get their tour at the next trigger | Sufficient for the base scenarios, which have one picker |
| O14 | There are two taxonomies: the CASOP taxonomy in ware\_ops\_algos and the online taxonomy in casim. Their class names overlap with different meanings (OBRP has batching\_routing in one and not in the other), and OBSRP in CASOP and OBRSP in the paper may name the same problem. The branch merged both into ware\_ops\_algos and changed the CASOP entries for SPRP and OBRP. | Branch change reverted; both stay separate until the user decides how to unify them. A unified registry would follow the dimensions of Pardo et al. (2024) without an online label (State model); cards of CASIM-only subproblems must stay inapplicable in CASOP (Task 4.12) |
| O15 | BaseWarehouseDomain mixes the warehouse (layout, storage, articles, resources), the workload (orders, planner information) and the task (problem class, objective), and it is a plain mutable class. In CASIM the task changes per decision while the warehouse stays fixed. | Unchanged in this plan; a separate change after the verification gate, because it touches the mapper, the data cards, all loaders and ware\_ops\_pipes |
| O16 | DynamicInfo subclassed WarehouseInfo at the base; on the branch it is a frozen dataclass. Features of the dynamic state belong on the data card, but at two levels: what a decision can see (decision\_features, fixed at setup) and what a snapshot contains (runtime features, such as the q−1 orders of O8). Runtime features would need applicability checks per decision, which changes the candidate set and with it the stable candidate names an RL action relies on. | This plan keeps decision\_features at setup. The same split applies to the belief: whether knowledge exists is static, its current value is read at decision time (State model) |
| O17 | Waiting and commitment both decide whether a tour is released now: the waiting stage through action wait, the commitment policy through DecisionMemory.waiting. Conjecture, not verified: if waiting only delayed start times, the commitment policy could decide alone, and WaitingSolution.action, DecisionMemory.waiting and their coupling would disappear. Under the Henn configuration a committed but delayed tour stays revisable (freeze\_from: pending), and the decision sees reserved pickers, so a new order still triggers a re-solve. | Task 4.8 provides the mechanism: a planned tour with a future start is queried again at that time, so a wait can be a later planned start or the absence of commitment. Task 6.4 removes DecisionMemory.waiting and reconsider\_at after the gate, only if the regression values stay identical |
| O18 | Forecasting and optimization over distributions are not modeled: PlannerInformation has no point-forecast or scenario variant, there is no forecasting subproblem, and the SolutionRanker evaluates a single objective value. Forecasts that depend on the decision itself would not fit a pipeline stage, because the forecast would then depend on the plan. | Outside this plan; see the forecasting document |
| O19 | Both `experiment_henn.py` and the stochastic waiting hook emit `OrderStreamClosed` at the last order date, so a policy learns at the last arrival that nothing follows. This is oracle information; a wait-k policy releases its last orders without delay, which affects completion times at the end of every run. Operationally closure is either announced in advance (an order cutoff) or not observable. Policies such as `OrderCountWaiting` only terminate with closure knowledge or a timeout. Which closure assumption Henn (2012) and Gil-Borrás make has not been checked. The paper-results commit aaa8333 used a third variant: the engine emitted FlushRemainingOrders once the event heap was empty with orders still buffered, set the closure flag and forced a decision; 7e47bbe replaced it by the emission at the last order date, so the branch's Henn runs differ from the published setup in this respect. | Task 6.5 after Task 6.4, without belief infrastructure: a scenario parameter `stream_closure` (`at_last_arrival`, labelled as oracle; `at_exhaustion`, the published setup; `announced`; `never`), a card feature for observable closure required by policies without a timeout, the variant recorded in every result, and a check of the closure assumption of the reproduced sources; the default `at_last_arrival` keeps regression values identical |

## Related concepts in literature and industry

The model draws its parts from rescheduling theory, sequential decision analysis and planning software.

| Source | Concept | Counterpart here |
| --- | --- | --- |
| Vieira, Herrmann and Lin (2003) | Rescheduling policy (periodic, event-driven, hybrid) separate from the rescheduling method (repair, regeneration) | Triggers and conditions are the policy; state information, revision scope and pipelines the method |
| Powell (2019) | State, decision in a feasible set X\_t(S\_t), exogenous information, transition function; policy classes | The feasible set is what passes `validation.check` under the revision scope. The `DecisionEngine` is the policy side, the `SimulationEngine` the transition function. The four policy classes map onto configurations (see Policy classes) |
| Ulmer, Goodson, Mattfeld and Thomas (2020) | Route-based MDP: the state holds planned routes, and only the executed part is irreversible | Tours have an executed prefix and a revisable suffix; insertion and rerouting are plan changes, not new problems |
| Ichoua, Gendreau and Potvin (2000) | Diversion of a vehicle on its way to the next stop | Not supported: a running tour completes its current edge |
| MPS time fences; Sahin, Narayanan and Robinson (2013) | Frozen, slushy and liquid zones; replanning frequency, frozen interval, horizon length | Rolling horizon = `every_s`, `WindowCommitmentPolicy.within_s` and `state_information.window.horizon_s`. `scheduled` combines firming a planned order and releasing it |
| Lorenz, Otto and Gendreau (arXiv:2409.12619) | Reoptimization on order arrival, with changes only at the depot or also underway | Depot reoptimization (`OBRSP`, scope from `pending` at `none`) and its `active_tour_admission` variant |
| Gil-Borrás, Pardo, Alonso-Ayuso and Duarte (2020) | Orchestration method (Algorithm 1) for the online order batching problem | Its loop splits into the simulator (new orders, picker availability, collection) and two decisions: batching, and selection with routing |
| Timefold continuous planning | Planning window with history, published, draft and unplanned parts | `scheduled` tours are published, `planned` tours the draft |
| Timefold pinning | `@PlanningPin`, `@PlanningPinToIndex`: a fixed entity or list prefix | The pinned prefix of a started tour; insertion only after it |
| Timefold field service routing | `freezeTime`, `pinNextVisitDuringFreeze`: a few named options | `RevisionScope(freeze_from, frozen_accepts)` instead of a matrix |
| SAP status management | Per-status rules on which business transactions are allowed | `PHYSICAL` owned by the system; a revision scope can only narrow it |
| Kubernetes CRDs and admission control | Schema validation at apply time, immutable fields | Reject an inadmissible configuration at load time and an inadmissible change at commit time, naming the rule |
| Terraform plan and apply | Show the plan before applying it | `casim check` prints the resolved decisions before a run |

### Policy classes

Powell (2019) defines a policy as any rule X(S\_t) that maps the state to a decision, and sorts policies into four classes. In CASIM, the class of a decision follows from its configuration, not from the algorithms in its pipeline: the same batching, routing and scheduling algorithms yield a myopic policy or a lookahead policy depending on state information, revision scope, commitment and triggers. Each class has tunable parameters θ, and CASIM is the simulator in which they are tuned.

A policy function approximation (PFA) maps the state directly to a decision through a parameterized rule:

```latex
X^{PFA}(S_t \mid \theta)
```

In CASIM, the waiting algorithms are PFAs. `OrderCountWaiting` releases the first batch once it contains k orders (wait-k, θ = k). `HennWaiting` releases a single batch j at 2r\_i + st\_i − st\_j, where r\_i is the arrival time of its critical order i, st\_i the service time of a tour with that order alone and st\_j the service time of the batch; θ is empty, because Henn derives the rule from a competitive analysis. The commitment rules are PFAs as well: `SchedulingCommitmentPolicy(n_jobs)` commits the first n\_jobs planned tours (θ = n\_jobs).

A cost function approximation (CFA) solves a deterministic optimization model whose objective or constraints are modified by θ:

```latex
X^{CFA}(S_t \mid \theta) = \operatorname*{arg\,min}_{x \in \mathcal{X}_t^{\theta}(S_t)} \bar{C}^{\theta}(S_t, x)
```

The Henn decision without waiting is a myopic CFA: on each trigger, the pipeline batches, routes and schedules the orders in the buffer and minimizes makespan, ignoring future arrivals. Reoptimization in the sense of Lorenz et al. is the same policy with planned and scheduled tours exposed and rebuildable. The change penalty is a CFA modification: with μ = `change_penalty`, a candidate that cancels or revises three planned tours must improve the objective by more than 3μ to be selected, which damps plan nervousness.

A value function approximation (VFA) adds an estimate of the value of the state after the decision:

```latex
X^{VFA}(S_t) = \operatorname*{arg\,min}_{x} \left[ C(S_t, x) + \bar{V}(S_t^{x}) \right]
```

CASIM has no VFA yet. It would enter as a `SolutionRanker` that adds V̄ of each candidate's post-decision plan, for example the expected cost of the open orders given a model of arrivals, or through the RL environment with a value-based learner. A policy-gradient learner that selects among the candidates is a PFA over pipeline choice instead.

A direct lookahead (DLA) optimizes an approximate model of the future over a horizon H, implements only the first decision and solves again:

```latex
X_t^{DLA}(S_t) = \operatorname*{arg\,min}_{x_t} \left[ C(S_t, x_t) + \tilde{\mathbb{E}} \left\{ \min_{\tilde{\pi}} \sum_{t'=t+1}^{t+H} C\big(\tilde{S}_{tt'}, \tilde{X}^{\tilde{\pi}}(\tilde{S}_{tt'})\big) \,\middle|\, S_t, x_t \right\} \right]
```

The rolling horizon of the strategy table is a deterministic DLA, and each part of the formula has a configuration key:

| DLA part | CASIM | Value in the strategy table |
| --- | --- | --- |
| Lookahead model over horizon H | Pipeline over the orders exposed by the window | `state_information.window: {by: due_date, horizon_s: 3600}` |
| Plan for t' > t | `planned` tours | kept between decisions |
| Implement only x\_t | Commitment policy | `WindowCommitmentPolicy(within_s=300)` |
| Solve again | Triggers and `Reconsider` | `OrderArrival`, `PickerIdle`, `every_s: 600` |
| Decisions already implemented | Revision scope | `scope: {freeze_from: scheduled, frozen_accepts: none}` |

The lookahead model is deterministic because it contains only known orders; with forecast orders from `PlannerInformation` it would remain deterministic but anticipative. `AnalyticStochasticWaiting` is the one stochastic lookahead in CASIM: it weighs releasing now against waiting for the next arrival, with the expected detour computed from a known exponential arrival process and a fixed S-shape route. Since its horizon is a single arrival, we classify it as a one-step stochastic DLA; it could also be read as a PFA whose rule is derived analytically.

Two parts of the model are outside Powell's classes. The revision scope shapes the feasible set X\_t(S\_t): `PHYSICAL` is a property of the warehouse, and narrowing it, as the frozen zone of a rolling horizon does, is a design choice of the policy. And the separation between algorithm and policy has one exception: a waiting algorithm is itself a PFA inside the pipeline. O17 would sharpen this, because the waiting PFA would then decide only start times, and the commitment policy would execute them.

## Sources

- [Timefold Solver: continuous planning](https://docs.timefold.ai/timefold-solver/latest/responding-to-change/continuous-planning)
- [Timefold Solver: modeling planning problems (pinning)](https://docs.timefold.ai/timefold-solver/latest/domain-modeling/modeling-planning-problems)
- [Timefold Solver: real-time planning](https://docs.timefold.ai/timefold-solver/latest/responding-to-change/real-time-planning)
- [Timefold field service routing: pinning visits](https://docs.timefold.ai/field-service-routing/latest/real-time-planning/real-time-planning-pinning-visits)
- [Lorenz, Otto, Gendreau: Picking operations in warehouses with dynamically arriving orders: how good is reoptimization? (arXiv:2409.12619)](https://arxiv.org/html/2409.12619v1)
- [Lorenz, Otto, Gendreau: On picking operations in e-commerce warehouses: insights from the complete-information counterpart (arXiv:2410.14316)](https://arxiv.org/html/2410.14316v1)
- [Gil-Borrás, Pardo, Alonso-Ayuso, Duarte (2020): GRASP with Variable Neighborhood Descent for the online order batching problem, Journal of Global Optimization 78, 295–325](https://doi.org/10.1007/s10898-020-00910-2)
- [Gil-Borrás et al. (2021): A heuristic approach for the online order batching problem with multiple pickers, Computers & Industrial Engineering 160, 107517](https://burjcdigital.urjc.es/items/9cef676b-b087-406f-845b-45e3479618fe/full)
- [Gil-Borrás et al. (2023): The time-window strategy in the online order batching problem, IJPR (abstract)](https://grafo.etsii.urjc.es/en/publication/gil-borr-s-2023/)
- [Branke, Middendorf, Noeth, Dessouky (2005): Waiting strategies for dynamic vehicle routing, Transportation Science 39(3)](https://ideas.repec.org/a/inm/ortrsc/v39y2005i3p298-312.html)
- [InsideOpt: We OR folks believe we support making better decisions. Do we?](https://insideopt.com/blogs/news/why-or-is-in-the-rut-it-is-in)
- [InsideOpt: Seeker for sequential decision problems](https://insideopt.com/blogs/news/seeker-for-sequential-decision-problems)
- [Sellmann 2026: Beyond the Bounds: Learning to Optimize for Impact, OR/MS Today](https://pubsonline.informs.org/do/10.1287/orms.2026.02.16/full/)
- [Kuhlemann, Sellmann and Tierney 2019: Exploiting Counterfactuals for Scalable Stochastic Optimization, LNCS (CP 2019), doi 10.1007/978-3-030-30048-7\_40](https://ris.uni-paderborn.de/record/14890)
- [Malitsky and Sellmann 2010: Stochastic Offline Programming, Int. J. on Artificial Intelligence Tools 19(4), doi 10.1142/S0218213010000236](https://doi.org/10.1142/S0218213010000236); conference version ICTAI 2009, doi 10.1109/ICTAI.2009.23
- [Kadioglu, Malitsky, Sellmann and Tierney 2010: ISAC, Instance-Specific Algorithm Configuration, ECAI 2010](https://dl.acm.org/doi/10.5555/1860967.1861114)
- [Ansótegui, Sellmann and Tierney 2009: A Gender-Based Genetic Algorithm for the Automatic Configuration of Algorithms, CP 2009](https://pub.uni-bielefeld.de/record/2958256)
- [Ansótegui, Sellmann, Shah and Tierney 2021: Learning to Optimize Black-Box Functions with Extreme Limits on the Number of Function Evaluations, LNCS 12931, doi 10.1007/978-3-030-92121-7\_2](https://portalrecerca.udl.cat/documentos/64ad4b237340c15e5130940b)
- [Bliek et al. 2023: The First AI4TSP Competition: Learning to Solve Stochastic Routing Problems, Artificial Intelligence 319, 103918](https://arxiv.org/abs/2201.10453)
- Cited from knowledge, not opened: Birge 1982, The value of the stochastic solution in stochastic linear programs with fixed recourse, Mathematical Programming 24; Kleywegt, Shapiro and Homem-de-Mello 2002, The sample average approximation method for stochastic discrete optimization, SIAM J. Optimization 12(2); Rockafellar and Uryasev 2000, Optimization of conditional value-at-risk, J. of Risk 2(3).
- [Nextmv: In conversation with Meinolf Sellmann, creator of Seeker](https://nextmv.io/blog/in-conversation-with-meinolf-sellmann-creator-of-seeker)
- [Optsicom: OOBP instances and results](https://grafo.etsii.urjc.es/optsicom/oobp)
- Cited from knowledge, not re-checked: Vieira, Herrmann, Lin (2003), Journal of Scheduling 6; Powell (2019), EJOR 275(3); Ulmer, Goodson, Mattfeld, Thomas (2020), EURO Journal on Transportation and Logistics 9; Ichoua, Gendreau, Potvin (2000), Transportation Science 34(4); Sahin, Narayanan, Robinson (2013), IJPR 51; Henn (2012), Computers & Operations Research 39, as reported by Lorenz et al.; Elmachtoub, Grigas (2022), Smart "predict, then optimize", Management Science 68(1); Mandi et al. (2024), Decision-focused learning: foundations, state of the art, benchmark and future opportunities, JAIR 80; Bertsimas, Kallus (2020), From predictive to prescriptive analytics, Management Science 66(3).
- Code: [kit-dsm/casim](https://github.com/kit-dsm/casim/tree/codex/revision-scopes) at `6cd1705` (base `67f9f31`), [kit-dsm/ware\_ops\_algos](https://github.com/kit-dsm/ware_ops_algos/tree/codex/revision-scopes) at `90c9364` (base `34bd157`).
