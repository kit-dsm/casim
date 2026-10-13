# CASIM revision scopes and configuration design

Oct 12, 2026 · @Janik Bischoff

## Summary

CASIM and `ware_ops_algos` are restructured so that the question "what may this decision change" has one explicit, declarative representation per decision, the revision scope, and so that every change a decision makes to the simulation state is a typed, validated command. This document is the blueprint for that restructuring. It defines the concepts and where each lives in the code, the configuration surface for users and agents, the control flow of one decision, and the order in which the changes are implemented.

Admissibility of a change has three layers with one owner each: what is physically irreversible (owned by the simulator, fixed), what the operator allows to change (the revision scope, configured per decision), and what an algorithm can respect (declared on its algorithm card). The effective decision space of a decision is the intersection of the three. Solvers keep returning their typed solutions; CASIM converts each solution into typed plan commands relative to the projected snapshot, validates every command against the three layers and live state, and commits by command type instead of by problem name. Plans persist as scheduled tours, so later decisions modify them instead of solving from scratch.

Under this model, rolling horizon, Lorenz-style reoptimization at the depot, insertion of incoming orders into running tours and full reoptimization of started tours are not different problems. They are the same base problems (OBP, OBRP, OBRSP) under different triggers, scopes and commitment policies. The problem class becomes a derived and checkable property, which also gives the existing applicability mapping and the CoSy synthesis a precise place in the flow: decision features derived from the scope are added to a per-decision data card before pipelines are synthesized.

The baseline below refers to CASIM at `67f9f31` and `ware_ops_algos` at `34bd157`. Implementation is on the `codex/revision-scopes` branch in each checkout; both pre-existing working trees were carried forward. The blueprint records the intended end state. The implementation record below states what currently runs and which parts remain open.

### Implementation record (2026-10-13)

The runnable reference configurations are `scenarios/scenario_henn_online/config/henn_online_config.yaml`, `scenarios/scenario_stochastic_waiting/config/stochastic_waiting_config.yaml`, and `scenarios/scenario_walk_or_wait/config/walk_or_wait_config.yaml`. Run their experiment modules with `viz.launch=false`; use `python -m casim check <config-directory> <config-name>` to resolve the same named decisions without running a simulation. `docs/revision_scopes_baseline.md` records exact commands, fixed-seed outcomes, and repair measurements.

Lifecycle types, scoped projections, transactional commands, named decision setup, precedence, versioned reconsideration, scheduled drafts, explicit release, waiting after solving, and incumbent-start repair are implemented. `TourView.as_admission_tour()` adapts at the pipeline boundary so the core `AdmissionTour` algorithm contract stays stable. `IncumbentProvider` supplies selected drafts to the repair stage. All repositories declare `solver.output`, and CoSy synthesizes one `PlanCandidateEndpoint` with a typed source bridge for that stage. The Henn repair repository synthesizes both full and incumbent-start pipelines. The source experiment scripts remain executable and produce their normal traces and metrics.

The projected decision domain uses typed context, plan, memory, and operations views; CoSy's `InstanceLoader` emits these parts directly. Legacy scenario loaders may still populate `DynamicInfo` in their initial domain, but the decision projection does not use it. The plan-candidate endpoint carries each typed stage solution; conversion assembles the complete plan after waiting. Incumbent-start batching performs cheapest feasible insertion into drafts. A typed positioning stage now follows scheduling and offers aisle patrol or center relocation. The top-level `on_violation` setting supports atomic `reject`, whole-decision `drop`, and one fresh-snapshot `repair` attempt. Release gates use operational break and dock facts. Step 9 extensions remain outside this delivery.

The runnable rolling Henn configuration is `decisions=rolling_horizon data_card=henn_rolling cosy_repo=rolling_horizon`. Orders receive a due date 5,400 seconds after arrival. The decision sees overdue buffered orders and those due in the next 3,600 seconds, plus all selected drafts. A fixed 600-second periodic grid keeps checking even when the window is empty. It ranks full rebuild and incumbent repair, stores one schedule, and releases drafts whose planned start is within 300 seconds. A release-window event with unchanged information acts on the draft without running CoSy. `decisions=release_only_example` shows the optional second named decision with no solver; it uses the same `resolve` path and reports zero pipelines.

Wait-0 uses `policy=wait_0 cosy_repo=paper_patrol`: the positioning component adds an empty depot-to-depot patrol job, `no_waiting` leaves its start unchanged, and `release_all` starts it. `policy=center cosy_repo=paper_center insertion_repo=center` moves an empty cart to an accessible center node and holds there. An admitted order reroutes from the observed position, including a partially traversed edge; an unused cart returns to the depot through `RecallPositioning` when the stream closes. Positioning adds no pick position. CASIM chooses to carry the cart during center movement; Lorenz et al. leave that handoff unspecified, so this example is a center relocation policy, not a reproduction of the entire Reopt* policy.

## Blueprint for implementers

This section is the normative summary. The sections after it define each part in the detail needed to implement it; where one of them contradicts this section, that is an error in the document. The following invariants hold after the implementation is complete.

1. Every state change caused by a decision is a command from `casim/decision_engine/commands.py`, and only `State.apply_all` applies commands. Event handlers, state adapters and solvers do not change plan state; exogenous events carry information only. Until step 9, `TruckDisruption.handle` is the one exception.
2. A command is applied only if it is admissible under the physical matrix in `lifecycle.py`, the decision's `RevisionScope` and live state (expected version, orders still free). The commands of one decision are applied together or not at all, and `on_violation` decides what happens otherwise.
3. Selection and scope are written only in lifecycle statuses, `Accepts` values, forward windows on a `TimeAttribute` and numbers with the unit in the key name. No configuration value names a class that branches on a problem name.
4. The problem class is derived at setup from the selection and the scope and is never configured. Outside synthesis, no code branches on it.
5. A solver sees the snapshot only: `DecisionContext`, `PlanView`, `DecisionMemory`, `OperationsView`, the static `WarehouseInfo` and the forecast in `PlannerInformation`. It never sees `State` or the `RevisionScope`, only their consequences in `TourView` (`accepts`, pinned prefix, origin).
6. A plan persists as tours with status `scheduled`, the draft. Only the commitment policy makes a tour binding, through `ReleaseTour`, and pickers start only released tours. A waiting policy sets start times on drafts; it never creates, changes or releases tours.
7. Conditions gate the solve of a decision, never its release.
8. Every decision has a version in `State`. Every command and every `Reconsider` event carries the version it expects, and a mismatch makes it stale.
9. Decisions due at one timestamp run one after another in `precedence` order, each on a snapshot that includes the commits of the decisions before it. Two decisions whose write sets overlap must be ordered by `precedence`.
10. Every name a user or an agent can write has one definition, and the JSON Schema and `casim catalog` are generated from these definitions. Every rule on a configuration lives with the component that has the knowledge (config schema, lifecycle data, `DomainAlgorithmMapper`, CoSy constraints); `casim check` runs the same setup as a run and only reports its findings.

The table maps the new or changed modules to their contents and to the implementation step that introduces them. The term and vocabulary tables in the section Model define the names.

| Module | Contents | Step |
| --- | --- | --- |
| `ware_ops_algos/domain_models/lifecycle.py` (new) | `TourStates` (moved, gains `released`), `OrderStates`, `PickListStates`, `ResourceStates`, `Accepts`, `TOUR_ORDER`, `PHYSICAL`, `accepts_for`, `TimeAttribute` | 1, 3 |
| `ware_ops_algos/domain_models/planning.py` (new) | `DecisionContext`, `PlanView`, `TourView`; conversion to the stable `AdmissionTour` contract at the pipeline boundary | 2 |
| `casim/state/views.py` (new) | `DecisionMemory`, `OperationsView` | 2 |
| `casim/interfaces.py` | `StateProjector` and `DecisionSolver` protocols at the simulation and solving boundaries | 3, 5 |
| `casim/simulation_engine/state_adapter.py` | the generic `StateAdapter`, projection extensions | 3 |
| `casim/config/schema.py` (new) | `RevisionScope` and selection (step 3); `DecisionConfig`, the discriminated unions and the configuration validators (step 5) | 3, 5 |
| `casim/decision_engine/commands.py` (new) | command types, violation codes | 4 |
| `casim/decision_engine/conversion.py` (new) | `to_commands` per solution type, matching of candidate tours to drafts | 4 |
| `casim/state/state.py`, `casim/state/apply.py` (new) | `State.apply_all`, validators, one apply function per command, `decision_versions` | 4, 6 |
| `casim/decision_engine/decision_engine.py` | `DecisionEngine.on_trigger`: solve, waiting per candidate, conversion, ranking, commitment | 4, 7 |
| `casim/setup.py` (new) | `resolve`, `DecisionSetup`, derivation of problem and decision features | 5 |
| `casim/cli.py` (new) | `casim schema`, `casim catalog`, `casim check` (reporting only) | 5 |
| `ware_ops_algos/algorithms/decision_features.py` (new), algorithm cards, `domain_algo_mapper` | decision features, `contains` operator, exclusion reasons as data, the merged `TAXONOMY` | 5 |
| `casim/simulation_engine/simulation_engine.py` | `run`, `_project`, `step`, precedence order, scheduling of `Reconsider` | 6 |
| `casim/events/` | `Reconsider`; `WaitExpired` and `WaitingOpportunity` removed | 6 |
| `casim/decision_engine/waiting.py` | waiting evaluation and release rules `release_all`, `release_next_per_picker`, `release_within` | 7 |
| `casim/pipelines/` | `IncumbentProvider`, one plan-candidate endpoint, stage-mode coherence constraint | 8 |
| `ware_ops_algos/algorithms/` | `positioning` components (step 7); cheapest-insertion admission, `IncumbentStartBatching`, scheduler heap seeded with availability (step 8) | 7, 8 |

After every step the reference scenarios run, and from step 5 on `casim check` reports no error for their configurations. The section Implementation order gives the acceptance criterion of each step. An agent that writes a configuration works from `casim schema` and `casim catalog` for what is syntactically allowed and iterates with `casim check --json` until no error remains. An agent that implements a step works from the invariants above, the module table and the step's acceptance criterion, and treats the open points of that step as hypotheses to validate, not as settled design.

## Model

A decision is admissible only if it is allowed by physics, by the operator's revision policy and by the capability of the algorithm that produced it. Each layer has one owner, and the user only ever edits the second.

&#91;embedded content: decision flow under a revision scope · setup and runtime\]

The scope is used twice: at setup it becomes decision features on the data card that restrict which pipelines are synthesized, and at runtime it drives, together with the state adapter selection, the projection and the validation of every command.

Where an existing CASIM concept fits, its name is kept. New names are used only for things that have no representation in the current code, and the only existing name that changes is `AdmissionTour`, which becomes `TourView`. The table lists each term, what it replaces, when it exists and where it lives. Types that algorithms consume live in `ware_ops_algos`, so algorithms stay independent of CASIM, as `AdmissionTour` already is.

| Term | What it is | Replaces today | Exists | Lives in |
| --- | --- | --- | --- | --- |
| decision | one config entry: `triggers`, `conditions`, `state_adapter`, `scope`, `solver`, `waiting`, `commitment`, `precedence` | one key under `decision_engine.problems` plus the same key under `simulation_engine.problems` | config | Hydra config, validated by `casim/config/schema.py` |
| `triggers`, `conditions` | when a decision runs, and whether its solve is worth running | the same names today | config | config; classes registered in `casim` |
| `waiting` | the start-time decision for each picker's next draft, applied to every candidate plan | the waiting stage inside a pipeline | config | config; algorithms in `ware_ops_algos/algorithms/waiting/` |
| `commitment` | the release rule that makes drafts binding | the commitment policy, which truncates the solution | config | config; rules in `casim` |
| `precedence` | the order of decisions due at the same timestamp | the order in which opportunities leave the event heap | config | config; read by `SimulationEngine` |
| `state_adapter` | one generic adapter, configured by a selection over lifecycle statuses instead of subclassed | the eleven adapter subclasses | config and runtime | `casim/simulation_engine/state_adapter.py` |
| `scope` | what the decision may change: lifecycle cut plus `frozen_accepts` | status lists in `ReORSPAdapter`, the RORSP branch of `commit_solution`, `TruckDisruption`; the insertion logic of the active-tour path | config | config; values from `lifecycle.py` |
| lifecycle statuses, `Accepts` | the vocabulary every selection and scope is written in | collection membership, booleans, `TourStates` | always | `ware_ops_algos/domain_models/lifecycle.py` |
| `DecisionContext`, `PlanView`, `OperationsView` | the parts of the snapshot | the fields of `DynamicInfo` | runtime, per decision | `ware_ops_algos/domain_models/planning.py`; `OperationsView` in `casim/state/views.py` |
| `TourView` | a tour as a solver sees it, with pinned prefix and `accepts` | the live `TourPlanningState` objects in `active_tours`; it converts to the unchanged `AdmissionTour` algorithm input | runtime | `ware_ops_algos/domain_models/planning.py` |
| `DecisionMemory` | what a decision remembers across decisions | `State.considered_active_orders` | runtime | `casim/state/views.py` |
| command | one typed state change with the version it expects | the branches of `commit_solution`, `commit_active_route`, the ATIP branch of `step` | runtime | `casim/decision_engine/commands.py` |
| physical matrix | what can never be changed, per status | asserts in `TourManager`, implicit rules in the commit code | fixed | data in `lifecycle.py`, enforced by the `casim` validator |
| `Reconsider` event | a decision's request to run again at a time, from its waiting policy, its release window or its periodic trigger | `WaitExpired`, `WaitingOpportunity`, `wait_version` | runtime | `casim/events/`, scheduled by `SimulationEngine.step` |

### Layer 1: physical admissibility (fixed, not configurable)

This layer states what cannot be undone in the modeled warehouse. It is a fixed matrix per plan object and status, in the spirit of an SAP status profile, stored as data in `lifecycle.py` and enforced by CASIM's validator. Users do not configure it, apart from scenario facts on the data card such as dedicated bins. The last column is the same matrix in the `Accepts` vocabulary used by `scope` and `TourView`. The PENDING row is derived from the current code and is not confirmed (open point O1).

| Tour status | Orders (membership) | Route | Picker | Start time | Cancel | Allowed `Accepts` |
| --- | --- | --- | --- | --- | --- | --- |
| PLANNED, ASSIGNED, SCHEDULED, RELEASED | free | free | free | free | yes | `rebuild`, `revise`, `insert`, `none` |
| PENDING (setup running) | free | free | fixed | fixed | yes | `revise`, `insert`, `none` |
| STARTED | append; remove only orders with no completed pick | suffix after pinned prefix | fixed | fixed | not modeled | `revise`, `insert`, `none` |
| DONE, CANCELLED | fixed | fixed | fixed | fixed | no | `none` |

The analogous rules for other objects are short. An order with a completed pick is pinned to its tour. An occupied cart bin is pinned to its order, a free bin is assignable. The pinned prefix of a started tour is its completed picks plus the edge currently traveled, which is exactly Timefold's `PlanningPinToIndex` semantics.

Concretely, these three rules follow from where items physically are. Once one item of an order is in a picker's cart, that order cannot move to another tour or back to the buffer, because the item sits on that cart; an order with nothing picked yet can still move. With `box_can_mix_orders: false`, each order occupies one cart bin, and a bin that holds items stays with its order until the tour ends, so admission may only use empty bins (`RemainingRouteAdmission` checks exactly this via `n_boxes` and `occupied_bins`). For the route, take a started tour depot, A, B, C, D, depot, where the picker has picked at A and B and is walking from B to C. Everything up to and including the edge from B to C is fixed, because it is done or underway and CASIM finishes the current edge (`RoutingOrigin.edge_destination`). Only the part after C may change, for example inserting a new location E between C and D. Timefold calls this pinning a list up to an index: elements before the index stay as they are, and new elements can only be placed after it.

This layer only says what is physically possible; it never permits anything by itself. An insertion into a running tour happens only if all four hold at once: a decision whose trigger fires (for example on `ActiveTourOpportunity`), a scope of that decision that allows it (`frozen_accepts: insert`), the physical conditions above (a free bin, all picks of the order ahead of the current position), and an algorithm in the pipeline that can produce it. The default scope is `frozen_accepts: none`, so without an explicitly configured insertion decision, started tours are never changed.

One consequence for configuration: since a tour that is `pending` or later can no longer get `rebuild`, `freeze_from` must be `planned`, `assigned`, `scheduled`, `released` or `pending`, and the schema rejects later values. `released` is physically as free as `scheduled`; it is binding by agreement, which is why it is protected through the scope and not through this matrix.

### Layer 2: revision scope (user-configured)

A revision scope is declared per decision and can only narrow layer 1. In the configuration it sits next to the state adapter's selection. The `state_adapter` entry selects which objects enter the snapshot: orders by status and optionally by a time window, tours by status, and resources by status (see the lifecycle table below). `scope` states what may change: a cut on the tour lifecycle (`freeze_from`) and what frozen tours still accept (`frozen_accepts: none | insert | revise`). The defaults are the conservative ones, `freeze_from: planned` and `frozen_accepts: none`, so a decision changes no existing tour unless its scope says so. Objects that are selected but frozen are projected as fixed context, because a solver usually needs them (a scheduler must know when a picker becomes free). The subsection Selection and scope, further below, defines the form of the scope.

The existing problem names survive as descriptions of a base problem plus a scope, not as keys the code branches on: RORSP is ORSP with `freeze_from: pending`, and ATIP is the class CASIM derives for a selection of started tours with `freeze_from: planned` and `frozen_accepts: insert`, OBRP with batching replaced by admission and routing by active-tour routing.

### What happens to the state adapters

The state adapter stays as a concept, but the eleven subclasses are replaced by one generic, CASIM-owned `StateAdapter` that is configured in YAML instead of subclassed. The engine builds the `DecisionContext`; the adapter reads the decision's `state_adapter` selection and `scope` and builds the rest of the snapshot: `PlanView` (free orders, pick lists, resources with the time they become available, and `TourView`s with pinned prefix and `accepts`), `DecisionMemory` and `OperationsView`. The non-trivial logic in today's adapters is kept, but moves into CASIM as constructors of these types; the residual-route and `RoutingOrigin` computation of `ActiveTourAdapter`, for example, becomes the constructor of `TourView` for started tours. Users no longer write classes that rebuild `SimWarehouseDomain`.

A user who needs data the state adapter does not provide registers a projection extension. An extension receives a read-only view of the state and returns additional frozen context objects, such as a congestion estimate, that components can declare as an input. It cannot filter or remove objects from the snapshot, because that would be a hidden scope again, and it cannot mutate state.

The eleven adapters differ only in the selection they apply, so each becomes a configuration of the one generic adapter plus a scope. The table gives the mapping derived from the code at 67f9f31.

| Adapter | Orders | Resources | Tours | Becomes |
| --- | --- | --- | --- | --- |
| `OBPAdapter`, `OSBPAdapter` | buffer | all | none | `state_adapter: {orders: [buffered], resources: [off_shift, idle, reserved, busy]}`; the two adapters are identical |
| `ReOSBPAdapter` | buffer | available | none | same, `resources: [idle, reserved, busy]` |
| `OrderWindowAdapter` | buffer | not occupied | none | same, `resources: [idle, reserved]` (ignores `available` today, see below) |
| `HennWaitingAdapter` | buffer | `available_for_planning` | none | same, `resources: [idle]` |
| `ORSPAdapter` | pick lists | available and not occupied | started, scheduled, assigned as context | `state_adapter: {pick_lists: [buffered], resources: [idle, reserved], tours: [assigned, scheduled, started]}`, `scope: {freeze_from: planned}` |
| `ReORSPAdapter` | pick lists plus batches of unstarted tours | available | unstarted tours | `state_adapter: {pick_lists: [buffered], resources: [idle, reserved, busy], tours: [planned, assigned, scheduled]}`, `scope: {freeze_from: pending}` |
| `ActiveTourAdapter` | buffer plus residual orders of started tours | pickers of started tours | started | `state_adapter: {orders: [buffered], tours: [started]}`, `scope: {freeze_from: planned, frozen_accepts: insert}` |
| `RLOSBPAdapter` | earliest due date only | not occupied | none | `state_adapter: {orders: [buffered], resources: [idle, reserved]}`; the truncation to one order moves into the RL environment |
| `RLORSPAdapter` | first pick list only | not occupied | started, scheduled, assigned | `state_adapter: {pick_lists: [buffered], resources: [idle, reserved], tours: [assigned, scheduled, started]}`; truncation moves into the RL environment |

#### What can be configured, and where the vocabulary comes from

Every plan object has an explicit, monotone lifecycle owned by CASIM, and selection and scope are written only in terms of these statuses. In the current code only tours have an explicit lifecycle (`TourStates`). An order's status is implicit in which collection holds it (`OrderManager._order_buffer`, `_pick_list_buffer`, a tour's `order_numbers`), and a resource's availability is spread over two booleans (`available`, `occupied`) and `TourManager.has_future_tours`. The last column of the table is the derivation that the state managers implement.

| Object | Statuses, in lifecycle order | Derived today from |
| --- | --- | --- |
| Order | `buffered`, `batched`, `planned`, `in_progress`, `done` | order buffer; pick-list buffer; member of a tour not yet started; member of a started tour; completed tour |
| Pick list | `buffered`, `planned`, `in_progress`, `done` | pick-list buffer; batch of an unstarted tour; batch of a started tour; completed tour |
| Tour | `planned`, `assigned`, `scheduled`, `released`, `pending`, `started`, `done`, plus terminal `cancelled` | `TourStates`; `released` is new (set by the commitment policy) |
| Resource | `off_shift`, `idle`, `reserved`, `busy` | `available` false; available, not occupied, no queued tour; available, not occupied, queued tour; occupied |

Five rules keep the configuration tight. First, every value in `state_adapter` or `scope` is a member of one of these enums, a forward window on a declared time attribute, or a number whose unit is in the key name (`horizon_s`, `within_s`); there are no free predicates and no code paths. A window covers \[now, now + `horizon_s`\], so it is only defined on `due_date` and `start_time`; every visible order's `order_date` lies in the past. Second, each object is selected in exactly one form. Orders inside a selected pick list or tour arrive with it, so `orders` selects only `buffered` orders and `pick_lists` only `buffered` pick lists, and a tour's orders and batch are reached through the tour. Third, the scope applies only to selected objects: anything not selected is untouchable, and the validator rejects commands that touch it. Fourth, `frozen_accepts` takes `none`, `insert` or `revise`; `rebuild` is what tours before the freeze cut get, and the physical matrix rejects scopes that ask for more than physics allows (for example inserting into a `done` tour). Fifth, ordering and truncation (`order_by`, `limit`) are not selection: they restrict an action space, so the RL adapters' one-order and one-pick-list inputs move into the RL environment and are rejected in `state_adapter`.

Taking `window` as an example of what a user sees: it has exactly two keys. `by` is `due_date` or `start_time`. `due_date` filters orders and pick lists (a pick list by its earliest due date), and `start_time` filters tours by their planned start; objects without that attribute are not filtered. `horizon_s` is a positive number of seconds, and the window covers \[now, now + `horizon_s`\]. The allowed values come from the `TimeAttribute` enum in `lifecycle.py` and reach the user through the generated JSON Schema, which gives completion in an editor with a YAML language server, and through `casim catalog`. Anything else fails when the configuration is loaded, before the simulation starts, with a message that names the field, the value and the allowed values. For example, `by: order_date` produces `state_adapter.window.by: 'order_date' is not allowed (allowed: due_date, start_time); every visible order's order_date lies in the past`. A misspelled key such as `horizon` instead of `horizon_s` is rejected the same way, because the schema forbids unknown keys.

Because the enums and the schema are the single definition, they can also be exported as JSON Schema. A user, or an agent writing a configuration, can then list exactly which keys and values exist and get the same error message `casim check` produces, instead of discovering the vocabulary by reading adapters. The resource statuses replace the five predicates; whether `OrderWindowAdapter`'s "not occupied", which ignores `available`, was intended is open point O2.

Every name a user or an agent can write, and every name that shows up in a log, has exactly one definition, and the config schema is generated from those definitions rather than written by hand. The placement rule is the same as for the snapshot types: what algorithms read goes to `ware_ops_algos`, what describes CASIM's own state, events or configuration stays in `casim`. Besides the four status enums, `lifecycle.py` therefore also holds the lifecycle order that `freeze_from` compares against, the `Accepts` values shared by `scope.frozen_accepts` and `TourView.accepts`, the physical matrix as data, and the time attributes a window may use. Neither package has a config or CLI module today, so the modules marked new are new.

| Concept | Values | Defined in | Discovered through |
| --- | --- | --- | --- |
| object statuses | `TourStates` (exists in `casim/domain_objects/tour_model.py`; moves to `lifecycle.py` and gains `released`), plus the new `OrderStates`, `PickListStates` and `ResourceStates` (lifecycle table above) | `ware_ops_algos/domain_models/lifecycle.py` (new) | enums in the JSON Schema for `state_adapter` and `scope.freeze_from`; `casim catalog` |
| selectable statuses | `orders: [buffered]`, `pick_lists: [buffered]`, any `TourStates` value except `cancelled`, any `ResourceStates` value | `casim/config/schema.py`, derived from the one-form rule | schema enums per key |
| lifecycle order | position of each `TourStates` value | `lifecycle.py` | `freeze_from` accepts only `planned`, `assigned`, `scheduled`, `released` or `pending` |
| `Accepts` | `rebuild`, `insert`, `revise`, `none`; `frozen_accepts` takes the last three | `lifecycle.py` | `scope.frozen_accepts`, `TourView.accepts` |
| physical matrix | allowed `Accepts` per `TourStates` value; membership of `in_progress` orders fixed | `lifecycle.py`, as data | the schema rejects scopes that ask for more than physics allows; the validator reads the same table |
| time attributes | `due_date`, `start_time` for windows; `order_date` for information only | `lifecycle.py` | `state_adapter.window.by` |
| units | seconds, written into key names (`every_s`, `horizon_s`, `within_s`) | `casim/config/schema.py` | key names themselves |
| decision features (data card section `decision_features`) | `routing_origin` (`depot`, `node`, `next_node`, `mid_edge`), `pinned_prefix`, `membership_ops` (`append`, `remove`), `max_active_tours`, `incumbent` | `ware_ops_algos/algorithms/decision_features.py` (new) | card schema for authors; `casim check` prints the derived features per decision |
| snapshot types | `DecisionContext`, `PlanView`, `TourView` | `ware_ops_algos/domain_models/planning.py` (new) | algorithm interfaces |
| `OperationsView`, `DecisionMemory` | per scenario, per decision | `casim/state/views.py` (new) | condition and extension signatures |
| status derivation | collection or flag to status | methods on `OrderManager`, `ResourceManager`, `TourManager` | internal, the only mapping from bookkeeping to statuses |
| triggers | event class names; `{every_s: n}`, a periodic reconsideration request (new) | `casim/events/`, registered by entry point | enum of `triggers` in the schema; `casim catalog` |
| conditions | condition classes and their parameters, referenced by `name` | `casim/simulation_engine/conditions.py`, registered by entry point | schema with parameter types |
| solver repositories | Hydra config group files listing components; components declared by algorithm cards (`component_name`) | `config/cosy_repo/*.yaml` per scenario, cards in `ware_ops_algos` | `casim catalog` lists repositories; `casim check` lists the applicable components per decision |
| objectives | `distance`, `makespan`, `tardiness`; `change_penalty` weight (new) | `casim/pipelines/solution_ranker.py` | schema enum for `solver.objective` |
| commitment and waiting policies | release rules `release_all`, `release_next_per_picker`, `release_within` (`within_s`) under `commitment`; `no_waiting`, `order_count` (`k`), `henn`, `analytic_stochastic` under `waiting`; both referenced by `policy` | release rules in CASIM's `commitment_policies.py` (rewritten); waiting algorithms in `ware_ops_algos/algorithms/waiting/policies.py` with their cards; the pipeline's waiting stage is removed | schema with parameter types |
| precedence | optional non-negative integer per decision; lower values run first among the decisions due at one timestamp | `casim/config/schema.py` | schema; `casim check` names the overlapping write sets that require it |
| commands and violation codes | `CreatePickList`, `CreateTour`, `CancelTour`, `InsertOrders`, `ReplaceSuffix`, `Reschedule`, `Reassign`, `ReleaseTour`, `Wait`, `RecordConsidered`; `stale`, `scope`, `physics`, `order_not_free` | `casim/decision_engine/commands.py` (new) | decision log; `on_violation` documentation |
| decision config schema | `DecisionConfig` and its parts, and the top-level `on_violation` (`reject`, `repair`, `drop`) | `casim/config/schema.py` (new), generated from the rows above | `casim schema`; `schemas/decision.schema.json` regenerated in CI |
| `casim check`, `casim schema`, `casim catalog` | CLI commands | `casim/cli.py` (new), console script in `pyproject.toml` | `casim --help` |

### Layer 3: algorithm capability (card-declared)

An algorithm card states which decision features the algorithm can respect, for example routing from a node other than the depot, honoring a pinned prefix, or handling more than one active tour. This is discussed with the applicability mapping below.

### Plan commands instead of problem-specific commits

After solving, CASIM converts the solution into commands relative to the snapshot the solver saw: `CreateTour`, `CancelTour`, `InsertOrders`, `ReplaceSuffix`, `Wait` (full list below). Each command carries the identity of the object it touches and the version it expects (`route_version`, and for started tours also the `cursor`), so stale commands are rejected with optimistic concurrency, as in event-sourced systems. A conversion function is defined per solution type, not per problem: a `SchedulingSolution` against a snapshot with rebuildable tours becomes "nothing for unchanged tours, Reschedule or Reassign for moved ones, CancelTour and CreateTour for the rest"; an `AdmissionSolution` becomes `InsertOrders` plus `ReplaceSuffix`.

Every command is then validated against layers 1 and 2 and against live state. On violation the configured response is `reject` (fail loudly, the default for research runs), `repair` (for example re-attach a dropped pinned prefix) or `drop` (skip the command and log it, closer to what a production WMS does). `State` commits by command type only, so neither `State` nor `SimulationEngine.step` branches on `problem_class`. `TruckDisruption` becomes a pure information event (`OrderRevision`, see Exogenous information and perturbations) and cancels no tours; a cancellation happens only through a decision that this event triggers and whose scope allows it.

#### Control flow of one decision

Control-flow steps 1 to 3 correspond to the current `SimulationEngine.run`, extended by per-timestamp coalescing and precedence. Control-flow step 4 is the current pipeline run with repair candidates added. Control-flow steps 5 to 8 replace the ranking inside `CoSySolver.solve` and the body of `SimulationEngine.step`, that is the calls to `State.commit_solution` and `State.commit_active_route` and the ATIP branch.

1. `SimulationEngine.run` pops an event, skips it if stale, and lets it update operational state (`event.handle`). It records every decision that lists the event's type among its triggers, and once all events of the current timestamp are handled, it runs each recorded decision once, in precedence order (see Precedence and write sets). Several arrivals at one time therefore cause one decision, not several.
2. For each decision to run, the engine builds its `DecisionContext` (the time, the trigger types recorded at this timestamp as `cause`, the decision's name, its version from `State`, whether the input stream is closed), and the decision's state adapter builds the snapshot from that context, the selection and the scope (`PlanView`, `DecisionMemory`, `OperationsView`). The decision's conditions are evaluated on that snapshot. A failed condition skips the solve and waiting (control-flow steps 4 and 5), not the decision: if the snapshot holds drafts, `run` still returns it, marked `solve=False`, and the commitment policy runs on the draft in control-flow step 6, so a tour whose delayed start has arrived is released even if no new orders came in. A decision with neither anything to solve nor a draft ends here.
3. `run` returns the snapshot to the caller, as today.
4. `DecisionEngine.on_trigger` runs the synthesized pipelines. Each candidate returns a complete plan for the selected objects, unchanged tours included: a repair pipeline modifies the incumbent, a full pipeline re-solves.
5. Waiting, conversion and ranking: if the decision has a waiting policy, CASIM applies it to every candidate, setting the start of each picker's earliest draft and collecting its `reconsider_at`. `to_commands(solution, snapshot)` then turns every candidate into commands, with one conversion function per solution type and never per problem name. The conversion matches candidate tours to incumbent tours by order overlap and uses a carried `tour_id`, as in `ActiveTourBatch`, only as a hint, so it can tell an unchanged or moved tour from a new one without requiring every algorithm to preserve identifiers; its tie rule is open point O3. `SolutionRanker` then picks one candidate by its objective plus a change penalty counted from its commands.
6. Commitment: the commitment policy, a release rule such as `release_next_per_picker` or `release_within`, adds `ReleaseTour` commands for the tours of the chosen plan that become binding now, and a `Wait` command for each reconsideration time requested by the waiting policy or by its own window. Tours it does not release stay `scheduled` as a draft for the next decision. `DecisionEngine.on_trigger` returns the plan commands together with these.
7. Validation: `SimulationEngine.step` hands all commands of the decision to `State.apply_all`, which checks each against the physical matrix, the decision's scope and live state (the expected version, whether the orders are still free) before executing any. A violation is handled by `on_violation`.
8. Commit: if no violation remains, `apply_all` executes the commands of the decision together or not at all. Each command returns its follow-up events, which the engine pushes onto the heap, and the command list goes to the decision log. `step` increments the decision's version and schedules reason-tagged waiting, release-window and periodic `Reconsider` checks, which closes the loop back to control-flow step 1.

#### What each command does

Most commands reproduce something the current commit code already does. `Reschedule` and `Reassign` are new; persistent plans need them to change a scheduled tour without cancelling it. `ReleaseTour` is new too and is produced only by the commitment policy. `RemoveOrders`, which `frozen_accepts: revise` needs, is added in implementation step 9.

| Command | Produced from | Checked before apply | Effect on State | Follow-up events | Replaces today |
| --- | --- | --- | --- | --- | --- |
| `CreatePickList(batch)` | `BatchingSolution` | orders still buffered | batch added to the pick-list buffer, its orders become `batched` | none | `BatchingSolution` branch of `commit_solution` |
| `CreateTour(key, route, picker_id, start, end)` | `SchedulingSolution` | orders or pick list still free (or freed by a `CancelTour` in the same decision), picker exists | tour created with status `scheduled` (draft), cart bins set, orders or pick list become `planned` | none until released | `add_sequencing_to_planning_state` and buffer clearing in `commit_solution`, `commit_waiting` |
| `ReleaseTour(tour, expected_version)` (new) | the commitment policy | tour `scheduled`, version unchanged | status `released`; the picker may start it | `PickerTourQuery` if the picker is free, as `SequencingDone.handle` does today | the truncation in `SchedulingCommitmentPolicy` |
| `CancelTour(tour_id, version)` | a plan that does not reproduce a rebuildable tour | tour `accepts: rebuild`, version unchanged | status CANCELLED, tour removed from the picker queue, its orders or pick list back to `buffered` or `batched` | none | RORSP branch of `commit_solution`, cancellation inside `TruckDisruption.handle` |
| `InsertOrders(tour_id, order_ids, version)` | `AdmissionSolution` assignments, repair pipelines | tour `accepts` is `rebuild`, `insert` or `revise`, orders still buffered, enough free bins, version unchanged | orders added to the tour, free bins assigned, orders become `planned` or `in_progress` | none | first half of `commit_active_route` |
| `ReplaceSuffix(tour_id, route, version)` | routing result for a changed tour | route starts at the tour's current origin (the depot for an unstarted tour), keeps the pinned prefix, covers all residual and inserted picks, ends at the depot, version unchanged | new route installed, cursor reset for started tours, travelled distance booked, `route_version` incremented | `TravelEvent` with the new version for a started tour | second half of `commit_active_route` |
| `Reschedule(tour_id, start, end, version)` (new) | `SchedulingSolution` that keeps a tour but moves it | tour `accepts: rebuild`, version unchanged | planned start and end changed, picker queue re-sorted | none | nothing; today the tour is cancelled and recreated |
| `Reassign(tour_id, picker_id, version)` (new) | `SchedulingSolution` that moves a tour to another picker | tour `accepts: rebuild`, target picker exists, version unchanged | tour moved between picker queues | none | nothing; today the tour is cancelled and recreated |
| `Wait(reconsider_at, reason)` | the waiting policy's `reconsider_at`, and `release_within` for its window entry | none | the requested time and reason are recorded for the decision | a versioned `Reconsider(decision, at, version, reason)` after commit | wait branch of `commit_waiting`, `DecisionEngine.solution_to_events`, `WaitExpired` |
| `RecordConsidered(pairs)` | `AdmissionSolution.considered_pairs` | none | pairs added to `DecisionMemory` | none | `state.considered_active_orders.update` in `step` |

For example, a decision with `freeze_from: pending` that rebuilds the unstarted tours emits a list such as `CancelTour(12, v3), CancelTour(14, v1), CreateTour(...), CreateTour(...)`. An active-tour insertion becomes `InsertOrders(7, [41], v2), ReplaceSuffix(7, route, v2), RecordConsidered(...)`, and the validator rejects the pair if tour 7 has moved to version 3 in the meantime.

### Commitment

Commitment decides which part of the chosen plan becomes binding now. The whole plan persists as state, and the commitment policy only releases tours of it.

#### Persistent plans

The plan is state, not a by-product of a solve. A decision applies its whole plan: every job becomes a `CreateTour` with status `scheduled`, held by `TourManager` like any other tour. Which part becomes binding is decided by the commitment policy, which releases tours (see below); later decisions keep released tours fixed through `freeze_from: released`, while scheduled tours stay revisable. At the next trigger the state adapter projects the planned tours (`tours: [planned, assigned, scheduled, released]`) as `TourView`s, so the incumbent plan is an input of the next decision rather than something to reconstruct.

Modifying that plan instead of solving from scratch needs three pieces. First, a repair pipeline among the CoSy candidates: insert new orders into existing tours with the same admission and rerouting components the active insertion uses (for an unstarted tour the origin is the depot and the pinned prefix is empty), then reschedule. Full re-solve pipelines stay in the repository as alternatives. Second, commands that change a tour in place: `InsertOrders` and `ReplaceSuffix` apply to any tour whose `accepts` allows it, and two new commands, `Reschedule` and `Reassign`, move a scheduled tour in time or to another picker without cancelling it. Third, the conversion emits nothing for tours that come back unchanged, so a repair that touches two tours produces commands for two tours. A `NewInformationCondition` (new) skips the solve, not the release, when nothing arrived since the last run. The selection must include `reserved` resources, since pickers with queued tentative tours are no longer `idle`.

A repair pipeline is a CoSy pipeline that starts from the incumbent draft instead of from loose orders, in the sense of Vieira et al.'s repair as opposed to complete regeneration. A full re-solve pipeline dissolves every rebuildable tour into its orders and runs batching, routing and scheduling from scratch. A repair pipeline has four stages; the subsection below shows which components exist:

1. `IncumbentProvider` (new) turns each draft `TourView` with `accepts: rebuild` into a batch that carries its `tour_id`, as `ActiveTourBatch` does for started tours.
2. An insertion stage assigns new buffered orders to those batches where capacity allows, and passes the rest to an ordinary batching algorithm for new batches. `RemainingRouteAdmission` exists but only admits an order whose picks already lie on the route; a cheapest-insertion admission is new.
3. Only changed batches are routed again: unstarted tours with any router from the depot, started tours with a reroutable router from their origin (exists for S-shape).
4. A list scheduler whose resources start at their free-from time keeps unchanged tours as fixed intervals and places changed and new ones.

Like every candidate it returns a complete plan for the selected tours, and the conversion turns it into `InsertOrders`, `ReplaceSuffix`, `Reschedule` and `CreateTour` for what changed only. The full re-solve stays in the repository as another candidate, and the change penalty in the ranker decides between them. On the cards, repair components require the incumbent decision feature, so they are only synthesized for decisions whose selection includes draft tours.

#### Repair pipelines in the CoSy template

`IncumbentProvider` reads selected rebuildable `TourView`s from `PlanView`. `IncumbentBatchingNode` passes their batches to `IncumbentStartBatching`, which tries each buffered order in every draft with enough cart capacity and chooses the smallest incremental route distance. Orders that do not fit form new FIFO batches. Empty incumbent input takes the ordinary FIFO path. The next routing and scheduling stages operate on those batches; the scheduler starts from each picker's projected availability.

The full batching pipeline remains an alternative in the same repository. CoSy's algorithm cards restrict the repair stage to decisions with the `incumbent` feature. The repair provider is a dependency of that stage, so it cannot be combined with a batching stage that ignores the incumbent. All repositories declare the desired `solver.output` stage. `PlanCandidateEndpoint` has one output key and a typed bridge for batching, routing, scheduling, or sequencing; the old per-class terminal tasks and suffix map are gone.

Conversion compares the chosen candidate with the snapshot. It matches tours by carried id and order overlap, using the lower tour id to break a tie, and emits `InsertOrders`, `ReplaceSuffix`, `Reschedule`, `Reassign`, `CreateTour`, or `CancelTour` for actual changes. An unchanged draft yields no edit command. The candidate's objective and command-based change penalty decide between full rebuild and repair. The fixed-seed Henn example builds two pipelines and selects four full and seven repair results; exact metrics appear in `docs/revision_scopes_baseline.md`.

CoSy components consume `PlanView`, `DecisionContext`, and `DecisionMemory` from `InstanceLoader`. Incumbent-start batching owns the cheapest feasible insertion for drafts. `AbstractPositioning` consumes the scheduling solution, and its aisle-patrol and center-relocation components emit a complete `PositioningSolution`. A reusable standalone admission stage for general unstarted tours remains separate design work.

#### Plan stability is decided in ranking

The commitment policy has the interface `apply(plan, snapshot) -> list[Command]`. It sees the chosen plan and the snapshot, not the other candidates, and it has no evaluator for the incumbent plan under the new information, whose objective is not comparable because it does not contain the newly arrived orders. A rule such as "apply the new plan only if it is better by a threshold" therefore cannot be decided there.

Plan stability is decided in `SolutionRanker`, which sees all candidates and the snapshot. It scores each candidate by its objective plus a change penalty, for example the number of changed tours or moved orders, which follows directly from the candidate's commands against the snapshot. With the repair pipeline among the candidates, "keep the plan unless re-solving is clearly better" is an ordinary ranking decision. The change penalty is new; the current ranker branches on the problem class instead.

#### Release rules

In a rolling horizon, and in model predictive control generally, the planner optimizes over the whole horizon but implements only the first part; the rest is replanned. Timefold names the two parts published and draft, MPS planning names the first one the frozen zone, and a WMS calls the act releasing work to the floor. Commitment is that act: deciding which part of the plan becomes binding now.

Tours created by a decision start as `scheduled`, which is the draft. The commitment policy looks at the chosen plan and emits `ReleaseTour` commands for the tours that become binding, moving them to a new status `released` between `scheduled` and `pending`. A picker only starts released tours. Later decisions protect released tours through the scope (`freeze_from: released`), and everything still scheduled remains a draft that the next decision may revise. The policy needs no objective and sees exactly what it needs, the plan and the snapshot; its interface becomes `apply(plan, snapshot) -> list[Command]`, returning `ReleaseTour` commands and, for `release_within`, a `Wait` for the time the next draft enters its window.

| Rule | Releases | Corresponds to |
| --- | --- | --- |
| `release_all` | every new tour immediately | the current `CommitAllPolicy` |
| `release_next_per_picker` | the earliest tour of each picker | `SchedulingCommitmentPolicy(n_jobs=1)` without discarding the rest; the release at a depot return in Lorenz-style reoptimization |
| `release_within` (`within_s`) | tours starting before now + `within_s`; requests reconsideration when the next draft enters the window | the frozen interval of a rolling horizon, MPS time fence, Timefold's `freezeTime` |
| not a release rule: `order_count`, `henn`, `analytic_stochastic` | nothing themselves; they set start times, which `release_within` respects | the waiting algorithms, configured under `waiting` (next subsection) |

Binding is a status, set by one component and read by the next decision, not a time computation repeated in the adapter and the validator. The commitment policy has a role that neither the ranker nor the scope can take over: the ranker decides which plan is best, the scope decides what the next decision may still change, and the commitment policy decides when a draft becomes binding.

#### Waiting policies set start times

In the order-picking literature, waiting is the time-window decision of online batching: once a batch exists, how long to wait before it is assigned to a picker and started, trading picker idle time against better batches (Henn, 2012; Gil-Borrás et al., 2023, who call it the time-window strategy). In dynamic vehicle routing, waiting strategies decide where and how long vehicles wait along their planned routes (Mitrović-Minić and Laporte, 2004; Branke et al., 2005). In both, waiting is a decision about the timing of planned work, and it needs forecasts or service times. Release is a different act, known from order release in production control and from the frozen zone of MPS planning: making planned work binding. The model keeps the two apart under these names. The table lists what each waiting algorithm in `ware_ops_algos/algorithms/waiting/policies.py` decides.

| Algorithm | Decides | Inputs | As a start time |
| --- | --- | --- | --- |
| `NoWaiting` | start the next batch now | nothing | now |
| `OrderCountWaiting(k)` | start once the batch holds k orders or the stream is closed | the batch | now, or on hold until the next arrival |
| `HennWaiting` | start now if a backlog exists, else at Henn's threshold | single-order service times from the router | now or the threshold |
| `AnalyticStochasticWaiting` | start after the optimal wait | the forecast in `PlannerInformation` | now plus the optimal wait |

Each of the four expresses the answer as a start time (now, later, or on hold), and none creates or changes a tour. That is a timing decision on the plan, which later decisions may revise; release fixes it. Waiting is therefore its own concept, with three consequences.

First, `waiting` is a per-decision setting next to `commitment`, for example `waiting: {policy: henn}`; omitted, it means `no_waiting`. The algorithms stay in `ware_ops_algos` with `problem_type: waiting`, their cards and their `WaitingInput -> WaitingSolution` interface. CASIM applies the policy to every candidate plan after the pipeline has run and before conversion and ranking, on each picker's earliest draft: a release keeps the planned start, a wait with `reconsider_at` moves the start to that time, and a wait without it puts the tour on hold until the next trigger. The ranker compares plans with waiting applied. Waiting is a configured setting, not a synthesized variable.

Second, commitment keeps only structural rules: `release_all`, `release_next_per_picker` and `release_within`. A waiting policy works best with a commitment that respects its start times; `release_within` with `within_s: 0` releases a tour when its possibly delayed start arrives. The existing Henn and stochastic configurations retain their earlier commitment rules for baseline parity. Derived scheduling is OBRSP with a waiting setting; `waiting` is outside the taxonomy's synthesized variables, and `ResultAggregationWaiting` and `commit_waiting` are removed. While a policy waits, the tour is a draft with a future start, and new arrivals revise it.

Third, a waiting policy may set start times but may not create or change tours. Sending an idle picker on a tour without orders is a positioning decision. In the taxonomy it is `problem_type: positioning`, an optional variable of OBRSP after scheduling. For each selected idle picker without assigned work, the stage may add a job with an empty batch and a positioning route. The implemented cards offer an all-aisle depot-to-depot patrol and a center relocation with a hold. Wait-0 combines the patrol component, `no_waiting`, `release_all`, and active insertion. Pipelines produce plans, waiting sets the start of each picker's next tour, and commitment releases.

The study of the complete-information counterpart by Lorenz, Otto and Gendreau ([arXiv:2410.14316](https://arxiv.org/html/2410.14316v1), 2024) supports this class. Its formal model has no repositioning variable, but the analysis of complete-information optimal solutions shows strategic relocation, the idle picker moving towards the location of an upcoming item, before 39 to 62 percent of orders. Their improved myopic policy Reopt\* implements a simple version: the idle picker moves to the warehouse center, which reduced the turnover gap by 4.3 percent on average and up to 14 percent. The paper's second intervention, adding an order to a started cart or replacing an unpicked one, is what `frozen_accepts: insert` and `revise` express, and the paper reports strategic waiting as harmful once these interventions are allowed. CASIM's center example carries an empty cart during relocation, holds at the center, and routes an admitted order from the observed position. The paper does not specify that cart handoff; this choice is CASIM's and the example does not reproduce the full Reopt* policy. Forecast-based positioning remains an extension (open point O16).

#### Inputs of the waiting policies

Waiting runs after the pipeline on each picker's earliest draft in the candidate plan. If the solve is skipped, the drafts keep their planned start times and only the release rule runs. `DecisionEngine._schedule_commands` builds `WaitingInput` from the typed snapshot and candidate schedule.

| Field | Read by | Current source |
| --- | --- | --- |
| `candidates` | all | the selected picker's earliest job in the candidate schedule |
| `current_time` | Henn, analytic | `DecisionContext.time` |
| `input_closed` | order count, Henn, analytic | `DecisionContext.input_closed` |
| `deadline_reached` | analytic | `Reconsider:wait` in `DecisionContext.cause` |
| `picker` | Henn, analytic | the selected `PlanView` resource |
| `layout`, `information` | analytic | the projected snapshot |
| `single_order_service_times` | Henn | CASIM routes each order alone with `SShapeRouting` |

Field sources at the waiting call.

`NoWaiting` and `OrderCountWaiting` work unchanged. `OrderCountWaiting` and Henn evaluate the earliest unreleased tour for each picker. The fixed-seed Henn and stochastic examples exercise persistent drafts; a multi-picker Henn run remains open point O7.

CASIM computes Henn's single-order service times with `SShapeRouting`, setup time and pick time. The runnable Henn repositories use S-shape routing. Passing the candidate router's exact identity into waiting would broaden the setting to other routers; that is not implemented here.

The analytic policy has all its inputs but raises on three preconditions: exactly one candidate, exactly q−1 known orders, and one line per order. `compute_segment_times` also reconstructs the progress of the candidate along its route and appears to assume an S-shape traversal, which holds in the current scenarios only because the stochastic scenario configures `SShape` alone (open point O17). In the model each precondition has a defined outcome instead of an exception. Fewer than q−1 orders returns wait without `reconsider_at`, which puts the tour on hold until the next arrival. More than one candidate means a backlog, so the policy starts the first now, as Henn does; the paper's model has a single batch, so this outcome is a design choice (open point O8). Multi-line orders are a scenario fact the card checks against the data card. The S-shape assumption becomes a requirement on the candidate's router, which a synthesis constraint can check because the waiting policy is fixed per decision.

The former `StartImmediatelyWaiting` class and card have been removed. Wait-0 selects the aisle-patrol positioning component and `no_waiting`.

In the model a wait is a future planned start on a draft together with a `Wait(reconsider_at)` command. New arrivals revise the draft under `freeze_from: released`, and the decision's `Reconsider` event runs it again.

Because waiting is applied before ranking, the ranker sees each candidate with its waiting applied, so under makespan or tardiness the plan is chosen with the wait priced in. Under a distance objective waiting changes nothing the ranker sees, and the choice of pipeline is independent of the policy.

The four waiting algorithms remain per-decision settings; positioning is a separate pipeline stage.

#### Waiting under a rolling horizon

Waiting and a rolling horizon can be configured together, and they compose rather than compete: the pipeline plans over the window, waiting sets the start of each picker's next draft, and `release_within` releases a tour once its start is less than `within_s` away. The frozen interval then bounds how long a wait stays revisable. A waiting policy that consumes arrivals (`order_count`, `henn`, `analytic_stochastic`) computes its wait assuming the batch can still grow until it starts; with `within_s > 0` the tour becomes binding `within_s` before its start, and orders arriving in that interval can join only through an insertion decision. The decision schema therefore warns when such a policy is combined with `within_s > 0`; the setting consistent with these policies is `within_s: 0`.

Whether they should coexist is an empirical question, and CASIM reproduces the experiments that answer it instead of forbidding the combination. Lorenz, Otto and Gendreau (arXiv:2409.12619) report Henn (2012) as finding that adding a waiting time to reoptimization at the depot did not improve results, use immediate reoptimization as their benchmark because it avoids the cost of waiting, and their complete-information study (arXiv:2410.14316) finds strategic waiting harmful once interventions in started tours are allowed. Under a rolling horizon with repair the default is therefore `no_waiting`, with waiting as an experimental treatment. One structural argument limits waiting as a separate layer: a scheduler that planned with forecast arrivals would set delayed starts itself, and waiting would duplicate its decision. CASIM has no such scheduler, since forecasts reach algorithms only through `PlannerInformation` (see Receding-horizon control), so the separate setting is needed for now.

#### Reconsideration as one mechanism

`Reconsider` replaces the former waiting-specific `WaitingOpportunity` and `WaitExpired` events. The old `wait_version` and `commit_waiting` path was removed. The decision version now makes superseded waiting and release-window checks stale; a fixed periodic grid supplies rolling solves independently of those checks.

Three parts of a decision request the same thing, "run this decision again at time t": its waiting policy; the periodic re-solve of a rolling horizon, a time the decision chooses for itself rather than an exogenous event; and `release_within`, which knows when the next draft enters its window. CASIM implements the three as one mechanism.

After a decision has committed, CASIM schedules reason-tagged `Reconsider(decision, at, version)` events for its requested wait and release-window times and the next tick on a fixed periodic grid. The version is incremented on each accepted decision. An earlier accepted decision makes its older checks stale and schedules fresh ones. A periodic check that finds no work still schedules the next grid tick; this is how an order can enter the horizon without another arrival. Events at one timestamp coalesce into one decision whose `DecisionContext.cause` records the reasons. `WaitExpired` and `WaitingOpportunity` have been removed.

`WaitingSolution.reconsider_at` supplies a waiting check. `release_within` supplies each future draft's window-entry check. `{every_s: n}` in `triggers` supplies the periodic grid. At a release-window check with unchanged planning information, the engine reuses the drafts and runs only the release rule. A periodic check can replan drafts even without a new arrival. Each reason stays visible in `DecisionContext.cause`.

#### Conditions and waiting policies

Conditions and waiting policies answer different questions. The current configurations use conditions as waiting rules: `NbrOrdersCondition(threshold=5)` in Henn's order-window scenarios is a fixed-size waiting rule (wait for k orders), and `NbrOrdersCondition(threshold=3)` in the stochastic waiting scenario exists because `AnalyticStochasticWaiting` raises unless exactly q−1 = 3 orders are known, so an algorithm's precondition sits in the user's configuration. The model separates the two on every axis:

|  | Condition | Waiting policy |
| --- | --- | --- |
| Evaluated | before solving | after solving, on each candidate |
| Sees | the snapshot, including its `OperationsView` | the candidate plan, the snapshot, forecasts, service times |
| Output | go or no-go | a start time for the picker's next tour, or wait until a time |
| Reconsideration | none; the next trigger re-evaluates it | `Wait(reconsider_at)` schedules one |
| Declared as | registered condition with parameters | algorithm with a card |
| Question answered | is a new solve worth running? | when should the next tour start? |

To keep them from overlapping, conditions are restricted to feasibility gates: something to plan exists (at least one order, pick list or candidate tour), a resource is available. Count thresholds above one express timing and are therefore waiting policies; the schema rejects `threshold > 1` on order and batch counts and points to `order_count`. An algorithm's own precondition belongs into the algorithm: the analytic policy returns "wait" while fewer than q−1 orders are known instead of raising, and its card states that it needs the forecast. Under these rules the order-window scenario becomes `waiting: {policy: order_count, k: 5}` with `commitment: {policy: release_within, within_s: 0}`, and the stochastic waiting scenario drops its count condition.

Concretely, three uses from the current configurations stay valid conditions. `NbrOrdersCondition(1)` and `NbrBatchesCondition(1)` skip the solve when there is nothing to plan, which also avoids the `None` result `on_trigger` cannot handle today (the Henn scenarios use them this way). `NbrPickersCondition(1)` skips the solve of a decision that needs an idle picker when none is available, as in the waiting and walk-or-wait setups. `ActiveTourReadyCondition` skips an insertion while the picker is mid-pick, because no route change can be executed until the pick completes; the next `PickComplete` triggers it again. Two existing conditions do not fit. `BreakCondition` and `DockCapacityCondition` do not say that planning is pointless, only that new work must not start: during a break or with a full dock, planning can go on and the plan should simply not be released. They therefore become a release gate in CASIM: no `ReleaseTour` is emitted while `OperationsView` reports a break or a full dock, independent of the policy, and the event that ends the block is a trigger of the decision, so the plan is ready when the break ends or the dock clears. Kept as conditions they would skip only the solve, and since the release still runs, they would no longer stop new work from starting.

#### Receding-horizon control

Model predictive control repeats four steps: observe the state, optimize over a horizon, implement the first part, shift the horizon. In the model they are the snapshot, the solve over `window` plus the incumbent draft, the commitment policy, and the next `Reconsider` or trigger. The rolling Henn run uses `every_s: 600` and `release_within` with `within_s: 300`.

1. A draft tour whose start falls between two solves must still be released in time, or its picker idles. `release_within` therefore requests a reconsideration at the time the next draft enters its window, so the decision runs again exactly then, and no rule relating `within_s` to `every_s` is needed. A decision may also omit `solver`; such a release-only decision runs only its commitment policy on the current draft, for example on `PickerIdle` when a picker finishes earlier than planned.
2. Pickers start only released tours. `PickerTourQuery`, which currently takes the next queued tour, skips `scheduled` ones. A released tour whose planned start lies in the past starts at the current time, because `PickerTourQuery` uses the later of the planned start and now.
3. `WaitingInput` carries one picker; CASIM calls the waiting policy once per selected picker with that picker's drafts in the candidate. The release rules are CASIM rules and need no picker field.
4. The incumbent must reach the algorithm. Repair pipelines read the draft `TourView`s; a full re-solve may ignore them. A metaheuristic can be warm-started without changing it where it already takes a start solution: `LocalSearchBatching` builds one through `start_batching_class`, so `IncumbentStartBatching` from the repair pipeline suffices. Algorithms without such a hook would need an initial-solution input.
5. True MPC plans against predicted disturbances. Here forecasts reach the algorithms only through `PlannerInformation`; there is no plan object for an expected but unknown order, so tentative capacity for future demand cannot be reserved in the plan, and there is no terminal cost for work beyond the window. Both are limitations of the plan model, not of the configuration.

A separate release-only decision is optional. It uses the same `resolve` path as planning but has zero pipelines; the runnable form is `decisions=release_only_example` in the Henn scenario:

```yaml
release_due:
  selection:
    tours: [scheduled]
    resources: [idle, reserved, busy]
  scope: {freeze_from: released}
  triggers: [{every_s: 300}]
  precedence: 20
  commitment: {rule: release_within, within_s: 300}
  waiting: {policy: no_waiting}
```

The Henn rolling run demonstrates this flow with full rebuild and incumbent repair candidates. Forecast-based capacity reservations and terminal costs remain separate algorithmic extensions.

## The configuration from the user's side

A scenario is configured in the files below, and each question has one home. The current code spreads the same questions over five overlapping places: the data card (which also names one problem class and one objective), the engine file (two parallel blocks per problem, `decision_engine.problems` and `simulation_engine.problems`), the CoSy repository files, the simulation parameters and Python hooks.

| Question | Where | Edited by | Example |
| --- | --- | --- | --- |
| What is the warehouse? | scenario data card, `config/data_card/*.yaml` | scenario author | layout, resources and carts, storage, objective; the decision's problem class is derived at setup |
| Where do data and exogenous events come from? | data card `source` (loader by dotted import path) and the scenario's stream | scenario author | `WalkOrWaitDataLoader`; order stream |
| Which algorithms exist and what can they handle? | algorithm cards in `ware_ops_algos` | algorithm author | requirements, including `decision_features` constraints |
| Which algorithms may a decision combine? | solver repository, `config/cosy_repo/*.yaml` | experimenter | list of CoSy components |
| When is which decision taken, on what, with what freedom? | `config/decisions/*.yaml`, replacing `config/engines/*.yaml` | experimenter | `triggers`, `conditions`, `selection`, `scope`, `solver`, `waiting`, `commitment` |
| What happens on an inadmissible command? | `on_violation`, next to `decisions` | experimenter | `reject` |
| Run settings | Hydra root config | experimenter | output directory, visualization, Luigi |

The decision entry has a direct counterpart in today's code; it merges the two engine blocks and changes the routing key from the problem class to the decision name.

| Decision key | Today | Built by | Runtime object |
| --- | --- | --- | --- |
| decision name | the problem class key, repeated in both engine blocks | `casim.setup.resolve`, called once per decision from `experiment_commons.py`, replacing `build_solvers`, `build_commitment_policies` and `build_simulation_problems` | `DecisionSetup`; `SimulationEngine.triggers_map` maps an event type to decision names, and `DecisionContext.decision` routes the snapshot to its setup |
| `triggers` | `simulation_engine.problems.*.triggers` | event name or dotted import path | `triggers_map` entries |
| `conditions` | `simulation_engine.problems.*.conditions` | Hydra target | `Condition` objects, called with the snapshot |
| `selection`, `scope` | `state_adapter: {_target_: ...Adapter}` | `StateAdapter(selection, RevisionScope)` | one adapter instance per decision |
| `solver` | `decision_engine.problems.*.solver` | `CoSySolver` with the decision's own data card | pipelines synthesized once at setup |
| `waiting` | waiting components inside the old solver repository | `WaitingConfig` and policy implementation | waiting policy in `DecisionSetup`, applied to every candidate in `DecisionEngine.on_trigger` |
| `commitment` | `decision_engine.problems.*.commitment_policy` | `CommitmentConfig` | release rule over the chosen plan, emitting `ReleaseTour` and `Wait` commands |
| `precedence` | none; the order in which opportunities leave the event heap | config schema | `SimulationEngine.precedence`, which orders the decisions due at a timestamp |
| `on_violation` (top level) | none; violations are not detected | config schema | handler called by `State.apply_all` |

The derived problem class drives applicability checks and is printed by `casim check`; it no longer selects a solver, a ranking branch or a commit branch.

## Configuring the four cases

The four target cases use one entry per decision with `triggers`, `conditions`, `selection`, `scope`, `solver`, `waiting`, `commitment` and `precedence`. `scope` controls permitted revisions. `waiting` is evaluated after solving, while `commitment` releases scheduled drafts. The table shows how the cases differ. The runnable YAML lives in `scenarios/*/config/decisions/`; the excerpt below is the Henn decision from `henn_waiting.yaml`.

| Case | Triggers | State adapter selection | Scope | Base problem | Commitment |
| --- | --- | --- | --- | --- | --- |
| Rolling horizon | `{every_s: 600}`, `OrderArrival`, `TourEnd`, `PickerArrival`, `PickerIdle`, `OrderStreamClosed` | overdue buffered orders and orders due within 3,600 s, all relevant incumbent drafts, idle, reserved and busy pickers | `freeze_from: released` | OBRSP; full rebuild and incumbent repair as candidates | `release_within` with `within_s: 300`; unchanged drafts produce no edit commands |
| Reoptimization at the depot (Lorenz) | `TourEnd`, `PickerIdle`, `OrderArrival` | buffered orders, unstarted tours, idle and reserved pickers | `freeze_from: pending` | OBRSP | `release_next_per_picker` |
| Insert incoming orders into running tours | `ActiveTourOpportunity` | buffered orders, started tours | `freeze_from: planned`, `frozen_accepts: insert` | OBRP (admission plus active-tour routing) | `release_all` |
| Reoptimize started tours | `OrderArrival`, disruption event | buffered orders, all unfinished tours | `freeze_from: pending`, `frozen_accepts: revise` | OBRSP | `release_all`; needs the `RemoveOrders` command (step 9) |

```yaml
OBRSPW:
  solver:
    _target_: casim.pipelines.pipeline_runner.CoSySolver
    output: scheduling
    solution_ranker:
      _target_: casim.pipelines.solution_ranker.SolutionRanker
      objective: ${data_card.objective}
    repo: ${cosy_repo}
  selection:
    orders: [buffered]
    tours: [planned, assigned, scheduled]
    resources: [idle, reserved]
  scope: {freeze_from: pending}
  triggers: [OrderArrival, OrderStreamClosed, PickerIdle, FlushRemainingOrders]
  conditions:
    - {_target_: casim.simulation_engine.conditions.NbrPickersCondition, threshold: 1}
    - {_target_: casim.simulation_engine.conditions.NbrOrdersCondition, threshold: 1}
    - {_target_: casim.simulation_engine.conditions.NewInformationCondition}
  precedence: 10
  waiting: ${cosy_repo.waiting}
  commitment: {rule: release_next_per_picker}
```

This excerpt uses the actual config model: `selection`, `commitment.rule`, Hydra `_target_` entries, and the solver repository interpolation. For a complete runnable configuration, use `python -m scenarios.scenario_henn_online.experiment_henn viz.launch=false`. Add `cosy_repo=repair_waiting_henn +decisions.OBRSPW.solver.solution_ranker.change_penalty=10` to run full and repair pipelines together. The corresponding `casim check` command is in the implementation record.

The table shows, for each field, where the behavior it controls lives in today's code. None of those places survives as the reader of the field: `ReORSPAdapter`, the RORSP branch of `commit_solution` and the other adapters disappear, and their logic moves into the generic state adapter, the validator or a registry. The column determines how the implementation step that ports a field is tested. A field whose behavior already exists is a port that can be tested for equal results on today's scenarios; a field marked new has no reference behavior and needs new code and new tests.

| Field | Behavior today lives in (to be replaced) | Port or new |
| --- | --- | --- |
| `triggers` | `triggers` in the simulation engine config | port; the periodic `{every_s}` form is new |
| `conditions` | `conditions`, instantiated by Hydra path | port; referenced by registered name instead of path |
| `state_adapter.orders`, `.pick_lists`, `.tours` | selection code inside the eleven adapter subclasses | port into the generic state adapter, now as lifecycle statuses with one form per object |
| `state_adapter.resources` | two booleans and `has_future_tours`, combined differently per subclass | port into four resource statuses; availability times are new |
| `state_adapter.window` | nothing | new; `due_date` or `start_time` only |
| `scope.freeze_from: pending` | `ReORSPAdapter` plus the RORSP branch of `commit_solution` | port; both places are deleted |
| `scope.frozen_accepts: insert` | `ActiveTourAdapter` plus `commit_active_route` | port; today limited to one picker and S-shape |
| `scope.frozen_accepts: revise` | nothing | new, needs `RemoveOrders` |
| `waiting.policy` | the waiting stage inside one pipeline (`AbstractWaiting`, `HennWaitingNode`, `commit_waiting`) | port; applied to every candidate instead of inside one pipeline |
| `solver.repo`, `solver.objective` | `decision_engine.problems.*.solver`, `SolutionRanker(objective)` | port |
| `solver.change_penalty` | nothing | new |
| `commitment` | `commitment_policy` (`SchedulingCommitmentPolicy.n_jobs`, which releases the first job and discards the rest, and `CommitAllPolicy`) | port of the intent; the released status and `ReleaseTour` are new |
| `precedence` | implicit order of opportunities in the event heap | new, required only for overlapping write sets |
| `on_violation` | nothing; violations are not detected | new |

The schema has no separate fields for picker or start-time freezes: a rebuildable tour may be rescheduled, reassigned or rebuilt, and the commands say which. `routing_origin: mid_edge` has no router behind it, and dedicated bins are a scenario fact on the data card, not policy. The problem class is not configured; `casim check` prints the derived one. `on_violation` is a single top-level setting next to `decisions`.

Each user-facing question has exactly one place. "When do we replan" is `triggers` and `conditions`. "What does the decision see" is `state_adapter`. "What may change" is `scope`. "Which algorithms" is `solver`. "When should a picker's next tour start" is `waiting`. "When does a draft become binding" is `commitment`, through the tours it releases. The adapter class name disappears from the config; the one generic state adapter is driven by its selection and the `scope`, and a projection extension remains available for data it does not provide.

### Precedence and write sets

`precedence` orders the decisions that are due at the same simulated time. A timestamp ends when the next event in the heap has a later time. Every decision triggered by an event of that timestamp, including events derived from it at the same time and `Reconsider` events, is then due exactly once. The due decisions run one after another, and each projects its snapshot only after the previous one has committed, so a later decision sees the effects of an earlier one. They run in ascending `precedence`; a decision without `precedence` runs after all decisions that have one, and ties are broken by decision name, so the order is always deterministic. `precedence` has no effect across timestamps, does not preempt a running decision and gives no other priority. In the worked example, `active_insertion` (precedence 1) admits order 7 at t = 31, and `planning`, which runs next, no longer sees order 7 as buffered.

The write set of a decision is derived from its selection and scope: `orders[buffered]` and `pick_lists[buffered]` if they are selected, and `tours[s]` for every selected tour status `s` whose `accepts` is not `none`. Two decisions overlap if their write sets share an element. Their commands can then invalidate each other, so their order must be explicit: a validator on the top-level decisions model in `casim/config/schema.py` requires for every overlapping pair that the order follows from `precedence` and not from the name tie-break, that is, at least one of the two carries a `precedence`, and if both do, the values differ. Loading the configuration fails otherwise, and `casim check` reports the finding with both decision names and the shared element. The rule ignores triggers, because whether two triggers coincide in simulated time cannot be decided from the configuration; it is conservative and may require precedences that never matter in a run (open point O10). Decisions with disjoint write sets need no `precedence`. Their order can still change what the later one reads, for example a picker's free-from time after an insertion, and the name tie-break keeps that order deterministic.

In the walk-or-wait configuration, `active_insertion` and `planning` both write `orders[buffered]` after one `OrderArrival`. In the current code their order depends on which of the two opportunities the arrival emits leaves the event heap first; `precedence` makes it explicit.

The scope also determines which decision variables are free, so CASIM derives and prints the problem class instead of asking the user to name it (see Applicability, taxonomy and synthesis).

### Case status

| Case | Implemented behavior | Remaining extension |
| --- | --- | --- |
| Insertion into running tours | `ActiveTourOpportunity`, admission, active routing, versioned `InsertOrders`/`ReplaceSuffix`, and precedence run in walk-or-wait; center admission routes from a moving or held picker | more router families and general draft admission |
| Reoptimization at the depot | unstarted drafts can be rebuilt or repaired; `IncumbentProvider`, release and resource availability are shared with rolling planning | a separate Lorenz experiment configuration |
| Rolling horizon | fixed-grid periodic and reason-tagged release-window checks; drafts, release commands, resource availability, stable matching, change penalty, and full/repair pipelines run in Henn | forecast-based capacity reservations and terminal costs |
| Reoptimize started tours | `TourView` carries a pinned prefix and observed origin; insertion is allowed | `RemoveOrders`, bin release rules and full started-tour revision (step 9) |

The rolling decision selects `busy` pickers: they enter `PlanView` with the time they are planned to become free, so a picker on a long tour stays visible to the next plan. Actual `TourEnd` and picker availability events trigger another solve. The admission component's handling of unstarted tours (depot origin, empty prefix) follows from its code but has not been run (open point O11).

## Applicability, taxonomy and synthesis

The existing machinery already has the right shape; it lacks inputs. `DomainAlgorithmMapper` filters cards by `problem_type` against the taxonomy's variables and then evaluates `requirements` per domain section with `equals`, `in` and comparison operators. `algorithm_applicability_constraint` applies that filter inside the CoSy constraint, so inapplicable components never enter a synthesized pipeline. What it cannot see today is anything dynamic, because the data card describes static domain features only.

### Splitting the dynamic domain object

`DynamicInfo` is split before decision features are added, because most of the hidden coupling lives in it. It currently has 13 fields that serve five different concerns, it inherits from the static `WarehouseInfo`, and `InstanceLoader` hands the whole object to every CoSy component as one pickle. No component declares which fields it reads, and every field has a default, so a component that needs a field an adapter did not fill gets an empty value instead of an error. The table lists what reads each field today.

| Field | Read by (code) | Concern | Home |
| --- | --- | --- | --- |
| `time` | waiting stage, decision engine, conditions | decision time | `DecisionContext` |
| `done` | waiting stage (`input_closed`), `NbrOrdersCondition` | decision time | `DecisionContext` |
| `wait_expired` | waiting stage; written by the engine after projection | trigger cause | `DecisionContext.cause` |
| `is_break` | `BreakCondition` | calendar | `OperationsView` |
| `n_staged_pallets` | `DockCapacityCondition` | operational aggregate | `OperationsView` |
| `active_tours` | `DockCapacityCondition` (length only) | plan state, as live CASIM objects | `PlanView` |
| `buffered_batches` | `PickListProvider`, `NbrBatchesCondition` | plan state | `PlanView` |
| `admission_tours` | admission, active-tour routing, `AdmittedTourBatch`, `ActiveTourReadyCondition`, logger | plan state for one algorithm family | `PlanView` (as `TourView`) |
| `active_candidate_ids` | admission, `ActiveTourReadyCondition` | free objects of the scope | `PlanView` |
| `considered_active_orders` | admission | memory of earlier decisions | `DecisionMemory` |
| `congestion_rate`, `current_picker` | nothing | unused | remove |

Two consequences are visible in the code. First, conditions run on the solver's projection, so a condition is only valid together with particular adapters: `DockCapacityCondition` calls `len(active_tours)`, which `OBPAdapter` sets to `None` and `ReORSPAdapter` fills with unstarted tours only. Second, `active_tours` exposes mutable `TourPlanningState` objects to the decision side, which works only because the snapshot is deep-copied; the frozen `AdmissionTour` next to it is the cleaner form of the same information.

The object is split into four parts with one owner each. `DecisionContext` holds when and why the decision happens: time, cause (the trigger type, or the reasons of a reconsideration such as an expired wait), whether the input stream is closed, the name of the decision and a decision version. `PlanView` is the scoped projection: free orders, pick lists, the resources in scope with the time they become available, and `TourView`s with pinned prefix and `accepts`, produced by the generic state adapter from the selection and the scope. `DecisionMemory` holds what a decision needs to remember across decisions, today only the considered (tour, order) pairs; it is owned per decision by CASIM and passed in explicitly. `OperationsView` holds operational facts that lie outside any decision's scope, today the break flag and the number of staged pallets. Static `WarehouseInfo` and the forecast in `PlannerInformation` stay as they are.

Conditions receive the complete projected `SimWarehouseDomain`, the same object the solver gets. By what they read, they fall into two groups. `NbrOrdersCondition`, `NbrPickersCondition`, `NbrBatchesCondition` and `ActiveTourReadyCondition` ask whether the scoped problem is worth solving and read `DecisionContext` and `PlanView`. `BreakCondition` and `DockCapacityCondition` ask about operational state that no adapter projects consistently; in the current code the dock count exists only because a scenario hook attaches a `DockManager` to `State`, read with `hasattr`. Those facts go into `OperationsView`, which a scenario fills through a projection extension and which conditions and solvers can both request. Conditions therefore get no separate world view; their snapshot carries those facts in its `OperationsView` part, separate from the scoped `PlanView`.

The `scope` block in the YAML is not one of these runtime objects. It is a `RevisionScope`, a configuration object fixed per decision at setup. It is used in three places: the state adapter reads it, together with the selection, to compute each `TourView`'s `accepts` value; the setup derives the data card's `decision_features` section from it; and the validator checks commands against it. A solver never sees the `RevisionScope` itself, only its consequences in `PlanView`.

#### Selection and scope

`RevisionScope` follows the split Timefold makes: a pin is data on the entity in the problem instance, and a separate policy (`freezeTime`, `pinNextVisitDuringFreeze`) decides which entities get pinned at each run. The `RevisionScope` is that policy, and the `accepts` value on each `TourView` is the pin. The policy is kept per decision, outside the solver input, because two decisions on the same state (insertion and depot reoptimization) need different pins on the same tours. Three distinctions determine its form.

The first is selection versus mutability. What a decision can see and what it may change are different questions, and frozen but visible objects are the normal case: a scheduler must see started tours to know when pickers become free. In Powell's terms, selection shapes the decision maker's state S\_t and mutability defines the action set X\_t(S\_t). They are two fields, the `state_adapter` selection and `scope`, so that "visible" never implies "changeable".

The second is the form of the mutability. A free status-by-attribute matrix lets users write combinations no operator wants, such as "started tours rebuildable, scheduled tours frozen". The tour lifecycle is monotone (PLANNED, ASSIGNED, SCHEDULED, RELEASED, PENDING, STARTED, DONE) and physical mutability only decreases along it. A scope can therefore be a cut on that lifecycle plus one flag, which is essentially Timefold's vocabulary:

```yaml
scope:
  freeze_from: released         # released and later are frozen, earlier statuses rebuildable
  frozen_accepts: none          # none | insert (append orders, replace suffix)
                                #      | revise (also remove orders without a completed pick)
```

Together with the selection, this covers all four cases from the table above (insertion is `freeze_from: planned` with `frozen_accepts: insert`), rules out the non-monotone combinations, and keeps the status-by-operation matrix where it belongs, as fixed physics. Because the scope applies only to selected objects, finer cases need no extra field either: "insert into scheduled tours but not into started ones" is a selection of `tours: [scheduled]` with `frozen_accepts: insert`.

The third is the data card. It is already per decision in practice, since it carries `problem_class` and `objective` and `setup_decision_engine` mutates one shared card for each decision. No new type is needed for this. The scenario's data card keeps only the warehouse description and loses `problem_class` and `objective`; setup builds a fresh copy per decision, sets the problem class derived from the scope and the objective from `solver.objective`, and fills its `decision_features` section. The mapper evaluates card requirements against that copy exactly as today, and the scenario card is never mutated.

The `DynamicInfo` split is also what makes the card constraints below precise. The data card section derived from the scope describes the structure of `PlanView` (which features exist, for example a pinned prefix or a non-depot origin), not runtime values, and it is called `decision_features` to keep it apart from both `DynamicInfo` and the decision's `scope`. If `InstanceLoader` emits one target per part, each CoSy component declares the parts it consumes, and a component that needs `TourView`s in a decision whose scope produces none fails at synthesis instead of running on an empty tuple.

For `active_insertion` the `decision_features` section of the decision's data card contains, for example, `routing_origin: next_node`, `pinned_prefix: true`, `membership_ops: [append]` and `max_active_tours: 1`. Facts such as dedicated bins are already on the card's static `resources` section and stay there. This needs a `decision_features` field on `DataCard` in the same shape as the other sections, a dict with `type` and `features`, since `_check_domain_requirements` reads exactly those keys. Two small mapper changes follow from its code: a `contains` operator for list-valued features, and a static default `decision_features` section on every data card, so that offline cards still satisfy the classic routers. Cards then declare what they can handle:

```yaml
# reroutable_walk_or_wait_sshape_routing.yaml
requirements:
  decision_features:
    type: [any]
    constraints:
      routing_origin: {in: [depot, node, next_node]}
      max_active_tours: {less_than_or_equal: 1}

# remaining_route_admission.yaml
requirements:
  decision_features:
    type: [online]
    constraints:
      membership_ops: {contains: append}       # contains is the new operator

# sshape_routing.yaml (classic, depot to depot)
requirements:
  decision_features:
    type: [any]                                # offline cards carry the static default section
    constraints:
      routing_origin: {equals: depot}
      pinned_prefix: {equals: false}
```

With that, a scenario that requires rerouting from mid-edge (diversion) gets a precise setup error ("no routing component supports routing\_origin=mid\_edge; candidates excluded: ReroutableWalkOrWaitSShape (in: depot, node, next\_node)") instead of the current generic "No CoSy pipeline for ...". The mapper's verbose mode already produces most of that message; casim check needs it as data, which is the refactor of filter described in section 2 of the worked example.

The taxonomy splits the problem key into a base class and a dynamic modifier. The base class names the free decision variables (OBP, OBRP, OBRSP, and so on) and controls algorithm applicability; every CoSy query uses `PlanCandidateEndpoint`. The modifier is the scope, and it is what distinguishes OBRP from the current ATIP. Admission is batching restricted to appending into existing batches, and active-tour routing is routing with a non-depot origin and a pinned prefix. Both keep their own `problem_type` on the cards (`admission`, `active_tour_routing`), since that keeps the CoSy component interfaces explicit, and CASIM derives them from the scope rather than from a problem name. Waiting leaves the synthesized variables, since it is a per-decision setting applied to every candidate (see Waiting policies set start times); OBRSPW is OBRSP with a waiting setting. One `TAXONOMY` lives in `ware_ops_algos`; the output stage is declared per decision and the plan-candidate endpoint is shared.

This also answers "at which point is which problem solved". At setup, each decision's scope fixes its base problem and dynamic features, the mapper and CoSy fix its set of pipelines, and `casim check` can print that table. At runtime, a trigger fires, the generic state adapter builds the snapshot under the scope, the pipelines run, and the result passes through command conversion and validation. Conditions that depend on runtime state, such as "more than one active tour" for a router declared for one, are checked per snapshot against the same card constraints, so a static declaration and a runtime guard share one source.

## Heterogeneous solvers: an analogue of Timefold's annotations

Timefold can annotate variables because the solver owns every move; a pinned entity is simply never offered to a move selector. CASIM's portfolio mixes S-shape, savings, local search, Gurobi models and policies, none of which share a move model. The guarantee therefore has to be assembled from three weaker pieces, each of which is cheap.

The first is the declared capability on the card, discussed above. It is a promise, not a proof, and its job is to keep obviously unsuitable algorithms out of the synthesized pipelines.

The second is input shaping. The algorithm receives only the free part of the problem as decision objects and the fixed part as immutable context. CASIM projects each selected tour as `TourView`, with status, pinned prefix, origin, available route suffix and `accepts`. At the admission pipeline boundary, `TourView.as_admission_tour()` supplies the established `AdmissionTour` input. This keeps the core `ware_ops_algos` algorithm contracts stable while giving CASIM one projection type for scheduled and started tours.

The third is output validation, which is the actual guarantee. Because the command conversion compares the output with the snapshot, CASIM can detect any change to a pinned object regardless of which algorithm produced it. This is the "verify" half that Timefold does not need and CASIM cannot avoid.

A closer analogue of annotations is possible with `typing.Annotated`: fields of the typed inputs and outputs are marked `Pinned`, `Free` or `Context` (the Timefold Python API uses the same mechanism, `Annotated[Timeslot, PlanningVariable]`), and a generic validator introspects which fields of an output may differ from the corresponding input. It is not part of the blueprint. Whether it pays off over explicit command conversion is open point O12, since most constraints that matter are relational (a prefix property, bin capacity) and do not live in a single field.

## End users and production use

For an end user the target is that a wrong configuration fails before the simulation starts, with a message that names the rule. Hydra composes YAML; `casim.setup.resolve` validates the `decisions` block with the Pydantic models in `casim/config/schema.py`, derives the problem class and decision features, checks card applicability and synthesizes pipelines. It rejects overlapping write sets with equal precedence and decisions without a declared output stage before an experiment starts.

`casim check <config>`, in the spirit of `terraform plan`, prints the resolved decision table: for each decision its triggers and conditions, the resolved state adapter selection and scope, the derived base problem and decision features, the number of synthesized pipelines, and for every excluded algorithm the constraint that excluded it. The user or agent sees what the system understood rather than what they meant to write. At runtime the counterpart is the command log: every commit recorded as commands with object ids and versions, which the Dash replay can render (pinned prefix greyed, inserted orders highlighted).

For extension without editing the repository, a data loader or event can be named by its dotted import path in configuration. `resolve` loads event types and `setup_experiment` loads the configured loader. `casim catalog` reports the shared lifecycle and installed algorithm cards. With the generic state adapter, the user rarely needs custom projection code; when they do, it is a typed extension that returns extra context objects and cannot mutate state, rather than a subclass that rebuilds the whole domain.

For a productive system two further properties matter. Solve time is not zero, so the floor moves while the solver runs; the `expected_version` on every command is what keeps a late solution from overwriting a tour that progressed in the meantime, in the same way Timefold applies problem changes to a working copy. And CASIM's commit becomes the natural boundary to a real WMS: the same commands could be sent to a WMS adapter instead of `State`, which is what makes the simulation a credible digital twin of the decision logic rather than of one scenario.

Today solve time is zero in simulated time. `DecisionEngine.on_trigger` measures the wall-clock `elapsed` but only logs it, `solution_to_events` stamps every follow-up event with `start_time_sim`, and `step` commits before the clock moves. A discrete event simulation does not need threads to model a solver that runs while the floor moves. The solver runs at once on the snapshot of time t, and its result is delivered by an event `DecisionApplied(decision, commands, at = t + d)`; between t and t + d the engine keeps processing events, and the commit happens when the delivery event fires. This is the main change: commit becomes an event instead of the tail of `step`. The latency d needs a model, and the choice decides reproducibility: a fixed value or the solver's configured time limit keeps runs deterministic, while the measured `elapsed` (possibly scaled to the target hardware) is realistic but varies between runs, so it should be recorded and replayable. A field such as `solver.latency: {model: fixed | time_limit | measured, value_s}` with default 0 keeps the current behavior. The latency extension is designed but not traced (open point O9).

What moved in the meantime is handled by three mechanisms the design already has. Every command carries `expected_version`, so a command on a tour that progressed during the solve is detected at delivery; `on_violation` decides whether it is rejected, dropped or repaired, where repair for `ReplaceSuffix` means rerouting from the new origin with the same order set. The scope can protect what will certainly change during the solve: a decision with latency d should freeze tours that start before t + d, which is the frozen interval of `release_within` with `within_s >= d`, so the solver never plans what the floor will execute before the plan arrives. And triggers of the same decision that fire while it is in flight need a rule: the design allows one solve per decision in flight, lets further triggers only mark the decision as dirty, and schedules a `Reconsider` at delivery if it is dirty. Solving on a state projected forward to t + d, as Timefold's real-time planning and the anticipation literature in dynamic routing do, would reduce rejections further but needs a forecast of the floor's progress; the design starts with snapshot-at-t plus validation. Truly concurrent solving only matters when the simulation runs paced to wall-clock time as an emulation or digital twin; in virtual time, computing at t and delivering at t + d is equivalent.

## Worked example: one scenario, end to end

This section sketches one decision sequence to explain snapshots, drafts, repair and release. Its times are illustrative. The runnable walk-or-wait configuration is `scenarios/scenario_walk_or_wait/config/walk_or_wait_config.yaml`; its decision files are `paper_intervention.yaml` and `paper_no_intervention.yaml` under `config/decisions/`. The Henn repair configuration is `scenarios/scenario_henn_online/config/cosy_repo/repair_waiting_henn.yaml`. Both are exercised by experiment entry points, not only tests.

### 1. What the user writes

A decision uses the schema shown in Configuring the four cases. The current files use `selection`, `commitment.rule`, Hydra `_target_` entries for the solver and conditions, and `waiting.policy`. The single source of the vocabulary is `casim schema`, while `casim catalog` lists lifecycle values and algorithm cards. Run the configurations directly:

```powershell
.venv\Scripts\python.exe -m casim check scenarios/scenario_walk_or_wait/config walk_or_wait_config
.venv\Scripts\python.exe -m scenarios.scenario_walk_or_wait.experiment simulation=explicit_example viz.launch=false
.venv\Scripts\python.exe -m scenarios.scenario_henn_online.experiment_henn cosy_repo=repair_waiting_henn +decisions.OBRSPW.solver.solution_ranker.change_penalty=10 viz.launch=false
.venv\Scripts\python.exe -m casim check scenarios/scenario_henn_online/config henn_online_config decisions=rolling_horizon data_card=henn_rolling cosy_repo=rolling_horizon
.venv\Scripts\python.exe -m scenarios.scenario_henn_online.experiment_henn decisions=rolling_horizon data_card=henn_rolling cosy_repo=rolling_horizon viz.launch=false
.venv\Scripts\python.exe -m scenarios.scenario_walk_or_wait.experiment simulation=explicit_wait0 policy=wait_0 cosy_repo=paper_patrol viz.launch=false
.venv\Scripts\python.exe -m scenarios.scenario_walk_or_wait.experiment simulation=explicit_wait0 policy=center cosy_repo=paper_center insertion_repo=center viz.launch=false
.venv\Scripts\python.exe -m casim check scenarios/scenario_henn_online/config henn_online_config decisions=release_only_example data_card=henn_rolling cosy_repo=rolling_horizon
```

### 2. What CASIM resolves: casim check

`casim check` composes the same Hydra configuration and calls the same `resolve` function as the experiment. `--json` returns machine-readable selection, scope, waiting, commitment and pipeline counts. For the runnable walk-or-wait example it finds one active insertion pipeline and one planning pipeline; their shared buffered-order write set has explicit precedences 0 and 10.

### 3. What the runnable traces show

The fixed-seed Henn repair run gives a concrete command trace. At 45.375 s it creates a scheduled tour and requests a reconsideration; at 90.75 s the release-only decision emits `ReleaseTour`. At 4,965.375 s a candidate extends existing drafts with `InsertOrders` and `ReplaceSuffix`, reschedules the affected work and releases the next tour. Both full and incumbent-start pipelines are synthesized, and the ranker selects four full and seven repair solutions over the run. The command and timing record is `kpis/decisions.json` under the Hydra output directory; the exact baseline run paths and aggregate measurements are in `docs/revision_scopes_baseline.md`.

The walk-or-wait wait-0 configuration is another executable case: `simulation=explicit_wait0 policy=wait_0 cosy_repo=paper_patrol` starts one empty patrol tour and later admits an order into it. `policy=center cosy_repo=paper_center insertion_repo=center` starts center movement and reroutes from the observed position when an order arrives. Both use the same scheduling, command, waiting and release path. The default explicit walk-or-wait example completes six orders in three tours and retains its baseline KPI values.

The fixed-seed rolling Henn run resolved two pipelines and completed 100 orders in 25 tours. It selected pipeline 0 35 times and pipeline 1 79 times; aggregate solve time was 3.88 s and candidate conversion time 0.17 s. The first accepted schedule appeared at 1,996.92 s, after its first known order entered the one-hour due-date horizon. Its trace includes `InsertOrders`, `ReplaceSuffix`, `Reschedule`, `ReleaseTour`, and release-window `Wait` requests. The optional `release_only_example` also finished all 100 orders; `release_due` built zero pipelines and emitted 55 accepted decisions, including `ReleaseTour` at 2,100 s. The 25 tours, 32,218 s makespan and 11,271 distance of the single-decision rolling run are intentional differences from the no-horizon Henn baseline, since 5,400-second due dates, repeated replanning and a five-minute frozen release window change the policy. The established Henn scenario remained at 100 orders and 26 tours; stochastic waiting remained at four orders and two tours; the default walk-or-wait example remained at six orders and three tours. The wait-0 patrol run started one empty tour and admitted one order after the empty start. The center run did the same from a moving cart, and its no-order closure test returned the cart without a pick.

For every accepted decision the runtime follows the same sequence:

1. `SimulationEngine.run` handles all operational events at the timestamp, then projects each due decision once in precedence order.
2. The solver sees the typed snapshot. Waiting evaluates each scheduled candidate; `to_commands` compares it with the selected drafts.
3. `State.apply_all` checks the command batch on a detached state and publishes it atomically. Accepted commands enter the decision log; reason-tagged, versioned `Reconsider` checks are scheduled at waiting, release-window and periodic times.
4. A picker starts only a released tour. Unreleased scheduled tours stay in the plan for the next solve or a release-only decision.

The stable `AdmissionTour` input to the existing algorithms is built from each `TourView` at the pipeline boundary. The scenario entry points in section 1 are the executable examples for this flow.

## Exogenous information and perturbations

Exogenous events carry information only; decisions change plans. Changes to the order stream, such as due dates pulled forward, orders injected or orders revealed early, are therefore modeled on the information side, separate from the replanning they trigger. In the current code the hooks of the IJPE scenario are where new information, a decision and bookkeeping are mixed, and the table uses them as the example. The IJPE scenario is not a reference scenario for the implementation, and this part is implemented in step 9 only if scenarios of this kind are still needed.

| Manipulation | What the code does today | In the model | Covered by commands and scopes? |
| --- | --- | --- | --- |
| Truck disruption | `TruckDisruption.handle` selects unstarted tours by due date until the TE volume is reached, sets them CANCELLED, overwrites `due_date` on the shared order objects and puts the batches back into the pick-list buffer | the event only announces the new deadline; the RORSP decision with scope `freeze_from pending` produces `CancelTour` and `CreateTour` commands, checked and logged | yes for the cancellation; the due-date change needs revision events (below) |
| Cross-day volume shift | a reset hook searches future `OrderArrival` events in the heap, mutates `order_date` and `due_date` in place, removes and re-inserts the events, and adds `WMSRun` and `VolumeShiftAcrossDay` | the shift is a perturbation of the exogenous stream, applied before the run; the replan is triggered by the revealed orders themselves | no, information side |
| Generated order injection | a reset hook appends generated orders to the initial domain for tracking and schedules `OrderArrival`, `WMSRun` and `OrderIngestion` | same as above, a perturbation that adds orders with an information time | no, information side |
| Marker events | `VolumeShiftAcrossDay` and `OrderIngestion` have empty handlers and exist only to trigger RORSP | not needed: the decision is triggered by `OrderRevision` and `OrderArrival` | yes |

The gap is that orders are mutable objects shared by the initial domain, the buffer, the tours and the tracker. Writing `o.due_date = self.to_time` changes all of them at once, and afterwards only the tracker knows the original value. The model applies the principle of the commands to information. Orders become immutable, and information about them arrives as typed events with an information time, for example `OrderRevision(order_id, due_date, known_at)`. State holds the current version and the history. A scenario is then a base stream plus a list of typed perturbations (`ShiftDueDates`, `InjectOrders`, `RevealEarly`), applied to produce the stream before the simulation starts, instead of hooks that edit the event heap. KPIs such as on-time rates can then be computed against both the original and the revised due date, without tracker calls inside hooks.

Whether the order selection in `TruckDisruption` is information or a decision is open point O14. Choosing which orders are pulled forward until the TE volume is reached could be part of the disruption (the customer states which orders move) or a planning decision (the warehouse chooses). In the first case it belongs to the perturbation; in the second it is a small selection problem that becomes its own decision with its own scope.

## Implementation checkpoints

Steps 1–8 are implemented on `codex/revision-scopes` in both checkouts. The reference commands, original revisions, fixed-seed outcomes, pipeline counts, and timing measurements are in `docs/revision_scopes_baseline.md`. Henn, stochastic waiting, and walk-or-wait experiment modules complete and write normal traces and metrics after the final checkpoint; `casim check` resolves the migrated Henn, stochastic, walk-or-wait, grocery, and IJPE configurations. The CASIM and `ware_ops_algos` test suites pass.

1. **Lifecycle.** Shared status enums, `PHYSICAL`, and `accepts_for` live in `ware_ops_algos`; CASIM derives order, tour, and resource states from its managers.
2. **Projection.** `DecisionContext`, `PlanView`, `TourView`, `DecisionMemory`, and `OperationsView` give decisions typed state. The stable `AdmissionTour` algorithm input is assembled from `TourView` at the pipeline boundary. CoSy components load the typed views directly.
3. **Scope.** One `StateAdapter` selects objects and applies `RevisionScope`. The eleven prior adapter subclasses are removed.
4. **Application.** Typed solutions become commands. `State.apply_all` validates the batch on detached state and publishes it atomically; accepted decisions are logged afterward. Tests cover stale versions, invalid batches, unchanged drafts, and violation handling.
5. **Setup.** Each named decision has one Hydra configuration and one `DecisionSetup` produced by `resolve`. The same setup supplies both engines. `casim schema`, `catalog`, and `check` use shared vocabulary and card data. A configured import path loads custom events or data loaders without a repository registry. `solver.output` declares the terminal stage.
6. **Timing.** The simulation coalesces triggers at a timestamp and runs decisions once in precedence order. Versioned `Reconsider` and periodic triggers replace waiting-specific timer events.
7. **Persistent plans and positioning.** Complete schedules are stored as drafts. Waiting adjusts each candidate before ranking; commitment emits `ReleaseTour`, and pickers start only released tours. A typed positioning stage adds empty patrol or center relocation jobs after scheduling. Operational break and dock facts gate release.
8. **Repair.** `IncumbentProvider` supplies selected drafts, `IncumbentStartBatching` inserts buffered orders at the smallest feasible route-cost increase, the scheduler uses projected picker availability, and conversion preserves stable tours. The command-based change penalty ranks repair against full rebuild. One `PlanCandidateEndpoint` accepts typed batching, routing, sequencing, or scheduling output. The Henn repair run selects both candidates and emits in-place commands.

The rolling extension adds due-date window selection that retains overdue work, a periodic grid that survives empty windows, independent reasons for waiting and release-window checks, an optional solver-free release decision, and a fixed-seed Henn run with full and repair pipelines. Step 9 remains separate: `RemoveOrders` with `frozen_accepts: revise`, typed information perturbations, solver latency, and optional field annotations. A standalone admission stage for general unstarted tours and forecast-based positioning remain future design work.

## Open points

The table tracks design decisions and evidence. A resolved item records the implemented behavior; the remaining items identify concrete work.

| ID | Open point | Status | Resolved in |
| --- | --- | --- | --- |
| O1 | The PENDING row of the physical matrix: picker and start time fixed while setup runs, membership and route free | implemented in shared lifecycle data; no reference run exercises a pending revision | step 1 |
| O2 | Whether `OrderWindowAdapter` intends "not occupied" without `available` | unknown | step 1 |
| O3 | Tie rule for matching candidate tours to existing tours by order overlap | resolved: largest positive overlap, then lower incumbent tour ID, then earlier candidate index; unchanged drafts emit no edit commands | step 8 |
| O4 | Seeding `earliest_free_assignment` with free-from times makes every list scheduler an extending scheduler | implemented; repair Henn scenario completes with drafts and unavailable resources | step 8 |
| O5 | Pipeline count and solve time added by repair variants | measured in final isolated fixed-seed Henn runs: one full pipeline at 0.118 s aggregate solve time; two full/repair pipelines at 0.218 s | step 8 |
| O6 | Cost of converting every candidate into commands | measured in final isolated fixed-seed Henn runs: 0.235 s full and 0.281 s repair aggregate conversion; application costs are recorded separately | steps 4 and 8 |
| O7 | Henn and order-count semantics under persistent drafts and multiple pickers | first scheduled draft per picker is evaluated; multi-picker Henn scenario coverage remains open | step 7 |
| O8 | The analytic policy starts the first of several candidates now | implemented in CASIM waiting evaluation; the analytical algorithm still receives one candidate | step 7 |
| O9 | Solver latency: `DecisionApplied`, one solve in flight per decision with a dirty flag, the latency models, the delayed reward in RL | designed, not traced | step 9 |
| O10 | The write-set rule may require precedences that never matter in a run | implemented conservatively; shared write sets with equal precedence are rejected by `resolve` | reviewed after step 6 |
| O11 | Admission into unstarted tours with depot origin and empty prefix | incumbent-start batching and `InsertOrders`/`ReplaceSuffix` exercise in-place draft growth; a standalone cheapest-insertion admission stage remains open | step 8 |
| O12 | `typing.Annotated` markers on input and output fields | not prototyped | step 9, optional |
| O13 | Freezes inside one tour (index pins) and order splitting cannot be expressed by a lifecycle cut | limitation of the model | no step planned |
| O14 | Whether the order selection in `TruckDisruption` is information or a decision | needs a modeling decision | step 9 |
| O15 | Positioning with a cart taken at the depot or carried along | resolved for CASIM: patrol and center both carry an empty cart; center admission reroutes from the observed position and an unused cart returns at stream closure | positioning extension |
| O16 | Lorenz et al.: the results section of arXiv:2410.14316, and the freeze rules of Reopt in arXiv:2409.12619 beyond plan changes at the depot | partially read | literature, no step |
| O17 | `compute_segment_times` appears to assume an S-shape traversal | stochastic scenario uses S-shape; a synthesis constraint for other routers remains open | step 7 |

Two further questions are empirical rather than open design points: whether repair beats full re-solve in plan quality, and whether waiting should be combined with a rolling horizon. The implementation is built so that CASOP experiments can answer both.

## Baseline at 67f9f31

This section records the state of the code that the implementation starts from and that the tables above call "today". The decision loop works as follows: `SimulationEngine.run()` yields a projected, deep-copied snapshot, conditions gate it, the `DecisionEngine` solves and applies a commitment policy, and `step()` commits. Trigger ownership is checked at setup, opportunities are versioned, and the waiting and admission algorithms have typed, CASIM-independent inputs and outputs. The properties below are the ones the implementation removes.

### The revision scope is defined three times

Which tours may be revised in the RORSP setting is written down as a list of tour statuses in three files. `ReORSPAdapter.transform_state` projects tours whose status is not in {STARTED, DONE, CANCELLED, PENDING} back into the pick-list buffer. `State.commit_solution` cancels exactly those tours when `problem_class == "RORSP"`. `TruckDisruption.handle` uses its own list {DONE, PENDING, STARTED} and cancels tours itself, inside an exogenous event. The three lists agree today, but nothing ties them together. A user who wants assigned but unscheduled tours to stay fixed has to change three files, one of which is an event that mixes new information with a decision.

### The problem class string carries three meanings

The problem class name is interpreted in at least six places: CASIM's `TAXONOMY`, the endpoint lookup, the solution file suffix in `CoSySolver._load_solutions`, the branches of `SolutionRanker.select_best`, the branches of `State.commit_solution`, and the ATIP branch of `SimulationEngine.step`. It therefore means "which subproblems are solved" (algorithmic), "what is projected" (adapter choice) and "how the result is applied" (commit). The commit branches also name OSBP variants (ReOSBP, RLOSBP, RLORSP) that are not in the taxonomy, and `ware_ops_algos` has a second `TAXONOMY` with different entries (it lists objectives and `batching_routing`, but neither ATIP nor OBRSPW). Adding a problem means editing all of these consistently.

### Adapters hide policy in projection

There are eleven adapter classes. Most of their roughly 500 lines rebuild the same `SimWarehouseDomain`; the differences are a few selection predicates. Resources are taken as all, `available`, `not occupied`, `available and not occupied`, or `available_for_planning`. Orders are the buffer, or the first buffered order by due date (`RLOSBPAdapter`). Tours are none, started, or not started. Each of these predicates is a modeling decision that changes what the planner may decide over, but none is visible in the config, which only names the adapter class.

### Solutions are full plans, commits infer the delta

Except for admission, every solver returns a complete plan (`BatchingSolution`, `SchedulingSolution`, `WaitingSolution`). The commit has to infer what changed from the solution type plus the problem name, for example "a scheduling solution under RORSP means cancel all unstarted tours and create new ones". `AdmissionSolution` is the exception: it is a delta of (tour, order) assignments, and the projected `AdmissionTour` already carries `route_version`, `routing_origin`, residual capacity and the remaining route as a frozen dataclass.

The branch documentation says `State.commit_active_route` checks tour, version, picker, origin, residual and completed picks, visible orders and bins. The code applies the route without comparing the projected `route_version` to the live one and without checking that admitted orders are still buffered; staleness is handled only at event level. With zero simulated solve time this has no effect, but it breaks once solve time is modeled or CASIM drives a real WMS.

Smaller points in the same direction: `setup_decision_engine` reuses one `DataCard` and mutates its `problem_class` per decision, the walk-or-wait data card declares OBRSPW while the engine also runs ATIP, and `DecisionEngine.on_trigger` returns `None` when no solution exists, which the experiment loop then fails to unpack.

The commitment policy truncates the solution. `SchedulingCommitmentPolicy(n_jobs=1)` keeps the job with the earliest start, `commit_solution` removes only that job's orders or pick list from the buffer (`clear_order_buffer_by_ids`, `clear_pick_list_buffer`), and the rest of the solved plan is dropped, so the next decision sees the same input and solves the same problem again. A wait returns `WaitingSolution(action="wait")`, but `commit_waiting` returns before storing anything, so the solved plan is discarded as well.

## Related concepts in literature and industry

The literature separates the questions "when do we decide", "what may change" and "what is released". The table maps the sources the model draws on to the CASIM component that takes each concept over.

| Source | Concept | What CASIM can take from it |
| --- | --- | --- |
| Vieira, Herrmann and Lin (2003), rescheduling framework | Rescheduling policy (periodic, event-driven, hybrid) is separate from the rescheduling method (repair vs. complete regeneration) | Trigger and condition are the policy; the revision scope is the method. They are configured independently. |
| Powell (2019), unified framework | State S\_t, decision x\_t in a state-dependent feasible set X\_t(S\_t), exogenous W\_{t+1}, transition; policy classes PFA, CFA, VFA, DLA | The revision scope defines X\_t(S\_t). Rolling horizon is a DLA policy, not a problem class. Exogenous events should not make decisions (TruckDisruption). |
| Ulmer, Goodson, Mattfeld and Thomas (2020), route-based MDP | The state contains the planned routes; a decision updates the plan, and only the executed part becomes irreversible | Tours are plan objects with an executed prefix and a tentative suffix; admission and suffix rerouting are plan updates, not new problems. |
| Ichoua, Gendreau and Potvin (2000), diversion | Redirecting a vehicle while it travels to its next stop | CASIM supports forward edge continuation only, i.e. no diversion. That is a structural feature on the data card (`routing_origin`), which becomes a user choice only once a router supports diversion. |
| MPS time fences; Sahin, Narayanan and Robinson (2013) | Frozen, slushy and liquid zones of a rolling plan; replanning frequency vs. frozen interval vs. horizon length | A rolling horizon is three parameters. In the configuration they are the periodic trigger (`every_s`), the release window of the commitment policy (`release_within`, `within_s`) and `state_adapter.window.horizon_s`. |
| Lorenz, Otto and Gendreau, Reopt (arXiv:2409.12619) | Reoptimize the not-yet-started part of the plan at decision epochs | In the model this is OBRSP with scope "everything not started is rebuildable", triggered at depot returns and arrivals. |
| [Timefold continuous planning](https://docs.timefold.ai/timefold-solver/latest/responding-to-change/continuous-planning) | Planning window stages: history (pinned), published (pinned or semi-movable), draft (movable), unplanned | The split between the scope a solver may change and the part that is published after solving. Released tours are the published part and scheduled tours the draft that the next decision may revise. |
| [Timefold pinning](https://docs.timefold.ai/timefold-solver/latest/domain-modeling/modeling-planning-problems) | `@PlanningPin` fixes an entity; `@PlanningPinToIndex` fixes a list prefix, inserts allowed only after it; the pin cannot change during solving | Exactly the semantics of a started tour: completed picks and the current edge are the pinned prefix, admission inserts into the suffix. |
| [Timefold field service routing](https://docs.timefold.ai/field-service-routing/latest/real-time-planning/real-time-planning-pinning-visits) | `freezeTime`, `pinNextVisitDuringFreeze` (ALWAYS / NEVER), shift-level `pinned` that also forbids adding visits, no unpinned visit between freeze time and a pinned visit | A small vocabulary of named options instead of matrices; the prefix property as an invariant the validator can check. |
| SAP status management (status profiles) | Per object status, each business transaction is allowed, allowed with warning, or forbidden | A status-by-operation matrix for tours and orders, owned by the system, which user policy can only narrow. |
| Kubernetes CRDs and admission control | Schema validation at apply time, immutable-field rules (`self == oldSelf`), validating admission webhooks | Reject an inadmissible config at load time and an inadmissible change at commit time, with a message naming the rule. |
| Terraform plan / apply | A plan shows which changes force replacement before anything is applied | `casim check` prints the resolved decision table, and every commit is logged as a list of commands ("cancel tour 12, create tour 31"). |

Two points stand out. First, Timefold can enforce pins inside the solver because there is one solver and it owns the moves. CASIM runs heterogeneous black-box heuristics, so the same guarantee has to be split into a declared capability, a shaped input and a checked output (see the section on heterogeneous solvers). Second, every industrial system in the table (Timefold checked against its documentation, SAP and Kubernetes from general knowledge) treats "what is frozen" as data attached to plan objects and evaluated by the platform, never as a property of the algorithm or of a named problem variant.

## Sources

- [Timefold Solver: continuous planning](https://docs.timefold.ai/timefold-solver/latest/responding-to-change/continuous-planning)
- [Timefold Solver: modeling planning problems (pinning)](https://docs.timefold.ai/timefold-solver/latest/domain-modeling/modeling-planning-problems)
- [Timefold Solver: real-time planning](https://docs.timefold.ai/timefold-solver/latest/responding-to-change/real-time-planning)
- [Timefold field service routing: pinning visits](https://docs.timefold.ai/field-service-routing/latest/real-time-planning/real-time-planning-pinning-visits)
- [timefold-solver on PyPI (Python API syntax)](https://pypi.org/project/timefold-solver)

  [Lorenz, Otto, Gendreau: On picking operations in e-commerce warehouses: insights from the complete-information counterpart (arXiv:2410.14316)](https://arxiv.org/html/2410.14316v1)

  [Lorenz, Otto, Gendreau: Picking operations in warehouses with dynamically arriving orders: how good is reoptimization? (arXiv:2409.12619)](https://arxiv.org/html/2409.12619v1)

  [Gil-Borrás et al. (2023): The time-window strategy in the online order batching problem, IJPR (abstract)](https://grafo.etsii.urjc.es/en/publication/gil-borr-s-2023/)

  [Branke, Middendorf, Noeth, Dessouky (2005): Waiting strategies for dynamic vehicle routing, Transportation Science 39(3)](https://ideas.repec.org/a/inm/ortrsc/v39y2005i3p298-312.html)
- Not re-checked here, cited from knowledge: Vieira, Herrmann, Lin (2003), Journal of Scheduling 6; Powell (2019), EJOR 275(3); Ulmer, Goodson, Mattfeld, Thomas (2020), EURO Journal on Transportation and Logistics 9; Ichoua, Gendreau, Potvin (2000), Transportation Science 34(4); Sahin, Narayanan, Robinson (2013), IJPR 51; Mitrović-Minić and Laporte (2004), Transportation Research Part B 38; Henn (2012), Computers & Operations Research 39, as reported by Lorenz et al.
- Code: [kit-dsm/casim](https://github.com/kit-dsm/casim/tree/codex/waiting-intervention-foundation) at `67f9f31`, [kit-dsm/ware\_ops\_algos](https://github.com/kit-dsm/ware_ops_algos/tree/codex/waiting-intervention-foundation).
