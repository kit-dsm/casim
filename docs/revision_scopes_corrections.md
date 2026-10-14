# Revision scopes: correction plan for the implementing agent

Version 2, 14 October 2026. This version replaces every earlier version of this file. Superseded instructions were deleted, not annotated; what was done is summarized in Section 8.1, and the evidence is in `docs/revision_scopes_status.md`. Future changes to this plan are made by rewriting the affected section, never by appending corrections.

The work happens in `kit-dsm/casim` and `kit-dsm/ware_ops_algos`, both on the branch `codex/revision-scopes`. Its base commits are casim `67f9f31` and ware_ops_algos `34bd157`; the first agent run started at casim `6cd1705` and ware_ops_algos `90c9364`, which the findings in Appendix A refer to. At the time of this version the branch heads are casim `a9dadff` and ware_ops_algos `dd4ffc9`, after Task 4.12.

Read this whole file before you change any code. The next work is Phase B (Section 8.2). Its priority is the reproduction of published results (Section 7.4): without it, every check only shows that CASIM agrees with itself. Phase B also records which numbers of today's scenarios the internal checks confirm. Nothing else continues before it is done.

## 1. Sources of truth

| Question | Source | On conflict |
|---|---|---|
| What the model is, which names exist, what each type contains | `docs/revision_scopes_design.md`, exported by the user | The design document wins over this file for names and semantics |
| Vocabulary of the architecture | The CASIM paper (WSC 2026): `SimulationEngine`, `DecisionEngine`, `StateAdapter`, adapted state snapshot, `CommitmentPolicy`, `SolutionRanker`, `CoSySolver`, `TourManager`, `TourPlanningState`, `DynamicInfo`, problem class, trigger, condition, state information | A name the paper uses is never replaced by a synonym |
| How to proceed, code rules, task order, acceptance criteria | This file | This file wins over the design document for process |
| CASOP conventions: algorithm cards, `configs.yaml`, module layout, the mapper | The CASOP paper (Sections 4.1 to 4.4) and ware_ops_algos `main` | A CASIM need never changes a CASOP convention silently; Section 9 applies |
| Review of the ware_ops_algos branch against `main` and the CASOP paper | `docs/ware_ops_algos_review.md`, exported by the user | Informative; the tasks derived from it are in this file |
| Whether results are correct | The checks and verification scenarios of Sections 7.1 to 7.3, derived from the model, and the literature benchmarks of Section 7.4 | A result that violates a check, a hand-derived value or a published value is wrong, whatever any earlier version produced |
| Whether a change altered behaviour | The baseline of Section 7.6 | Every difference from the baseline is either named by the task as expected or a defect |
| Progress | `docs/revision_scopes_status.md`, kept by you | It records what is done with evidence; this file records what is to be done |
| Existing code style and existing names | The base commits (Section 4) | Follow the existing style and keep existing names |

If the design document, this file and the code disagree in a way these rules do not settle, stop and ask (Section 9). Never edit the design document yourself.

## 2. Goal

At the end of this plan:

1. Correctness follows from the design and is checked by machine (Section 7): the lifecycle and the revision scope are verified exhaustively, every run is checked against plan and resource invariants, KPIs are recomputed independently, and small verification scenarios match values derived by hand.
2. Every behaviour change is visible: each scenario configuration has a recorded baseline, and every task reports its difference from it.
3. `ware_ops_algos.planning` contains the rules of the plan (lifecycle, `Accepts`, `PHYSICAL`, plan views, state information and revision scope, decision features, commands with conversion and validation, invariants) and imports no simulator, CoSy, Luigi, Hydra or pydantic.
4. casim contains the two engines of the paper and everything that decides when something happens or changes live state.
5. All algorithm logic lives in ware_ops_algos algorithms that inherit from the base class of their subproblem, have an algorithm card, and are selected through `DomainAlgorithmMapper`. casim's engine, setup, state, conditions and component templates contain no algorithm-specific knowledge, and the engine adds no rule that the configuration does not state.
6. Every structured value is typed. A user cannot write a configuration that loads but means something else; `casim check` and `woa check` report every detectable mistake with the key, the value and what is allowed.
7. The scenarios share one structure, one entry point and one output format (Phase 6b).

## 3. Ground rules

### 3.1 Never

- Never change what an existing algorithm computes, unless a task says so.
- Never weaken a check, an exception or a guard to make a test pass. Never filter `DomainAlgorithmMapper` rejection reasons. Never skip, `xfail` or delete a failing test without the user's approval.
- Never treat a number as correct because an earlier version produced it, and never accept a baseline difference that the task does not name or whose cause you have not shown.
- Never update a baseline file in the same commit as the change that alters it (Section 7.6).
- Never change an expected value in a verification scenario to match the code.
- Never add a configuration key, enum value, type, field or command that neither the design document nor a task of this file defines. If one is needed, stop and ask.
- Never rename a name that Section 6 does not list.
- Never put algorithm-specific names, strings, constants or computations into casim's `simulation_engine`, `state`, `setup.py`, `config` or the component templates. Components in `casim/pipelines/subproblems/` only instantiate an algorithm and pass its inputs.
- Never hard-code a concrete algorithm inside another algorithm, a component or an engine. Pass the configured instance in.
- Never use `state.__dict__`, `deepcopy` as a transaction, `getattr`/`hasattr` on your own types, private attributes of another class, function-local imports to dodge cycles, or attributes set after construction.
- Never leave dead code, commented-out code you added, or "for later" modules.
- Never commit data: no instances, result tables, distance matrices, archives or papers, whatever their licence. Data is obtained by a download script (`scripts/fetch_data.py <benchmark>`) that writes into an ignored directory and checks the SHA-256 of every file against a committed `SHA256SUMS.txt`; a committed `SOURCE.md` names the URL, the date, the licence and the citation. Data without a public URL (the IJPE data, the 4D4L simulator, papers behind a paywall) is placed by the user; `SOURCE.md` says where it comes from, and the script only checks it. Data already committed on `main` stays until Task 6b.3 moves it behind the script.
- Never change existing scenario data.

### 3.2 Always

- Keep tests green after every commit, with the whole suite in one process: `python -m pytest -q` in each repository.
- One commit per task, with a message naming the task and the findings (`Task 4.12: configure incumbent router through configs.yaml (F48)`). Push after each task.
- Update `docs/revision_scopes_status.md` after each task: status, commits, evidence, baseline difference.
- Move a module with `git mv` and keep its name.
- When unsure whether code is the user's or a previous agent's, check `git log -S` and say what you found.

## 4. Code style

The base code is the reference for style. Imitate it and keep new code at least as plain.

Domain objects are `@dataclass` types, `frozen=True` for every snapshot part, view, input and command. Each domain has its own subpackage, `domain_models/<domain>/models.py`, with an `__init__.py` that re-exports. Data-card domains inherit from `BaseDomainObject` with `tpe` and `get_features()`. The plan types live in `ware_ops_algos/planning/`. Vocabularies are `class X(str, Enum)`; where order matters, the order is a module constant next to the enum with a comparison helper.

Every subproblem has an abstract base class in its package that binds the input and solution types, following `class Batching(Algorithm[list[WarehouseOrder], BatchingSolution], ABC)` in `batching/batching.py`. Every algorithm subclasses the base of its subproblem, implements `_run`, declares `algo_name`, receives its collaborators as typed constructor parameters, lives in the one module of its subproblem, is exported from the package's `__init__.py`, and has an algorithm card. Parameterized algorithms use the configuration mechanism of `configs.yaml` (component options and value options), never one card or one class per parameter value.

casim keeps its existing structure. Managers own state through explicit methods. Events implement `handle(state) -> list[Event]`. Hydra `_target_` instantiates configured components. CoSy components subclass the abstract stage of their subproblem and override a `_get_inited_*` hook.

Functions are short, exit early and name their values. Every public function and attribute has a type hint. Use `logging`, not `print`. Docstrings are one line saying what something is; comments explain why. Dispatch over solution types with `functools.singledispatch` or a typed registry, not `isinstance` chains. Compare enums with enums (`tour.accepts is Accepts.REBUILD`).

## 5. Target architecture

This section describes the end state. Section 8 says which task reaches which part of it.

### 5.1 Responsibilities

| Package | Owns | May import |
|---|---|---|
| `ware_ops_algos.domain_models` | Static warehouse domain as at the base, plus the optional `Dock` | nothing outside the package |
| `ware_ops_algos.algorithms` | Algorithms, subproblem base classes, inputs and solutions, algorithm cards and `configs.yaml` | `domain_models` |
| `ware_ops_algos.planning` | Lifecycle, `Accepts`, `PHYSICAL`, `TourView`, `TourProgress`, `PlanView`, `Window`, `StateInformation`, `RevisionScope`, decision features, commands, `to_commands`, `validation.check`, invariants | `domain_models`, `algorithms` |
| `ware_ops_algos.taxonomy`, `ware_ops_algos.domain_algo_mapper` | CASOP taxonomy and applicability, as at the base | as at the base |
| `casim` | Events, `State` and its managers, `apply`, `TourPlanningState`, `SimWarehouseDomain` with `DynamicInfo`, `DecisionContext`, `ReconsiderReason`, `DecisionMemory`, `StateAdapter`, `SimulationEngine`, triggers, conditions, `DecisionEngine`, `DecisionResult`, `DecisionSolver`, commitment policies, `SolutionRanker`, the online taxonomy with `variant` and the stage registry, configuration, `CoSySolver` and its components, the CLI, trackers, visualization, the scenario commons of Phase 6b | `ware_ops_algos`, cosy, luigi, hydra, pydantic |

Nothing in ware_ops_algos imports casim, cosy, luigi, hydra or pydantic (Task 5.0 adds pydantic to ware_ops_algos only for the card schema, in a module that no algorithm imports). `algorithms` does not import `planning`, with one exception defined by the design document: `AdmissionInput.tours` holds `TourView`s after Task 4.13. `casim.decision_engine` imports nothing from `casim.state`, `casim.simulation_engine` or `casim.events`. Only conditions read `DecisionContext.cause`. The import-boundary tests enforce these rules.

### 5.2 Module layout

New, moved or changed modules of the end state only. The design document defines the fields of each type.

```text
ware_ops_algos/src/ware_ops_algos/
  domain_models/                Dock(capacity, pallets_per_tour) as optional part of BaseWarehouseDomain
  algorithms/
    algorithm_interfaces.py     AdmissionInput, AdmissionSolution, WaitingInput, WaitingSolution,
                                PositioningInput, PositioningSolution, ActiveTourRoutingSolution,
                                OrderSplittingSolution
    algorithm_cards/configs.yaml  includes IncumbentStartBatchingSShape and the OrderCountWaiting values
    waiting/waiting.py          Waiting, NoWaiting, OrderCountWaiting, HennWaiting, AnalyticStochasticWaiting
    admission/admission.py      Admission, RemainingRouteAdmission, CenterPositionAdmission
    positioning/positioning.py  Positioning, AislePatrolPositioning, CenterRelocationPositioning
    routing/routing.py          ReroutableRouting(Routing)
    routing/walk_or_wait.py     WalkOrWaitSShapeRouting and its rerouting variant
    batching/incumbent.py       IncumbentStartBatching, router as configuration option
    card_schema.py              pydantic schema of cards and configs.yaml (Task 5.0)
  planning/
    lifecycle.py                TourStates, ResourceStates, OrderStates, Accepts, TimeAttribute,
                                TOUR_ORDER, TOUR_TRANSITIONS, PHYSICAL
    models.py                   TourView (with TourProgress), PlanView
    scope.py                    Window, StateInformation, RevisionScope
    decision_features.py        decision_features(state_information, scope)
    commands.py                 CreatePickList, CreateTour, CancelTour, ReplaceSuffix, Reschedule,
                                CommitTour, Violation, ViolationCode
    conversion.py               to_commands
    validation.py               check(commands, plan), check_scope(state_information, scope)
    invariants.py               check_plan(plan), check_progress(before, after) -> tuple[Violation, ...]
  taxonomy/taxonomy.py          as at 34bd157

casim/src/casim/
  domain_objects/sim_domain.py  SimWarehouseDomain, DynamicInfo(context, plan, memory), DecisionContext,
                                ReconsiderReason, DecisionMemory
  domain_objects/tour_model.py  TourPlanningState with earliest_start, planned_start, planned_end,
                                actual_start, actual_end
  decision_engine/              decision_engine.py, commitment_policies.py, solution_ranker.py, interfaces.py
  events/operational_events.py  OrderArrival, OrderStreamClosed, PickerArrival, PickerDeparture, PickerIdle,
                                Dispatch, Timer, Reconsider, TourStart, TravelEvent, NodeArrival,
                                PickComplete, TourEnd, TruckDeparture, ActiveTourOpportunity
  events/decision_events.py     PickListCreated
  state/                        managers, DockManager, apply.py (check, apply), views.py (OperationsView
                                with the dock fill and capacity until Phase 6b)
  simulation_engine/            simulation_engine.py, state_adapter.py, conditions.py
  config/                       schema.py, data_card.py
  setup.py                      resolve(cfg, data_card) -> dict[str, DecisionSetup]
  cli.py                        casim schema | catalog | check [--all] | run | new (run and new in Phase 6b)
  pipelines/
    pipeline_runner.py          CoSySolver; stage maps derived from the registry
    taxonomy.py                 online problem classes, variant(), stage registry
    problem_based_template.py   one abstract stage per subproblem, AbstractActiveTourRouting
    subproblems/                one module per subproblem; no sequencing module
  scenarios/                    run loop, stream builders, perturbations, common plots (Phase 6b)

casim/tests/verification/            checks of Section 7
casim/tests/baseline/                recorded baselines of Section 7.6
ware_ops_algos/tests/verification/   exhaustive checks of lifecycle, Accepts, PHYSICAL and scope
```

## 6. Names

### 6.1 Renames

Each rename is made completely, everywhere it occurs (code, configs, tests, docs, schema), in one commit.

| Old name | New name | Status | Reason |
|---|---|---|---|
| `selection`, `SelectionConfig` | `state_information`, `StateInformation` | done | Property 4 of the CASIM paper |
| `commitment: {rule, within_s}` | `commitment_policy: {_target_, ...}` | done | The paper's `CommitmentPolicy` |
| `TourStates.SCHEDULED` as draft, `TourStates.RELEASED` | `PLANNED`, `SCHEDULED` | done | Base meaning of the statuses |
| `ReleaseTour`, `InsertOrders` + `ReplaceSuffix`, `Reassign` | `CommitTour`, `ReplaceSuffix(tour_id, route, added_order_ids)`, `Reschedule` | done | One command per change |
| Reconsider strings | `ReconsiderReason` | done | No string encodes a state |
| decision keys `OBRSPW`, `ATIP` in scenario configs | `OBRSP`, `active_tour_admission` | done (4.11) | Waiting is a subproblem of OBRSP; a second decision is named after what it does |
| `PickListDone` | `PickListCreated` | done (4.11) | It reports the effect of a command |

### 6.2 Names that stay

Do not rename these: `DecisionEngine`, `DecisionEngine.on_trigger(state_snapshot, action=None)`, `DecisionEngine.solver_map`, `DecisionEngine.commitment_policies`, `DecisionResult`, `SimulationEngine.run()` returning `(done, state_snapshot)`, `SimulationEngine.step`, `StateAdapter`, `SimWarehouseDomain`, `SimWarehouseDomain.dynamic_warehouse_info`, `DynamicInfo`, `AdmissionInput`, `TourManager`, `TourPlanningState`, `CommitmentPolicy`, `CommitAllPolicy`, `SchedulingCommitmentPolicy(n_jobs)`, `WindowCommitmentPolicy(within_s)`, `SolutionRanker`, `CoSySolver`, `Condition.get_decision`, `TourStates`, `ResourceStates`, `OrderStates`, `Accepts`, `PHYSICAL`, `TOUR_ORDER`, `TOUR_TRANSITIONS`, `RevisionScope`, the config key `scope` with `freeze_from` and `frozen_accepts`, `TourView`, `PlanView`, `DecisionContext`, the decision features `incumbent` and `routing_origin`, the problem classes of both taxonomies, the waiting algorithm names, `WaitingSolution.action`, `PickerArrival`, `PickerIdle`, `OrderStreamClosed`, `TruckDeparture`, `TruckDisruption` (until Task 6b.1).

### 6.3 Names that go

Already removed: `TourStates.ASSIGNED`, `Accepts.REVISE`, `TimeAttribute.ORDER_DATE`, `PickListStates`, `RevisionRights`, `PlanEntry`, `OperationalPlan`, the keys `on_violation`, `waiting`, `release_gates` and `solver.output`, `expected_*` fields, the commands `Wait`, `RecordConsidered` and `RecallPositioning`, `PlanView.active_candidate_ids`, the loose `DynamicInfo` fields, the positioning fields, `casim/decision_engine/waiting.py`, `RoutingDone`; in Task 4.11 the sequencing stage (`AbstractSequencing`, `SequencingPlanCandidate`, `subproblems/sequencing.py`, the `sequencing` entry of OBRSP) and the `scenario` config group of the Henn scenario; in Task 4.12 `WaitForOne` to `WaitForFour`, `WaitFor0` to `WaitFor4`, `_ActiveSShapeRerouting`, the extra waiting, admission and positioning modules and the hand-written `routing_class` in a card.

To be removed, by task:

| Task | Names |
|---|---|
| 4.13 | `AdmissionTour`, `to_admission_tour`, `ActiveTourBatch`, `ActiveTourBatchingSolution`, `InvariantViolation`, `AbstractBatchProvider`, `_ActiveTourRoutingNode` |
| 4.14 | `SequencingDone`, `PickerTourQuery`, `BreakStart`, `BreakEnd`, `ShiftStart`, `BreakCondition`, casim's use of `Resource.occupied`, `mark_picker_occupied`, `mark_picker_free`, `_busy_until`, `State.available_for_planning` (unused, a second definition of idle), `state.is_break`, `state.break_duration`, `TourPlanningState.start_time` and `end_time_planned`, the `picker_available` argument |
| 4.15 | `FlushRemainingOrders`, `WMSRun`, `OrderIngestion`, `VolumeShiftAcrossDay`, `TruckArrival`, the `simulation`, `policy` and `routing` config groups, the duplicate `BaseTourEvent` in `events/base_events.py`, `warehouse_info: {type: online}` in data cards |
| 6b | `OperationsView`, the scenario hooks, the five experiment scripts, the grocery scenario (unless the user decides otherwise) |

## 7. Verification

Three questions decide whether CASIM's results can be trusted, and each has its own evidence. Is the simulation internally sound? Layers 7.1 to 7.3 answer this with exhaustive checks, invariants, a recomputation of every KPI from the tour records and small scenarios whose values are derived by hand. Does CASIM reproduce what others have published? Layer 7.4 answers this with literature benchmarks on public instances. Did a change of this branch alter behaviour? Layers 7.5 and 7.6 answer this by comparison with the base commits and with a recorded baseline.

No earlier output of this project is a reference for correctness: not the base commits, not the paper-results commit `aaa8333`, not the branch head. The scenarios were written before any of these checks existed, and several are known to contain defects (F26, F28, F57, F60 and the grocery and IJPE points of Task 4.9). A number counts as confirmed only when a check of 7.1 to 7.4 confirms it.

### 7.1 Exhaustive checks over finite domains

Exist since Phase 2 for the lifecycle, `Accepts`, `PHYSICAL` and the scope (ware_ops_algos), and for `variant` and the write sets (casim). They check: `TOUR_TRANSITIONS` only moves forward in `TOUR_ORDER` and reaches every status from `planned`; `PHYSICAL` never increases along `TOUR_ORDER`; `check_scope` reports no violation exactly when the scope's level never exceeds `PHYSICAL` for an exposed status; a tour's `accepts` level never increases as its status advances; `variant` is defined or rejects with a named reason for every feature combination; setup rejects overlapping write sets without distinct precedences.

### 7.2 Invariants after every step

`ware_ops_algos.planning.invariants` reads a `PlanView` taken with full state information and returns `Violation` objects, never raising. `check_plan` verifies that every order is in exactly one place, that one picker's tours do not overlap, that every `scheduled`, `pending` and `started` tour has a picker and times, and that a started tour's route visits every unpicked position and its boxes fit its cart. `check_progress` verifies that status changes are in `TOUR_TRANSITIONS`, that executed prefixes and completed picks never change, that terminal tours never change and that time never decreases.

After Task 4.14, casim also checks resources after the last event of every timestamp: at most one `pending` or `started` tour per resource; no tour became `pending` before its earliest start or while its resource was absent; no present, non-busy resource has a first queued tour that could start; idle, busy and absent time per resource sum to the horizon; the planned times of every queue equal its projection, except for a resource absent without known return.

A fixture runs both sets of checks after every event and every `step` of every scenario and test configuration (Task 5.2). A Hypothesis test runs random small instances with random accepted configurations, and a `RuleBasedStateMachine` drives random command sequences, where `validation.check` must reject exactly the commands that would violate an invariant.

### 7.3 KPI recomputation and verification scenarios

KPIs are recomputed in `casim/tests/verification/` from the tour records alone: distance from the route edges in the layout graph, tour duration from the time model, order completion as the end of the tour that delivers the order. They must equal the trackers' values in every run of 7.2.

The time model is the design document's: a `ScheduledJob` occupies its picker from `start_time`, when the setup begins, to `end_time = start_time + tour_setup_time + distance / speed + time_per_pick × picks`. `Dispatch` makes the tour `pending` once its picker is free and present and its earliest start, if any, is reached, and the route starts `tour_setup_time` later. Write the model once in the test module's docstring and derive every expected value from it. If it is not precise enough, stop and ask.

Verification scenarios, each small enough to derive every expected value by hand, with the derivation in the docstring before the test runs:

1. One picker, one order: start, end, distance, completion (fails on the base because of the double setup, F28).
2. Two orders, cart capacity two, `NoWaiting` against `OrderCountWaiting(2)`.
3. `HennWaiting` with one batch: `reconsider_at` equals Henn's threshold computed by hand, and all jobs of that picker are delayed accordingly.
4. With `SchedulingCommitmentPolicy(n_jobs=1)`, the second tour stays `planned`, a later arrival is merged into it, and the scheduled tour is unchanged.
5. A candidate that changes a `scheduled` tour under `freeze_from: scheduled` is removed with `ViolationCode.ACCEPTS`, and the next candidate is selected.
6. Admission to a started tour: unchanged executed prefix, a suffix that visits the added order, distance prefix plus suffix.
7. `WindowCommitmentPolicy(within_s)`: committed tours and times, and the requested `reconsider_at`.
8. Periodic trigger: decision times on the grid from the simulation start, none after the order stream closes.
9. `AnalyticStochasticWaiting`: the waiting time equals the closed-form optimum, computed independently.

### 7.4 Literature benchmarks

A literature benchmark is a public instance set, or a documented instance generator, together with values that a publication reports for it, run in CASIM under the protocol of that publication. Without them, every check of 7.1 to 7.3 only shows that CASIM agrees with itself. The register below lists every benchmark known so far. You keep it, with sources, protocol and status, in `docs/literature_benchmarks.md` (Task B.3); extend it whenever you find another one, and never drop one silently.

The benchmarks differ in what they can show, and each report states the kind:

- Exact: the same instances, a deterministic method that CASIM can run, and per-instance values. CASIM must reproduce them.
- Statistical: a published generator and published averages of a deterministic rule. CASIM regenerates instances with the same generator and must reproduce the averages within their sampling error.
- Bound: published instances and per-instance values of a method that CASIM does not have (a metaheuristic, a best-known value). CASIM's rule must not be better than a proven optimum and should not be better than a best-known value; the gap is reported.

| # | Benchmark | Instances and source | Published values | Kind |
|---|---|---|---|---|
| L1 | The seven CASOP sets (SPRP, SPRP-SS, BahceciOencan, HennWaescher, MuterOencan, Foodmart, Kris), as static instances | ware_ops_pipes `data/instances/README.md` and `data/reference/README.md` (Heßler and Irnich; Wahlen and Gschwind 2023, per routing policy including S-shape; Wahlen 2026; Briant et al. on the van Gils instances) | Per instance; CASOP's own results for the same pipelines | Exact |
| L2 | Henn (2012), online order batching with FCFS and other batch selection rules, S-shape and largest gap | Generator in the paper: 10 aisles with 45 cells per side, order sizes uniform on 5 to 25 items, n = 30, 60, 90, 120 orders over eight hours with exponential interarrival times, capacity 45 or 75 items, 48 length units per minute, 6 items per minute, 3 min setup; 50 instances per class. Working paper: journals.ub.ovgu.de/index.php/FEMM-WPS/article/view/204; journal version in Computers & Operations Research 39 (2012) | Averages per class in minutes, Tables 7.1 to 7.4 of the working paper: maximum completion time and turnover time for FCFS with the selection rules FIRST, SHORT, LONG and SAV (for example FCFS, FIRST, S-shape, 30 orders, capacity 45: 467 min completion, 44 min turnover). Two rows of the S-shape block are both labelled 120/45; report which one matches | Statistical |
| L3 | OOBP of Gil-Borrás et al. (2020): 144 instances, 64 of Henn and Wäscher and 80 of Albareda-Sambola et al. (2009), four-hour horizon with arrival times | `oobp_instances.zip` and `experiments.xlsx` of grafo.etsii.urjc.es/optsicom/oobp.html; the 64 Henn instances are in `scenarios/scenario_henn_online/data/oobp/`, the 80 Albareda instances are not yet in the repository | Per instance: GRASP+VND and best known, maximum completion time and maximum turnover time | Bound |
| L4 | D'Haen, Braekers and Ramaekers (2023), integrated order picking under dynamic order arrivals, IJPR 61(10), doi 10.1080/00207543.2022.2078747 | OSF, doi 10.17605/osf.io/7r4gn (CC BY 4.0): two-block warehouse with 12 aisles, 1 m/s, 180 s setup, 10 s per pick, batches of at most 10 orders, 3 or 5 pickers, orders with truck departures as due dates, 72 factor combinations with 10 instances each; author version at documentserver.uhasselt.be/bitstream/1942/37567/2/authorversion.pdf | Per instance on OSF for the LNS: total tardiness and pick time | Bound; it is the published benchmark closest to CASIM's truck and due-date setting |
| L5 | *Walking vs. Waiting*, radar standard case | KIT RADAR dataset "Benchmark Instances for Manual Warehouse Order Picking", loaded by `simulation=radar_standard_case`; the instances are sound. The published values were produced with the simulator of `project_4D4L/2_Stochastic_Waiting`, which has known defects in the intervention (the paper says only orders on the remaining route are inserted, but the simulator can reverse the route) | Values reported for the standard case per wait-k policy (paper in `docs/literature/`) | Reproduction with the original simulator, then a decision-by-decision comparison with CASIM; differences are attributed with the checks of 7.1 to 7.3 |
| L6 | Lorenz et al. (2024, 2025), reoptimization and the complete-information counterpart, arXiv 2409.12619 and 2410.14316 | 2,600 small instances in an electronic companion, if it is accessible | Exact complete-information optima on small instances | Bound (lower bound for every online policy); only if the companion is available |

L1 is the strongest check of the layout, the distance, the time model and plan execution, and it tests whether the ware_ops_algos branch still computes what `main` computes. In CASIM, a static instance is the case in which every order arrives at time 0, the stream closes at time 0, one decision solves the whole instance with `CommitAllPolicy`, and the pickers execute the plan. For a pipeline that CASOP runs on the instance, CASIM must reproduce CASOP's objective exactly: the distance for the routing and batching sets, the total time (setup, picking and travel, summed over tours) for Kris. Where CASOP's value equals the published value, CASIM must equal the published value. No CASIM result may be below a value marked proven optimal for the same objective and routing policy.

L2 is the only published check of a simple online rule that CASIM can run as published; it tests the online event logic, the arrival process and both objectives. L3 and L4 place CASIM's rules on published instances. For L3 the protocol is Table 4 of Gil-Borrás et al. as listed in `docs/oobp_assumptions.md`, with the depot of the benchmark page (bottom left for Henn and Wäscher; as stated per warehouse for Albareda-Sambola). For L4 the protocol is Section 5 of the paper. CASIM spends no simulated time on solving, whereas the published metaheuristics use real-time budgets; for a rule without search this has no effect, and the report says so.

Running every instance of every benchmark is not feasible. Each benchmark is run on a subset chosen by a stated strategy: stratified by the factors that define the instance set (for example size, capacity, storage policy, layout, arrival pattern, number of pickers), with every stratum represented, a fixed seed, all instances with a proven optimum or a best-known value where the set marks them and they fit the budget, and a size justified by the run time measured on a pilot and, for statistical benchmarks, by the confidence interval the comparison needs. The strategy, the resulting instance lists and the expected run time are part of `docs/literature_benchmarks.md` and are approved by the user before the runs.

Never adjust a parameter of CASIM to match a published value. A difference is explained by a stated difference of protocol, a defect, or not at all, and the report says which.

### 7.5 The base commits as a diagnostic

The base results at casim `67f9f31` and ware_ops_algos `34bd157`, reproduced on 13 October 2026, are a point of comparison, not a target:

| Scenario command | Orders | Tours | Decisions | Makespan | Distance | Mean order completion |
|---|---:|---:|---:|---:|---:|---:|
| `python -m scenarios.scenario_henn_online.experiment_henn viz.launch=false` | 100 | 25 | 26 | 34,702 | 11,603 | 14,584.174 |
| `python -m scenarios.scenario_stochastic_waiting.experiment_stochastic_waiting viz.launch=false` | 4 | 2 | | 87.085 | 76 | 54.835 |
| `python -m scenarios.scenario_walk_or_wait.experiment simulation=explicit_example viz.launch=false` | 6 | 3 | | 167 | 160 | 107.167 |

`scripts/compare_with_base.py` runs a scenario on the base and on the branch and prints, per decision, the inputs, candidates, waiting result and committed tours, and the first decision where they diverge. Acceptable causes of a difference are the fix of the double setup (F28), a base defect shown by a check or a verification scenario, or a change the design document makes.

### 7.6 Baseline at the branch head

The baseline records what every scenario configuration does at a given commit, defects included, so that every later change shows its effect. It is a change detector, not a statement of correctness; what it says about correctness is limited to the result of the checks it records.

- `scripts/baseline.py record` runs every runnable configuration of the inventory (Task B.1) and the benchmark runs of Tasks B.4 and B.5 with fixed seeds and writes one file per configuration to `casim/tests/baseline/<scenario>__<configuration>.json`: the commit of both repositories, the command, the KPIs (orders, tours, decisions per decision name, makespan, distance, mean order completion, idle time per picker), the sequence of committed tours (id, picker, orders, start, end), a digest of the normalized event trace (event type, time, tour id, picker id, in processing order), and the result of the checks: the number of violations of `check_plan` and `check_progress` per `ViolationCode` with the first violation of each code, and every KPI whose recomputation (7.3) differs from the tracker's value.
- Each file states one status: `consistent` when there is no violation and no recomputation difference, otherwise `inconsistent`. `consistent` means that the run contradicts neither the invariants nor its own tour records; it does not mean that the configuration models what its README claims. That is what the verification scenarios test.
- `scripts/baseline.py compare` runs the same configurations and prints, per configuration, every KPI that differs, the first committed tour that differs, whether the trace digest differs and whether the status or the violation counts changed. `tests/test_baseline.py`, marked `@pytest.mark.baseline`, fails on any difference.
- Every task report contains this comparison. A task may only produce the differences it names as expected, and it may never turn a `consistent` configuration `inconsistent`. After the user has accepted the report, the baseline is re-recorded in a separate commit with the message `Baseline: re-record after Task <n> (<reason>)`.

## 8. Work plan

### 8.1 Done

| Phase | Tasks | Result |
|---|---|---|
| 0 | 0.1, 0.2 | Base results reproduced (7.5); one-process test suites; solver cache test; cosy-luigi pinned |
| 1 | 1.1 to 1.5 | Dead code, unused mapper capability path and copied design document removed; one mapper requirement path |
| 2 | 2.1 to 2.9 | `planning` package with lifecycle, views and scope; `TourManager.advance` as only status writer; simulator types back in casim; generic `StateAdapter`; exhaustive and invariant tests |
| 3 | 3.1 to 3.16 | Commands, conversion and validation in `planning`; commitment policies; decision features; online taxonomy with `variant`; ranker; `DecisionSolver`; typed configuration; setup; two-phase `apply`. Task 3.17 (all scenarios on the new keys) continues as Tasks 4.11, 5.1 and 6.1 |
| 3b, 3c | all | OOBP reference and assumptions (`docs/oobp_assumptions.md`); audit of moved modules; official OOBP files in `data/oobp/` |
| 4 | 4.0 to 4.12 | Subproblem base classes; waiting as a pipeline stage with the whole schedule; injected routers; analytical precondition; positioning as ordinary tours; incumbent router from its card; card and base check; name sweep; setup counted once (F28); `Dock` and `DockManager` (F42); one order-count waiting card (F43); residual admission after picks (F44); configuration and template fixes with the Henn sweep restored as decision alternatives (4.11, F45, F54, F56); CASOP conventions in ware_ops_algos with configured cards and the applicability lock (4.12, F46 to F49). Tasks 4.11 and 4.12 were done before a baseline existed; Phase B checks them afterwards |

Two results of Task 4.9 wait for the user and are handled in Task 6b.6: a `TruckDeparture` with `capacity: null` releases no pallets, and the grocery truck capacity counts orders while staging counts pallets.

### 8.2 Phase B: benchmarks before further refactoring (do this first)

Phase B establishes what the scenarios do today, which of their numbers can be trusted, and whether CASIM reproduces published results, before Tasks 4.13 to 4.15 change more code. It uses the three kinds of evidence of Section 7: internal checks, literature benchmarks, and comparison with earlier and recorded runs. Expect that some of today's scenarios and some earlier results are wrong. Phase B finds and documents defects with evidence; it does not fix them, and it changes no existing behaviour and no existing scenario file. It adds checks, tests, loaders, scenarios for the benchmarks and documentation.

Work from the current heads; B.1 is done (`feb7e69`). Commit each step separately. Stop early and report if a static benchmark of B.4 gives a different value in CASIM than in CASOP for the same pipeline and instance, because every later result depends on that agreement.

- [x] **B.1** Inventory (done, `feb7e69`). List every scenario configuration: each scenario's default configuration and every combination its sweeps and documented commands select (for example the walk-or-wait policies and routings, the Henn repair repository, the three stochastic waiting repositories). For each, run `casim check` and the scenario, and record in `docs/baseline.md`: the command, whether it runs, and if not, the error and its cause (missing loader, missing data, undefined problem class, emptied file). Rename `docs/ware_ops_algos branch review.md` to `docs/ware_ops_algos_review.md` with `git mv`.
- [ ] **B.2** Internal checks (7.2, 7.3). Write the KPI recomputation in `casim/tests/verification/` and a run option that calls `check_plan` and `check_progress` after every event and collects the violations without raising. If the engine has no place to call them without changing behaviour, add one that is a no-op when unused, and show with the base scenarios of 7.5 that the results are unchanged. Then write the nine verification scenarios of 7.3, each with its derivation in the docstring before the test runs. A scenario that fails is not fixed here: mark it `pytest.mark.xfail(strict=True, reason=...)`, show whether the code or the derivation is wrong, and name the task that fixes it (for example Task 4.14 for start times after `Dispatch`, Task 6.5 for the stream end) or propose one.
- [ ] **B.3** Literature register and data. This task and B.4 and B.5 are the priority of Phase B; do them before B.6 to B.9. Write `docs/literature_benchmarks.md` with one section per benchmark of the register in 7.4: the full citation and DOI, the URL of instances and results, the licence, a protocol table (each parameter, its value, and the page or table of the source that states it), the published values CASIM is compared with (file and columns, or table numbers), what CASIM needs that does not exist yet (loader, KPI, card, layout feature), and the status. Search for further benchmarks with published instances or generators and published values for online order batching, picker routing or scheduling, and add every one you find with the same fields; say where you searched. Obtain the instances and result files through `scripts/fetch_data.py` as Section 3.1 requires, including the Albareda-Sambola instances of L3 and the OSF files of L4. If a source cannot be reached from your machine, list it and ask the user for the files. Then measure the run time of one instance per benchmark (and per pipeline for L1), propose the subset of every benchmark as Section 7.4 requires, with the instance lists and the expected total run time, report the register and the subsets, and wait for the user's approval before B.4.
- [ ] **B.4** Exact static benchmarks (L1). Add the scenario `scenario_static_benchmarks`, which loads an instance of a CASOP set with the loader CASOP uses in ware_ops_pipes (do not write a new parser; add ware_ops_pipes as a development dependency of casim, and stop and ask if that is not possible), releases all orders at time 0, closes the stream at time 0, and solves one decision with `CommitAllPolicy` and a given CoSy repository. For each of the seven sets use the approved subset of B.3, drawn from the instances that CASOP's comparison `experiments/evaluation/03_compare_vbs_to_references.py` covers with a reference value, and list it in the scenario's README. For each set run at least the pipeline whose CASOP result is closest to the reference and the S-shape pipeline where the set has one. Run the same pipelines in CASOP with ware_ops_pipes at its pinned ware_ops_algos commit. Report per set: CASIM against CASOP (required: equal, to a relative tolerance of 1e-9), CASOP against the reference, and every CASIM result below a proven-optimal reference.
- [ ] **B.5** Online literature benchmarks (L2 to L6, as far as B.3 obtained them). Build what each needs without changing existing behaviour: a loader per instance format (for L3 the archive's order files, `settNN.txt` and arrival files of all groups; for L2 a seeded generator of the paper's instances), the KPIs maximum completion time, maximum turnover time (`max(completion_time - order_date)`) and mean turnover time, picker speed as an instance value instead of the hard-coded 1, and the batch selection rules the papers use as scheduling cards (for L2 the rules FIRST, SHORT, LONG and SAV as Henn defines them; for L3 the batch containing the oldest order). Report each new card's CASOP applicability; the changes of the applicability lock that they cause are expected and are the only ones allowed. Then run, under the checks of B.2:
    - L2: the approved number of generated instances per class of Tables 7.1 to 7.4 (the paper uses 50) for FCFS with each selection rule and both routings; report the mean and its 95 % confidence interval next to the published value for every cell.
    - L3: the approved subset of the 144 instances with FCFS batching, the oldest-order selection and S-shape, no waiting; report per instance and in aggregate against GRASP+VND and best known for both objectives.
    - L4: if the OSF files were obtained, the approved subset of the instances whose layout CASIM can represent (say which it cannot, for example the two-block layout) with an earliest-due-date rule; report tardiness and pick time against the LNS per instance and in aggregate.
    - L5: obtain the original simulator of `project_4D4L/2_Stochastic_Waiting` (the walk-or-wait README refers to it; ask the user if it is not on your machine) and run it on the standard case for every wait-k policy. First establish whether it reproduces the published values; report each value as reproduced or not. Then run CASIM on the same instances and compare the two simulators decision by decision and tour by tour: arrivals seen, batches, routes, insertions into started tours, start and end times, KPIs. For every difference, show which simulator is right using the model of the design document and the checks of 7.1 to 7.3 (invariants, KPI recomputation, hand-derived scenarios), and attribute it: a defect of the original simulator (for example a route reversed after an insertion although the paper inserts only orders on the remaining route), a documented difference of CASIM (the intervention difference of the README), or a defect of CASIM. Do not change either simulator. Where CASIM is shown to be right, report the corrected values next to the published ones; they are a result of this work, not a failure to reproduce.
    - L6: only if the electronic companion was obtained.

    Read the sources for every aggregate value of a rule that CASIM can run exactly and reproduce those as well. A value below a proven optimum or below a best-known value: stop and report. The existing Henn scenario keeps its converted files; whether its study moves to the archive instances is decided after this report.
- [ ] **B.6** Earlier results of this project (7.5). For each scenario, list in `docs/baseline.md` the earlier numbers it should be compared with, and give each one status: reproduced; differs, with a cause shown (a defect at the earlier commit, a defect at the head, or a deliberate change of this plan); or differs without a shown cause. Fix nothing.
    - The base commits, with `compare_with_base.py`.
    - The Henn waiting-strategy study: check out `aaa8333` in a separate worktree, run its sweep for two instances and both strategies, and confirm whether `scenarios/scenario_henn_online/scripts/plots/waiting_strategy/summary.csv` reproduces; check whether that sweep and `summary.csv` also omit `71s-100-60-0`, and why; run the sweep restored in Task 4.11 at the head for the same instances and compare with both; run the B.2 checks on `aaa8333` if they apply without changes, and report whether those runs are consistent. Known causes of differences: the stream end (Task 6.5), the double setup (F28), the snapshot order (F26). State in the report that the study's instances are the converted files, not the published OOBP instances (F60).
- [ ] **B.7** Record the baseline (7.6). Write `scripts/baseline.py` and `tests/test_baseline.py`, and record every runnable configuration of B.1 and the benchmark runs of B.4 and B.5 with the checks of B.2. Run `compare` twice to show that the recording is deterministic; if it is not, find the source of nondeterminism and report it before anything else.
- [ ] **B.8** Check Tasks 4.11 and 4.12 afterwards, which were done before a baseline existed. In separate worktrees at casim `2b08a5f` with ware_ops_algos `bb9c174` (before Task 4.11) and at the commits of Task 4.11, copy in the scripts of this phase and the engine change of B.2 if there is one, change nothing else, record, and compare each recording with the next one up to the head, mapping the old decision keys to the new ones. Expected: for Task 4.11, no difference except the decision names and the restored Henn sweep, which has no earlier recording; for Task 4.12, none. Report every other difference with the task and the change that caused it, and revert nothing.
- [ ] **B.9** Replace `docs/revision_scopes_baseline.md` by a pointer to `docs/baseline.md` and delete its outdated tables. Mark `docs/waiting_intervention_foundation.md` as superseded by the design document, or delete it if nothing links to it.

Report the literature part (B.3 to B.5) as soon as it is done, before B.6: the register of `docs/literature_benchmarks.md` with the status of every benchmark, the table of the static benchmarks (set, instances, CASIM against CASOP, CASOP against the reference, results below a proven optimum), and one table per online benchmark (published value, CASIM value, kind, status). Wait for the user's answer, then do B.6 to B.9 and report the scenario configurations (runs, status `consistent` or `inconsistent` with the violation codes, verification scenarios and their result, earlier numbers and their status). After the user has accepted that report, continue with Task 4.13. A defect that Phase B finds and no task of this plan fixes is added to Appendix A by the user, not by you.

### 8.3 Phase 4, remaining tasks

Each task report includes the baseline comparison (7.6) and lists the differences it expected.

- [ ] **4.13** Types under the state model (F50, F52, F62). Expected baseline difference: none, except the configurations in which `validation.check` rejected a candidate with "Added orders are not in the exposed order buffer" while the added orders came from a tour cancelled by the same candidate; they include both 61s-80-30-0 configurations, which become runnable. List every configuration that changes.
    - `validation.check` treats orders of tours cancelled by the same command set as released work for `CreateTour` (`free_ids`), but checks the additions of `ReplaceSuffix` only against the order buffer. Check both against the same set of released work. A test with two drafts merged into one, the case of 61s-80-30-0 at t = 2255.438 reduced to its essentials, derived by hand.
    - The execution fields of a started tour form one frozen `TourProgress`; `TourView.progress: TourProgress | None`.
    - Admission: `AdmissionInput` holds `orders`, `tours` (started `TourView`s with `progress`), `candidate_order_ids`, `considered_pairs` and `input_closed`; `AdmissionSolution` holds `assignments`, `considered_pairs` and `return_tour_ids`. Delete `AdmissionTour`, `to_admission_tour`, `ActiveTourBatch` and `ActiveTourBatchingSolution`. Move the closure rule of `AdmittedTourBatch` (`problem_based_template.py`, `context.input_closed and view.holding`) into both admission algorithms: a holding tour without active orders is returned once the stream is closed. `AdmittedTourBatch` only assembles batches from `assignments` and `return_tour_ids`.
    - `InvariantViolation` merges into `Violation`; `ViolationCode` gets the invariant codes.
    - CoSy stages: `AbstractActiveTourRouting(BaseComponent)` with `batching_sol = CoSyLuigiTaskParameter(AdmittedTourBatch)` replaces the mixin `_ActiveTourRoutingNode`; depot routing depends on `AbstractBatching` only; `AbstractBatchProvider` goes. A test shows that a repository with FiFo, `AdmittedTourBatch`, `SShape` and `ReroutableSShape` synthesizes exactly the two valid pipeline families.
    - Compare lifecycle values with enums in all CoSy nodes and `IncumbentProvider`; filter a decision's tours through one function instead of three copies.
    - `AbstractWaiting` takes the picker of the first job; make the single-picker assumption a requirement of the waiting cards that need it and fail clearly otherwise.
    - One helper builds any `Routing` from layout and resources, used by every node that builds a router.
    - Keep `DynamicInfo`, `SimWarehouseDomain`, `DecisionContext` and `DecisionMemory`. Report the number of types before and after.
- [ ] **4.14** One dispatch mechanism for resources (F55). Read the sections "Resources and dispatch" and "Planned and executed times" of the design document first. Expected baseline difference: Henn, walk-or-wait and stochastic waiting identical except idle time, whose definition changes; report idle time before and after, and whether any decision selects a different set of resources because a draft no longer reserves its picker.
    - Defects at `640b5ce`, found by reading the code: `Reschedule` emits no event, so a tour moved to another idle picker never starts and a tour moved earlier starts at the old time; an earlier tour committed for a picker that waits for a later one sends no query; the break is one global flag that the first `BreakEnd` clears for all pickers, and idle pickers never take it; no wake-up checks availability, and `PickerArrival(picker_available=True)` wakes no picker; `occupied` is set at commit; the grocery hook's `PickerArrival` defaults to `picker_available=False`.
    - A resource's queue is the ordered list of its committed tours. Resource state is derived: `BUSY` with a `pending` or `started` tour, otherwise `OFF_SHIFT` if absent, otherwise `RESERVED` if its queue is not empty, otherwise `IDLE`. `ResourceManager.resource_status` computes this classification, checking `BUSY` before `OFF_SHIFT` because a tour is never interrupted, and every part of casim that asks whether a resource is free, absent or reserved calls it: the snapshot filter of `state_information.resources`, the `Dispatch` handler, the emission of `PickerIdle`, the tracker and the resource checks of 7.2. No other code reads `occupied`, `available` or the queue to answer that question. The facts it reads are held in casim, not on the shared domain objects: presence by `ResourceManager` (warehouse state, changed only by `PickerArrival` and `PickerDeparture`), the queue and the tour statuses by `TourManager` (plan state). casim stops writing and reading `Resource.available` and `Resource.occupied`, which stay in the shared domain model of ware_ops_algos, where no algorithm reads them; list both for removal (Section 9.1). `ResourceStates` stays in `ware_ops_algos.planning.lifecycle` as the vocabulary of `state_information.resources`; its docstring, like that of `OrderStates`, says that it is a classification derived by the simulator and has no transitions, which only `TourStates` has.
    - Availability: a resource is absent until its first `PickerArrival(picker_id)`. `PickerDeparture(picker_id, duration: float | None)` makes it absent from its next free point on; with a duration it returns by itself (a break), without one at the next arrival (a shift end). A tour is never interrupted. Hooks emit one departure per picker instead of `BreakStart`; the IJPE hook emits arrivals for the pickers working that day and departures without duration for the others.
    - `Dispatch(picker_id, version)` replaces `PickerTourQuery` and the wake-up of `SequencingDone`. Each resource has a queue version, increased by every commit, reschedule (old and new picker), cancel of a committed tour and availability change; an older `Dispatch` is stale. `apply`, `TourEnd`, `PickerArrival` and the effective start and end of a departure emit one. The handler leaves a busy or absent resource alone, makes a pending departure effective, and otherwise takes the first queued tour: none, the resource becomes `IDLE`; its earliest start in the future, it queues `Dispatch` then; otherwise the tour becomes `pending` and `TourStart` follows after the setup. `Dispatch` keeps the priority of the event it replaces (0 from `apply`, 1 otherwise) and cannot be a trigger. `PickerIdle` is emitted once per transition into `IDLE`.
    - Times: `TourPlanningState` gets `earliest_start` (binding, optional), `planned_start`, `planned_end` and `actual_start`, `actual_end`; `start_tour` no longer overwrites the plan. `TourView.start_time` and `end_time` hold the planned times, and `TourView` gains `earliest_start`. `CreateTour` and `Reschedule` get `earliest_start: float | None`, set by the decision engine from the decision key `start_times: binding | forecast` (default `binding`, today's behaviour). The engine projects a resource's queue after every start, end, queue or availability change (first tour at the later of its earliest start and the time the resource becomes free, each later tour at the later of its earliest start and its predecessor's planned end); the projection never reorders the queue. `StateAdapter` reads `available_from` from it. The tracker records planned times at commit.
    - `TruckDeparture` no longer emits `PickerArrival`; configurations that need a decision at a departure name `TruckDeparture` as trigger. Triggers on `ShiftStart` or `BreakEnd` become `PickerArrival`. The tracker computes idle, busy and absent time from the log of state changes; idle means present and not busy.
    - The resource checks of 7.2, and tests with two pickers each: a tour rescheduled earlier; a tour moved to another idle picker; an earlier commit while the picker waits for a later tour; a break with one picker busy and one idle; a shift end with queued tours; a departure and arrival at one timestamp; a tour that ends early under `binding` and under `forecast`; a tour that ends late.
- [ ] **4.15** Hidden rules (F57). Apply in this order, with a baseline comparison after each step; each step names its expected difference.
    - `FlushRemainingOrders` is a trigger in five decisions, no code emits it, and the engine lets it bypass the conditions. Delete the event, the engine clause and the trigger entries. Expected: none. (At `aaa8333` the engine emitted it when the event heap ran empty with orders buffered; Task 6.5 restores that behaviour as `stream_closure: at_exhaustion`.)
    - The engine lets `ReconsiderReason.WAIT` bypass the conditions. Keep it until Task 6.4. Expected: none.
    - `StateAdapter` switches the due-date window off once the stream is closed. Remove the exception; `casim check` rejects a window `by: due_date` when the data card declares no `due_date`. Expected: differences only in configurations with a due-date window; list them.
    - `StateAdapter` adds the pickers of exposed started tours to the snapshot even when `state_information.resources` excludes them. They stay reachable through the tour view and leave `snapshot.resources`; components that need them read `TourView.picker_id`. Expected: none.
    - `StateAdapter` hard-codes `ResourceType.HUMAN` and `OrderType.STANDARD`; take both from the state. Expected: none.
    - The stochastic waiting decision encodes the precondition of `AnalyticStochasticWaiting` (q − 1 known orders) as `NbrOrdersCondition(threshold=3, allow_when_done=true)`, so its no-waiting and Henn variants also wait for three orders. Remove the condition; the analytic algorithm returns `wait` until its precondition holds. Expected: the analytic variant unchanged, the two other variants changed; report them.
    - Timer events with empty handlers (`WMSRun`, `OrderIngestion`, `VolumeShiftAcrossDay`, `TruckArrival`) become one event `Timer(name)` and one trigger form `{timer: <name>}`. A scenario declares fixed timers once as `timers: {<name>: {every_s, offset_s, until_s} | {at_s: [...]}}`; a hook may add `Timer(name)` at a computed time. The IJPE `t + 1e-6` markers go: `OBP` gets a precedence below `RORSP`'s 20. Expected: none for the runnable scenarios.
    - Instance values in the `simulation` groups move into the data card's `source` section; the `routing` and `policy` groups of walk-or-wait go, because a repository file now lists its components and configured cards; `warehouse_info: {type: online}` leaves the data cards. Expected: none.
    - The schema rule "scheduling decision" derives from `variant()` instead of naming `ORSP` and `OBRSP`; `DecisionTracker` records decision name and problem class separately; the duplicate `BaseTourEvent` goes. Expected: none.

- [ ] **4.16** Remove what nothing uses (F63). Expected baseline difference: none. Found by reading the branch at `b44fd8a`; check each item again before deleting it, and keep it if anything references it.
    - Delete: the empty `pipelines/subproblems/assignment.py`; `tests/verification/reference/center_before_task44.json`, which no test reads; the committed cache `tests/data/instances/caches/vis_input/inital_domain.pkl`; `tests/data/instances/29s-40-30-0.txt` and its arrival file, which no configuration names; `tests/test_henn_offline.py`, which asserts nothing and is covered by the baseline; `tests/test_tour_model.py`, which only constructs a `TourPlanningState`; the unused functions `io_helpers.load_json`, `io_helpers.dump_json`, `viz.app.picker_pos`, `viz.gantt_chart.plot_gantt_from_jobs` and `viz.kpis.active_picklist_order_ids`; the committed plots and table of the Henn study (`scripts/plots/waiting_strategy/*.png`, `summary_table.tex`), keeping `summary.csv` as the reference of Task B.6.
    - Order splitting: `OrderSplitter` in the template, `SimpleSplitter` in `subproblems/order_splitting.py` (with a hard-coded pallet size) and the `OSBP` entry of `TAXONOMY` are used by no configuration. Delete them unless the user decides that CASIM supports order splitting (Section 9.1).
    - `tests/test_dummy_rl_env.py` asserts `info["done_at_reset"], True`, which reads as a comparison but is an assertion with a message; assert the value. Let it use a scenario configuration instead of `tests/config/` and `tests/data/`, then delete both directories.
    - Remove the commented-out code (`events/operational_events.py`, `state/tour_manager.py`) and replace the `print` calls in `src/` by `logging`.
    - Keep the CoSy components that no repository file names yet (`OrderNrFiFo`, `LSBatchingNNFiFo`, `LSBatchingSShapeFiFo`, `ClarkAndWrightNN`, `ClarkAndWrightSShape`, `TSPRouting`, `RatliffRosenthal`, `ERDScheduler`): they are the portfolio the static benchmarks of Task B.4 run. Each must be run by B.4 or by a smoke test; report any that is not.

Phase 4 is done when the tests pass, the name sweep test passes, and the Phase 4 report lists every baseline difference with its cause, every card whose CASOP applicability differs from `main` (none expected), and the type counts of Task 4.13.

### 8.4 Phase 5: verification gate

- [ ] **5.0** Enforce the conventions mechanically.
    - A pydantic schema for algorithm cards and `configs.yaml` in ware_ops_algos (closed vocabularies for `problem_type`, `implementation.type` and `objective`, `extra="forbid"` on the implementation, every constrained feature also listed in `features`), and cross-card checks (unique `algo_name` and `component_name`, `class_name` resolves to a subclass of the base for its `problem_type`, every `configs.yaml` entry resolves).
    - `woa check` runs them and the applicability lock of Task 4.12 against the data cards of ware_ops_pipes, copied as fixtures; a difference fails, and `woa check --update-lock --reason "..."` records a reason.
    - `casim check` calls the same checks; `casim check --all <scenario>` composes every file of every config group and sweep of a scenario.
    - One stage registry in `casim/pipelines/taxonomy.py` maps each subproblem to its abstract stage and plan-candidate bridge; `pipeline_runner.py` derives its maps from it; the stale `endpoint` entries leave `TAXONOMY`; `casim check` compares registry and `variant()`. Find out why `build_pipelines` filters synthesized pipelines to configured classes; remove the filter if its cause can be fixed in the class definitions, otherwise document it next to the filter.
    - `AGENTS.md` in both repositories with the conventions of Section 4, the CASOP conventions, and the rule that `woa check`, `casim check`, the tests and the baseline comparison pass before every commit; a CI job on every branch.
- [ ] **5.1** The Henn decision follows the design document's example: no `NewInformationCondition` (F27). Stochastic waiting and walk-or-wait keep their conditions after Task 4.15. Expected: Henn changes; explain the difference with `compare_with_base.py`.
- [ ] **5.2** The remaining checks of 7.1, and the invariant fixture and Hypothesis runs of 7.2. No verification scenario of 7.3 may remain `xfail`; a scenario that still fails names its cause and waits for the user.
- [ ] **5.3** Run `compare_with_base.py` on the three scenarios of 7.5 and explain every difference.

Do not continue to Phase 6 while a verification test or a literature benchmark fails or a difference has no demonstrated cause. When the user has accepted the differences, re-record the baseline; it then serves as the regression reference.

### 8.5 Phase 6: extensions on the corrected core

- [ ] **6.1** Port `rolling_horizon.yaml` as the design document's rolling-horizon strategy, with `problem_class: OBRSP`: rolling horizon is a decision policy (trigger, window, scope, commitment), not a problem class. Use the triggers of the strategy table or justify each additional one, and state its `start_times` mode with a reason in the README. `henn_rolling` synthesizes due dates (`due_offset_s: 5400`) for the due-date window; label them synthetic in its data card and the README (O10). Port `release_only_example.yaml` as `commit_only_example.yaml`, and the walk-or-wait patrol and center relocation examples. Add each to the baseline.
- [ ] **6.2** Run them under the invariant fixture and report their KPIs.
- [ ] **6.4** Waiting as a plan fact (O17), after the user has accepted Phase 5. A wait is a later binding start or the absence of commitment. Remove `DecisionMemory.waiting` and `reconsider_at` if and only if the Henn, analytic waiting and walk-or-wait baselines stay identical; otherwise stop and report.
- [ ] **6.5** Stream closure (O19, F51). Replace the hard-coded `OrderStreamClosed` at the last order date by the scenario parameter `stream_closure` with the values `at_last_arrival` (labelled as oracle in results, default, keeps the baseline), `at_exhaustion` (the behaviour of `aaa8333`: closure when the event heap is empty with orders buffered, and one forced decision), `announced` (a cutoff time known from t = 0) and `never`. Add a card feature for observable closure, which policies without timeout that rely on closure (such as `OrderCountWaiting`) require. Comparisons with published Henn numbers use `at_exhaustion`. Report which closure assumption Henn (2012) and Gil-Borrás et al. make; do not guess it.

Task 6.3 of version 1 is part of Task 6b.6.

### 8.6 Phase 6b: one scenario structure

Read the section "Scenarios and exogenous information" of the design document first. Today the five scenarios share the engine but nothing else: five entry scripts with copies of the main loop, three hook styles, loaders that read different config groups, scenario-specific outputs, plots that partly run the simulation; the grocery loader does not exist; the IJPE hooks rewrite the event heap and the domain; and `TruckDisruption` cancels tours in its handler, so an exogenous event changes the plan without a decision (F58).

- [ ] **6b.1** Exogenous information as one typed stream. A loader returns W_0, B_0 and an `ExogenousStream`, a time-ordered tuple of events from a closed set: `OrderArrival`, `OrderUpdate(order_id, due_date=None, cancelled=False)`, `PickerArrival`, `PickerDeparture`, `DepartureUpdate(departure_id, time=None, capacity=None)`, `Timer` and `OrderStreamClosed`. Handlers change W or B only, never the plan. Departures belong to W (the `Dock` holds the schedule, which also replaces `OperationsView`), and the engine emits `TruckDeparture` from it. Hooks are deleted.
- [ ] **6b.2** Perturbations as declared, typed and seeded transformations of the stream, listed under `perturbations:` and applied before the run: `inject_orders`, `reveal_early`, `update_due_dates`, `move_departure`, `absence`, each with a selector by due date, arrival window, customer or volume, each recording its changes in the trace.
- [ ] **6b.3** One layout and a generator: `README.md`, `config/<name>.yaml` with exactly the groups `data_card`, `decisions`, `cosy_repo`, `sweep`, one `loader.py`, optional `metrics.py` and `plots.py` (which reads output files only), and a `data/SOURCE.md` with `SHA256SUMS.txt`; the data itself is fetched by `scripts/fetch_data.py` or placed by the user, never committed (Section 3.1). Move the data committed on `main` behind the script here. `casim new <name>` creates the skeleton; shared code moves into `casim.scenarios`.
- [ ] **6b.4** One entry point and one output format: `casim run <scenario> [overrides]` writes `kpis.json`, `decisions.json`, the trace, the output of `metrics.py` and the Gantt chart; `--dash` starts the dashboard. `scripts/baseline.py` uses `casim run`. `casim check --all` checks the layout.
- [ ] **6b.5** Port Henn online, walk-or-wait and stochastic waiting, one commit each, with their baselines unchanged. Delete the grocery scenario unless the user decides otherwise.
- [ ] **6b.6** Reintegrate the IJPE study (F59) once the user has added the original data, the original loaders and the IJPE paper to `docs/`. Read the paper and list every simulated number it reports. Then, stopping after each step:
    1. Reproduction: run the original code on the original data at the commit the paper used (ask if neither the paper nor the history says which), compare with the paper, port the scenario (original loaders as `loader.py`, hooks as perturbations, the cancellation of `TruckDisruption` as a decision), run again and explain every difference by a fix of this plan, a changed semantics or a defect of the original. Where a published number can only be reproduced by restoring a defect, present the options.
    2. Data audit: units (including 1.6 TE per order), day boundaries, duplicate or missing lines, due dates before arrival, articles without location, pickers per day against the records, the configuration values, the two points of Task 4.9. Report each with evidence and its effect; do not change the data.
    3. Closeness to the WMS records: define the measures first (per day tours, picks per tour, tour durations, utilization, completed picks over time, lateness against departures), calibrate on a subset of days and report on the others, and state what the model cannot represent.

    The data may be confidential and is placed by the user (Section 3.1); `data/SOURCE.md` says where it lives.

### 8.7 Phase 7: documentation

- [ ] **7.1** `docs/verification.md`: the verification layers, the time model, the derivations, the base comparison, the baseline and its accepted differences, the open points the implementation touches (O2, O3).
- [ ] **7.2** In casim's README, "Configuring a decision", with one annotated configuration per strategy of the design document under `tests/config/strategies/`, each checked by a test with `casim check`.
- [ ] **7.3** `casim catalog` lists tour statuses and transitions, `Accepts` and `PHYSICAL`, commitment policies, problem classes with subproblems, decision features, conditions, triggers and timers, and algorithm cards with requirements.

## 9. Stop and ask

Stop and ask when:

- CASIM and CASOP give different values for the same pipeline on a static benchmark instance, or a result lies below a proven optimum or below a best-known value of the online benchmark;
- a verification scenario and the code disagree and you cannot show which is wrong;
- a baseline difference appears that the task does not name, or a difference from the base or a published number has no cause you can show;
- the time model is not precise enough to derive a value;
- a task needs a type, field, key, command or enum value that neither the design document nor this file defines;
- the design document, this file and the code disagree, or a statement here is false (say which and how you checked);
- keeping an existing algorithm's result requires changing it;
- a task would touch CASOP behaviour or change the applicability lock;
- a task touches one of the pending user decisions of Section 9.1.

Ask in a short message: what you found, the evidence (command and output), and the options you see. Then wait. Do not stop for a factual slip that changes no task (a version number, a date, a count); note it in the report and continue.

### 9.1 Pending user decisions

Leave these as they are on the branch and list them in each phase report until the user decides:

1. The removed `due_date` requirement of `spt_scheduling.yaml` (F46).
2. The semantics of `Resources.get_features()`, which reports cart values or `None` instead of `True` (F49).
3. The per-picker processing time in `earliest_free_assignment` (F34).
4. The objective vocabulary of the cards (24 cards say `Distance`; the CASOP paper leaves constructive methods without objective).
5. Whether the grocery scenario is deleted or rebuilt (Phase 6b). At `b44fd8a` it has no loader and no data, its repositories are inapplicable to ORSP, and none of its seven configurations runs.
9. Whether CASIM supports order splitting (Task 4.16).
6. Whether rolling horizon on the Henn instances, with synthetic due dates, is kept (Task 6.1).
7. Whether the stochastic waiting scenario stays a scenario or becomes a verification fixture.
8. Whether `Resource.available` and `Resource.occupied` are removed from the shared domain model once casim no longer uses them (Task 4.14).

## 10. Done criteria

- All phases are reported, and every finding of Appendix A is fixed or deferred by the user.
- CASIM equals CASOP on every static benchmark run of Task B.4, no online benchmark of Task B.5 has a value below a proven optimum or best-known value, and every benchmark of the register in `docs/literature_benchmarks.md` is run or has a stated reason why not.
- `python -m pytest -q` passes in one process in both repositories, including exhaustive, invariant, resource, KPI recomputation, verification, baseline, import-boundary, AST and state-machine tests.
- `casim check --all` passes for every scenario, and `woa check` passes with an applicability lock equal to that of `main` or changed only with the user's recorded approval.
- No name of Section 6.3 is referenced anywhere.

## Appendix A: findings

Findings F1 to F41 come from the review of `6cd1705` / `90c9364`, F42 to F59 from the reviews during Phases 3 and 4, F60 and F61 from the review of the benchmarks on 14 October 2026, F62 from Task B.2, F63 from a review of unused code on 14 October 2026.

| # | Finding | Fixed by | Status |
|---|---|---|---|
| F1 | Henn deviation of `6cd1705` (26 tours, 12 decisions, six single-order tours), caused by F4, F26, F27 | F4, F26, F27; comparison | 5.1, 5.3 |
| F2 | Problem class derived from solver output, selection and component names | Declared `problem_class`, `variant()` | done |
| F3 | Setup filtered mapper reasons, mapped waiting names, special-cased conditions | Removed | done |
| F4 | Waiting received one candidate; the engine duplicated Henn's rule | Whole schedule | done |
| F5 | Analytical waiting turned its precondition into a silent wait | Error restored; precondition in the algorithm (4.15) | 4.15 |
| F6 | Ranker with invented weights, no feasibility filter, last key wins for batching | `changed`, feasibility, check error | done |
| F7 | `SShapeRouting` hard-coded, `resources[0].pick_cart` | Router injection; configured router (4.12) | done |
| F8 | Dead operational types, pipeline executor, loose fields, capability path | Deleted | done |
| F9 | Positioning strings and route computation in casim | Ordinary tours, router | done |
| F10 | `decision_engine/waiting.py` mixed gates, waiting and commitment | Waiting stage, commitment policies | done |
| F11 | Waiting delay retimed only the first job | All jobs of the picker | done |
| F12 | String-encoded reconsider causes | `ReconsiderReason` | done |
| F13 | An event could trigger only one decision | Tuple of decisions | done |
| F14 | Unobservable statuses, unused levels | Trimmed lifecycle | done |
| F15 | `TourPlanningState` in ware_ops_algos | Back in casim | done |
| F16 | `TourView` with positioning fields and admission bridge | Flat `TourView`; admission takes `TourView` (4.13) | 4.13 |
| F17 | Untyped snapshot parts | Typed `DynamicInfo` | done |
| F18 | `PlanView.active_candidate_ids` | Removed | done |
| F19 | `DecisionSolver` unused | Protocol | done |
| F20 | `SimulationEngine` assembled after construction, repair path | Constructor, removal | done |
| F21 | `casim check` imported `scenarios` | `config/data_card.py` | done |
| F22 | `apply_all` deep-copied and overwrote `State` | Two-phase check and apply | done |
| F23 | Tour status written in nine places | `TourManager.advance` | done |
| F24 | `expected_*` fields could never fail | Removed | done |
| F25 | Order-dependent tests with one cosy-luigi build | Regression test, pinned build | done |
| F26 | Snapshot order changed FIFO input | Sorted by `(order_date, order_id)` | done |
| F27 | Commit-only path with stale start time; `NewInformationCondition` in Henn | Memory and policies done; Henn config | 5.1 |
| F28 | Setup counted twice | Counted once | done |
| F29 | Copied design document; baseline doc presenting the Henn deviation as intended | Deleted; `docs/baseline.md` (B.9) | B.9 |
| F30 | Two mapper requirement implementations | One | done |
| F31 | `henn_rolling` synthesizes due dates | Labelled synthetic | 6.1 |
| F32 | Flat positioning module | Package | done |
| F33 | No algorithm rebuilds and inserts at once | `variant` rejects, O2 | done |
| F34 | Per-picker processing time in `earliest_free_assignment` | Homogeneous results tested; semantics by user decision | 9.1 |
| F35 | `Wait` and `RecordConsidered` as commands | `DecisionResult` fields | done |
| F36 | `InsertOrders` and `Reassign` | One `ReplaceSuffix`, merged `Reschedule` | done |
| F37 | `ReleaseGates` for IJPE only | Removed; IJPE in 6b.6 | 6b.6 |
| F38 | Enum comparisons through strings | Enum comparisons | 4.13 |
| F39 | `RoutingDone` made a decision's outcome depend on an event | Rejected at setup | done |
| F40 | CASOP taxonomy changed on the branch | Restored | done |
| F41 | Simulator domain types in ware_ops_algos | Back in casim | done |
| F42 | Dock attached by hooks, `hasattr` branches | `Dock`, `DockManager` | done |
| F43 | Four cards and components for `OrderCountWaiting` | One card; value option (4.12) | done |
| F44 | Admission after a pick failed | Validation of unpicked orders | done |
| F45 | Numeric waiting aliases, inconsistent card lookup | 4.11, 4.12 | done |
| F46 | SPT card lost `due_date` | User decision | 9.1 |
| F47 | Two CASIM cards applicable in CASOP classes | Applicability lock | done (4.12); 5.0 |
| F48 | Parameterized algorithms bypass `configs.yaml`; module layout and mixin deviate from `main` | Configuration mechanism, layout | done |
| F49 | `None` features absent; `get_features()` semantics; mapper docstrings | Docstrings; semantics by user decision | done (4.12); 9.1 |
| F50 | 40 new types without a stated model | Consolidation | 4.13 |
| F51 | Stream closure at the last order date is oracle information | `stream_closure` | 6.5 |
| F52 | CoSy stages typed too loosely; closure rule in an assembly node | Stage typing, admission fields | 4.13 |
| F53 | Hard-coded stage maps, stale endpoints, unexplained filter | Stage registry | 5.0 |
| F54 | Dead sequencing stage, invalid rolling configuration, committed outputs | Removal, check | done (4.11); 6.1 |
| F55 | Picker dispatch defects | Derived resource state, availability, `Dispatch` | 4.14 |
| F56 | Henn sweep emptied, stale sweep path and README, group files never composed | Restore, corrections, `casim check --all` | done (4.11); 5.0 |
| F57 | Hidden rules outside the decision model | Removal | 4.15 |
| F58 | No common scenario structure; exogenous events that change the plan | Phase 6b | 6b |
| F59 | IJPE never checked against paper and data | Reintegration | 6b.6 |
| F60 | The Henn online scenario and the waiting-strategy study of `aaa8333` use converted Heßler-Irnich files whose item positions differ from the published OOBP instances (`docs/oobp_source_comparison.md`); only the arrival files are identical, so their results are not comparable with published OOBP values | Archive loader and OOBP protocol; decision on the study after Phase B | B.5 |
| F61 | No result of the branch has been compared with a published value; Phase 3b documented the OOBP protocol but ran no benchmark | Literature register and benchmarks (7.4) | B.3 to B.5 |
| F62 | `validation.check` rejects a `ReplaceSuffix` that adds orders from a draft cancelled in the same candidate, although it accepts a `CreateTour` with those orders; with one candidate, the decision fails (61s-80-30-0) | One rule for released work | 4.13 |
| F63 | Files, functions, tests and committed outputs that nothing uses | Removal | 4.16 |
