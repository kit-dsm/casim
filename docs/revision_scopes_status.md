# Revision scopes status through Task 4.12

This is an audit against the current `revision_scopes_corrections.md`, through Phase 3b. Evidence names are casim commits unless prefixed `ware:`. A row marked “moved to a later task” records an explicit phase boundary or user-approved sequencing deviation, not a completed check. The historical base results are diagnostics; Phase 5 establishes correctness.

## Phase 4 progress

| Task | Status | Evidence and remaining work |
|---|---|---|
| 4.0 | done | ware:`2676b78`, `ed97e5f`; casim:`2b482a0`; waiting, admission and positioning bases. Module consolidation remains in 4.12. |
| 4.1–4.2 | done | ware:`a05bf88`; casim:`1a1c7ab`, `0111acc`; whole-schedule waiting and injected Henn router. |
| 4.3 | done | ware:`a8b5142`; casim:`c07839d`; analytical precondition and cart capacity. |
| 4.4 | done | ware:`2ca5f67`; casim:`9458945`; ordinary positioning tours and return routing. |
| 4.5–4.7 | done | ware:`f29ff63`, `1c7aef9`; casim:`6387f0b`, `6853ab1`; incumbent router, card/base check and name sweep. |
| 4.8 | done | ware:`f2e467e`; casim:`3295930`; setup starts at scheduled start. |
| 4.9 | done | F42; `tests/verification/test_dock_fixture.py` compares the full prechange event, dock-fill, decision and KPI record for seeds 17 and 29 and truck capacities 2 and `null`. All four comparisons pass. ware adds optional `Dock`; casim owns `DockManager`, and the IJPE loader reads `simulation.dock.K_dock`. Grocery loader wiring is deferred to Phase 6b by the user's instruction. |
| 4.10 | done | ware:`20cb91e`; casim:`640b5ce`; one order-count waiting card and component. Task 4.12 replaces temporary component aliases. |
| 4.11 | done | Removed dead sequencing, renamed `PickListCreated` and the five decision files, restored alternative Henn strategies as `decisions` files, corrected stale references, and removed tracked outputs. One-instance sweep and 62 casim tests pass; comparison below. |
| 4.12 | done | Five configured CoSy components are generated from resolved CASOP cards; wait 0–4 and both Henn pipelines keep their KPIs. The seven CASOP applicability sets equal `main` (counts 19, 15, 19, 24, 19, 8, 13); Task 5.0 commits the permanent lock. Module layout, rerouting base, mapper docstrings, and `OrderSplittingSolution` are corrected. Four user decisions remain untouched. |
| 4.13–4.15 | not done | Complete in order after Task 4.12. |

Task 4.12 makes `walk_or_wait_sshape_routing` and `fifo_scheduler` inapplicable to CASOP by requiring the `routing_origin` decision feature, which the seven CASOP data cards lack. CASIM's corresponding data cards provide it. A comparison of the applicable card-name sets for each of those seven data cards against `main` shows no additions or removals after this change. `IncumbentStartBatchingSShape` merges the SShape requirements through `configs.yaml`; `OrderCountWaiting1` through `OrderCountWaiting4` resolve the value option. The prechange committed code and current code have byte-identical `kpis.json` for `wait_0` through `wait_4` on `explicit_example` and for `repair_waiting_henn` on `29s-40-30-0.txt`. The baseline Henn pipeline still has makespan 22533.874, 30 tours, mean tour duration 563.566667 and distance 11057, as in Task 4.11. The task generated its components without scenario-data changes.

The shipped CASOP source has no reference to order splitting outside its implementation package and no order-splitting card; `OrderSplittingSolution` is now placed with the other algorithm interfaces. I propose removing order splitting from the CASOP library in a later task unless a supported CASOP use case is identified; I did not add a card with no consumer. The following Task 4.12 decisions remain untouched: whether to restore the `due_date` requirement of `spt_scheduling.yaml`, whether to change `Resources.get_features()` semantics, whether to change per-picker processing time in `earliest_free_assignment`, and whether to revise the cards' objective vocabulary.

Task 4.9 retained two known issues: a `TruckDeparture` with `capacity: null` releases no staged pallets, and grocery truck capacity counts orders while staging counts pallets per tour. The plan mentions a `TruckArrival` dock branch, but no such event exists in the current core; this factual slip changes no Task 4.9 work. The plan's original instruction to wire both loaders is superseded by the user's explicit direction to defer the missing grocery loader to Phase 6b.

Task 4.11 corrected the Section 6.1 decision-key status: it had been marked done because removal of `OBRSPW` and `ATIP` as problem classes was mistaken for renaming them as decision names. The five decision files, their runtime references, tests, and current documentation now use `OBRSP` and `active_tour_admission`. The historical `aaa8333` Henn configurations are alternatives under the same decision key. The unused `rolling_horizon` files remain; `casim check ... decisions=rolling_horizon --json` rejects `problem_class: rolling`.

The `29s-40-30-0.txt` Henn sweep compares as follows against `scripts/plots/waiting_strategy/summary.csv` (the old `scenario_henn_order_window_waiting` label denotes the order-window alternative, not a combined strategy):

| Strategy | Old → current makespan | Old → current mean tour duration | Old → current utilization | Tours | Dock mean / max, old → current |
|---|---:|---:|---:|---:|---:|
| Henn waiting | 22533.874 → 22533.874 | 563.566667 → 563.566667 | 0.764089 → 0.764089 | 30 → 30 | 0 / 0 → 15.860469 / 30 |
| Order window (5) | 22391.988 → 22438.988 | 594.142857 → 595.821429 | 0.773911 → 0.774405 | 28 → 28 | 0 / 0 → 14.790495 / 28 |

The order-window differences in makespan and total tour duration are both exactly 47 seconds; the tour count is unchanged. This is consistent with the final route changing when `OrderStreamClosed` is emitted at the last order arrival, whereas `aaa8333` flushed at event-heap exhaustion. Task 6.5 restores `at_exhaustion`; no closure rule was changed in 4.11. The dock numbers now count staging in every `State` since Task 4.9, including warehouses without a `Dock`; the historical plot was made before that change.

The Task 4.9 fixture recorded 298 events, 13 decisions, three completed tours, distance 126 and makespan 86 in each seeded run, both before and after the change. Dock fill ended at 1 with truck capacity 2 and at 3 with `capacity: null`; the full fill trace and tour records are locked in `tests/verification/dock_baseline.json`. Neither scenario config triggers a decision directly on `TruckDeparture`. The departure's emitted `PickerArrival` triggers grocery `ORSP` through `triggers_map`; IJPE has no `PickerArrival` trigger, so it triggers no decision there.

## Tasks through Phase 3b

| Task | Status | Evidence and remaining work |
|---|---|---|
| 0.1 | done | Separate base checkout and `docs/revision_scopes_baseline.md`, Section 7.4 KPIs reproduced. |
| 0.2 | done | `1f1fa9a`, `af1a942`; `tests/test_solver_cache_isolation.py`, one-process full suites. |
| 1.1 | done | ware:`e66d4a8`, `3ce009d`; operational plan types deleted and `TourPlanningState` moved. |
| 1.2 | done | ware:`02d7885`; unused pipeline DAG executor deleted. |
| 1.3 | done | ware:`3faf07a`; mapper capability path removed, `AlgorithmCard.capabilities` retained. |
| 1.4 | done | ware:`8405404`; one mapper requirement path and mapper tests. |
| 1.5 | done | `adac710`; copied design document removed. |
| 2.1 | done | ware:`c0ed627`; `planning/lifecycle.py`, `test_lifecycle_exhaustive.py`. |
| 2.2 | done | `d1f1447`; `TourManager.advance`, AST and Hypothesis lifecycle tests. |
| 2.3 | moved to a later task | `b6eb7e6` moved `TourPlanningState`; user directed `positioning_mode` and `positioning_hold` to stay until 4.4. |
| 2.4 | moved to a later task | ware:`4af567c`, `5e190b1`; typed buffers/context complete; positioning fields remain until 4.4 by user direction. |
| 2.5 | done | `27826f4`, `d30b895`; typed `sim_domain.py` and loader constructors; ware:`54e7217` restores static domain models. |
| 2.6 | done | ware:`7fc4660`, `9a5002b`; admission component constructs `AdmissionTour`; user approved `TourView.active_order_ids` and removing unused `AdmissionTour.route_version`. |
| 2.7 | done | `d1f1447`; generic `simulation_engine/state_adapter.py`, manager methods and sorted orders. |
| 2.8 | done | `dedb385`; `OperationsView` has only the two operational facts. |
| 2.9 | done | ware:`1d95b16`; exhaustive lifecycle and hand-built invariant tests. User clarified duplicate detection in `check_plan` and loss detection in `check_progress`; full-run fixture is Phase 5. |
| 3.1 | done | `99baea8` and ware:`f652088`; old ranker compatibility shim removed in Phase 3c. |
| 3.2 | done | ware:`7d3a75b`, `4af567c`; `planning/scope.py`, exhaustive scope test and schema boundary. |
| 3.3 | done | `b6be5a5`, `d1f1447`; `commitment_policies.py`, `test_commitment_policies.py`. Ready-time meaning changes in 4.8. |
| 3.4 | done | ware:`7731b2a`; `decision_features.py` and `test_decision_features.py`. |
| 3.5 | done | ware:`54e7217` restores taxonomy byte-for-byte to `34bd157`; `7c388af`, `test_taxonomy_variant.py` hold online classes. |
| 3.6 | moved to a later task | ware:`0f64f90` defines six plan commands; temporary casim-internal `RecallPositioning` stays until 4.4 by user direction. |
| 3.7 | done | ware:`f652088`; `planning/conversion.py`, `tests/test_planning_conversion.py`. |
| 3.8 | done | ware:`9040494`, `54e7217`; pure `planning/validation.py`, `test_validation.py`. |
| 3.9 | done | `99baea8`, `d1f1447`; `decision_engine/solution_ranker.py`, setup checks for multiple batching candidates. |
| 3.10 | done | `99baea8`; `decision_engine/interfaces.py`, stable candidate names tested by solver tests. |
| 3.11 | done | `d1f1447`; named solver/ranker/policy maps and typed `DecisionResult`, import boundary test. |
| 3.12 | done | `d1f1447`; `config/schema.py`, `config/data_card.py`, committed `schemas/decision.schema.json`. |
| 3.13 | done | `d1f1447`; `setup.resolve`, routing-only rejection, write-set tests in `tests/test_revision_scope_contract.py`. |
| 3.14 | done | `d1f1447`; constructor collaborators and multi-decision trigger ordering in `simulation_engine.py`. |
| 3.15 | moved to a later task | `d1f1447` has two-phase `state/apply.py`; the temporary positioning recall apply path stays through 4.4 by user direction. |
| 3.16 | done | `d1f1447`; per-decision waiting/reconsideration in `State` and `DecisionMemory`. |
| 3.17 | moved to a later task | `d1f1447` adapts `BaseEnv`, scripts, CLI and schema; Phase 5 ports the base scenarios and Phase 6 ports IJPE and extension configs for the all-config `casim check` gate. |
| 3b.1 | done | `6836b50`; `scripts/oobp_reference.py`, `data/oobp_reference.csv`. Spreadsheet has no per-instance ILS field, per user clarification. |
| 3b.2 | done | `6836b50`; `docs/oobp_assumptions.md` maps each paper assumption and records conflicts. |
| 3b.3 | done | `6836b50`; comparison protocol in `docs/oobp_assumptions.md`, with runs explicitly assigned to 6.0. |

## Appendix A findings

| Finding | Status | Evidence or later task |
|---|---|---|
| F1 | moved to a later task | 5.1–5.3 Henn port and causal comparison. |
| F2 | done | `7c388af`, `d1f1447`; declared class and `variant`. |
| F3 | moved to a later task | Setup name rules removed in `d1f1447`; final algorithm-name sweep is 4.7. |
| F4 | done | `1a1c7ab`, ware:`a05bf88`; whole-schedule waiting stage. |
| F5 | moved to a later task | 4.3 restores the analytic precondition and card. |
| F6 | done | `d1f1447`; ranker and candidate check. |
| F7 | moved to a later task | Henn router injected in `0111acc`; incumbent router and cart are 4.5. |
| F8 | done | Phase 1 commits plus typed `DynamicInfo` in Phase 2. |
| F9 | moved to a later task | 4.4 removes positioning strings/recall/shortest path; 4.7 checks casim. |
| F10 | done | `1a1c7ab`, `d1f1447`; waiting stage and commitment policies. |
| F11 | done | ware:`a05bf88`; waiting retimes the picker schedule. |
| F12 | done | `d1f1447`; `ReconsiderReason`. |
| F13 | done | `d1f1447`; tuple trigger map and precedence. |
| F14 | done | ware:`c0ed627`; trimmed lifecycle. |
| F15 | done | `b6eb7e6`; simulator tour model back in casim. |
| F16 | moved to a later task | Flat `TourView`/admission bridge done; positioning fields intentionally stay until 4.4. |
| F17 | done | ware:`4af567c`, `5e190b1`, `dedb385`; typed snapshot and operations. |
| F18 | done | ware:`4af567c`; no active-candidate duplicate. |
| F19 | done | `99baea8`; solver protocol. |
| F20 | done | `d1f1447`; simulation constructor and repair removal. |
| F21 | done | `d1f1447`; data-card loading in casim config. |
| F22 | done | `d1f1447`; manager-backed two-phase apply. |
| F23 | done | `d1f1447`; tour status writer test. |
| F24 | done | ware:`0f64f90`; expected fields removed. |
| F25 | done | `1f1fa9a`, `af1a942`; two-solver regression and pinned cosy-luigi. |
| F26 | done | `d1f1447`; StateAdapter sorts `(order_date, order_id)`. |
| F27 | moved to a later task | Per-decision memory/policy done; Henn config/behavior comparison is 5.1–5.3. |
| F28 | moved to a later task | 4.8 time model. |
| F29 | moved to a later task | Copied design deleted in 1.5; baseline doc rewritten in 7.1. |
| F30 | done | ware:`8405404`; mapper tests. |
| F31 | moved to a later task | 6.1 labels synthetic rolling due dates. |
| F32 | moved to a later task | 4.4 positioning package. |
| F33 | done | `7c388af`; O2 rejection and exhaustive variant test. |
| F34 | done | ware:`54e7217`; homogeneous scheduler regression tests. |
| F35 | done | ware:`0f64f90`, `d1f1447`; waiting/considered data in result/memory. |
| F36 | done | ware:`0f64f90`; one ReplaceSuffix and merged Reschedule. |
| F37 | moved to a later task | Release gates removed; IJPE README note is 6.3. |
| F38 | moved to a later task | Lifecycle/planning comparisons converted in Phase 2–3; final algorithm/positioning sweep is 4.4–4.7. |
| F39 | done | `d1f1447`; routing-only setup rejection, no RoutingDone event. |
| F40 | done | ware:`54e7217`; `git diff 34bd157 -- src/ware_ops_algos/taxonomy/taxonomy.py` is empty. |
| F41 | done | `27826f4`, `d30b895`; simulator domain moved back and re-export removed. |
| F42 | done | Task 4.9, typed dock and exact synthetic trace comparison. |
| F43 | done | ware:`20cb91e`; casim:`640b5ce`; one parameterized waiting card. |
| F44 | done | ware:`0d47fe0`; casim:`c94c410`; residual admission after completed picks. |
| F45 | done | Task 4.11 follow-ups; waiting aliases move to 4.12 and dispatch checks to 4.14 as directed. |
| F46 | awaiting user decision | `spt_scheduling.yaml` due-date requirement left untouched in Task 4.12. |
| F47–F49 | done, with named decisions pending | Task 4.12 card applicability, configured components, module layout and mapper docstrings; the four named decisions above remain untouched. |
| F50 | moved to a later task | Task 4.13 type consolidation. |
| F51 | moved to a later task | Task 6.5, timing of stream closure. |
| F52 | moved to a later task | Tasks 4.11 and 4.13, admission choice and CoSy stage typing. |
| F53 | moved to a later task | Task 5.0, one stage registry. |
| F54 | done | Task 4.11 removed dead sequencing and tracked Henn output; `rolling_horizon` is retained for Task 6.1 and rejected by `casim check`. |
| F55 | moved to a later task | Task 4.14, dispatch and resource availability. |
| F56 | moved to a later task | Task 4.11 restored the alternative Henn sweep and corrected stale references; Task 5.0 checks all configurations. |
| F57 | moved to a later task | Task 4.15, hidden decision rules. |
| F58–F59 | moved to a later task | Phase 6b scenario structure and IJPE study. |

## Section 5 rules

| Rule | Status | Evidence or later task |
|---|---|---|
| 5.1 static domain ownership | done | ware static `domain_models`; `tests/test_import_boundaries.py`. |
| 5.1 algorithm inputs independent of planning | done | ware algorithm interfaces and import boundary test. |
| 5.1 planning owns lifecycle, scope, commands and invariants | done | ware `planning/`, exhaustive/validation tests. |
| 5.1 CASOP taxonomy and mapper unchanged from base | done | ware taxonomy exact base diff; mapper tests and user approval for 1.3–1.4. |
| 5.1 casim owns both engines and live state | done | casim engine/state packages and import boundary test. |
| 5.1 only conditions read `context.cause` | done | `tests/verification/test_import_boundaries.py`. |
| 5.1 every module respects forbidden import directions | done | Both repositories' AST import tests. |
| 5.2 ware algorithm interfaces | moved to a later task | Waiting/admission/active-tour inputs live in `algorithm_interfaces.py`; positioning mode removal is 4.4. |
| 5.2 ware waiting base and policies | done | ware:`2676b78`, `a05bf88`, `test_subproblem_bases.py`. |
| 5.2 ware admission base | done | ware:`2676b78`, `test_subproblem_bases.py`. |
| 5.2 ware positioning package | moved to a later task | 4.0/4.4; current flat module is temporary. |
| 5.2 ware optional depot router | moved to a later task | 4.4 first tests ReroutableSShapeRouting and adds a router only if needed. |
| 5.2 ware incumbent batching injection | moved to a later task | 4.5. |
| 5.2 ware planning/lifecycle/models/scope/features | done | `planning/`, lifecycle/scope/features tests. |
| 5.2 ware planning commands/conversion/validation/invariants | done | `planning/`, command/conversion/validation/invariant tests. |
| 5.2 ware taxonomy restoration | done | Exact diff against `34bd157` is empty. |
| 5.2 casim sim_domain/tour_model | moved to a later task | Domain and tour state moved back; positioning fields stay until 4.4. |
| 5.2 casim decision engine/ranker/interfaces/policies | done | `decision_engine/`, import and policy tests. |
| 5.2 casim decision process events | done | `SequencingDone` and `PickListCreated`; `RoutingDone` deleted. |
| 5.2 casim state managers/apply/views | moved to a later task | Generic apply/views in place; temporary recall apply path stays until 4.4. |
| 5.2 casim simulation engine/adapter/conditions | done | `simulation_engine/`, adapter and decision tests. |
| 5.2 casim schema/data card/setup/CLI | done | `config/`, `setup.py`, `cli.py`, generated schema. |
| 5.2 casim CoSy taxonomy/runner/waiting/admission | done | `pipelines/`, online variant and waiting tests. |
| 5.2 casim positioning component | moved to a later task | 4.4. |
| 5.2 listed removals | moved to a later task | All listed non-positioning modules are gone; recall/positioning paths await 4.4 by user direction. |

## Section 6 names

| Rule | Status | Evidence or later task |
|---|---|---|
| 6.1 selection/state information | done | ware `StateInformation`, casim `DecisionConfig.state_information`. |
| 6.1 commitment configuration | done | `config/schema.py`, `commitment_policies.py`. |
| 6.1 planned/scheduled lifecycle | done | ware `planning/lifecycle.py`, exhaustive tests. |
| 6.1 ReleaseTour/CommitTour | done | ware `planning/commands.py`. |
| 6.1 InsertOrders/ReplaceSuffix | done | ware `planning/commands.py`, conversion/validation tests. |
| 6.1 Reassign/Reschedule | done | ware `planning/commands.py`. |
| 6.1 membership_ops removal | done | ware `planning/decision_features.py`. |
| 6.1 waiting service-time field | done | ware `WaitingInput`, `HennWaiting` injected router. |
| 6.1 OBRSPW/ATIP config keys | done | Task 4.11 renamed all five decision files and their runtime, test, and current documentation references. Previously marked done by confusing problem-class removal with decision-key renaming. |
| 6.1 reconsider cause enum | done | casim `sim_domain.py`, `operational_events.py`. |
| 6.1 ranker move | done | casim `decision_engine/solution_ranker.py`; Phase 3c removes old shim. |
| 6.1 command/conversion moves | done | ware `planning/`; no active imports from old paths. |
| 6.1 lifecycle/planning split | done | ware `planning/`, casim `sim_domain.py`. |
| 6.1 simulator domain/tour moves | done | casim `sim_domain.py`, `tour_model.py`. |
| 6.1 DecisionSolver move | done | casim `decision_engine/interfaces.py`. |
| 6.2 protected names | done | Named API remains in casim and ware; contract tests cover `DecisionEngine.on_trigger`, `SimulationEngine.run/step` and planning enums. |
| 6.3 TourStates.ASSIGNED | done | ware `planning/lifecycle.py`. |
| 6.3 Accepts.REVISE | done | ware `planning/lifecycle.py`. |
| 6.3 TimeAttribute.ORDER_DATE | done | ware `planning/lifecycle.py`. |
| 6.3 PickListStates | done | ware `planning/lifecycle.py`. |
| 6.3 RevisionRights/PlanEntry/OperationalPlan | done | ware Phase 1 deletion. |
| 6.3 on_violation/waiting/release_gates/solver.output keys | done | Active schema rejects them; historical generated outputs are inventoried below. |
| 6.3 command expected fields | done | ware `test_commands.py`. |
| 6.3 Wait/RecordConsidered | done | ware commands and casim decision result. |
| 6.3 RecallPositioning | moved to a later task | Temporary casim-internal command retained through 4.4 by user direction. |
| 6.3 PlanView.active_candidate_ids | done | ware `PlanView`. |
| 6.3 loose DynamicInfo fields | done | casim `DynamicInfo(context, plan, memory)`. |
| 6.3 positioning_mode/positioning_hold | moved to a later task | User directed removal in 4.4 with center return. |
| 6.3 decision_engine/waiting.py | done | Deleted; CoSy waiting stage restored. |
| 6.3 RoutingDone | done | Deleted; setup rejects routing-only output. |

## Section 7.1–7.2 verification rules

| Rule | Status | Evidence or later task |
|---|---|---|
| 7.1 transitions forward/reachable/terminal | done | ware `test_lifecycle_exhaustive.py`. |
| 7.1 PHYSICAL monotonicity | done | ware `test_lifecycle_exhaustive.py`. |
| 7.1 every scope × exposed-status set | done | ware `test_lifecycle_exhaustive.py`. |
| 7.1 accepts never increases with status | done | ware `test_lifecycle_exhaustive.py`. |
| 7.1 variant total-or-named-rejection/ordered | done | casim `test_taxonomy_variant.py`. |
| 7.1 every pair of decision write sets | done | casim `test_revision_scope_contract.py`. |
| 7.2 `check_plan` order membership | done | ware `planning/invariants.py`, `test_plan_invariants.py`; user clarified no unseen-order completeness check. |
| 7.2 `check_plan` picker tour nonoverlap | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 `check_plan` committed tour assignment/times | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 `check_plan` started route and occupied boxes | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 `check_progress` legal status changes | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 `check_progress` executed prefix and completed picks | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 `check_progress` terminal tours immutable | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 `check_progress` time and order conservation | done | ware `planning/invariants.py`, `test_plan_invariants.py`. |
| 7.2 fixture after every step and event of every scenario | moved to a later task | 5.2. |
| 7.2 Hypothesis instances/configurations | moved to a later task | 5.2. |
| 7.2 command/event RuleBasedStateMachine | moved to a later task | 5.2 extends the existing TourManager machine. |

## Untouched tracked scripts and docs

The `scripts/` directory does not yet exist; `scripts/compare_with_base.py` is assigned to Phase 5. Generated image/CSV/TeX files under `scenarios/scenario_henn_online/scripts/plots/` are historical plot outputs, not executable scripts. All other tracked files under `scenarios/*/scripts/` and `docs/` that Phases 2–3b did not change are listed here. “Holds” means the file remains useful as a historical result or source and is not an active configuration.

| File | Verdict |
|---|---|
| `docs/revision_scopes_baseline.md` | Holds the base diagnostic; Phase 7 rewrites it as `docs/verification.md`. Historical old command names describe the old branch. |
| `docs/waiting_intervention_foundation.md` | Holds historical foundation notes; its opening now labels it historical. |
| `scenarios/scenario_grocery_retailer/scripts/__init__.py` | Still imports as a package marker. |
| `scenarios/scenario_grocery_retailer/scripts/experiment_plots.py` | Compiles; figure helpers hold until given tracker data. |
| `scenarios/scenario_henn_online/scripts/make_waiting_plots.py` | Compiles; holds for recorded multirun output and writes plots when executed. |
| `scenarios/scenario_ijpe/scripts/make_results_domain_validation.py` | Compiles; Phase 3c fixes its deleted `build_data_loader` import. Full execution awaits IJPE data validation in 6.3. |
| `scenarios/scenario_ijpe/scripts/make_results_scenarios_multirun.py` | Compiles; holds for recorded multirun output. |
| `scenarios/scenario_ijpe/scripts/make_summary_scenarios_multirun.py` | Compiles; holds for recorded multirun output. |
| `scenarios/scenario_henn_online/scripts/plots/waiting_strategy/scaling_makespan_orders.png` | Holds historical plot. |
| `scenarios/scenario_henn_online/scripts/plots/waiting_strategy/scatter_avg_utilization.png` | Holds historical plot. |
| `scenarios/scenario_henn_online/scripts/plots/waiting_strategy/scatter_final_makespan.png` | Holds historical plot. |
| `scenarios/scenario_henn_online/scripts/plots/waiting_strategy/summary.csv` | Holds historical plot source. |
| `scenarios/scenario_henn_online/scripts/plots/waiting_strategy/summary_table.tex` | Holds historical plot source. |

In ware_ops_algos there is no tracked `scripts/` or `scenarios/*/scripts/` tree. Its untouched tracked documentation is:

| File | Verdict |
|---|---|
| `docs/Makefile` | Still holds as the Sphinx build entry point. |
| `docs/make.bat` | Still holds as the Windows Sphinx build entry point. |
| `docs/source/_static/favicon.png` | Still holds as a documentation asset. |
| `docs/source/_static/favicon.svg` | Still holds as a documentation asset. |
| `docs/source/algorithms.rst` | Still holds; API directives name present algorithm modules. |
| `docs/source/authors.rst` | Still holds as author metadata. |
| `docs/source/conf.py` | Still holds as Sphinx configuration; documentation was not rebuilt in this audit. |
| `docs/source/examples.rst` | Still holds; links the current example notebook. |
| `docs/source/extending.rst` | Still holds; references the present `examples/custom_batching.py`. |
| `docs/source/getting_started.nblink` | Still holds; links `examples/getting_started.ipynb`. |
| `docs/source/index.rst` | Still holds as the documentation table of contents. |

Untracked user files remain untouched: casim `.env`, `.hydra/`, `.worktrees/`, `docs/research/`, the two user-owned revision markdown files, `docs/technical_report/`, `docs/visualizations/`, `scenarios/scenario_dm/`, `scenarios/scenario_ijpe/generator/private/`, `tmp/`, `tmp_output/`; ware `.worktrees/` and the three `examples/stochastic_waiting_*.py` files. The ignored ware `docs/build/` contains generated documentation, including an old loader import, and is left untouched. The ignored `domain_models/operational/__pycache__/` is the only remaining physical content at that removed package path. The ware `pyproject.toml` edit predates this audit and is excluded from its commits.

## Phase 3c checks

`ruff check --select F401,F811,F841` passed over casim source/tests/scenario scripts and ware source/tests. `vulture --min-confidence 80` reported no high-confidence dead code in the changed packages. Public imports in ware `__init__.py` files are intentionally re-exports. The cleanup commits are ware:`102dee4` and casim `75fa182`. The original Henn archive and per-instance comparison are in `scenario_henn_online/data/oobp/` and `docs/oobp_source_comparison.md` (`6eaf5ee`); all 194 committed source blobs were checked byte-for-byte against the ZIP. The archive loader gap is recorded there for Task 6.0.

The pre-existing grocery/IJPE hooks attach an optional dock manager to `State` after construction. Phase 3c preserves that behavior and its existing optional dock lookup; changing the hook contract would be a separate design decision during the IJPE work in 6.3.
