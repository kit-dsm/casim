# Revision scopes status at Phase 3c

This is an audit against the current `revision_scopes_corrections.md`, through Phase 3b. Evidence names are casim commits unless prefixed `ware:`. A row marked “moved to a later task” records an explicit phase boundary or user-approved sequencing deviation, not a completed check. The historical base results are diagnostics; Phase 5 establishes correctness.

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
| 5.2 casim decision process events | done | `SequencingDone` and `PickListDone`; `RoutingDone` deleted. |
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
| 6.1 OBRSPW/ATIP config keys | done | casim online taxonomy and active-tour decision config. |
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
