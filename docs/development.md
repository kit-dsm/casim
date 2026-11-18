# Development

This document is the one backlog of CASIM: what is being done now, what is open, what is
deferred and why, and which design decisions are still open. The rules every commit follows are
in `AGENTS.md`; the model is described in `docs/concepts.md` and what has been checked in
`docs/validation.md`.

## Now: a stable base for the IJPE revision

The IJPE revision (due 27 December 2026, plan in `docs/ijpe/plan.md`) needs a stable core. The
core is therefore finished to a frozen base, tagged `core-ijpe-base`, from which the branch
`codex/ijpe-revision` is created. After the freeze:

- the IJPE branch may change the core only for the changes its plan lists in Section 4
  (C1 disruptions as exogenous information, C2 affected-order flag, C3 decision log with
  candidate scores and solve times, C4 `picks.csv` and service measures, C5 the two dock fixes, C6 one definition of
  the tours a replanning decision may revise, C7 departures in request order),
  each with a test and with every other baseline unchanged;
- any other change to the core needs the user's approval first;
- the effect of every core change is reported on the baseline tiers and on the IJPE reference
  run.

Steps to the freeze:

| Step | State |
|---|---|
| Planned drafts follow their resource's committed queue, because every commitment policy reads their times | done |
| `tours.csv` leaves the start of a never-started draft or cancelled tour empty | done |
| The audit reports the orders still open at the end of a run | done |
| Documentation reduced to README, concepts, scenarios, validation and this backlog | done |
| Comparison with the base version (`docs/validation.md`): walk-or-wait and analytic waiting identical, Henn differs with four known causes | done |
| Full tier (60 runs equal), tests (134 pass) and `casim check --all`; tag `core-ijpe-base`; `codex/ijpe-revision` created from the tag | done |

The first task on the IJPE branch, C1, is done there: `TruckDisruption` changes the due dates
of the orders the source names and cancels nothing; cancellation happens only through the
replanning decision's revision scope. The replanning decision does not cancel the same tours as
the old handler, which cancelled tours in the affected due-time window in dictionary order until
the transport-unit volume was reached; with `freeze_from: pending` every unstarted tour is
revisable. The reproduction (P2.2) records this as a changed definition.

## Deferred until after the IJPE revision

Each item says what and why. The order is a proposal.

### Core correctness

- **Remaining hidden rules.** The state adapter hard-codes the resource type (human) and order
  type (standard) instead of taking them from the state. The analytic waiting decision encodes
  the rule's precondition (q − 1 known orders) as a condition, so its no-waiting and Henn
  variants also wait for three orders; the algorithm should return "wait" until its
  precondition holds. Timer events with empty handlers (`WMSRun`, `OrderIngestion`,
  `VolumeShiftAcrossDay`, `TruckArrival`) should become one `Timer(name)` event with a
  declared schedule. The old `simulation`, `policy` and `routing` configuration groups of the
  IJPE and walk-or-wait setups, `warehouse_info: {type: online}` in the data cards, the schema's
  hard-coded list of scheduling problem classes and a duplicate `BaseTourEvent` should go. These
  items touch IJPE event types, which is why they wait.
- **Decompose the Henn difference from the base version.** Walk-or-wait and analytic waiting
  equal the base; Henn (makespan 34,702 against 30,668.75, 25 against 27 tours) differs through
  four changes made before the regression baseline existed (setup counted once, the whole
  schedule for the waiting rule, the sorted order buffer, persistent planned tours). Applying
  them one at a time to the base would show each one's share.
- **Henn decision without `NewInformationCondition`.** The design's Henn example has no such
  condition; removing it changes the Henn results, which must then be explained against the
  base version.
- **Remaining verification.** A test fixture that runs the plan invariants over every scenario
  and test configuration; Hypothesis tests on random small instances; a state machine that
  drives random command sequences, where validation must reject exactly the commands that would
  break an invariant; the remaining exhaustive checks.
- **Waiting as a fact of the plan.** A wait would be a later binding start or the absence of a
  commitment, so the decision memory's waiting flag and reconsideration time could go; only if
  the Henn, analytic waiting and walk-or-wait baselines stay identical.
- **Stream closure.** The order stream is closed at the last order's arrival, which is oracle
  information. A scenario parameter with the values `at_last_arrival` (default, labelled as
  oracle), `at_exhaustion` (closure when the event heap is empty with orders buffered, as in
  the paper-results version), `announced` and `never`, a card feature for policies that need an
  observable closure, and a check of the closure assumption of Henn (2012) and Gil-Borrás et al.
  (2020).

### Scenarios and exogenous information

- **Exogenous events as one typed stream.** A source returns the initial state and a
  time-ordered stream from a closed set of events (order arrival, order update, picker arrival
  and departure, departure update, timer, stream closure) whose handlers change the warehouse or
  the belief, never the plan; the dock holds the departure schedule.
- **Perturbations.** Declared, seeded transformations of the stream (inject orders, reveal
  early, update due dates, move a departure, absence), each recording its changes in the trace,
  so a disruption study needs configuration, not code.
- **Parked configurations.** Two configuration sets that never ran wait on the branch
  `deferred/parked-configs` (directory `docs/deferred/`): the rolling-horizon strategy on the
  Henn instances (it declared a problem class `rolling`, which the schema rejects; it should be
  ported with `problem_class: OBRSP`, its synthetic due dates of arrival plus 5,400 s labelled as
  such, and its `start_times` mode stated) and Henn's long-horizon L2 cells
  (`decisions_l2_long.yaml` with its own solver). Port each, run it under the invariant fixture
  and add it to the baseline. `release_only_example` should then become `commit_only_example`,
  and the walk-or-wait patrol and centre-relocation examples get their own configurations.

### Benchmarks

- Recount the L3 runs below a best-known value, and recompute the L2 cell tables over all 20
  replications, after the order-count gate correction.
- L2 parts not run: the selection rules SHORT, LONG and IGNORE, Largest Gap routing, 50
  replications, the truncated arrival reading and the printed rate 0.08625, and the working
  paper's cells with LONG and SHORT. Their configurations stay in `scenarios/henn2012`.
- The 16 working-paper truncated L2 pilot records made before those cells moved to the
  working-paper decision differ as expected; record them again once the difference is accepted
  (the slim baseline keeps one such record).
- L4 (D'Haen et al. 2023) needs two-block layouts, trucks and due dates as departures, tardiness
  and pick time. L5 (RADAR) needs the commit of the original simulator that produced the
  published results, and its speed and pick time.
- The Kris loader of ware_ops_pipes merges intersections of aisles without products; its fix is
  the user's, after which the Kris L1 records are recorded again. Until then a Kris difference
  that this defect explains is not a CASIM defect.

### Conventions and tooling

- A pydantic schema for algorithm cards and `configs.yaml` in ware_ops_algos with closed
  vocabularies and cross-card checks, a `woa check` command, and an applicability lock that pins
  CASOP's applicable cards per data card against `main` (today one card differs, the first-fit
  FCFS batching added on purpose).
- One stage registry in `casim/pipelines/taxonomy.py` from which the pipeline runner derives its
  maps; the stale `endpoint` entries go; find out why `build_pipelines` filters synthesised
  pipelines to configured classes.
- `AGENTS.md` in ware_ops_algos, and a CI job on every branch.
- Speed: about 3 s of every run is importing, mostly because ware_ops_algos' layout module loads
  pandas, matplotlib and networkx; the baseline script starts a new process per run.
- Documentation of each strategy of `docs/concepts.md` as an annotated configuration checked by
  `casim check`, and a `casim catalog` that lists every status, policy, condition, trigger and
  card.

### Model extensions (not planned)

Decision delay (a decision's commands applied after its solve time), admission and rebuilding in
one decision, preemption and shared tours, forecasts and optimisation over distributions, one
registry for CASOP's and CASIM's taxonomies, and separating warehouse, workload and task in the
domain object. Each is a known limit stated in `docs/concepts.md`.

## Open design decisions

These wait for the user and are left as they are until decided.

1. Whether the SPT scheduling card requires `due_date` again (the requirement was removed).
2. The semantics of `Resources.get_features()`, which reports cart values or `None` instead of
   `True`.
3. The per-picker processing time in `earliest_free_assignment`.
4. The objective vocabulary of the algorithm cards (24 cards say `Distance`; the CASOP paper
   leaves constructive methods without objective).
5. Whether `Resource.available` and `Resource.occupied` leave the shared domain model; CASIM no
   longer uses them.
6. A command line for creating and inspecting scenarios (`casim init`, a results view).
7. Whether rolling horizon on the Henn instances, with synthetic due dates, is kept.
8. Whether the analytic waiting scenario stays a scenario or becomes a verification fixture.

Decided: the grocery scenario was deleted; order splitting is not supported and was removed.
