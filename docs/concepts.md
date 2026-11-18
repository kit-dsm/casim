# Concepts

This document describes the model CASIM implements: what the simulated state is, what a decision
is and which properties configure it, how far a decision may change earlier plans, how a plan is
committed and executed, and what a run writes. It describes the code as it is. How a scenario is
set up is in `docs/scenarios.md`; what has been checked is in `docs/validation.md`.

## Two repositories

CASIM uses two packages.

- `ware_ops_algos` holds everything that needs no simulator: the warehouse domain (layout,
  articles, storage, orders, resources), the algorithms with their algorithm cards, and the
  package `ware_ops_algos.planning`, which defines the plan, its lifecycle, the revision scope,
  the commands that change a plan, their validation and the plan invariants. Nothing in
  `ware_ops_algos` imports CASIM. A researcher with another simulator can build a plan view, run
  the algorithms, convert the result into commands, validate them and apply them to their own
  state.
- `casim` holds time and process: events, the live state, the simulation engine, the decision
  engine, triggers and conditions, commitment policies, the ranking of candidate solutions, the
  CoSy pipeline components and the command line.

Each decision is a CASOP problem whose instance is taken from a plan that persists between
decisions. CASOP (the static framework CASIM builds on) maps one instance to one solution; CASIM
adds the plan as an object that lives across decisions and is partly irreversible, and the
decision as a recurring, configured CASOP problem.

## State

The state at time t has three parts.

- The warehouse W: layout, articles, storage, the orders revealed so far, and the resources with
  position and presence (shift, break, absence), as well as the dock.
- The plan P: pick lists and tours with their status, planned times and execution progress. The
  executed part of a tour is the irreversible part of the plan.
- The belief B: what the decision maker knows about the future, for example whether further
  orders can arrive. Whether such knowledge exists is a static property of the scenario and is
  stated on the data card; its current value (the order stream is closed) arrives as an event
  and is read by an algorithm at decision time as `input_closed`.

Exogenous events change W or B. Only decisions change the plan. One exception remains in the
code: the disruption event of the IJPE study cancels tours itself. The IJPE revision replaces it
with an event that changes only due dates and orders, and leaves cancellation to the replanning
decision.

## Decisions

A decision is configured by these properties. The first five are those of the CASIM paper; the
revision scope is added.

| Property | Configuration key | What it answers |
|---|---|---|
| Problem class and algorithms | `problem_class`, `solver` | Which problem the pipelines solve and with which algorithms |
| Trigger events | `triggers` | Which events make the decision due |
| Conditions | `conditions` | When solving is worthwhile |
| State information | `state_information` | Which part of the state the decision sees |
| Committed components | `commitment_policy` | Which part of the selected solution becomes binding |
| Revision scope | `scope` | How far the decision may change what earlier decisions planned |

Further keys: `change_penalty` (the cost of changing an existing tour in the ranking),
`precedence` (the order of decisions due at the same time) and `start_times` (whether committed
starts are binding, see below). A scenario may contain several decisions; each has a name. By
convention the name is the problem class when a scenario has one decision of that class
(`OBRSP`), and otherwise names what the decision does (`active_tour_admission`).

### State information

`state_information` has five fields.

- `orders`: whether the order buffer is exposed.
- `pick_lists`: whether the pick-list buffer is exposed.
- `tours`: the set of tour statuses exposed.
- `resources`: the set of resource states exposed (`idle`, `reserved`, `busy`, `off_shift`).
  Only these resources appear in the snapshot. A decision that revises a started tour reaches
  that tour's picker, with its cart and speed, through the tour view (`TourView.picker`).
- `window`: an optional limit to a horizon ahead of now. With `by: due_date` it keeps orders and
  pick lists due within the horizon, throughout the run, and requires the data card to declare
  `due_date` on orders. With `by: start_time` it keeps tours that start within the horizon.

What a decision sees is defined by `state_information` alone, and when it runs by `triggers` and
`conditions` alone. One rule of the engine remains outside this model: a due reconsideration of
a waiting decision solves regardless of its conditions.

### Conditions

A condition reads the snapshot and returns whether the decision should solve. The available
conditions are: `NbrPickersCondition`, `NbrOrdersCondition` (counts the work open to the
decision: buffered orders and the orders of planned tours it may rebuild), `NbrBatchesCondition`,
`NewInformationCondition`, `VisiblePlanningWorkCondition`, `ActiveTourReadyCondition` and
`DockCapacityCondition`. If a condition fails, the decision computes no candidates and only
applies its commitment policy to the existing planned tours.

### Triggers

A decision is due when one of its trigger events occurred at the current time, when its own
reconsideration time is reached, or at its periodic tick (`{every_s: n}`, anchored at the start
and stopped once the order stream is closed). Due decisions run after all events of a timestamp,
in order of `precedence` (lower first, unset after, ties by name). Each decision runs at most once
per timestamp.

## Tour lifecycle

A tour's status moves only forward.

| Status | Meaning | Entered through | Next statuses | What it still accepts |
|---|---|---|---|---|
| `planned` | Part of the plan, not committed; a later decision may revise it | creating a tour | `scheduled`, `cancelled` | rebuild |
| `scheduled` | Committed; the picker will execute it | committing it | `pending`, `cancelled` | rebuild |
| `pending` | The picker sets up at the depot | dispatch | `started` | insert |
| `started` | The picker is underway | tour start | `done` | insert |
| `done` | Delivered at the depot | tour end | none | none |
| `cancelled` | Removed before it started | cancelling it | none | none |

Only `TourManager.advance` writes a status, and it rejects any other transition. Pickers execute
only committed tours. The status of an order (buffered, batched, planned, in progress, done) is
derived from the buffers and the tours, not stored.

## Revision scope

What a tour still accepts is one of three ordered levels: `none < insert < rebuild`. `insert`
allows adding orders and replacing the unexecuted suffix of the route; completed picks, the edge
being travelled, orders with a completed pick and occupied boxes stay fixed. `rebuild` allows any
change: cancelling, changing membership, route, picker or times. Removing orders from a started
tour is not supported.

The process sets an upper bound per status, the physical bound: `rebuild` for planned and
scheduled tours, `insert` for pending and started tours, `none` for done and cancelled tours. A
decision's scope `RevisionScope(freeze_from, frozen_accepts)` narrows it: tours before
`freeze_from` in the lifecycle accept `rebuild`, tours from `freeze_from` on accept
`frozen_accepts` (`none` or `insert`). A scope that gives an exposed status more than the
physical bound is rejected when the configuration loads.

```yaml
scope: {freeze_from: scheduled, frozen_accepts: none}   # planned tours may be rebuilt, committed tours are fixed
scope: {freeze_from: pending, frozen_accepts: insert}   # planned and scheduled tours may be rebuilt, running tours accept orders
```

## From candidates to commitment

When a decision runs, the decision engine:

1. lets the solver return candidate solutions, if the conditions hold;
2. converts each candidate into the commands it implies against the snapshot (`CreatePickList`,
   `CreateTour`, `CancelTour`, `ReplaceSuffix`, `Reschedule`, `CommitTour`);
3. removes every candidate whose commands touch tours outside the state information, exceed a
   tour's level or the physical bound, or change what a started tour has executed;
4. ranks the remaining candidates by their objective plus `change_penalty` times the number of
   existing tours the candidate cancels, revises or reschedules (new tours do not count);
5. lets the waiting stage of the selected solution decide whether to release or wait;
6. lets the commitment policy choose which planned tours to commit.

The commands of one decision are applied together or not at all. The uncommitted part of a
solution stays in the plan as `planned` tours, which a later decision sees and may revise.

| Commitment policy | Commits | Asks to run again |
|---|---|---|
| `CommitAllPolicy` | every planned tour | never |
| `SchedulingCommitmentPolicy(n_jobs)` | the first `n_jobs` planned tours by planned start | never |
| `WindowCommitmentPolicy(within_s)` | planned tours that start within `within_s` | when the next tour becomes ready, minus `within_s` |
| `PullCommitmentPolicy` | the first planned tour of each visible (free) resource | never; the next free resource triggers it |

A decision without a solver only commits.

## Synthesis per decision

Each decision gets its own data card: the scenario's data card plus decision features derived
from its state information and scope. `incumbent` is true if planned or scheduled tours that may
be rebuilt are exposed. `routing_origin` is `live` if started tours that accept `insert` are
exposed together with the order buffer, and `depot` otherwise. The features decide which
subproblems are synthesised: a decision that admits orders to running tours (`live`, no
incumbent) gets item assignment, admission and active-tour routing; a depot decision gets the
problem class's other subproblems. An algorithm card applies if its subproblem is in that list
and the mapper accepts its requirements against the decision's data card. CASIM's problem
classes and their subproblems are listed in `casim/pipelines/taxonomy.py`; the OBRSP class has
`item_assignment`, `batching`, `routing`, `scheduling`, `waiting`, `positioning`, `admission` and
`active_tour_routing`. A CoSy repository file lists the components a decision may combine;
synthesis builds every valid pipeline from them.

## Resources and dispatch

A resource's queue is the ordered list of its committed tours. A planned tour may name a resource
but reserves nothing and is never dispatched. A resource's state is derived and never stored: it
is busy if it has a pending or started tour, otherwise off shift if it is absent, otherwise
reserved if its queue is not empty, and otherwise idle.

Presence changes through two events. `PickerArrival` makes a resource present.
`PickerDeparture(duration)` makes it absent from its next free moment, at once or after its
current tour; with a duration it returns by itself (a break), without one it stays absent until
the next arrival (a shift end). A tour is never interrupted.

One mechanism hands work to a resource. Every change of a resource's queue or presence, and every
tour end, emits `Dispatch`. Dispatch leaves a busy or absent resource alone, makes a pending
departure effective, and otherwise starts the first queued tour once its earliest start, if any,
is reached. A resource that becomes idle emits `PickerIdle` once, which a decision can use as its
trigger.

Push and pull. A push decision commits tours into the queues ahead of time; a queued tour waits
for its resource, also while the resource is away, and stays queued if it never returns. A pull
decision (`PullCommitmentPolicy`) lets each free resource take one tour when it becomes free;
the rest stays planned for the next free resource, so no work waits with a resource that is
away. Pull needs the decision to see only idle resources and to be triggered by `PickerIdle` and
`PickerArrival` (the schema checks both); it solves only when a free resource sees it. A
component that plans for resources the decision does not see (`RecordedPlan`, which gives each
tour to its recorded picker, and `SAVScheduler`, defined for one picker) declares
`supports_pull = False`, and setup rejects it in a pull decision. A run reports in
`service.json` (`stranded`) the committed tours that never started because their resource was
away at the end, the failure that push can produce and pull cannot.

## Planned and executed times

A committed tour carries three kinds of time: an optional binding earliest start, the planned
start and end (the forecast later decisions read), and the actual start and end. With
`start_times: binding`, the default, a committed tour's scheduled start is its earliest start, so
the plan is a timetable. With `start_times: forecast`, it has no earliest start, so the plan is a
sequence per resource and the next tour starts as soon as the resource is free.

After every start, end and change of a queue or of presence, the engine projects the resource's
queue: the first tour is planned at the later of its earliest start and the time the resource
becomes free, every later tour at the later of its earliest start and its predecessor's planned
end; each keeps its planned duration and the order never changes. The resource's planned tours
follow its queue by the same rule, so the times a commitment policy reads stay feasible.

The time model counts the setup once:

```text
end = start + setup + distance / speed + picks * time_per_pick
```

`start` is when the picker begins the setup at the depot. Without disturbances the executed
times equal the planned ones.

## Execution

The simulation engine processes the event heap. After the last event of a timestamp it runs the
due decisions, one after another; each sees the commits of the decisions before it. Applying a
decision checks the commands against the live state, changes the state through its managers,
stores the decision's memory (waiting state, reconsideration time), and emits the process events
the changes need: `Dispatch` for changed queues, `PickListCreated`, and `TravelEvent` for a
running tour whose suffix was replaced. Solving takes no simulated time.

## Events and sources

A scenario's source provides the initial warehouse and belief and the exogenous events in time
order: order arrivals, picker arrivals and departures, the closure of the order stream, and the
scenario's own events (for example truck departures). The library sources in `casim.sources`
read the public formats of the benchmarks (Heßler and Irnich's Henn files, the OOBP archive, the
CASOP sets, Foodmart, Kris, RADAR); a scenario on new data adds its own `source.py`. The order
stream is closed at the last order's arrival, which gives a policy information it would not have
in a real warehouse; other closure rules are on the backlog (`docs/development.md`).

## Outputs

`casim run` writes per run:

- `kpis.json`: makespan, mean order completion time, tours and orders completed, mean tour
  duration and batch size, orders, tours and picks per hour, total and per-picker distance, idle,
  busy and absent time per picker, delayed orders, dock utilisation, total tour and processing
  time;
- `decisions.json`: the number of decisions and the chosen pipelines;
- `tours.csv`: per tour picker, status, start, end, setup, distance, picks and orders;
- `orders.csv`: per order arrival, due date and completion;
- `service.json`: completed and open orders (all, affected, unaffected), presence, busy, idle
  and absent time per picker, and `stranded`, the committed tours (and their orders) left with a
  resource that is away at the end;
- `trace.json`: the digest of the event trace, which identifies a run;
- with `--audit`, `audit.json`: the committed tours, completed tours, route legs, the findings of
  the plan invariants and resource checks, and the orders still open at the end;
- with a `reference.csv`, `comparison.csv`: the run's values against the published ones.

## Checks during a run

With `--audit`, CASIM checks after the last event of every timestamp and after every decision:
every order is in exactly one place; a picker's tours do not overlap; status changes follow the
lifecycle; executed prefixes and finished tours never change; time never decreases. It also
checks the resources: at most one pending or started tour per resource; no tour became pending
before its earliest start or while its resource was absent; no free resource leaves a startable
tour waiting; the planned times of every queue equal its projection. At the end, idle, busy and
absent time sum to the horizon per resource, and the orders still open are listed.

## Known limits of the model

- No algorithm rebuilds planned tours and admits orders to running tours in one decision;
  admission to running tours is a second decision.
- A decision runs at most once per timestamp, so events its own commit produces at that time do
  not make it due again.
- Removing orders from a started tour, diverting a picker mid-edge, preempting a tour, and tours
  shared by several resources are not modelled.
- Solving takes no simulated time, so computation delay is not modelled.
- There is no candidate that keeps the plan unchanged; the incumbent enters through repair
  pipelines such as `IncumbentStartBatching`.
- `SchedulingCommitmentPolicy(n_jobs=1)` commits one tour per decision, not one per picker.
- Waiting is stored in the decision's memory rather than as a fact of the plan.
- Forecasts and optimisation over distributions are not modelled.
- `AnalyticStochasticWaiting` assumes the walk-or-wait S-shape route, and a card cannot require a
  component of another subproblem; its card description states this.
- `IncumbentStartBatching` assumes one cart type for all pickers and rejects mixed carts at run
  time.
- Decision features are fixed at setup. Features of a snapshot's content (for example "exactly
  q − 1 orders are known") are not checked by the mapper; the algorithm checks them at decision
  time.

## Related concepts

The model takes its parts from rescheduling theory, sequential decision analysis and planning
software.

| Source | Concept | Counterpart in CASIM |
|---|---|---|
| Vieira, Herrmann and Lin (2003) | Rescheduling policy (periodic, event-driven, hybrid) separate from the rescheduling method (repair, regeneration) | Triggers and conditions are the policy; state information, revision scope and pipelines the method |
| Powell (2019) | State, decision in a feasible set, exogenous information, transition function; four policy classes | The feasible set is what passes validation under the revision scope; the decision engine is the policy, the simulation engine the transition function |
| Ulmer, Goodson, Mattfeld and Thomas (2020) | Route-based MDP: planned routes in the state, only the executed part irreversible | Tours with an executed prefix and a revisable suffix |
| Ichoua, Gendreau and Potvin (2000) | Diversion of a vehicle on its way | Not supported: a running tour completes its current edge |
| MPS time fences; Sahin, Narayanan and Robinson (2013) | Frozen, slushy and liquid zones; replanning frequency and horizon | Rolling horizon as periodic trigger, `WindowCommitmentPolicy` and `window` |
| Lorenz, Otto and Gendreau (arXiv:2409.12619) | Reoptimisation on order arrival, at the depot or underway | Depot reoptimisation and the admission decision |
| Gil-Borrás, Pardo, Alonso-Ayuso and Duarte (2020) | Orchestration loop of the online order batching problem | The loop splits into the simulator and two decisions |
| Timefold continuous planning and pinning | Published and draft parts of a plan; pinned prefixes; freeze time | Scheduled tours are published, planned tours the draft; the pinned prefix of a started tour; the revision scope |
| SAP status management | Per-status rules for allowed transactions | The physical bound, which a scope can only narrow |

In Powell's terms the class of a policy follows from a decision's configuration, not from its
algorithms. The waiting rules and the commitment rules are policy function approximations
(`OrderCountWaiting` with k orders; `HennWaiting`, which releases a batch at a time derived from
its critical order; `SchedulingCommitmentPolicy` with `n_jobs`). A decision that batches, routes
and schedules the known orders and minimises makespan is a myopic cost function approximation;
the change penalty modifies its cost. CASIM has no value function approximation yet; it would
enter as a ranker that adds the value of each candidate's post-decision plan. The rolling horizon
is a deterministic direct lookahead: the window is the lookahead horizon, planned tours are the
plan beyond now, the commitment policy implements only the first part, triggers solve again, and
the revision scope fixes what is already implemented. `AnalyticStochasticWaiting`, which weighs
releasing now against waiting for the next arrival under a known arrival process, is a one-step
stochastic lookahead. The revision scope, which shapes the feasible set, lies outside Powell's
classes.

Sources: Timefold Solver documentation (continuous planning, pinning, real-time planning,
field-service pinning); Lorenz, Otto and Gendreau, arXiv:2409.12619 and arXiv:2410.14316;
Gil-Borrás et al. (2020), Journal of Global Optimization 78, 295-325; Gil-Borrás et al. (2021),
Computers & Industrial Engineering 160, 107517; Gil-Borrás et al. (2023), IJPR; Branke,
Middendorf, Noeth and Dessouky (2005), Transportation Science 39(3); Vieira, Herrmann and Lin
(2003), Journal of Scheduling 6; Powell (2019), EJOR 275(3); Ulmer, Goodson, Mattfeld and Thomas
(2020), EURO Journal on Transportation and Logistics 9; Ichoua, Gendreau and Potvin (2000),
Transportation Science 34(4); Sahin, Narayanan and Robinson (2013), IJPR 51; Henn (2012),
Computers & Operations Research 39. The design discussion of forecasting and optimisation under
uncertainty (Sellmann and co-authors, Birge, Kleywegt et al., Elmachtoub and Grigas, Bertsimas
and Kallus) is in the separate document "Forecasting and uncertainty in CASIM".
