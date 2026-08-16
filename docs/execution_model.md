# CASIM execution and state model

This document is the concise ownership map for online dispatch,
reoptimization, waiting, and active-tour intervention.

## Model boundaries

| Layer | Owns | Does not own |
|---|---|---|
| `ware_ops_algos` | Static domain models, algorithm inputs, algorithms, and solution objects | Mutable simulation execution state |
| CoSy pipeline | Preparing algorithm input, invoking components, and aggregating solutions | Orders, picker motion, cart occupancy, event validity, or commitment |
| CASIM `State` and managers | The current operational truth | Algorithm-internal search state |
| CASIM events | A timed request to apply one state transition | A second copy of operational state |
| Scenario | Data/configuration, hooks, experiment loop, and concise outputs | Replacement solvers or generic execution mechanisms |

Algorithm calls are synchronous. Their wall-clock runtime is reported, but
the simulation clock does not advance while a decision is computed.

## Configuration dimensions

CASIM configurations describe independent choices. An intervention is a
replanning scope, not a scenario type.

| Dimension | Typical values |
|---|---|
| Order information | complete backlog; backlog plus arrivals; arrivals only |
| Decision timing | before execution; periodic/nightly; event-driven; policy-controlled waiting; disruption-triggered |
| Planning input | one dispatch window; due-time horizon; all visible backlog |
| Commitment | complete plan; first `n` jobs; planning horizon |
| Replanning | future work only; unstarted work; active route; active batch |
| Termination | drain; fixed horizon |

The checked-in scenarios are readable reference combinations, not an
exhaustive mode catalogue:

| Scenario | Information and timing | Commitment and replanning |
|---|---|---|
| `scenario_reopt` | arrivals only, event-driven | limited commitment; depot-only future work |
| `scenario_henn` | arrivals only, policy-controlled waiting | limited commitment; depot-only future work |
| `scenario_ijpe` | historic/generated backlog, WMS planning, continued operations and disruptions | complete or horizon commitment; optionally replan unstarted work |
| `scenario_intervention_stress` | arrivals only, event-driven | limited commitment; active route or active batch |
| `scenario_dynamic_operations` | backlog plus arrivals, nightly or periodic | complete/time-fenced commitment; buffered, unstarted, or active-route replanning |

### Decision bindings

Each entry in `engines.problems` is a `(problem_class, replanning)` pair.
`replanning` selects how far into already-committed work the variables may
change. The exposure vocabulary (what the adapter projects) is derived from
this pair; user YAML never names the low-level projection terms.

| `problem_class` | `replanning` | Exposes | Research params |
|---|---|---|---|
| `OBP` | `none` | all orders, all pickers | `limit` |
| `ORSP` | `none` | buffered batches, nonactive pickers | `due_horizon_s`, `limit` |
| `ORSP` | `unstarted` | buffered + replannable batches, available pickers | `due_horizon_s`, `limit` |
| `OBRSP` | `none` | buffered orders, dispatchable pickers | `limit` |
| `OBRP` | `none` | buffered orders, dispatchable pickers | `limit` |
| `ORP` | `active` | residual active tour | `congestion_penalty` |
| `OBRP` | `active` | residual active tour + buffered orders | `congestion_penalty`, `limit` |

Any other combination is rejected at setup. A `(problem_class, replanning)`
pair may appear at most once per engine.

A backlog-based or periodic study may also use active-route intervention. It
must explicitly select the intervention trigger, active-tour adapter, and
compatible algorithms; no separate intervention overlay is required.

## Construction and preparation

A scenario's `input` configuration names a `DataLoader`. `casim.setup`
instantiates it through Hydra, passes the configured source paths to
`DataLoader.load`, and gives the resulting `SimWarehouseDomain` to
`SimulationEngine.reset`.

`build_runtime(cfg)` is the shared construction boundary. Before the event
loop it constructs the simulation problems, adapters, triggers, conditions,
commitment policies, and solvers. Every solver is then prepared against the
effective `DataCard`. For CoSy this includes applicability filtering and
pipeline synthesis; direct solvers validate and cache only their own static
requirements. Hydra construction and pipeline synthesis therefore do not
occur in the decision loop.

`SimWarehouseDomain` is the initial planning/domain input. On reset, CASIM
creates its own runtime `State`; the domain is not the mutable source of truth
during execution.

## Applicability before execution

The checked-in `DataCard` owns static facts: item-level data, layout and
shortest paths, equipment, cart dimensions and bin count, bin mixing, and
inventory availability. Each existing `StateAdapter` declares the planning
features its snapshot provides. During scenario setup CASIM copies the card,
adds those declared features, and sets the decision problem class.

`DomainAlgorithmMapper` compares that effective card with `AlgorithmCard`
requirements and capabilities before Maestro constructs a pipeline. For
example, an active residual route requires residual picks and arbitrary
origin/end support; active-batch insertion additionally requires partial-batch,
locked-owner, residual-capacity, and empty-bin support. Only applicable
configured components enter the CoSy repository. An explicitly configured
repository that cannot form a pipeline fails during setup.

This separates three concerns:

- `DataCard`: what the warehouse data and static equipment can represent;
- `StateAdapter`: what CASIM promises to project for this decision;
- `AlgorithmCard`: what an algorithm can solve.

CoSy tasks prepare inputs and execute algorithms. They do not decide
applicability. Adapters enforce operational snapshot invariants but do not
inspect algorithm cards.

## Runtime ownership

| Owner | Mutable source of truth |
|---|---|
| `State` | Current time, completion flags, picker state, dock state, and atomic cross-subsystem operations |
| `OrderManager` | Buffered, committed, and completed orders |
| `TourManager` | Planned/queued/active/completed tours and each active tour's execution state |
| `StorageManager` | On-hand quantities and per-tour reservations |
| `LayoutManager` | Static layout plus runtime edge/zone/pick-location occupancy and FIFO waiters |
| `Tracker` | Observations/KPIs after state transitions; never decision authority |

`TourPlanningState` owns the mutable execution state of one tour:

- active batch and cart-bin ownership;
- completed and remaining concrete picks;
- annotated route and cursor;
- executed route prefix;
- active edge or active atomic pick;
- route version and pending intervention flags.

`CartBinState` is runtime state. `PickCart` remains static equipment
configuration. An owned bin is not reusable, even before it locks; its first
confirmed pick locks it.

## Normal tour event chain

```text
SequencingDone
  → commit_scheduling_solution
  → TourStart
  → TravelEvent            (starts an edge)
  → NodeArrival            (confirms the edge and advances position)
  → PickComplete           (confirms one concrete pick)
  → ...                    (travel/pick repeats)
  → TourEnd                (completes orders and releases the picker)
```

Events carry a tour ID and route version. An event whose version no longer
matches the active tour is stale and performs no mutation.

Travel is interruptible. While an edge is active, the tour stores its origin,
destination, distance, start time, and expected arrival. Picking is atomic;
an arrival during a pick is handled after `PickComplete`.

## Decision flow

```text
trigger event
  → SimulationEngine selects the configured problem
  → StateAdapter creates a detached planning snapshot
  → DecisionEngine.solve runs the prepared CoSy or direct solver
  → DecisionEngine.commit applies commitment policy
  → DecisionEngine creates process events
  → SimulationEngine executes it synchronously at the trigger time
  → process event calls one State operation
  → managers update the operational truth
```

Adapters project state; they do not commit or cancel work. Solutions describe
plans; they do not authorize operational changes.

`DecisionEngine.on_trigger` is the normal `solve → commit` convenience path.
A study with an actual intermediate policy may use the phases explicitly. The
Henn study solves candidate batches, applies its waiting rule, and calls
`commit` only when dispatching. An external or learned controller constructs
an `AlgorithmSolution` and enters through the same `commit` phase.

The process event never waits in the operational heap while decision latency
is zero. Therefore a second event at the same timestamp cannot observe orders
or batches selected by a first decision as still buffered.

## Buffer lifecycle and planning scope

```text
raw order buffer
  → commit_batching_solution
batch buffer
  → commit_scheduling_solution
queued tour
  → start_tour
active tour
  → complete_tour
completed order
```

The order history is an audit catalogue, not a work queue. Each commitment
validates its complete transition before mutation; a stale repeated process
event fails without partially moving work.

The adapter controls planning input. The commitment policy controls only the
returned solution. `OrderWindowAdapter.max_orders` prevents a pull policy from
solving a 100-order backlog merely to dispatch one cart. `ORSPAdapter` and
`ReORSPAdapter` use `max_batches` and `due_horizon_s` for push/rolling windows.
An unlimited window is explicit policy semantics for Lorenz Reopt or Henn, not
an accidental consequence of committing one job.

`SchedulingCommitmentPolicy` can apply a time fence, a per-picker prefix, and
a final global job limit. It never chooses solver input, changes assignments,
or mutates state.

## Depot-only reoptimization

Arriving orders enter `OrderManager`'s buffer. When the configured picker/order
conditions hold, `OrderWindowAdapter` exposes buffered work and available
pickers. Scheduling commitment creates tours and commits their orders. An
active tour is unchanged until depot return.

The order threshold is a dispatch trigger, not cart capacity or commitment
size. Cart capacity limits a batch; commitment controls how much of a returned
plan becomes operational. For a finite stream, `FlushRemainingOrders` closes
the input and lets residual work bypass only the order/batch threshold.
Resource and availability conditions still apply.

`completion_mode: drain` runs until buffers, tours, reservations, and committed
orders are resolved; exhausting the event queue with unfinished work is an
error. `completion_mode: horizon` stops at `horizon_time`, reports remaining
work, and uses `horizon_complete` rather than ordinary success.

## Active-tour intervention

An `InterventionRequest` is created only by a relevant operational trigger:

- ordinary arrivals do not trigger routing-only replanning;
- actual edge, zone, or pick-location blocking may trigger routing-only replanning;
- insertion mode requests at most one active tour with a genuinely empty bin
  per arriving order;
- arrivals during a pick set a pending request handled after that pick;
- repeated arrivals do not duplicate a request already pending for a tour.

`ActiveTourRoutingAdapter` projects only:

- the actual node or interpolated edge position;
- the remaining picks and active batch;
- cart-bin ownership and locks;
- currently buffered orders and available inventory;
- the targeted tour ID and expected route version.

`ActiveRouteReplacement` validates the target and returned plan. An unchanged
executable suffix is a no-op and does not increment the version. A changed
suffix may be accepted even when its static distance is longer, because an
occupied-edge cost snapshot may deliberately avoid blocking.

For a real replacement, `State.commit_active_plan` validates and applies
inventory reservations, buffered-order commitment, bin ownership, remaining
picks, and the new route suffix atomically. The version increments once,
making old suffix events stale.

## Current supported insertion semantics

- non-mixing `DimensionType.ORDERS` carts;
- one complete order per stable bin;
- insertion only into genuinely unassigned bins;
- owned or locked bins are never displaced;
- completed orders retain their bins until depot return;
- inventory for inserted picks must be reservable;
- completed picks and the executed route prefix are immutable.

Mixed bins, splitting an order across bins, reusing a bin during a tour,
interrupting a pick, and lock-aware general batching remain open work.

## Layout and congestion contract

`LayoutData`/`LayoutNetwork` remain the only graph abstraction. Graphs may be
undirected or directed and irregular, provided nodes have two-dimensional
`pos`, edges have numeric `weight`, picks map to nodes, and distance/predecessor
data are consistent. A synthetic mid-edge origin connects only to the legal
forward endpoint in a directed graph.

The generated dynamic-operations study uses an aisle-centerline e-commerce
model with symmetric access to picks. It is not the IJPE grocery graph, whose
left/right rails and pallet faces represent costly aisle-side changes.

Optional existing graph attributes provide coarse congestion:

- node `pick_capacity` limits simultaneous picks;
- edge `capacity` limits traversals;
- edge `zone_id` shares capacity across a constrained zone.

No metadata means unconstrained movement. These are operational
approximations, not a pedestrian model of overtaking or following distance.

## Where to look

- shared construction and solver preparation: `src/casim/setup.py`
- event loop: `src/casim/simulation_engine/simulation_engine.py`
- runtime facade: `src/casim/state/state.py`
- active-tour execution: `src/casim/domain_objects/tour_model.py`
- managers: `src/casim/state/`
- operational events: `src/casim/events/operational_events.py`
- commitment events: `src/casim/events/decision_events.py`
- planning projections: `src/casim/simulation_engine/state_adapter.py`
- decision solve/commit seam: `src/casim/decision_engine/decision_engine.py`
- externally controlled batching: `src/casim/envs/order_batching.py`
- CoSy routing task: `src/casim/pipelines/subproblems/picker_routing.py`

## Controlled decisions

`OrderBatchingEnv` controls one configured problem class and receives an
already-built `SimulationEngine` and `DecisionEngine`. It returns the existing
planning snapshot and its resolved `WarehouseOrder`s as a tuple. All other
snapshots continue through `DecisionEngine.on_trigger`, so routing and
scheduling remain ordinary CASIM decisions.

The learning code builds its `BatchingObservation` directly from those
semantic objects. Selecting order IDs creates the existing ware-ops
`BatchingSolution`, which enters through `DecisionEngine.commit`. The
route-aware decoder uses the fixed router's side-effect-free `score` method;
only the committed downstream routing decision calls `solve`.

`ExperimentTracker` maintains accrued total flow time incrementally from order
arrivals and completions. The batching reward is simply the negative increase
in that operational KPI divided by the number of orders.

The dependency direction is deliberate: `ware_ops_algos` provides domain
objects and algorithms; CASIM owns operational state, projections, commitment,
and its pipeline taxonomy; learning consumes the controlled-decision boundary;
scenarios compose concrete studies and data sources.

## Scenario input versus DataCard

`input` selects how an initial domain is constructed: loader class, source
paths or generator profile, seed, and loader arguments. A DataCard is a
handwritten declaration of static technical context used for applicability.
It does not contain loader targets, paths, or mutable runtime values.

To add a study, start from the nearest maintained scenario, keep warehouse-
specific parsing and geometry in its loader, and configure only experimental
values that really vary. The scenario index is
`scenarios/README.md`.
