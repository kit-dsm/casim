# What this branch actually changes

For the published *Walking vs. Waiting* strategy mapping and configured
wait-k examples, see `scenarios/scenario_walk_or_wait/README.md`. The
paper's wait-0 now starts an empty all-aisle tour through the same configured
waiting and CASIM commit path; an explicit delayed-arrival example makes its
active-tour admission visible. The
`scenario_stochastic_waiting` path below is an exploratory analytical
integration. Its active insertion is automatic and is **not** the explainer
mail's Phase 2 completion-time admission decision; it is not evidence of
end-to-end parity for that method.

This records `codex/waiting-intervention-foundation` against `main`. The branch is **not a cherry-pick of `codex/playground`**. It reimplements selected active-tour behavior in main's existing tour, event, adapter, and configured CoSy flow. Waiting and forecast support are separate additions. The WSC and IJPE papers describe the configured decision pipeline and replanning of **unstarted** tours; active-tour insertion and causal stochastic waiting are proposed extensions.

## What came from playground

| Playground behavior used as the reference | Implementation here | Difference from playground |
| --- | --- | --- |
| A started tour has a mutable execution record with remaining picks and a route version. | `TourPlanningState` now holds remaining/completed picks, edge timing and position, cart bins, and `route_version`; `TourManager` initializes it. | Main's existing tour state was extended. No new tour model or compiler. |
| A decision sees a residual route from the picker's actual location, including mid-edge. | `ActiveTourAdapter` projects visible orders and uncompleted picks. `RoutingOrigin` carries position, forward edge endpoint, and remaining edge distance into the existing S-shape router. | One picker, S-shape routing, and forward edge continuation only. |
| Completed travel and picks survive replacement; old events cannot execute a superseded route. | `TravelEvent` records an edge; `NodeArrival` completes travel; `PickComplete` consumes a residual pick. Versioned tour events ignore stale versions. | A smaller change within main's operational events, without playground's intervention request/response machinery. |
| CASIM validates and installs the new unserved route. | `State.commit_active_route` checks tour/version/picker, actual origin, residual and completed picks, visible new orders, bins, and depot return; it installs the suffix and increments the version. | No playground atomic rollback, inventory reservation, unchanged-route detection, bin locking, general routing-only replanning, or multi-picker support. |

These are **behavioral ports, not verbatim source ports**. Playground's `commit_active_plan`, intervention request and replacement types, tests, and compiler were not copied. Playground is a reference for the listed cases, not a general parity claim.

## What was introduced for this integration

| Addition | Concrete change |
| --- | --- |
| Common waiting contract | `ware_ops_algos` adds `WaitingInput` and `WaitingSolution`. NoWaiting, HennWaiting, and AnalyticStochasticWaiting use it through algorithm cards. They accept domain data and do not import CASIM. |
| Configured waiting stage | CASIM adds a CoSy waiting component after candidate routing and scheduling. The decision engine handles `WaitingSolution` and CASIM commits by solution type. Henn's former special router and commitment policy leave the active path. |
| Decision/commit boundary | `HennWaitingAdapter` no longer retains a snapshot or clears orders; `ReORSPAdapter` no longer cancels tours during projection. Adapters project detached data. `State.commit_solution` and `State.commit_active_route` change live state. |
| Waiting/stream events | Versioned `WaitExpired` events trigger reconsideration; `OrderStreamClosed` reveals closure of a finite stream when processed. Decisions do not read future events. |
| Forecast data | `WarehouseInfo` and its card carry assumed arrival process, mean interarrival time, order size, and pick-location distribution. A loader supplies the forecast; `OrderArrival` hooks provide realized orders. |
| Configured scenario | `scenario_stochastic_waiting` has a Hydra root, engine config, CoSy repositories, data card, loader, hooks, and an experiment with the existing build → reset → run → decide → step → report flow. Henn's configuration was migrated to this waiting stage. |

The **stochastic mathematics** comes from a different source: `2_Stochastic_Waiting/Algorithms/analytic_progress` on the `ware_ops_algos` `feature/waiting-uncertainty` branch. `core.py`, `detour.py`, `continuous_calculation.py`, and `models.py` were moved and adapted for package imports; `interval_builder.py` was reworked; `optimal_waiting.py` is new integration code using that analytical cost model. `policies.py` maps **currently visible** orders and the forecast to that calculation. This is different from the original retrospective calculation that knows a realized future insert order. The numerical waiting results were not rerun.

Two more `ware_ops_algos` additions, `order_splitting` and `data_loaders`, restore imports expected by CASIM main against this main-based worktree. `order_splitting.py` matches the existing feature-branch file. These are compatibility additions, not playground behavior or waiting theory; review them before merge.

## Boundary between release and active insertion

In the configured single-picker scenario, an `OrderArrival` can trigger both
problem classes. `OBRSPW` is eligible while an order is buffered and the picker
is free; it can wait or release a scheduled job. Releasing creates a tour and
removes its orders from the buffer. `OBRP` becomes eligible only after that tour
has actually started and a new order is buffered. Its configured batching
algorithm checks cart capacity and remaining-route membership; rejection leaves
the order buffered. The interval between release and `TourStart` has
no tour-revision policy; later orders stay buffered during it.

The simulation engine now rejects an event that makes both problems eligible,
instead of silently using their order in the YAML file. The expected detour in
the analytical waiting model does not select or constrain the actual FIFO and
S-shape insertion route. Their behavioral and numerical alignment is unproven.

## Scope and evidence

The earlier foundation work already had a substantial source footprint, chiefly
from the analytical waiting modules and Henn migration. The configured paper
scenario here is an additional, narrower strategy mapping; it does not make
that whole branch a small change.

`tests/test_waiting_intervention_contract.py` checks node and mid-edge insertion, arrival during a pick, preserved work, stale events, detached input, and same-time commitment through the configured path. Configured NoWaiting, analytical waiting, and Henn smoke runs and the existing CASIM tests passed during implementation. These checks establish an integration path. They do **not** establish numerical parity for Henn, equivalent stochastic results, or parity with all playground intervention behavior; those claims need separate evidence before a technical report makes them.

Deferred: playground's compiler, general intervention request protocol, exact/general routing, multiple active pickers, atomic inventory and bin ledger, congestion, and broad configuration changes. The current supported case is one active picker inserting a visible new order into an S-shape residual route. Configuration still selects triggers, conditions, adapter, solver, and commitment policy; scenario code seeds events and reports results.
