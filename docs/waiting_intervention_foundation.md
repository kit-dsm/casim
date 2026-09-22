# What this branch actually changes

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

## Scope and evidence

The branch changes **42 CASIM files** and **22 `ware_ops_algos` files** relative to main. About 2,500 of the 2,822 added `ware_ops_algos` lines are analytical waiting modules. Calling the whole change “small” was misleading. The intended architecture is focused, but the source footprint and Henn migration are substantial.

`tests/test_waiting_intervention_contract.py` checks node and mid-edge insertion, arrival during a pick, preserved work, stale events, detached input, and same-time commitment through the configured path. Configured NoWaiting, analytical waiting, and Henn smoke runs and the existing CASIM tests passed during implementation. These checks establish an integration path. They do **not** establish numerical parity for Henn, equivalent stochastic results, or parity with all playground intervention behavior; those claims need separate evidence before a technical report makes them.

Deferred: playground's compiler, general intervention request protocol, exact/general routing, multiple active pickers, atomic inventory and bin ledger, congestion, and broad configuration changes. The current supported case is one active picker inserting a visible new order into an S-shape residual route. Configuration still selects triggers, conditions, adapter, solver, and commitment policy; scenario code seeds events and reports results.
