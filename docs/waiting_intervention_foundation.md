# Waiting and active-tour intervention: implementation boundary

## Relation to the papers

The WSC CASIM paper (Bischoff et al., 2026, Sections 3.1–3.3) establishes the event-driven simulation state, configured triggers and conditions, state adapters, CoSy-Luigi pipelines, and commitment policies. The IJPE preprint (Kunze et al., *Picking a Strategy*, Sections 3.1–3.3) applies those mechanisms to OBP, ORSP, and RORSP. Its replanning returns **unstarted** tours to the decision pool; started and completed tours remain fixed. Neither paper reports the active-tour insertion or the stochastic waiting results implemented here.

This branch extends those mechanisms in two narrow ways. Waiting becomes the last CoSy decision stage after routing and scheduling. A started tour may replace its **unserved** route when a new order becomes visible. The WSC and IJPE descriptions place some state changes in adapter cleanup; this implementation moves them into `State.commit_solution` and `State.commit_active_route`. That is an implementation refinement proposed here, not a claim that the papers used this exact commit boundary.

## Decision and operational ownership

The configured `SimulationEngine` processes operational events and owns mutable `State`. An adapter only projects a detached `SimWarehouseDomain`; it never cancels a tour, clears a buffer, or stores a previous snapshot. The configured `DecisionEngine` selects a CoSy pipeline and returns a solution. CASIM validates and commits that solution before scheduling its process or travel events. `ware_ops_algos` receives warehouse domain data and returns a solution without importing CASIM or reading its event heap. CASIM dispatches by solution type, not by algorithm name.

For ordinary batching and scheduling, `State.commit_solution` performs the former adapter cleanup. For RORSP, `ReORSPAdapter` projects the unstarted tour batches and commitment cancels those tours. For an active tour, CASIM keeps the actual position on the current edge, completed and remaining picks, cart-bin ownership, and a route version. `ActiveTourAdapter` shows the router only visible new orders and the residual picks. CASIM accepts a replacement only if its version and origin still match, its route finishes the current edge and ends at the depot, all unserved picks remain, completed picks are absent, new orders are visible, and bins are available. Superseded tour events do nothing. A pick in progress completes before insertion is considered.

The currently configured active insertion uses the existing S-shape router on a conventional single-block layout with one picker. It is a foundation for interventionist routing, not a general multi-picker reoptimizer or a new routing heuristic.

## Waiting and uncertainty

`WaitingInput` contains scheduled candidates, current time, a picker and layout, the domain forecast, an explicit stream-closed flag, and whether a valid waiting deadline has arrived. `WaitingSolution` either releases selected jobs or gives a reconsideration time. NoWaiting, Henn, and AnalyticStochasticWaiting use that same contract. Their algorithm cards specify their requirements, and CoSy selects the components from the scenario repository. CASIM versions waiting deadlines, so an older timer cannot trigger a newer decision. The Henn-specific routing and commitment classes are no longer the active path; Henn's single-order service times are computed with the router selected upstream in the same pipeline.

The **forecast** available to a decision maker lives in `WarehouseInfo` and the data card: an exponential arrival process, mean interarrival time, one order line per order, and uniform independent pick locations for the analytical policy. The loader supplies these assumptions. The **realized** arrivals remain `OrderArrival` events seeded by hooks. The finite benchmark also seeds `OrderStreamClosed`; `done` changes only when that operational event is processed, never because the simulator inspected its future event heap. A decision snapshot contains only orders whose arrival events have already occurred.

The causal analytical policy reuses the existing single-line expected-cost calculation to decide when to reconsider a visible candidate batch. The original `2_Stochastic_Waiting` scripts and their retrospective insertion calculation can inspect a realized insert order. They are a different method and are not called by this online policy. This branch does not claim numerical equivalence between them or new stochastic performance results; their existing numerical evidence is not rerun here.

## Scenario and evidence

`scenario_stochastic_waiting` follows the existing scenario layout: a Hydra root selects the data card, engine configuration, CoSy repositories, and simulation inputs; its loader builds the domain; hooks seed order, picker, and stream-closure events; the experiment reads as **build → reset → run → decide → step → report**. Trigger, condition, adapter, solver, and commitment selection remain in engine configuration. Henn uses the same waiting-stage path and its existing experiment structure.

Parity with the playground is scoped to the operational invariants needed here: node and mid-edge insertion, arrivals during a pick, completed work preservation, stale events, detached snapshots, and same-time route commitment. The focused tests check those invariants through the configured scenario. They do not claim parity with the playground's multi-picker or exact-routing features. Configured smoke runs completed for analytic waiting, NoWaiting, and Henn; the Henn run used the repository's 40-order fixture. The existing CASIM test directory also passes. These establish integration behavior, not reproduction of the published experimental tables. Existing CASIM scenario semantics require their own instance-specific comparison before any publication claim.

## Why these additions are small

| Addition | Concrete need and current callers | Smaller option and why it fails |
| --- | --- | --- |
| `WaitingInput` / `WaitingSolution` | The three configured waiting policies need one algorithm-level release contract. | A CASIM-specific callback would make `ware_ops_algos` depend on the simulator and leave no common policy interface. |
| `AbstractWaiting` and three CoSy components | The three current policies must be selectable by existing cards and repositories. | Calling a policy in the experiment would bypass configured pipeline selection. |
| `RoutingOrigin` on `Route` | The active S-shape route needs the actual mid-edge position and remaining edge distance. | A depot-only route or node snap would invent travel and lose the picker's position. |
| Tour route version and residual picks | Current `TravelEvent`, `NodeArrival`, and `PickComplete` must distinguish live work from superseded plans. | Removing events from the heap cannot reliably undo already scheduled same-time events or account for completed picks. |

The compiler, generic solver or scenario frameworks, inventory ledger, congestion, broad configuration redesign, and multi-picker intervention remain separate proposals. Easier configuration and richer inventory state may be valuable for an industry transfer project, but they have no current caller in this integration.
