# Analytical waiting prototype

This configured scenario exercises the single-line, exponential-arrival
waiting calculation from the explainer mail. It is **not** the wait-k study in
*Walking vs. Waiting*; that study is mapped in
`../scenario_walk_or_wait/README.md`.

The analytical special case starts with exactly q−1 known orders. For the
four-bin example the engine waits until three orders are visible before asking
`AnalyticStochasticWaiting` for a departure time. The policy rejects any
other batch size while the stream is open; closure releases a final partial
batch. The forecast in `PlannerInformation` is separate from the
realised event stream in `simulation/reference.yaml`.

The loader checks the configured three-order waiting gate, the exclusive
decision opportunities, and the remaining-route admission component. Luigi
logs task errors at `ERROR` level, so invalid policy input is visible.

The mail's Phase 2 is **not implemented here**. `WaitingOpportunity` runs the
Phase 1 waiting policy when a picker can take a new batch.
`ActiveTourOpportunity` runs a separate rule from the original simulator:
admit an arrived order only if a cart bin is free and all its picks still lie
on the active route. The domain algorithm returns accepted order IDs; the configured CoSy assembly component supplies the accepted tour batch to routing. It does
**not** compare deterministic detour and order
completion time with the next batch. The four-order fixed stream is an
integration example, not evidence for the two-phase method.

An order arrival emits one opportunity based on operational state: active tour
or no active tour. Conditions check whether candidates and a usable picker are
present; adapters only project the state. During a pick, active admission is
deferred until `PickComplete`. CASIM alone commits an accepted residual route.
The original offline Phase 2 calculation assumes a fixed three-order window,
knows the fourth order when evaluating insertion, and searches future route
positions. An online version needs a current route position and completed work,
the now visible order, deterministic detour from that state, and a defined
next-batch completion-time comparison. Those semantics are not supplied by the
current retrospective calculator.
Do not use its metrics as a validation or publication result for that method.

## Dash replay

From the CASIM project root, run:

```powershell
.venv\Scripts\python.exe -m scenarios.scenario_stochastic_waiting.experiment_stochastic_waiting viz.launch=true
```

After the simulation finishes, open `http://127.0.0.1:8050/`. Use the slider
to inspect the event stream. Decision frames follow the triggering event:
`OBRSPW wait` at t=2, `OBRSPW release` at t≈11.085, and `OBRP admit` at t=12.
The replay shows the picker, order status, and committed tour membership at
each frame. It does not plot the analytical waiting calculation or evaluate
the Phase 2 insertion choice.
