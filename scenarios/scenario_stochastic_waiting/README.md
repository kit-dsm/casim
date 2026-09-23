# Analytical waiting prototype

This configured scenario exercises the single-line, exponential-arrival
waiting calculation from the explainer mail. It is **not** the wait-k study in
*Walking vs. Waiting*; that study is mapped in
`../scenario_walk_or_wait/README.md`.

The analytical special case starts with exactly q−1 known orders. For the
four-bin example the engine waits until three orders are visible before asking
`AnalyticStochasticWaiting` for a departure time. The policy rejects any
other batch size while the stream is open; closure releases a final partial
batch. The forecast in `WarehouseInfo` is separate from the
realised event stream in `simulation/reference.yaml`.

The mail's Phase 2 is **not implemented here**. The configured OBRP path
automatically routes eligible buffered orders into an active tour; it does not
compare adding an order with leaving it for the next batch using deterministic
detour and order completion time. The four-order fixed stream is an integration
example, not a stochastic experiment or evidence for the two-phase method.
Do not use its metrics as a validation or publication result for that method.
