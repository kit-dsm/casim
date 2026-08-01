# Multi-picker intervention stress scenario

This scenario is separate from `scenario_reopt`. It runs one deterministic
64-order fixture on a conventional two-block graph with eight pickers and
three non-mixing order bins per cart.

## Configuration

The configuration follows the same separation as the established scenarios:

- `input/` selects the loader and generated fixture.
- `data_card/` describes static technical context.
- `cosy_repo/` contains the ordinary depot-dispatch CoSy pipeline.
- `intervention_repo/` contains the active-tour pipeline, if any.
- `engines/` binds problems, triggers, adapters, solvers, and commitment.
- `intervention_stress_config.yaml` contains only shared experiment settings.

Every run initially uses:

`GreedyIA → FIFO → TSP routing → SPT scheduling`

The supported comparisons are:

| Run | Engine | Active-tour repository |
|---|---|---|
| Depot-only reoptimization | `reopt` | `reopt` |
| Routing-only exact TSP | `routing` | `routing_tsp` |
| Empty-bin insertion + TSP | `insertion` | `fill_tsp` |
| Empty-bin insertion + nearest neighbour | `insertion` | `fill_nn` |

Run the default insertion/nearest-neighbour case:

```powershell
python -m scenarios.scenario_intervention_stress.experiment_intervention_stress
```

Run another comparison member:

```powershell
python -m scenarios.scenario_intervention_stress.experiment_intervention_stress engines=reopt intervention_repo=reopt
python -m scenarios.scenario_intervention_stress.experiment_intervention_stress engines=routing intervention_repo=routing_tsp
python -m scenarios.scenario_intervention_stress.experiment_intervention_stress engines=insertion intervention_repo=fill_tsp
```

`engines` selects the operational policy. `intervention_repo` selects the
compatible active-tour CoSy components. They are separate intentionally;
changing a routing component must not silently change CASIM trigger or
commitment semantics.

The default order threshold is `2`. It controls when depot dispatch is
requested; it is neither cart capacity nor the number of jobs committed from
a solution. A finite stream always emits `FlushRemainingOrders`, so a
threshold of `64` or greater forms a large final wave and still drains all
orders. Such a wave can leave no arrival after dispatch and therefore no
useful arrival-driven intervention opportunity; the result records that as a
configuration warning.

Use Hydra overrides to study this directly:

```powershell
python -m scenarios.scenario_intervention_stress.experiment_intervention_stress engines.simulation_engine.problems.OBRSP.conditions.1.threshold=64
```

Before the first event, the selected adapter's projected planning features are
combined with the static data card. `DomainAlgorithmMapper` then filters the
configured algorithm components. Exact TSP and nearest neighbour declare
residual-route support; Residual FIFO declares residual active-batch support.
An incompatible intervention repository therefore fails during setup rather
than inside a CoSy task.

## Data loading

The checked-in fixture is
`data/intervention_stress.json`. Loading follows the normal scenario path:

1. `experiment_intervention_stress.py` calls `setup_scenario(cfg)`.
2. `setup_scenario` instantiates the loader declared in `config/input/`
   through Hydra.
3. `ReoptDataLoader` is reused directly because this fixture uses its JSON
   schema and multiblock construction.
4. `ReoptDataLoader.load(instance_path=...)` builds the multiblock layout,
   orders, inventory, separate picker/cart objects, and the initial domain.
5. `add_orders_hook` adds those orders to CASIM as timed `OrderArrival`
   events.

The stress fixture reuses `ReoptDataLoader` because it uses the same explicit
JSON schema and multiblock domain construction. No implicit domain builder or
scenario-local replacement is involved.

## Intervention behavior

For insertion runs, one arrival requests at most one active tour that still
has a genuinely empty bin. Full carts do not create intervention work.
Residual FIFO may fill available bins and the configured router computes the
remaining suffix.

CASIM installs the suffix when an order was inserted or the executable suffix
changed. An unchanged suffix is recorded as `replaced=false`; the route
version stays unchanged and existing operational events remain valid.

Each run writes `result.json`, `decisions.jsonl`, Hydra metadata, KPI files
under `work/kpis`, and, when enabled, a compressed replay under `viz`.
Algorithm runtime is measured wall-clock time and does not advance simulation
time. The replay includes concrete completed/remaining pick identities and
the two intervention-pending flags, so a skipped pick or duplicate request can
be diagnosed without reconstructing it from route geometry.

Launch a recorded replay with:

```powershell
python -m casim.viz.app <run-directory>/viz
```

For the general ownership and event flow, see
[`docs/execution_model.md`](../../docs/execution_model.md).
