# Dynamic operations reference study

This scenario connects the multi-day IJPE/grocery setting to online
reoptimization without adding another experiment framework. It has a seeded
initial backlog, continuing arrivals, varying picker shifts, breaks, due-time
windows, an urgent burst, and one disruption of unstarted work.

## Run

```powershell
python -m scenarios.scenario_dynamic_operations.experiment_dynamic_operations engines=nightly_static
python -m scenarios.scenario_dynamic_operations.experiment_dynamic_operations engines=periodic_rolling
python -m scenarios.scenario_dynamic_operations.experiment_dynamic_operations engines=unstarted_reopt
python -m scenarios.scenario_dynamic_operations.experiment_dynamic_operations engines=routing_intervention
```

Use `simulation=smoke` for a quick run or
`simulation=six_day_stress` for the 1,400-order profile. Hydra multiruns work
normally, for example:

```powershell
python -m scenarios.scenario_dynamic_operations.experiment_dynamic_operations -m engines=nightly_static,periodic_rolling,unstarted_reopt,routing_intervention simulation=three_day
```

## Shape and data flow

The scenario follows the same visible shape as the maintained publication
scenarios:

```text
config/{data_card,engines,cosy_repo,simulation}
data/
loader.py
scenario_specific_hooks.py
experiment_dynamic_operations.py
```

Hydra instantiates `DynamicOperationsLoader` from `config/input/`. The loader
reads the selected JSON generation profile, builds the ordinary multiblock
layout, and calls the public `generate_orders` function with the configured
seed. The experiment then explicitly performs setup, reset, run, decide,
step, and final JSON output.

The planning pipeline is constant across comparisons:

```text
GreedyIA → DueDate → NearestNeighbourhood → EDDScheduler
```

`WMSRun` commits batching synchronously before a same-time `PlanningRun` can
project the batch buffer. The policies vary decision timing, planning window,
commitment, and replanning scope—not the algorithms.

| Policy | Planning input | Commitment | Replanning |
|---|---|---|---|
| `nightly_static` | daily visible backlog | all returned jobs | new work |
| `periodic_rolling` | six-hour due window | four-hour fence, one job per picker | buffered batches |
| `unstarted_reopt` | same window plus eligible queued tours | same | buffered and unstarted work |
| `routing_intervention` | same | same | unstarted work plus blocked active suffixes |

## Physical abstraction

The generated graph is an e-commerce aisle centerline: picks are reached
symmetrically from one rail. Pick locations have capacity one; connector edge
capacity is an optional coarse narrow-area restriction. This is intentionally
different from the IJPE grocery graph, which represents left/right aisle rails
and pallet-facing nodes. Both satisfy the same `LayoutNetwork` contract.

The capacity model does not claim detailed overtaking, walking-speed
interaction, or calibrated congestion. Routing-only intervention is triggered
by actual blocking, never merely by an arriving order. An unchanged suffix is
not installed and therefore does not create another route version.

## Outputs

Each run writes `result.json`, `decisions.jsonl`, Hydra metadata, and optional
Dash replay files. Decision records distinguish raw/batch buffers, projected
candidates, returned jobs, committed jobs, and deferred jobs. Algorithm and
decision wall-clock times are measurements; simulation decision latency is
zero.
