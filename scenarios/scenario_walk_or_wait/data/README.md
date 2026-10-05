# Published walk-or-wait instances

`radar/selected/Instances` preserves the directory structure of the local
[RADAR instance bundle](https://radar.kit.edu/radar/en/dataset/mwsv59v8sk9sqaan).
The 100 `Orders/OFAT/Standard_case/instance_*.json` files contain 100,597
orders in total, the same count as `Results/OFAT/Standard_Case/Results_Standard_Case.csv`.
Each order file names its NetworkX layout pickle and article-to-pick-node JSON
mapping relative to itself. The loader reads those files directly; it does not
regenerate orders or replace the benchmark geometry with the example grid.

From the CASIM checkout, with the sibling `ware_ops_algos` checkout installed
in the existing `uv` environment:

```powershell
uv run python -m scenarios.scenario_walk_or_wait.experiment simulation=radar_standard_case policy=wait_1
uv run python -m scenarios.scenario_walk_or_wait.experiment simulation=radar_standard_case policy=wait_2 'simulation.instance_path=scenarios/scenario_walk_or_wait/data/radar/selected/Instances/Orders/OFAT/Standard_case/instance_25.json'
```

To run the 100-instance standard case through the same configured experiment:

```powershell
1..100 | ForEach-Object {
    $i = $_
    uv run python -m scenarios.scenario_walk_or_wait.experiment simulation=radar_standard_case policy=wait_1 "simulation.instance_path=scenarios/scenario_walk_or_wait/data/radar/selected/Instances/Orders/OFAT/Standard_case/instance_$i.json" "experiment.output_dir=tmp/radar/instance_$i" "hydra.run.dir=tmp/radar/instance_$i"
    if ($LASTEXITCODE -ne 0) { throw "Benchmark instance $i failed" }
}
```

The `radar_standard_case` configuration selects the original 8-by-16 graph,
its depot and weighted edges, four order bins, one picker, unit walking speed,
one time unit per picked item, and the published arrival and due dates.
`routing=s_shape` selects `WalkOrWaitSShapeRouting`, which follows the
`2_Stochastic_Waiting` simulator's final-aisle and cross-aisle route construction.
`engines=paper_intervention` assigns waiting and
active-tour opportunities to their configured CoSy problems. The order and
picker hooks seed the event stream. The benchmark pickle is deserialized as
the original simulator does, so this input should come from a trusted source.

The CSV reports means over 100 instances. An earlier exploratory CASIM run
completed all 100 standard-case files with temporary routing switches copied
from the stochastic simulator fork. Those switches have been removed, so the
following CASIM numbers do not describe the current configuration. Means use
one value per instance, as in the CSV; batch and order counts are sums.
MLPI divides travelled distance by the number of order lines, matching the
original simulator's `nr_items` statistic.

| WOC 1 + S-Shape | MOCT | MLPI | TARD | Batches | Orders |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published CSV | 125.62 | 20.58 | 0.41 | 41,288 | 100,597 |
| Earlier fork-matched CASIM run | 135.22 | 20.82 | 0.434 | 40,083 | 100,597 |

The published 95% interval for MOCT is 122.62–128.62; CASIM's sample interval
is 132.14–138.29. This is a substantive numerical mismatch, not an
instance-selection error. MLPI differs by 0.24, while TARD is close. CASIM
writes each run's `paper_metrics.json` and KPI tracker under the Hydra output
directory. A single instance result must be compared with the original
simulator on the *same instance*, not with the 100-instance mean.
The run behind the table is in `tmp/radar_paper_full/wait_1/`; its
`summary.json` and per-instance metrics are local, uncommitted outputs.

Exact numerical parity with the original intervention runs is currently
limited by its position update. In `2_Stochastic_Waiting/Simulation/core.py`,
the forward `tour` is reversed before being stored as `travel_tour`.
`Utils/utils.py::get_current_or_next_node` walks this reversed list, and
`Algorithms/batching.py::FIFOBatching.make_decision` tests admission against
its suffix. On standard instance 1, order 10 arrives at t=295.767 and is
added to batch 5, although
its pick at (2, 13) was passed at about t=264.515 on the tour that started
at t=248.515.
CASIM projects the actual forward position and leaves order 10 for a later
tour. This explains a concrete difference in batch membership without
weakening CASIM's operational state model to imitate a backward traversal.
For this instance, the original simulator reports 408 batches, MOCT 136.71 and
MLPI 20.18; CASIM reports 409 tours, MOCT 146.07 and MLPI 20.40.
With wait-2 on the same instance, MOCT is 148.69 in the original simulator
and 165.48 in CASIM. The configured wait-0 and wait-2 CASIM runs both complete
all 1,021 orders; wait-0 starts empty tours as expected.
After admission, the original routing action calculates a new whole-batch
route from the graph's depot and replaces the completion event with one at
the tour's **original start time plus the new full route duration**. It does
not physically send the picker back to the depot. CASIM retains completed
travel and picks, then routes the unserved work from the observed position.
The two completion-time calculations agree if the new whole-batch route has
exactly the path and pick time already executed as its prefix. The source does
not enforce that prefix condition, but we have not shown how often it fails
or whether it contributes to the aggregate gap. The already-passed admission
above is the concrete discrepancy established for instance 1.

The raw and selected benchmark files are local research inputs ignored by
Git. Their paths stay relative to this scenario so that the order metadata
continues to resolve without a converter.
