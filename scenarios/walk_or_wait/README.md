# Walking vs. Waiting in CASIM

This scenario implements the **wait-k experiments** described in Section 3 of
*Walking vs. Waiting: Performance Impact of Waiting Strategies in Manual
Warehouses*. The default source, written out in `scenario.yaml`, is a small
integration example; `variants=radar_standard_case` loads the published
standard-case orders, storage assignment and layout directly (source `radar`,
`casim.sources.radar`). The latter can be compared with the
paper's results, subject to the intervention difference documented below.

## What maps to what

| Paper rule | Original `project_4D4L/2_Stochastic_Waiting` | Here |
| --- | --- | --- |
| FCFS order batching, four order bins in the baseline | `Algorithms/batching.py::FIFOBatching` | `ware_ops_algos.FifoBatching`, selected by the CoSy waiting repo |
| FCFS release and next available picker | `Algorithms/dispatching.py::GreedyDispatcher` and the batch-ready events | `FIFOScheduling` through `FIFOScheduler`; with one picker, assignment is unambiguous |
| Wait until at least k orders, k = 1, 2, 3, 4 | `Algorithms/waiting.py::WaitingOrderCountPolicy` | One `OrderCountWaiting` card with four value configurations; `cosy_repo=wait_1` ... `wait_4` selects generated CoSy components for those values. |
| Wait-0: leave immediately with an empty route, then take reachable arrivals | `StartImmediatelyPolicy`, `AllNodes`, and the dummy batch | The `AislePatrolPositioningNode` adds an empty scheduled job after scheduling; `no_waiting` and `release_all` start it. `cosy_repo=paper_patrol variants=explicit_wait0` shows an arrival while it walks. |
| Center relocation with a carried empty cart | Lorenz et al. describe idle movement to the center but leave cart handoff unspecified | `CenterRelocationPositioningNode` moves to an accessible center node and holds. `CenterPositionAdmissionNode` can route an order from the observed position. An unused cart returns at stream closure. This is CASIM's cart choice, not a full Reopt* reproduction. |
| S-Shape, Return, Largest Gap, Nearest Neighbour | Four classes in `Algorithms/routing.py` | Configured `ware_ops_algos` routing algorithms; the `wait_<k>` repositories use `WalkOrWaitSShape`, the original waiting simulator's S-Shape traversal; `wait_4_return`, `wait_2_nearest_neighbour` and `wait_1_largest_gap` use the others |
| Optional intervention on arrival: free bin and every new pick on remaining route; reroute after admission | `FIFOBatching.make_decision` and `Simulation/core.py::__create_event_on_routing_decision` | `RemainingRouteAdmission` in the configured OBRP CoSy repo; CASIM validates and commits only an accepted residual route |
| Mean order completion time, distance per item, tardiness | The paper's simulation statistics | `metrics.py`, functions of the run's `orders.csv`, `tours.csv` and `kpis.json` |

`RemainingRouteAdmission` is the intervention admission check, not the regular
FIFO batcher. The regular `FiFo` component batches buffered orders for a free
picker. Admission checks a newly arrived order against a *started* tour's
unserved path and cart bins. Its CoSy stage returns accepted order IDs as an
`AdmissionSolution`. A separate `AdmittedTourBatch` component assembles the
batch required by the existing router without making another decision. CASIM
then commits or leaves the order buffered.

The intervention configuration gives `ActiveTourOpportunity` to the insertion
decision and `OrderArrival` to planning. Both can run at one timestamp in
precedence order. The active-tour condition checks for a visible candidate
and waits for an in-progress pick to finish. The adapter projects the residual
tour; admission remains in the configured algorithm.
For waiting, the engine checks picker readiness; the configured wait-k algorithm
alone decides whether the visible batch should depart.

`metrics.py` also gives `empty_tours_started` and
`orders_admitted_after_empty_start`, so the wait-0 example can be checked
without inferring it from completion-time averages.

The original batcher changes the transported batch while deciding. Here CoSy
decides from a detached state projection. CASIM records the attempted arrival,
keeps rejected orders in the buffer, and commits an accepted replacement route.
Completed travel and picks remain operational state; superseded route events
are ignored by route version. This is an architectural translation of the
paper's rule, not a source-code copy.

## Configured runs

```bash
python -m casim run scenarios/walk_or_wait
python -m casim run scenarios/walk_or_wait variants=radar_standard_case
python -m casim run scenarios/walk_or_wait cosy_repo=paper_patrol variants=explicit_wait0
python -m casim run scenarios/walk_or_wait cosy_repo=paper_center decisions=paper_intervention_center variants=explicit_wait0
python -m casim run scenarios/walk_or_wait cosy_repo=wait_3
python -m casim run scenarios/walk_or_wait decisions=paper_no_intervention cosy_repo=wait_4_return
python -m casim run scenarios/walk_or_wait decisions=paper_no_intervention cosy_repo=wait_2_nearest_neighbour
```

`variants=radar_standard_case source.instance=radar/selected/Instances/Orders/OFAT/Standard_case/instance_25.json`
selects another standard-case instance; `python scripts/fetch_data.py radar`
restores the data. The patrol run starts tour 1 at t=0 with zero orders; the
default `wait_1` run has no empty start. With `cosy_repo=wait_0` on the default
six-order input, the tour also starts empty, but order 0 arrives at the same
t=0 and is admitted immediately; aggregate metrics alone do not reveal the
empty start.

`scenario.yaml` selects the data card, the decisions and the CoSy repository;
`decisions/paper_intervention.yaml` holds the admission repository of its
second decision. The source seeds the arrivals and the picker's availability;
`casim run` owns build → reset → run → decide → step. A newly arrived order
can be rejected by the configured remaining-route admission algorithm;
rejection is a real decision and leaves the order for a later batch. There is
no insertion policy hidden in the runner.

The benchmark loader uses the published standard-case input. Its configured
S-Shape is a separate, named `ware_ops_algos` algorithm following the traversal
in `2_Stochastic_Waiting/Algorithms/routing.py`. The earlier numerical comparison
used temporary switches for the same traversal; the named algorithm still needs
a completed comparison before those numbers can be attributed to it. See
[`data/README.md`](data/README.md) for the input and intervention parity limit.

## Supported boundary and explicit failures

- **Wait-0 is supported for the configured one-block, S-Shape, one-picker
  case.** The original simulator represents its empty walk with a zero-quantity
  dummy order. Here the route traverses every aisle without inventing an order
  or a pick. It is generated by the configured positioning component, committed as
  a normal empty tour, and can receive orders through active-tour admission.
  `explicit_wait0.yaml` starts the shift at time 0 and the first order at time
  8, making the empty walk observable. This preserves the operational rule;
  exact path and numerical parity with the original dummy-order implementation
  have not been established. Use `paper_intervention` to admit work into it.
- **Intervention is supported with S-Shape and one picker only.** The current
  active-route router has an observed-position entry point for S-Shape. The
  other routing components can start tours but cannot yet reroute their
  residual tour from an arbitrary position. Selecting one with
  `decisions=paper_intervention` raises an explanatory error. Multi-picker
  assignment is not configured; a one-picker FIFO scheduler is not evidence
  of parity for the paper's multi-picker cases.
- **The benchmark configuration reads published input directly.** It loads
  each order JSON, its relative article mapping and its NetworkX graph pickle.
  The layout conversion assumes the selected 8-by-16 standard-case graph.
  Missing files raise ordinary file errors; other graph families are not
  supported by this conversion. See
  [`data/README.md`](data/README.md) for the files and observed parity limit.
- **Arrivals during a pick are considered after that pick completes.** CASIM
  keeps the pick in progress and projects a residual route at `PickComplete`.
  The original simulator updates the batch at arrival. It also traverses a
  reversed tour when estimating the active picker's position, which can admit
  an order whose pick location was already passed. CASIM retains the actual
  forward position and reroutes only the unserved suffix. The original
  routing action recomputes the whole batch from the depot after admission
  and times its completion from the original tour start. That calculation
  can agree when the newly planned route preserves the executed prefix; its
  contribution to the measured gap has not been established. The confirmed
  already-passed admission prevents exact parity claims. The intervention
  example rejects equal arrival
  timestamps rather than assigning an arbitrary order.
- **The paper's full experiment grid is not reproduced.** The paper varies
  arrivals, order-size distributions, layout, storage policy, due dates,
  workload, and picker count. The configured example fixes these inputs.
  The standard-case configuration uses the published orders and layout, but
  other OFAT and LHS factor levels still need their own validated configs.

The explainer mail's **optimal stochastic wait** is a later, separate method.
It requires exactly q−1 known orders and a Phase 2 completion-time admission
decision after the missing order arrives. The paper's wait-k experiment has
neither of those rules. The existing `scenario_stochastic_waiting` is still an
exploratory integration path; its automatic active routing is not that Phase 2
decision and must not be presented as a faithful implementation of the mail.
