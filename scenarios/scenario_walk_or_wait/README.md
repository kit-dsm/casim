# Walking vs. Waiting in CASIM

This scenario implements the **wait-k experiments** described in Section 3 of
*Walking vs. Waiting: Performance Impact of Waiting Strategies in Manual
Warehouses*. It is a configured CASIM experiment, not a reproduction of the
paper's numerical tables. The six explicit orders in `simulation/explicit_example.yaml`
are a readable integration example. They are not an OFAT or LHS instance.

## What maps to what

| Paper rule | Original `project_4D4L/2_Stochastic_Waiting` | Here |
| --- | --- | --- |
| FCFS order batching, four order bins in the baseline | `Algorithms/batching.py::FIFOBatching` | `ware_ops_algos.FifoBatching`, selected by the CoSy waiting repo |
| FCFS release and next available picker | `Algorithms/dispatching.py::GreedyDispatcher` and the batch-ready events | `FIFOScheduling` through `FIFOScheduler`; with one picker, assignment is unambiguous |
| Wait until at least k orders, k = 1, 2, 3, 4 | `Algorithms/waiting.py::WaitingOrderCountPolicy` | `OrderCountWaiting` through `WaitForOne` ... `WaitForFour`; `policy=wait_1` ... `wait_4` |
| Wait-0: leave immediately with an empty route, then take reachable arrivals | `StartImmediatelyPolicy`, `AllNodes`, and the dummy batch | `StartImmediatelyWaiting` creates an empty `ScheduledJob` with an all-aisle walking route; CASIM starts a normal tour with no order or pick. `policy=wait_0 simulation=explicit_wait0` shows an arrival while it walks. |
| S-Shape, Return, Largest Gap, Nearest Neighbour | Four classes in `Algorithms/routing.py` | Existing `ware_ops_algos` routing algorithms, selected with `routing=...` |
| Optional intervention on arrival: free bin and every new pick on remaining route; reroute after admission | `FIFOBatching.make_decision` and `Simulation/core.py::__create_event_on_routing_decision` | `RemainingRouteAdmission` in the configured OBRP CoSy repo; CASIM validates and commits only an accepted residual route |
| Mean order completion time, distance per item, tardiness | The paper's simulation statistics | `experiment.py::report`, written to `paper_metrics.json` after the event stream ends |

`RemainingRouteAdmission` is the intervention admission check, not the regular
FIFO batcher. The regular `FiFo` component batches buffered orders for a free
picker. Admission checks a newly arrived order against a *started* tour's
unserved path and cart bins. It returns accepted order IDs through a dedicated
admission interface. The configured CoSy node converts those IDs into the
batch required by the existing router; it makes no admission choice. CASIM
then commits or leaves the order buffered.

The intervention configuration gives `WaitingOpportunity` to OBRSPW and
`ActiveTourOpportunity` to OBRP. An arrival produces one of these based on
whether a tour has started. The active-tour condition checks for a visible
candidate and waits for an in-progress pick to finish. The adapter only
projects the residual tour; admission remains in the configured algorithm.

The `paper_metrics.json` file also reports `empty_tours_started` and
`orders_admitted_after_empty_start`, so the wait-0 example can be checked
without inferring it from completion-time averages.

The original batcher changes the transported batch while deciding. Here CoSy
decides from a detached state projection. CASIM records the attempted arrival,
keeps rejected orders in the buffer, and commits an accepted replacement route.
Completed travel and picks remain operational state; superseded route events
are ignored by route version. This is an architectural translation of the
paper's rule, not a source-code copy.

## Configured runs

From the CASIM checkout with its `uv` environment and the sibling
`ware_ops_algos` project installed:

```powershell
uv run python -m scenarios.scenario_walk_or_wait.experiment
uv run python -m scenarios.scenario_walk_or_wait.experiment policy=wait_0 simulation=explicit_wait0
uv run python -m scenarios.scenario_walk_or_wait.experiment policy=wait_3
uv run python -m scenarios.scenario_walk_or_wait.experiment engines=paper_no_intervention policy=wait_4 routing=return
uv run python -m scenarios.scenario_walk_or_wait.experiment engines=paper_no_intervention policy=wait_2 routing=nearest_neighbour
```

To see the initially empty tour, run
`uv run python -m scenarios.scenario_walk_or_wait.experiment policy=wait_0 simulation=explicit_wait0 viz.launch=true`
and open `http://127.0.0.1:8050/` after the simulation finishes. The
`TourStart` frame at t=0 shows tour 1 with zero orders; the default `wait_1`
run has no empty start. With `policy=wait_0` on the default six-order input,
the tour also starts empty, but order 0 arrives at the same t=0 and is
admitted immediately; aggregate metrics alone do not reveal the empty start.

The Hydra root selects the data card, engine config, CoSy repos, waiting
component, routing component, and explicit orders. The hooks only seed arrivals
and picker availability. The experiment follows build → reset → run → decide →
step → report. A newly arrived order can be rejected by the configured
remaining-route admission algorithm; rejection is a real decision and leaves
the order for a later batch. There is no insertion policy hidden in the
experiment script.

## Supported boundary and explicit failures

- **Wait-0 is supported for the configured one-block, S-Shape, one-picker
  case.** The original simulator represents its empty walk with a zero-quantity
  dummy order. Here the route traverses every aisle without inventing an order
  or a pick. It is generated by the configured waiting component, committed as
  a normal empty tour, and can receive orders through active-tour admission.
  `explicit_wait0.yaml` starts the shift at time 0 and the first order at time
  8, making the empty walk observable. This preserves the operational rule;
  exact path and numerical parity with the original dummy-order implementation
  have not been established. Wait-0 with `paper_no_intervention` raises an
  explicit error because the empty tour cannot take arriving orders there.
- **Intervention is supported with S-Shape and one picker only.** The current
  active-route router has an observed-position entry point for S-Shape. The
  other routing components can start tours but cannot yet reroute their
  residual tour from an arbitrary position. Selecting one with
  `engines=paper_intervention` raises an explanatory error. Multi-picker
  assignment also raises; a one-picker FIFO scheduler is not evidence of
  parity for the paper's multi-picker cases.
- **The demonstration uses explicit orders and a unit grid.** Direct import
  of the published OFAT/LHS instance set requires an explicit converter for
  its layout geometry and storage mapping. `simulation.source` other than
  `explicit` raises. Nothing in this example is silently sampled or inferred
  from a published instance's metadata.
- **Arrivals during a pick are considered after that pick completes.** CASIM
  keeps the pick in progress and projects a residual route at `PickComplete`.
  The original simulator updates the batch at arrival. This timing difference
  prevents numerical parity claims for intervention runs with positive pick
  time. The same-time order of arrival and operational events also needs a
  dedicated parity study; the intervention example rejects equal arrival
  timestamps rather than assigning an arbitrary order.
- **The paper's full experiment grid is not reproduced.** The paper varies
  arrivals, order-size distributions, layout, storage policy, due dates,
  workload, and picker count. The configured example fixes these inputs.
  Its three reported metrics have the paper's definitions, but its values
  must not be compared with the published numerical results.

The explainer mail's **optimal stochastic wait** is a later, separate method.
It requires exactly q−1 known orders and a Phase 2 completion-time admission
decision after the missing order arrives. The paper's wait-k experiment has
neither of those rules. The existing `scenario_stochastic_waiting` is still an
exploratory integration path; its automatic active routing is not that Phase 2
decision and must not be presented as a faithful implementation of the mail.
