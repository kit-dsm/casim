# Henn OOBP Scenarios

This directory contains CASIM configurations for the Henn online order batching benchmark. The scenarios demonstrate how different waiting and release policies can be represented through CASIM’s configurable decision infrastructure.

We provide two scenario variants which can be adjusted in config/henn_online_config.yaml.

## 1. Order-window waiting

`scenario_henn_order_window_waiting` implements a simple accumulated-order waiting strategy. Decisions are triggered by `OrderArrival`, `PickerIdle`, and `FlushRemainingOrders`, but the solver is only invoked once at least five open orders are available.

The solver produces candidate jobs, and `SchedulingCommitmentPolicy(n_jobs=1)` commits the first selected job. This represents the rule: wait until at least `N` open orders are available, then release one job.

## 2. Henn waiting rule

`scenario_henn_waiting` reimplements the waiting rule from Henn’s online batching algorithm. Decisions are considered whenever at least one picker and one open order are available.

The waiting decision is represented by the `HennWaiting` commitment policy. The pipeline first forms candidate batches, routes them, and sequences them. If multiple candidate batches exist, the first batch according to the sequencing rule is released. If only one candidate batch exists, `HennWaiting` may postpone its release according to the Henn release-time rule.

## CoSy repository

The baseline Henn waiting repository uses the following components:

```yaml
components:
  - casim.pipelines.problem_based_template.InstanceLoader
  - casim.pipelines.subproblems.item_assingment.GreedyIA
  - casim.pipelines.subproblems.batching.FiFo
  - casim.pipelines.subproblems.picker_routing.SShapeHenn
  - casim.pipelines.subproblems.sequencing.SPTSequencing
  - casim.pipelines.problem_based_template.ResultAggregationSequencing
```

`SShapeHenn` implements the `HennWaitingPickerRouting` template. In addition to routing each candidate batch, it computes the service-time information required by the Henn waiting rule:

- `st_j`: service time of the candidate batch
- `st_i`: service time of each order if picked alone

These values are stored on the corresponding `PickList` and are used by the `HennWaiting` commitment policy.

## Modeling choices

The Henn waiting rule requires service times of individual orders inside a candidate batch. Since these values are not available from the final batch route alone, the Henn routing component computes them explicitly.

The release decision is modeled as a commitment policy. This keeps the benchmark implementation modular:

```text
batching   -> forms candidate batches
routing    -> computes batch and single-order service times
sequencing -> orders candidate jobs
commitment -> releases one job, possibly with postponed start time
```

The order-window scenario instead represents waiting directly through `NbrOrdersCondition`, making it a simple accumulated-order release policy.

We provide a sweeped run over multiple instances. We vary the described waiting policies.

The instances are obtained from "Hessler" while the order arrival times can be found in "". 

![image](.sctipts/plots/scatter_final_makespan.pdf)