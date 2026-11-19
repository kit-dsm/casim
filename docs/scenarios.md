# Scenarios

A scenario is one warehouse case that CASIM runs: a literature benchmark, a study on company
data, or a small case derived by hand. This document describes how a scenario is structured,
what every configuration field means, and how to write a new one. The model behind the fields is
in `docs/concepts.md`.

## Structure

A scenario lives in `scenarios/<name>/` and has exactly this structure:

| Path | Required | Content |
|---|---|---|
| `README.md` | yes | The sections question, data source, decisions, how to run, expected outputs, reference result |
| `scenario.yaml` | yes | The name, the source and the default file of each configuration group |
| `data_card/`, `decisions/`, `cosy_repo/` | yes | One `.yaml` file per alternative |
| `sweep/`, `variants/` | no | Sweeps over parameters; named variants that override several settings |
| `source.py` | only for `source: {name: local}` | A class `ScenarioSource` with `initial_state()` and `events()` |
| `metrics.py` | no | Functions of the output directory that compute further measures |
| `plots.py` | no | Figures from output files; it never runs a simulation |
| `reference.csv` | no | Published or derived values to compare with |
| `data/SOURCE.md`, `data/SHA256SUMS.txt` | yes | Where the data comes from and the checksum of every file |

A scenario contains no simulation loop and no hooks: `casim run` runs it, writes the outputs and
compares them with `reference.csv`. Data files are never committed; `scripts/fetch_data.py`
downloads the public datasets into the ignored `data/` directories and checks them against the
checksums, and confidential data is placed there by the user.

## Writing a new scenario

1. Copy `scenarios/minimal/`. Copy no other scenario.
2. Set the source in `scenario.yaml`. If your data is in a format a library source reads
   (`hessler_irnich`, `oobp`, `casop`, `foodmart`, `kris`, `radar`), name it. A new public format
   becomes a library source in `casim.sources`, registered in `casim.sources.base.LIBRARY`, not a
   loader inside the scenario. Data that only this scenario uses gets `source: {name: local}`
   and a `source.py`.
3. Write the data card, the decisions and the CoSy repository (below).
4. Describe the data in `data/SOURCE.md` and list its checksums in `data/SHA256SUMS.txt`.
5. Put published or derived values into `reference.csv`, each with its kind.
6. Run `python -m casim check --all`; it must pass. The test suite runs it on every scenario.

## Running

```bash
python -m casim run scenarios/<name> [overrides] [--audit] [--output-dir DIR]
python -m casim check --all
```

Overrides use Hydra syntax: `decisions=order_window` picks another file of a group,
`source.filepath=29s-40-30-0.txt` sets a value, `variants=albareda` applies a variant, and
`sweep=<file>` runs every combination of a sweep file. Without `--output-dir`, outputs go to
`scenarios/<name>/outputs/run/` (ignored by git). The outputs are listed in `docs/concepts.md`.

`casim check --all` checks every tracked scenario: the required and allowed files, the five
groups, the source (a library source by name, or a `source.py` with `ScenarioSource`), that no
scenario file imports the simulation loop, the checksums of the data present, the format of
`reference.csv`, and that every file of every group composes and validates, alone or with one
variant. `casim check <dir> [overrides]` prints, per decision, the problem class, the decision
features, the subproblems, the number of pipelines, the algorithm cards excluded with their
reasons, the write set and the precedence.

## `scenario.yaml`

```yaml
defaults:
  - _self_
  - data_card: minimal       # default file of each group
  - decisions: obrsp
  - cosy_repo: fifo_sshape
  - optional variants: null  # when the scenario has variants

name: minimal
source:
  name: local                # a library source, or local
  shift_start_s: 0.0         # every further key is a parameter of the source
  ...
```

The parameters under `source` are those of the source's constructor; each library source
documents them in its module under `src/casim/sources/`. CASIM adds the settings every run
shares (output directory, cache directory, the `--audit` flag) itself.

## Data card (`data_card/`)

The data card describes the warehouse case for the applicability check of the algorithm cards,
in the format of CASOP. Each section has a `type` and a list of objects with features; a feature
without a value means "present", a feature with a value states it.

| Section | Describes | Example features |
|---|---|---|
| `name`, `problem_class`, `objective` | the case, the problem class (`OBRSP`) and the objective the ranker uses | `makespan`, `mean_order_completion_time`, `tardiness` |
| `layout` | the layout type and its parameters | `n_aisles`, `n_blocks`, `distance_matrix`, `start_node` |
| `articles` | the article data | `article_id` |
| `orders` | the order data | `order_date`, `due_date`, `order_id` |
| `resources` | the pickers and their carts | `speed`, `time_per_pick`, `tour_setup_time`, `n_resources`, `capacities`, `n_boxes` |
| `storage` | the storage policy | `dedicated`, `amount`, `article_id` |
| `warehouse_info` | the planning setting | `type: online` |
| `information` | knowledge about the future | `type: process_information` with the arrival process |

A feature an algorithm requires must be declared here, or the algorithm is not applicable. For
example, a due-date window or an EDD scheduler needs `due_date` on orders, and the two analytic
waiting rules need `n_resources: 1`.

## Decisions (`decisions/`)

A decisions file holds one entry per decision, keyed by the decision's name:

```yaml
OBRSP:
  problem_class: OBRSP
  state_information:
    orders: true
    pick_lists: false
    tours: [planned]
    resources: [idle, reserved]
  scope: {freeze_from: pending, frozen_accepts: none}
  solver:
    _target_: casim.pipelines.pipeline_runner.CoSySolver
    repo: ${cosy_repo}
  commitment_policy:
    _target_: casim.decision_engine.commitment_policies.SchedulingCommitmentPolicy
    n_jobs: 1
  precedence: 10
  triggers: [OrderArrival, OrderStreamClosed, PickerIdle]
  conditions:
    - {_target_: casim.simulation_engine.conditions.NbrPickersCondition, threshold: 1}
    - {_target_: casim.simulation_engine.conditions.NbrOrdersCondition, threshold: 1}
```

| Key | Meaning | Values |
|---|---|---|
| `problem_class` | which problem the pipelines solve | a class of `casim/pipelines/taxonomy.py`, usually `OBRSP` |
| `state_information.orders` | expose the order buffer | `true`, `false` |
| `state_information.pick_lists` | expose the pick-list buffer | `true`, `false` |
| `state_information.tours` | tour statuses the decision sees | subset of `planned`, `scheduled`, `pending`, `started`, `done`, `cancelled` |
| `state_information.resources` | resource states the decision sees | subset of `idle`, `reserved`, `busy`, `off_shift`, or `[]` |
| `state_information.window` | limit to a horizon | `{by: due_date \| start_time, horizon_s: n}` |
| `scope.freeze_from` | first status that is frozen | a tour status |
| `scope.frozen_accepts` | what frozen tours still accept | `none`, `insert` |
| `solver` | which pipelines produce candidates | `CoSySolver` with `repo`; omitted for a decision that only commits |
| `commitment_policy` | which planned tours are committed | `CommitAllPolicy`, `PlanOnlyPolicy`, `SchedulingCommitmentPolicy` with `n_jobs`, `WindowCommitmentPolicy` with `within_s`; omitted when the decision sees no planned tours |
| `change_penalty` | cost per changed existing tour in the ranking | number at least 0, default 0 |
| `precedence` | order among decisions due at one time | integer at least 0, lower first |
| `start_times` | committed starts are binding earliest starts or forecasts | `binding` (default), `forecast` |
| `triggers` | events that make the decision due | event names such as `OrderArrival`, `OrderStreamClosed`, `PickerIdle`, `PickerArrival`, `TourEnd`, `ActiveTourOpportunity`, `TruckDeparture`, and at most one `{every_s: n}` |
| `conditions` | when solving is worthwhile | condition classes of `casim.simulation_engine.conditions` with their parameters |

The schema (`python -m casim schema` prints it) rejects unknown keys and values, a scope that
gives an exposed status more than the process allows, `frozen_accepts: rebuild`, and a window
without a horizon. Setup also rejects a problem class CASIM does not know, a combination of
features no pipeline can serve, two decisions that write the same part of the plan without
distinct precedences, a due-date window on a data card without due dates, and a configured
component whose algorithm card the data card does not satisfy (with the reasons).

## CoSy repository (`cosy_repo/`)

A repository file lists the components a decision's pipelines may combine, by their dotted path:

```yaml
components:
- casim.pipelines.problem_based_template.InstanceLoader
- casim.pipelines.problem_based_template.OrdersProvider
- casim.pipelines.subproblems.item_assingment.GreedyIA
- casim.pipelines.subproblems.batching.FiFo
- casim.pipelines.subproblems.picker_routing.SShape
- casim.pipelines.subproblems.scheduling.FIFOScheduler
- casim.pipelines.subproblems.waiting.NoWaitingNode
```

CoSy builds every pipeline these components allow for the decision's subproblems. With several
alternatives for one subproblem (for example two batching rules), the decision gets one
candidate per pipeline and the ranker chooses. The available components are in
`src/casim/pipelines/subproblems/`; components for configured algorithm cards (for example
`OrderCountWaiting1` to `4`) are generated into `subproblems/generated/` by
`python -m casim.pipelines.subproblems.generate_configured_components`.

## Sweeps (`sweep/`) and variants (`variants/`)

A sweep file maps override keys to a value or a list of values; `casim run` runs the product of
all lists, each into its own output directory, and records a failed combination in `error.txt`:

```yaml
source.filepath: [29s-40-30-0.txt, 30s-40-45-0.txt]
decisions: [henn_waiting, order_window]
```

A variant file is a Hydra package (`# @package _global_`) that overrides several settings at
once, for example another data card and another source format:

```yaml
# @package _global_
defaults:
  - override /data_card: albareda
source:
  format: albareda
```

## Reference values (`reference.csv`)

`reference.csv` has the columns `kpi` (a key of `kpis.json`, or a function of that name in
`metrics.py`), `published`, `kind` and `source`. The kind is one of:

- `exact`: the value must be met exactly (a hand derivation, an equality with another code);
- `statistical`: compared through a confidence interval over replications;
- `bound`: CASIM runs a simpler rule than the published method and must not beat the value;
- `reproduction`: the value of a specific earlier implementation.

Every further column is a selector, a dotted key of the configuration (for example
`source.filepath`); a row applies to a run when all its selectors match. `casim run` writes
`comparison.csv` with the run's value, the published value and the difference.

## Scenarios in the repository

| Scenario | Case |
|---|---|
| `minimal` | One picker, one order, every value derived by hand; the template |
| `henn_waiting` | Henn's (2012) waiting rule and an order window on Heßler and Irnich's converted Henn instances |
| `walk_or_wait` | The wait-k and intervention rules of "Walking vs. Waiting" on a small example and the RADAR standard case |
| `analytic_waiting` | The analytic stochastic waiting rule on a four-order example |
| `casop_static` | The seven CASOP benchmark sets as static instances (benchmark L1) |
| `henn2012` | Henn's (2012) online batching experiments (benchmark L2) |
| `oobp` | The online order batching benchmark of Gil-Borrás et al. (2020) (benchmark L3) |
| `ijpe` | The IJPE grocery-retailer study (see `docs/ijpe/plan.md`); the submitted version is on the tag `archive/scenario-ijpe-submitted` |
| `scenario_dhaen_online` | Data of the D'Haen et al. (2023) benchmark only; not yet runnable |

`scenario_dhaen_online` is exempt from `casim check --all` until it has the structure above.
