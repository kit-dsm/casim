# CASIM: Context Aware Simulation for Warehouse Operations

CASIM is a discrete-event simulation of manual order picking in which planning decisions are
configured, not programmed. A decision (when to batch, route, schedule, wait or admit orders to
a running tour) is defined by its trigger events, its conditions, the part of the state it sees,
how far it may revise earlier plans, and what it commits. Each decision is solved by pipelines
that CoSy synthesises from the algorithms of `ware_ops_algos`, which also serves the static
framework CASOP. CASIM is described in the paper "CASIM: Context Aware Simulation for Warehouse
Operations" (Winter Simulation Conference 2026).

## Setup

CASIM needs `ware_ops_algos` cloned next to it, on the matching branch (on this branch,
`codex/ijpe-revision`), and [uv](https://docs.astral.sh/uv/) (Python 3.13 is installed by uv).

```bash
git clone -b codex/ijpe-revision https://github.com/kit-dsm/casim.git
git clone -b codex/ijpe-revision https://github.com/kit-dsm/ware_ops_algos.git
cd casim
uv sync --locked --extra dev
```

`python -m casim run scenarios/minimal` runs without any data. The benchmark data is not
committed; download it, with its checksums checked, before running the other scenarios or the
tests:

```bash
uv run --locked python scripts/fetch_data.py all
uv run --locked pytest
```

`fetch_data.py` also takes one dataset: `henn_waiting`, `oobp`, `dhaen`, `casop`,
`casop_instances` or `radar`. The tests skip the slow baseline comparison by default
(`pytest -m baseline` runs it). Optional extras: `--extra viz` for the Dash replay and the plots,
`--extra rl` for the Gymnasium environment.

## Command line

```bash
python -m casim run <scenario_dir> [overrides] [--audit] [--output-dir DIR]
python -m casim check --all
python -m casim check <scenario_dir> [overrides] [--json]
python -m casim schema
python -m casim catalog
```

- `run` runs a scenario and writes `kpis.json`, `tracker.json`, `decisions.json`,
  `service.json`, `trace.json`, the facts tables `facts/tours.csv`, `facts/orders.csv` and
  `facts/picks.csv` and, with a `reference.csv`, `comparison.csv`. `--audit` checks the plan and the
  resources after every timestamp and writes `audit.json`. Overrides use Hydra syntax
  (`decisions=order_window`, `source.filepath=29s-40-30-0.txt`).
- Sweeps: a scenario's `sweep/` files list values to combine; `run ... +sweep=<name>` runs every
  combination into its own output directory.
- `check --all` checks the structure and every configuration of every scenario; `check
  <scenario_dir>` prints the decisions as CASIM resolves them, with the pipelines and the
  algorithm cards it excludes and why.
- `schema` prints the JSON schema of a decisions file; `catalog` lists statuses, policies and
  algorithm cards.

## Scenarios

| Scenario | Case |
|---|---|
| [minimal](scenarios/minimal/README.md) | One picker, one order, every value derived by hand; the template for new scenarios |
| [henn_waiting](scenarios/henn_waiting/README.md) | Henn's (2012) waiting rule and an order window on converted Henn instances |
| [walk_or_wait](scenarios/walk_or_wait/README.md) | The wait-k and intervention rules of "Walking vs. Waiting", with the RADAR standard case |
| [analytic_waiting](scenarios/analytic_waiting/README.md) | The analytic stochastic waiting rule on a four-order example |
| [casop_static](scenarios/casop_static/README.md) | The seven CASOP benchmark sets as static instances (benchmark L1) |
| [henn2012](scenarios/henn2012/README.md) | Henn's (2012) online batching experiments (benchmark L2) |
| [oobp](scenarios/oobp/README.md) | The online order batching benchmark of Gil-Borrás et al. (2020) (benchmark L3) |
| `ijpe` | The IJPE grocery-retailer study, under revision (`docs/ijpe/`) |

## Documentation

- [docs/concepts.md](docs/concepts.md): the model: state, decisions and their properties,
  revision scope, commitment, tour lifecycle, dispatch, events and outputs.
- [docs/scenarios.md](docs/scenarios.md): how a scenario is structured, the meaning of every
  configuration field, and how to write a new one.
- [docs/validation.md](docs/validation.md): what has been verified and how, the literature
  benchmarks with their open differences, the defects found, and the baseline tiers.
- [docs/development.md](docs/development.md): the backlog and the open design decisions.
- [docs/ijpe/](docs/ijpe/): the plan, status and data audit of the IJPE revision.
- [AGENTS.md](AGENTS.md): the rules every contributor follows.
