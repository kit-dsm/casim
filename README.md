# CASIM: Context Aware Simulation for Warehouse Operations 

## What is this?
This repository contains the implementation of CASIM as described in the paper "CASIM: Context Aware Simulation for Warehouse Operations" submitted to the Winter Simulation Conference 2026. 
## Setup

This project depends on `ware_ops_algos`, which must be cloned adjacent:

```bash
git clone https://github.com/kit-dsm/casim.git
git clone https://github.com/kit-dsm/ware_ops_algos.git
cd casim
uv sync --locked --extra dev
uv run --locked pytest
```

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 (installed automatically by uv).

## Scenarios

A scenario is one warehouse case in `scenarios/<name>/`, run with

```bash
python -m casim run scenarios/<name> [overrides]
python -m casim check --all
```

Every scenario has the same structure (`AGENTS.md`); `scenarios/minimal/` is
the template for a new one.

- [minimal](scenarios/minimal/README.md): one picker, one order, values derived by hand.
- [henn_waiting](scenarios/henn_waiting/README.md): Henn's (2012) waiting rule and an order window on converted Henn instances.
- [walk_or_wait](scenarios/walk_or_wait/README.md): the wait-k and intervention rules of *Walking vs. Waiting*, with the RADAR standard case.
- [analytic_waiting](scenarios/analytic_waiting/README.md): the analytical stochastic waiting example.
- [oobp](scenarios/oobp/README.md), [casop_static](scenarios/casop_static/README.md): the literature benchmarks L3 and L1 (`docs/literature_benchmarks.md`); `scenario_henn2012/` (L2) is ported next.
- `scenario_ijpe/`: the IJPE study, ported in Task 6b.6.
