# CASIM: Context Aware Simulation for Warehouse Operations 

## What is this?
This repository contains the implementation of CASIM as described in the paper "CASIM: Context Aware Simulation for Warehouse Operations" submitted to the Winter Simulation Conference 2026. 
## Setup

This project depends on `ware_ops_algos`, which must be cloned adjacent:

```bash
git clone https://github.com/kit-dsm/casim.git
git clone https://github.com/kit-dsm/ware_ops_algos.git
cd casim
uv sync --extra dev
uv run pytest
```

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 (installed automatically by uv).

## Experiments

Experiment setups live under `scenarios/`. Their common research axes and
the procedure for adapting CASIM are summarized in
[`scenarios/README.md`](scenarios/README.md).

`scenarios/scenario_henn` contains the maintained reproduction of Henn
Algorithm 4.1 and its benchmark comparison.

`scenarios/scenario_intervention_stress` contains the multi-picker comparison
of depot-only reoptimization, routing-only intervention, and active-batch
insertion with exact-TSP or nearest-neighbour residual routing.

`scenarios/scenario_dynamic_operations` is the generated backlog-plus-arrivals
reference study. It compares nightly planning, periodic rolling commitment,
unstarted-work reoptimization, and optional active-route intervention on the
same seeded multi-day demand.

The current execution flow, state ownership, manager responsibilities, and
intervention semantics are summarized in
[`docs/execution_model.md`](docs/execution_model.md).

Install `--extra viz` for the optional Dash replay and `--extra rl` for the
experimental, currently non-maintained RL environment. The core simulator does
not depend on either stack.
