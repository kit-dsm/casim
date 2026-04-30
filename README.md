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

##  Experiments

Two experiment setups can be found in scenarios/.
In scenarios/scenario_grocery_retailer you can find the results of a real-life using from an european grocery retailer. 
We provide the configuration, experiment scripts and detailed results. 

In scenarios/scenario_henn_online we reproduce two classic waiting strategy approaches as detailed in Henn 2012. 
This can be used as a starting point to get familiar with the framework.