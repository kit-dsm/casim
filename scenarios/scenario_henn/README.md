# Henn Algorithm 4.1 on the W5 benchmark

This scenario follows the standard CASIM experiment structure:

1. Hydra composes the data, engine, CoSy, batching, selection, and experiment
   configuration.
2. `experiment_commons` constructs the loader, simulation, and decision engine.
3. `simulation.reset(hooks=build_sim_hooks(cfg))` installs the order stream and
   the scenario-local wake-up trigger.
4. CoSy generates candidate batches and S-shape routes.
5. One function applies Henn Algorithm 4.1.
6. Only a dispatch calls `SimulationEngine.step()`. A wait leaves the order
   buffer, tours, and picker unchanged.

## Algorithms

The configurable batching variants are:

- `fcfs`: the ware-ops FIFO implementation.
- `cw_like`: the ware-ops Clark-and-Wright savings implementation.
- `ls`: the ware-ops local-search implementation.

`cw_like` and `ls` are sensitivity-study approximations. They are not claimed
to be the paper's exact C&W(ii) and ILS algorithms. Selection is configurable
as `first`, `short`, `long`, or `sav`. Every experiment uses S-shape routing.

Service time is calculated from the actual CoSy route:

```text
180 seconds setup + route distance / 0.8 + 10 seconds per item
```

Single-order service times are produced by routing that order alone through
the same CoSy pipeline.

## Commands

```powershell
python -m scenarios.scenario_henn.experiment_henn `
  experiment=single `
  experiment.instance_id=H_abc1_40_29 `
  batching=fcfs selection=short

python -m scenarios.scenario_henn.experiment_henn `
  experiment=reference_set `
  batching=fcfs selection=short

python -m scenarios.scenario_henn.experiment_henn --multirun `
  experiment=reference_set sweep=matrix
```

After a matrix sweep, combine the twelve job summaries with:

```powershell
python -m scenarios.scenario_henn.scripts.make_summary <sweep-directory>
```

## Result semantics

The workbook contains Gil GRASP/VND and Best Known completion and maximum
turnover objectives. These are cross-algorithm benchmarks, not Algorithm 4.1
reproduction targets. Comparisons therefore use `EQUAL`, `BETTER`, and
`WORSE`. Reference batch membership, dispatch times, and routes are
`NOT_AVAILABLE`.

The W5 instances also differ from the numerical experiment behind Henn Tables
7.1–7.4. The matrix summary checks whether broad qualitative patterns are
similar, but it does not claim numerical reproduction of those tables.
