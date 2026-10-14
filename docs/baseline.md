# Phase B baseline inventory

Recorded at casim commit `50c1a00` and ware_ops_algos commit `dd4ffc9` on 14 October 2026.
This file is the scenario inventory of Task B.1. Task B.6 records the full baselines;
Task B.5 adds the earlier-results comparison; Task B.8 replaces `docs/revision_scopes_baseline.md`
with a pointer here.

## Summary

| | Configurations | Run | Fail |
|---|---:|---:|---:|
| Total | 60 | 45 | 15 |

## scenario_henn_online

Entry point: `python -m scenarios.scenario_henn_online.experiment_henn`
Config: `henn_online_config`
Defaults: `data_card=henn`, `decisions=henn_waiting`, `cosy_repo=baseline_waiting_henn`

| # | Configuration | Command (overrides) | casim check | Runs? | Error / cause if not |
|---|---|---|---|---|---|
| 1 | default | (none) | ok | yes | |
| 2 | repair | `cosy_repo=repair_waiting_henn +decisions.OBRSP.change_penalty=10` | ok | yes | |
| 3 | rolling | `data_card=henn_rolling decisions=rolling_horizon cosy_repo=rolling_horizon` | fail | no | Undefined problem class: `problem_class 'rolling'` is not in the schema enum. Task 6.1 ports it as `OBRSP`. |
| 4 | release_only | `decisions=release_only_example` | ok | yes | |
| 5 | 29s-40-30-0 / henn_waiting | `data_card.source.filepath=29s-40-30-0.txt decisions=henn_waiting` | ok | yes | |
| 6 | 29s-40-30-0 / order_window | `data_card.source.filepath=29s-40-30-0.txt decisions=order_window` | ok | yes | |
| 7 | 30s-40-45-0 / henn_waiting | `data_card.source.filepath=30s-40-45-0.txt decisions=henn_waiting` | ok | yes | |
| 8 | 30s-40-45-0 / order_window | `data_card.source.filepath=30s-40-45-0.txt decisions=order_window` | ok | yes | |
| 9 | 31s-40-60-0 / henn_waiting | `data_card.source.filepath=31s-40-60-0.txt decisions=henn_waiting` | ok | yes | |
| 10 | 31s-40-60-0 / order_window | `data_card.source.filepath=31s-40-60-0.txt decisions=order_window` | ok | yes | |
| 11 | 32s-40-75-0 / henn_waiting | `data_card.source.filepath=32s-40-75-0.txt decisions=henn_waiting` | ok | yes | |
| 12 | 32s-40-75-0 / order_window | `data_card.source.filepath=32s-40-75-0.txt decisions=order_window` | ok | yes | |
| 13 | 37s-60-30-0 / henn_waiting | `data_card.source.filepath=37s-60-30-0.txt decisions=henn_waiting` | ok | yes | |
| 14 | 37s-60-30-0 / order_window | `data_card.source.filepath=37s-60-30-0.txt decisions=order_window` | ok | yes | |
| 15 | 38s-60-45-0 / henn_waiting | `data_card.source.filepath=38s-60-45-0.txt decisions=henn_waiting` | ok | yes | |
| 16 | 38s-60-45-0 / order_window | `data_card.source.filepath=38s-60-45-0.txt decisions=order_window` | ok | yes | |
| 17 | 39s-60-60-0 / henn_waiting | `data_card.source.filepath=39s-60-60-0.txt decisions=henn_waiting` | ok | yes | |
| 18 | 39s-60-60-0 / order_window | `data_card.source.filepath=39s-60-60-0.txt decisions=order_window` | ok | yes | |
| 19 | 40s-60-75-0 / henn_waiting | `data_card.source.filepath=40s-60-75-0.txt decisions=henn_waiting` | ok | yes | |
| 20 | 40s-60-75-0 / order_window | `data_card.source.filepath=40s-60-75-0.txt decisions=order_window` | ok | yes | |
| 21 | 61s-80-30-0 / henn_waiting | `data_card.source.filepath=61s-80-30-0.txt decisions=henn_waiting` | ok | no | `ValueError: OBRSP: no feasible candidate solution` at `decision_engine.py:103`. Runtime defect: no candidate survives validation for this instance. Document as defect. |
| 22 | 61s-80-30-0 / order_window | `data_card.source.filepath=61s-80-30-0.txt decisions=order_window` | ok | no | Same: `ValueError: OBRSP: no feasible candidate solution`. |
| 23 | 62s-80-45-0 / henn_waiting | `data_card.source.filepath=62s-80-45-0.txt decisions=henn_waiting` | ok | yes | |
| 24 | 62s-80-45-0 / order_window | `data_card.source.filepath=62s-80-45-0.txt decisions=order_window` | ok | yes | |
| 25 | 63s-80-60-0 / henn_waiting | `data_card.source.filepath=63s-80-60-0.txt decisions=henn_waiting` | ok | yes | |
| 26 | 63s-80-60-0 / order_window | `data_card.source.filepath=63s-80-60-0.txt decisions=order_window` | ok | yes | |
| 27 | 64s-80-75-0 / henn_waiting | `data_card.source.filepath=64s-80-75-0.txt decisions=henn_waiting` | ok | yes | |
| 28 | 64s-80-75-0 / order_window | `data_card.source.filepath=64s-80-75-0.txt decisions=order_window` | ok | yes | |
| 29 | 69s-100-30-0 / henn_waiting | `data_card.source.filepath=69s-100-30-0.txt decisions=henn_waiting` | ok | yes | |
| 30 | 69s-100-30-0 / order_window | `data_card.source.filepath=69s-100-30-0.txt decisions=order_window` | ok | yes | |
| 31 | 70s-100-45-0 / henn_waiting | `data_card.source.filepath=70s-100-45-0.txt decisions=henn_waiting` | ok | yes | |
| 32 | 70s-100-45-0 / order_window | `data_card.source.filepath=70s-100-45-0.txt decisions=order_window` | ok | yes | |
| 33 | 72s-100-75-0 / henn_waiting | `data_card.source.filepath=72s-100-75-0.txt decisions=henn_waiting` | ok | yes | |
| 34 | 72s-100-75-0 / order_window | `data_card.source.filepath=72s-100-75-0.txt decisions=order_window` | ok | yes | |

Sweep: `sweep/waiting_strategy_sweep.yaml` covers the 15 instances above with `decisions=henn_waiting,order_window`.
Instance `71s-100-60-0.txt` exists in `data/` but is not in the sweep.

## scenario_walk_or_wait

Entry point: `python -m scenarios.scenario_walk_or_wait.experiment`
Config: `walk_or_wait_config`
Defaults: `decisions=paper_intervention`, `policy=wait_1`, `routing=s_shape`, `cosy_repo=paper_waiting`, `insertion_repo=paper_remaining_route`, `simulation=explicit_example`

| # | Configuration | Command (overrides) | casim check | Runs? | Error / cause if not |
|---|---|---|---|---|---|
| 35 | default | (none) | ok | yes | |
| 36 | wait_0 | `policy=wait_0 simulation=explicit_example` | ok | yes | |
| 37 | wait_2 | `policy=wait_2 simulation=explicit_example` | ok | yes | |
| 38 | wait_3 | `policy=wait_3 simulation=explicit_example` | ok | yes | |
| 39 | wait_4 | `policy=wait_4 simulation=explicit_example` | ok | yes | |
| 40 | radar_wait_1 | `simulation=radar_standard_case policy=wait_1` | ok | yes | |
| 41 | patrol | `policy=wait_0 simulation=explicit_wait0 cosy_repo=paper_patrol` | ok | yes | |
| 42 | center | `policy=center simulation=explicit_wait0 cosy_repo=paper_center insertion_repo=center` | ok | yes | |
| 43 | no_intervention_return | `decisions=paper_no_intervention policy=wait_4 routing=return` | ok | yes | |
| 44 | no_intervention_nearest | `decisions=paper_no_intervention policy=wait_2 routing=nearest_neighbour` | ok | yes | |
| 45 | no_intervention_largest_gap | `decisions=paper_no_intervention routing=largest_gap` | ok | yes | |

No sweep directory. Configurations from README.md.

## scenario_stochastic_waiting

Entry point: `python -m scenarios.scenario_stochastic_waiting.experiment_stochastic_waiting`
Config: `stochastic_waiting_config`
Defaults: `decisions=waiting`, `cosy_repo=analytic_waiting`, `insertion_repo=active_insertion`, `simulation=reference`

| # | Configuration | Command (overrides) | casim check | Runs? | Error / cause if not |
|---|---|---|---|---|---|
| 46 | default (analytic_waiting) | (none) | ok | yes | |
| 47 | henn_waiting | `cosy_repo=henn_waiting` | ok | yes | |
| 48 | no_waiting | `cosy_repo=no_waiting` | ok | yes | |

No sweep directory.

## scenario_grocery_retailer

Entry point: `python -m scenarios.scenario_grocery_retailer.experiment_grocery_retailer`
Config: `grocery_retailer_config`
Defaults: `decisions=scenario_grocery_fixed_arrival`, `cosy_repo=return`

| # | Configuration | Command (overrides) | casim check | Runs? | Error / cause if not |
|---|---|---|---|---|---|
| 49 | default (return) | (none) | ok | no | Missing loader: `ModuleNotFoundError: No module named 'scenarios.scenario_grocery_retailer.grocery_retailer_loader'`. The loader class referenced by the data card does not exist. |
| 50 | midpoint | `cosy_repo=midpoint` | fail | no | Card applicability: `ORSP: configured cards excluded: FiFo: problem type 'batching' not in ['ORSP', 'routing', 'scheduling']`. The cosy_repo includes batching and item-assignment cards inapplicable to ORSP. |
| 51 | largest_gap | `cosy_repo=largest_gap` | fail | no | Same card-applicability error. |
| 52 | nearest_neighbour | `cosy_repo=nearest_neighbour` | fail | no | Same. |
| 53 | sshape | `cosy_repo=sshape` | fail | no | Same. |
| 54 | adaptive_routing | `cosy_repo=adaptive_routing` | fail | no | Same. |
| 55 | full_portfolio | `cosy_repo=full_portfolio` | fail | no | Same. |

Sweep: `sweep/cosy_repo_sweep.yaml` covers `return, midpoint, largest_gap, nearest_neighbour, sshape, adaptive_routing` with `decisions.ORSP.conditions.1.threshold=98`.
All grocery configurations fail because the loader does not exist and the cosy_repo cards are inapplicable to ORSP. Deferred to Phase 6b.

## scenario_ijpe

Entry point: `python -m scenarios.scenario_ijpe.scenario_grocery_retailer`
Config: `grocery_retailer_config` (local to `scenario_ijpe/config/`)
Defaults: `decisions=scenario_obp`, `cosy_repo=ushape`, `simulation=scenario3`, `sweep=scenario_sweep` (active)

| # | Configuration | Command (overrides) | casim check | Runs? | Error / cause if not |
|---|---|---|---|---|---|
| 56 | default (scenario3) | (none) | ok | no | Missing data: `FileNotFoundError: data\order_stream_historic.csv`. The config defaults to `sweep=scenario_sweep` (active, not optional), so Hydra enters multirun mode. |
| 57 | baseline | `simulation=baseline` | ok | no | Same: missing `data\order_stream_historic.csv`. |
| 58 | scenario1 | `simulation=scenario1` | ok | no | Same. |
| 59 | scenario2 | `simulation=scenario2` | ok | no | Same. |
| 60 | scenario4 | `simulation=scenario4` | ok | no | Same. |

Sweep: `sweep/scenario_sweep.yaml` covers `simulation: baseline, scenario1, scenario2, scenario3, scenario4`.
All IJPE configurations fail because the data file `data\order_stream_historic.csv` is not in the repository. Deferred to Task 6b.6.

## Defects found

| Config | Defect | Task that fixes |
|---|---|---|
| henn__rolling | `problem_class 'rolling'` is invalid; should be `OBRSP` | 6.1 |
| henn__61s-80-30-0 (both strategies) | `ValueError: OBRSP: no feasible candidate solution` — no candidate survives validation | document; propose task |
| grocery (all) | Loader `scenarios.scenario_grocery_retailer.grocery_retailer_loader` does not exist | 6b |
| grocery (non-default cosy_repos) | Cards inapplicable to ORSP (batching, item_assignment) | 6b |
| ijpe (all) | `data\order_stream_historic.csv` missing | 6b.6 |
| ijpe (all) | Config defaults to active sweep, forcing multirun mode for single runs | 6b |
