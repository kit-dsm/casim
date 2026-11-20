# IJPE submission: grocery retailer case

The code of the IJPE submission (PROECO-D-26-03325) is the tag `ijpe-2026`
(casim `7b6a1f3`). `main` adds only what is needed to reproduce it: the
converter that built the order stream (`generator/private/`, not in the
repository at submission time), an ignore rule for the confidential data and
this README. The revised model is developed on `codex/ijpe-revision` and is
not compared with this code.

## What you need

- casim at `main` (or `ijpe-2026` plus `generator/private/` from `main`).
- ware_ops_algos at `32cdff5`, its last commit before the submission (tag
  `ijpe-2026` in ware_ops_algos).
- A Python environment with casim's dependencies and `openpyxl`.
- The partner's files of June 2026, which are confidential and never
  committed: `OrdPick_REWE051_2026_KW16.xlsx` (WMS export) and
  `LBH_REWE051_2026_KW16.xlsx` (article dimensions). Place both in
  `scenarios/scenario_ijpe/data/historic_source/`; everything in
  `scenarios/scenario_ijpe/data/` except `layout.csv` is ignored by git.

`pyproject.toml` installs `../ware_ops_algos` as an editable package. Either
check out `32cdff5` there, or keep that checkout as it is and put a checkout of
`32cdff5` first on `PYTHONPATH` together with this repository (separator `;`
on Windows, `:` elsewhere):

```bash
git -C ../ware_ops_algos worktree add ../ware_ops_algos-ijpe 32cdff5
export PYTHONPATH="$PWD;$PWD/src;$PWD/../ware_ops_algos-ijpe/src"
export PYTHONUTF8=1   # Windows: the runs print characters cp1252 cannot encode
```

## Run

From the repository root, build the order stream (writes
`data/order_stream_historic.csv`, `data/calibrated_generator.json` and the
anonymisation map `data/historic_source/id_maps_new.json`):

```bash
uv run --with openpyxl python -m scenarios.scenario_ijpe.generator.private.generate_historic_order_stream
```

From `scenarios/scenario_ijpe/`, run the five scenarios as the paper did (a
Hydra multirun into `outputs/multirun/paper_results/`), or one of them:

```bash
python scenario_grocery_retailer.py viz.launch=false
python scenario_grocery_retailer.py simulation=scenario2 viz.launch=false
```

From `scenarios/scenario_ijpe/scripts/`, write the paper's tables and Gantt
figures to `outputs/paper_kpi_eval/tardiness/`:

```bash
python make_results_scenarios_multirun.py
```

## Order generator and public order stream

- `generator/public/generate_order.py` draws new orders from a calibration:
  lines per order from the empirical distribution, articles by pick
  frequency, a quantity per article. Scenario 3 uses it. The calibration
  (`data/calibrated_generator.json`) is fitted by the converter above
  (`fit_public_generator_calibration`); it is derived from the partner's data
  and not in the repository.
- `generator/private/generate_public_order_stream.py` writes the canonical
  order stream with article, order, store and picker numbers replaced by
  pseudonyms and the recorded times cleared, and checks that orders, lines and
  articles are preserved (`data/order_stream_public.csv`). It needs the
  partner's files as input; its rows are the recorded order lines.

```bash
uv run --with openpyxl python -m scenarios.scenario_ijpe.generator.private.generate_public_order_stream
```

Neither the calibration nor the pseudonymised stream is published yet.

## Reproduced results

Run on 20 November 2026 as above (Python 3.13, the third-party packages of
that date). The stream has 1,422 orders and 63,305 lines (manuscript,
Table 3). The sweep takes about 30 s.

| Manuscript | Reproduced | Paper |
|---|---|---|
| Table 4, baseline | 1,422 orders, makespan 124.17 h, processing 441.27 h, utilisation 47.65 %, on time 100 % | 1,422, 124.18 h, 441.27 h, 47.75 %, 100 % |
| Table 5, decisions | 12, 13, 13, 14, 14 | identical |
| Table 6, Scenario 1 | 72 affected, 100 % on time, 2 delayed, average wait 0.40, if delayed 10.4, maximum 17.8 min | 72, 100 %, 2, 0.49, 12.7, 20.5 |
| Table 6, Scenario 2 | 47 affected, 27.7 % on time, 1 delayed, 1.90, 98.7, 98.7 min | 50, 30.0 %, 1, 1.82, 94.5, 94.5 |
| Table 6, Scenarios 3 and 4 | 4 and 20 affected, all on time, no delay | identical |

Structure and most values reproduce; Scenario 1 and 2 waits, Scenario 2's
affected orders and utilisation differ slightly. The cause is not
determined. Candidates: the stream is built from the export as delivered on
11 June 2026, whose picks sheet the converter used under another name, and
the third-party packages are newer than at submission.

Table 6's "delayed departures" are hourly due-time groups, not trucks: every
distinct due time of any scenario is one departure, its wait is the latest
completion of the orders with that due time minus the due time, and the
average runs over all due times, empty ones included
(`compute_truck_waiting_times` in `scripts/make_results_scenarios_multirun.py`).
