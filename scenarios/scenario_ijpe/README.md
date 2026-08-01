# IJPE grocery operations study

This is the maintained grocery publication study. Its loader deliberately
keeps warehouse-specific assumptions local: eight encoded aisles, left/right
travel rails, pallet-facing pick nodes, top/bottom crossings, special
`-90`/`-99` markers, RB storage ranges, a manual middle cross aisle, and
millimetre geometry.

Current simplifying assumptions are explicit in the input profiles:

- zero-cost depot connectors;
- effectively unlimited inventory (`9999`);
- a dedicated first location for each article;
- one order-bin cart;
- due windows reconstructed from rounded historic completion times.

The U-shape card applies only to this two-sided ladder context; it does not
claim intervention compatibility.

## Inputs

The default is a small deterministic public canonical stream:

```powershell
python -m scenarios.scenario_ijpe.experiment_ijpe
```

Use the anonymized six-day generated profile or an external historic stream:

```powershell
python -m scenarios.scenario_ijpe.experiment_ijpe input=generated_six_day
python -m scenarios.scenario_ijpe.experiment_ijpe input=historic input.load.orders_path=C:/data/order_stream.csv
```

Runtime simulation reads canonical CSV, never Excel. Convert and calibrate raw
workbooks explicitly:

```powershell
python -m scenarios.scenario_ijpe.generator.private.prepare_historic_stream \
  --picks C:/data/picks.xlsx --master C:/data/master.xlsx \
  --output C:/data/order_stream.csv \
  --calibration-output C:/data/calibration.json --seed 42 --days 6
```

The same canonical rows feed historic replay, seeded generation, normal
arrivals, and emergency-order injection. Scenario variants retain the existing
truck, cross-day volume, emergency-order, picker-schedule, WMS, and planning
events, but operational mutation is performed by `State`.
