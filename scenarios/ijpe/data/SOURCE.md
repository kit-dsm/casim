# Data: the WMS export of the IJPE case

One workbook, `wms_export_kw16.xlsx`: the order-picking export of the
retailer's warehouse for 11 and 13 to 17 April 2026, with the warehouse's own
pivot of transport units per day and its lists of disruption orders. It is
confidential (docs/ijpe/plan.md, Section 1): it is never committed,
pushed or quoted at order, line or picker level. The user places it here
under this name; `SHA256SUMS.txt` lists its checksum. There is no public
source and `scripts/fetch_data.py` cannot restore it.

`casim.sources.wms_export` reads it; `scripts/ijpe/convert.py` caches the sheets as parquet in `cache/`
(also ignored):

```bash
uv run --with openpyxl python -m scripts.ijpe.convert
```

The workbook's content is described in `docs/ijpe/data_audit.md`.

`canonical/` holds the export as canonical tables, the only form the
simulation, the fit and the validations read (`casim.sources.order_stream`,
`scenarios/ijpe/source.py`, `day_lines`): `orders.csv`, `lines.csv` (the
replayed cohort with the recorded login, start, end and sheet row),
`records.csv` (every record of the export), `excluded.csv`, `lists.csv`,
`templates/S3.csv` and `layout.csv`. They hold order-, line- and login-level
data, so they are confidential and ignored like the export;
`python -m scripts.ijpe.canonical` writes them from the cache. Only that
script and the data audit (`scripts/ijpe/audit.py`) read the workbook.

`layout.csv` is the warehouse's grid layout, a copy of the tracked
`scenarios/scenario_ijpe/data/layout.csv`, read by `casim.sources.grid_layout`.

`non_picking/<day>.json` holds the fit of breaks, tour setup and line time for
each study day on the other five days, written by
`python -m scripts.ijpe.fit_non_picking` from the canonical tables, and
`week.json` the fit on all six days for `scenarios/ijpe_week`. It holds samples
taken from the export at picker-day and line level, so it is confidential and
ignored like the export; rerun the script to restore it.
