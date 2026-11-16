# Data: the WMS export of the IJPE case

One workbook, `wms_export_kw16.xlsx`: the order-picking export of the
retailer's warehouse for 11 and 13 to 17 April 2026, with the warehouse's own
pivot of transport units per day and its lists of disruption orders. It is
confidential (docs/ijpe_revision_plan.md, Section 1): it is never committed,
pushed or quoted at order, line or picker level. The user places it here
under this name; `SHA256SUMS.txt` lists its checksum. There is no public
source and `scripts/fetch_data.py` cannot restore it.

`scenarios/ijpe/analysis/export.py` caches the sheets as parquet in `cache/`
(also ignored):

```bash
uv run --with openpyxl python -m scenarios.ijpe.analysis.export
```

The workbook's content is described in `docs/ijpe/data_audit.md`.
