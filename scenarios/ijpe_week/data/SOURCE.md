# Data: the IJPE week

This scenario holds no data of its own. Its source reads the confidential WMS
export of the IJPE case and the fit of non-picking time from
`scenarios/ijpe/data/` (see `scenarios/ijpe/data/SOURCE.md`): the cached
sheets in `cache/`, `layout.csv`, and `non_picking/week.json`, which
`python -m scripts.ijpe.fit_non_picking` writes. Nothing is committed or
restored by `scripts/fetch_data.py`.
