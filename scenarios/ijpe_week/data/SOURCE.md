# Data: the IJPE week

This scenario holds no data of its own. Its source (`order_stream`,
`source.tables`) reads the canonical tables of the IJPE case in
`scenarios/ijpe/data/canonical/` (`scripts/ijpe/canonical.py`, see
`scenarios/ijpe/data/SOURCE.md`), and its `process_times` rule the fit
`scenarios/ijpe/data/non_picking/week.json` (`scripts/ijpe/fit_non_picking.py`).
Nothing is committed or restored by `scripts/fetch_data.py`.
