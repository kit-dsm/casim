# Working on casim

These rules apply to every contributor, human or agent. The plan in
`docs/revision_scopes_corrections.md` and the design document
`docs/revision_scopes_design.md` take precedence; Task 5.0 extends this file
with the code conventions.

## Scenarios

A scenario is one warehouse case that CASIM runs: a literature benchmark, a
study on company data, or later a shadow of a running WMS. It lives in
`scenarios/<name>/` and has exactly this structure:

- `README.md`: what the case is, its source or paper, how to run it.
- `scenario.yaml`: the source (`source: {name: ..., parameters}`) and the
  Hydra groups `data_card`, `decisions`, `cosy_repo`, `sweep`, `variants`,
  each a directory of `.yaml` files.
- `source.py` only if no library source of `casim.sources` fits
  (`source: {name: local}`, class `ScenarioSource` with `initial_state()` and
  `events()`).
- Optional: `metrics.py` (functions of the output files), `plots.py` (reads
  output files only), `reference.csv` (published values with their kind:
  exact, statistical, bound, reproduction).
- `data/SOURCE.md` and `data/SHA256SUMS.txt`. Data is never committed: it is
  fetched by `scripts/fetch_data.py` or placed by the user, and checked
  against the checksums.

A scenario contains no simulation loop and no hooks: `casim run` owns the loop,
the outputs and the comparison with `reference.csv`.

1. A new case starts from a copy of `scenarios/minimal/`. No other existing
   scenario is copied.
2. A new data format becomes a library source in `casim.sources` (registered
   in `casim.sources.base.LIBRARY`), not a loader inside a scenario.
3. `python -m casim check --all` must pass. The test suite runs it on every
   tracked scenario.

`scenarios/scenario_ijpe/` keeps its old structure until Task 6b.6.

## Before every commit

- `python -m pytest -q` passes in one process. It skips the `baseline` marker
  (every scenario against its recorded baseline, slow); run those
  with `python -m pytest -m baseline` at the end of a task.
- `python -m casim check --all` passes.
- `python scripts/baseline.py compare` (and `compare-benchmark` for a change
  that can touch a benchmark) shows only the differences the task names.
