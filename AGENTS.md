# Working on casim

These rules apply to every contributor, human or agent. The model is described in
`docs/concepts.md`, the scenario structure in `docs/scenarios.md`, what has been verified in
`docs/validation.md`, and the backlog in `docs/development.md`.

## Scenarios

A scenario is one warehouse case in `scenarios/<name>/` with the structure of
`docs/scenarios.md`: `README.md` (question, data source, decisions, how to run, expected
outputs, reference result), `scenario.yaml`, the groups `data_card/`, `decisions/`, `cosy_repo/`,
optionally `sweep/` and `variants/`, `source.py` only for a local source, optional `metrics.py`,
`plots.py` and `reference.csv`, and `data/SOURCE.md` with `data/SHA256SUMS.txt`.

1. A new case starts from a copy of `scenarios/minimal/`. No other scenario is copied.
2. A new data format becomes a library source in `casim.sources` (registered in
   `casim.sources.base.LIBRARY`), not a loader inside a scenario.
3. A scenario contains no simulation loop and no hooks; `casim run` owns the loop and the
   outputs.
4. Data is never committed: `scripts/fetch_data.py` downloads public data, the user places
   confidential data. IJPE data at order, line or picker level never enters a committed file.
5. `python -m casim check --all` must pass. `scenarios/scenario_dhaen_online/` is exempt until
   it has this structure.

## Before every commit

- `python -m pytest -q` passes in one process (the baseline comparison is skipped by default).
- `python -m casim check --all` passes.
- `python scripts/baseline.py compare-fast --workers 4` shows only the differences the change
  names. At the end of a change, run the full tier (`compare` and `compare-benchmark` for `l1`,
  `l2`, `l3`, see `docs/validation.md`).
- A change is committed with its explained baseline differences; re-recorded baselines go into a
  separate commit titled "Baseline: re-record ...".
- Check `git diff --cached --stat` before committing; other sessions may share the index.

## Core changes during the IJPE revision

The core (`src/casim`, `ware_ops_algos`) is frozen at the tag `core-ijpe-base` until 27 December
2026. On `codex/ijpe-revision`, only the core changes listed in Section 4 of `docs/ijpe/plan.md`
are allowed, each with a test and with all other baselines identical. Any other core change needs
the user's approval first.

Do not change ware_ops_pipes; its loader defects are fixed there by the user.

## Writing

Documents use plain, full sentences, no em dashes, and separate what is verified from what is
assumed. Documents describe the current state; history belongs in git. No code or data goes
under `docs/`.
