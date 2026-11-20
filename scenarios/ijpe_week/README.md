# IJPE week: one order stream with exogenous rules

The six study days of the IJPE case (11 and 13 to 17 April 2026) as one
simulation: the orders of a day are released at 05:30 of that day, and an
order still open at the end of a shift is picked in the next one. This is
the base of the disruption and policy experiments (`docs/ijpe/plan.md`, P2
and P4). The single-day replay and its validation stay in `scenarios/ijpe`.

## Data

The canonical tables of the IJPE case (`scenarios/ijpe/data/canonical/`,
confidential, ignored by git), written once from the WMS export by
`python -m scripts.ijpe.canonical`: `orders.csv` (order, day, release, store),
`lines.csv` (order lines at their pick nodes, with the recorded login and
times the replay uses), `lists.csv` (the warehouse's disruption lists),
`templates/S3.csv` and the layout. The library source `order_stream` reads
them (`source.tables`); the line and setup times come from
`scenarios/ijpe/data/non_picking/week.json` (`scripts/ijpe/fit_non_picking.py`).
Only the conversion script reads the workbook.

## What the world does (`exogenous`)

Declared in `scenario.yaml` and applied by `casim.sources.exogenous`:

- `roster`: every day 05:30 to 14:30, 12, 13, 14, 16, 13 and 4 pickers: the
  record's picker-hours per day divided by the 9-hour shift (the record has
  21, 17, 16, 19, 15 and 7 logins, many present for part of the day). Breaks
  08:30 to 09:00 and 11:30 to 12:00 for everyone, a simple assumption near
  the fitted median starts; change them here;
- `due_dates`: the synthetic dispatch plan (`docs/ijpe/departure_fallback.md`):
  a store's slot at 10, 12, 14 or 16 h minus 30 minutes of staging;
- `times`: each order line's and setup's time drawn from the fit with
  `seed`; the pickers plan with the means.

Variants add the paper's disruptions on the warehouse's lists:

| Variant | Rule |
|---|---|
| `scenario1` | Monday 07:30: list S1 due at 10:00 (`due_change`; orders due earlier keep their deadline) |
| `scenario2` | Thursday 10:30: list S2 due at 11:00 |
| `scenario3` | Wednesday 08:30: the S3 template order arrives 3 times (6 TU / 2.6), due 10:00 (`order_injection`) |
| `scenario4` | Thursday's orders of list S5 arrive on Wednesday 06:30, due 10:00 (`order_move`) |
| `scenario2_4` | both (P4.4) |

## Decisions

`edd` (default) and `fifo`: the pull of `scenarios/ijpe` (pick lists of one
order, routed once with U-shape; each idle picker takes the next tour, every
idle picker at once, `per_resource`), by earliest due date or first in first
out.

## How to run

```bash
python -m scripts.ijpe.canonical            # once: the canonical tables
python -m scripts.ijpe.fit_non_picking      # once: the fit, also week.json
python -m casim run scenarios/ijpe_week
python -m casim run scenarios/ijpe_week variants=scenario1 exogenous.times.seed=7
python -m casim run scenarios/ijpe_week decisions=fifo variants=scenario4
```

## Expected outputs

CASIM's outputs; `facts/orders.csv` marks the affected orders. Times are
seconds from midnight of 11 April; day `d` starts at `(d - 11 April) * 86400`.
Seed 0, EDD, no disruption: 632 orders by 10:00 against the WMS's 626 over
six days, 938 by 12:00 against 1,031; 108 orders carry over to the next
morning, 30 Friday orders are open at the end of the stream.
