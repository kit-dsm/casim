# IJPE week: one order stream and the world around it

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
them (`source.tables`), wrapped by this scenario's `source.py`; the line and setup times come from
`scenarios/ijpe/data/non_picking/week.json` (`scripts/ijpe/fit_non_picking.py`).
Only the conversion script reads the workbook.

## What the world does (`source.py`)

Parameters under `source` in `scenario.yaml`, produced by `source.py`:

- `roster`: every day 05:30 to 14:30, 12, 13, 14, 16, 13 and 4 pickers: the
  record's picker-hours per day divided by the 9-hour shift (the record has
  21, 17, 16, 19, 15 and 7 logins, many present for part of the day). Breaks
  08:30 to 09:00 and 11:30 to 12:00 for everyone, a simple assumption near
  the fitted median starts; change them here;
- `due_dates`: the synthetic dispatch plan (`docs/ijpe/departure_fallback.md`):
  a store's slot at 10, 12, 14 or 16 h minus 30 minutes of staging;
- `times`: each order line's and setup's time drawn from the fit with
  `seed`; the pickers plan with the means.

Variants add the disruptions of plan P2.1 (`docs/ijpe/scenarios.md`) under
`source.disruptions`, each with a `type` (`due_change`, `order_move`,
`order_injection`; `source.py` describes them). Each
moves or adds a volume: the TE the partner's mail names times 0.20, the area's
share of the warehouse's volume, in TE of 0.445 m³ of goods (`volume` of the
canonical lines). `python -m scripts.ijpe.disruptions` writes the selections
into the canonical folder: whole stores of the original departure's slot,
largest first (`*_v1_fixed`) and one random selection per replication within
15 % of the target (`*_v1_r00`, ...), for V = 0.5 to 4 times the target
(`*_v<V>_*`); Scenario 3's order comes from the order generator G3. Every
scenario deadline is the departure minus the 30 minutes of staging.

| Variant | Rule |
|---|---|
| `scenario1` | Monday 07:30: stores of the 14:00 slot ("13:00"), 24 TE, due 09:30 (10:00 departure; `due_change`) |
| `scenario2` | Thursday 09:30: stores of the 14:00 slot, 20 TE, due 10:30 (11:00 departure) |
| `scenario2_short` | the same requested at 10:30, when the goods are due (the submitted timing) |
| `scenario3` | Wednesday 08:30: one new store order of 1.2 TE from G3, due 09:30 (`order_injection`) |
| `scenario4` | Thursday's stores of the 10:00 slot, 40 TE, arrive on Wednesday 06:30, due 09:30 (`order_move`); next-day picking ahead is not in the export |
| `scenario2_4` | both (P4.4); the two selections share no order |

A sweep overrides the selection:
`source.disruptions.scenario2.orders.list=S2_v2_r03`,
`source.disruptions.scenario3.template=S3_v4_fixed`. The selections are targets; the
achieved TE of each list is printed by the script (the fixed rule overshoots
small targets, for example +21 % for Scenario 2 at V = 1).

## Decisions

`edd` (default) and `fifo`: the pull of `scenarios/ijpe` (pick lists of one
order, routed once with U-shape; each idle picker takes the next tour, every
idle picker at once, `per_resource`), by earliest due date or first in first
out.

## How to run

```bash
uv run --with openpyxl python -m scripts.ijpe.canonical   # once: the canonical tables
python -m scripts.ijpe.disruptions          # once: the P2.1 selections
python -m scripts.ijpe.fit_non_picking      # once: the fit, also week.json
python -m casim run scenarios/ijpe_week
python -m casim run scenarios/ijpe_week variants=scenario1 source.times.seed=7
python -m casim run scenarios/ijpe_week decisions=fifo variants=scenario4
```

## Expected outputs

CASIM's outputs; `facts/orders.csv` marks the affected orders. Times are
seconds from midnight of 11 April; day `d` starts at `(d - 11 April) * 86400`.
Seed 0, EDD, no disruption: 632 orders by 10:00 against the WMS's 626 over
six days, 938 by 12:00 against 1,031; 108 orders carry over to the next
morning, 30 Friday orders are open at the end of the stream.
