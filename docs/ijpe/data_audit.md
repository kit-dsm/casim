# Data audit of the IJPE case (Phase R0)

Phase R0 of `docs/ijpe/plan.md`: what the WMS export contains, before
any of it is modelled. This document holds aggregates only. The tables are
written by

```bash
uv run --with openpyxl python -m scripts.ijpe.convert   # once: workbook -> parquet
python -m scripts.ijpe.audit                           # tables -> scenarios/ijpe/outputs/audit/
```

and every number below is taken from that output (table names in brackets).
"Order" means a WMS order number (AUFTRAGSNR) and "store-day" means one
customer's orders of one day. The submitted model refers to casim `main`,
`scenarios/scenario_ijpe/`.

## Findings in brief

1. **Period.** Saturday 11 April and Monday 13 to Friday 17 April 2026: six
   working days, 65,291 lines, 1,525 orders, 403 stores, 45 pickers. 11 April
   lies in ISO week 15, the other days in week 16. The workbook's pivot labels
   11 April "Sonntag" (Sunday); it was a Saturday.
2. **1,422 and 1,419.** The model drops aisle 10 (area 10) and one article,
   which removes 103 orders (1,525 to 1,422). Figure 8 drops a further 3 orders:
   single-line orders whose only line has start = end, so their historic span
   is 0 and `valid_rows` (span > 0) excludes them.
3. **An order is a zone order of a store, picked by one picker.** 1,517 of
   1,525 orders (99.5 %) have exactly one picker; no order is split among
   pickers by zone. Instead the WMS splits a store-day into one order per area
   (area 1 = aisles 1–6, area 7 = aisles 7–8, area 10 = aisle 10): 97 % of
   store-days with several orders have one order per area, 98.5 % have a
   different picker per order, and in 39 % these orders are picked at
   overlapping times. The zones and parallel pickers of the manuscript's
   Section 4.1 exist, at the level of the store-day, not of the order.
4. **Pickers keep several orders open.** Of the time a picker has an order
   open, 89 % is spent with one order, 8 % with two and 3 % with three or four.
   57 of 106 picker-days alternate between orders. The model's single-order
   tour is the dominant case but not the only one.
5. **A line's time interval includes the walk to it.** Within an order, 83 %
   of lines start at exactly the second the previous line ended (95 % within
   1 s). The start of a line is the confirmation of the previous one; a line's
   duration is travel plus pick. R1 has to model the two together or separate
   them by distance.
6. **Booking date versus pick date.** 147 orders have 295 lines (0.5 %) booked
   on the day after they were picked. Every such line splits a position that
   was picked on the earlier day, 99.7 % have the order's picker, 98 % lie
   inside the order's time interval of the earlier day, and 52 % collide with
   other lines of the same picker on the booking date. The workbook's daily
   order counts and the warehouse's disruption lists filter by booking date and
   so count these orders on two days.
7. **Breaks are not at 8:00 and 11:00 for everyone.** Gaps longer than 15 min
   occur from 6:00 to 13:30; 47 of 106 picker-days have one overlapping
   8:00–8:30 and 39 overlapping 11:00–11:30.
8. **Observed busy share.** Over all picker-days, the union of line intervals
   is 62 % of the time between a picker's first and last line (69 % median per
   picker-day); with gaps over 5 min counted as idle it is 70 %. The paper's
   47.75 % is a simulation output (active over available time) and is not
   comparable until R2.3 defines both on the same basis.
9. **A TU is 2.6 orders by definition.** The workbook computes transport units
   as `distinct orders × 2.6` and TU per hour as `TU / 7.5`; no TU is measured.
   The warehouse sized its disruption lists with this factor (120 TU: 48
   orders, 200 TU: 76). The submitted simulation converts with 1.6 TU per
   order and selects orders by due time, which yields 72 orders for 120 TU in
   Scenario 1 and 20 orders for 200 TU in Scenario 4: exactly the ratios
   reviewer 2 questions (R0.5).
10. **Scenario 2 is 37 stores, not two.** The warehouse's list for 16 April
    (100 TU before 13:00) holds 38 orders from 37 stores. The submitted
    configuration uses 120 TU for this scenario.
11. **No departures in the export.** No column names a truck, route, tour or
    departure, and no store-to-truck relation can be derived. R3 needs the
    departure plan per store from the user (stop point of Section 9).

## What the data says about splitting, batching and sequencing

The model needs three WMS decisions per day: how a store's demand is split
into orders, how orders are grouped into the tours of a picker (batching and
assignment), and in which sequence a tour's lines are picked. The export
records the outcome of each, line by line, but not the decision rules.

| Decision | Columns that record it | Recoverable? |
|---|---|---|
| Split of a store's demand | KUNDENNR (store), NEU_DATUM / pick date, area, AUFTRAGSNR | Yes, the outcome: a store-day's orders are its zone orders, one per area in 97 % of multi-order store-days; their order numbers are mostly consecutive (median gap 1, fully consecutive in 62 %), so the WMS creates them together. The rule beyond "one order per zone and store-day" (for example a size limit: 56 orders cover all three areas) is not recorded. |
| Batching and assignment | PERS_NR (picker), BEGINN_ZEIT / ENDE_ZEIT, PALNR | Only as inferred from times. There is no batch, wave, pick-list or tour column. LFDNR is a running line number across all orders (order ranges overlap in 99.6 %; it matches the time order within an order for 41 % of lines), not a batch key. PALNR is the load carrier: 2,757 of 2,760 pallets hold one order, so a picker carrying several orders carries several pallets. A recorded tour is therefore defined from the picker's timeline: lines in time order, cut where no order of the picker is open; overlapping orders (11 % of order time) form one tour. |
| Sequence | AUFTRAGSPOS, Q_PLATZ, start times | Yes. The WMS numbers an order's positions along the walking path (position order equals location order for 99.5 % of lines), and pickers mostly follow it: 97 % of consecutive picks of an order are in non-decreasing position order, 35 % of orders exactly so. The replay uses the recorded time order. |

Consequence for the model: the replay treats orders as split and batched
upstream ("pre-batched"): it takes the store's zone orders as given and the
recorded tours as the batching decision. Modelling the split is possible
from the data in its outcome (one order per store-day and zone), but not its
exceptions; modelling the batching would need a rule the export does not
contain, and can only be calibrated against the inferred tours.

## R0.1 Inventory

### Workbook

| Sheet | Content |
|---|---|
| `OP Daten KW16_2026` | 65,291 order lines, 22 columns (below) |
| `TE pro h` | pivot per booking date: distinct orders and pallets, TU (formula), and the TU volume of the five disruptions |
| `S1` … `S5` | the warehouse's disruption lists: order numbers (S1, S2, S4, S5) or a template order (S3) |

### Columns of the line sheet [r01_columns]

| Column | Meaning | Type, unit | Missing | Distinct |
|---|---|---|---|---|
| NEU_DATUM | booking date | date | 0 | 6 |
| LFDNR | running line number | integer, unique | 0 | 65,291 |
| AUFTRAGSNR | order number; first six digits YYMMDD of its creation date | string | 0 | 1,525 |
| AUFTRAGSPOS | position within the order | integer | 0 | 144 |
| ARTIKELNR | article number; last four digits are the package code | string | 0 | 1,250 |
| ARTBEZ1 | article description | string | 0 | 913 |
| Q_PLATZ | pick location: site(3) area(2) aisle(2) house(2) place(2) level(2) | string, 13 digits | 0 | 1,016 |
| BEGINN_ZEIT, ENDE_ZEIT | line start and end | clock time, 1 s | 0 | 25,511 / 25,577 |
| PERS_NR | picker | string | 0 | 45 |
| LHM_TYP | load carrier type | code 38, 39, 40, 41, 43 | 0 | 5 |
| MENGE_IST | picked quantity | packages | 0 | 170 |
| GEWICHT_SOLL | target weight of the line | kg (assumed; median 9.3 per line) | 0 | 3,602 |
| VOLUMEN_KOLLI | volume of one package | unit not stated (median 13.8; consistent with litres) | 0 | 937 |
| TYP | line type | E 59,617, P 3,117, S1E 2,554, NP 3 | 0 | 4 |
| PALNR | pallet (load carrier) number | string | 0 | 2,760 |
| KUNDENNR | customer (store) | string | 0 | 403 |
| area, aisle, house, place | location parts, equal to the digits of Q_PLATZ in every line | integer | 0 | 3, 9, 50, 38 |
| Lücken Auftrag | formula: gap from this row's end to the next row's start if both rows have the same picker and order and the gap exceeds 10 s | duration | 56,758 | 668 |

The meaning of TYP and LHM_TYP codes is not documented in the export. The
master data with article dimensions (length, width, height), which the
submitted loader merged, is not part of this workbook and is not in the
checkout.

### Days [r01_days]

By pick date (the date holding most of an order's lines; finding 6). The last
two columns count by booking date, as the workbook does.

| Date | Weekday | ISO week | Lines | Orders | Stores | Articles | Locations | Pickers | Pallets | First line | Last line | Orders by booking date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-04-11 | Saturday | 15 | 10,136 | 259 | 137 | 997 | 891 | 25 | 419 | 05:54 | 12:22 | 259 |
| 2026-04-13 | Monday | 16 | 10,546 | 275 | 205 | 971 | 861 | 17 | 484 | 05:46 | 14:29 | 275 |
| 2026-04-14 | Tuesday | 16 | 13,480 | 291 | 180 | 1,019 | 894 | 19 | 547 | 05:40 | 14:29 | 322 |
| 2026-04-15 | Wednesday | 16 | 16,109 | 325 | 194 | 1,060 | 912 | 20 | 649 | 05:40 | 14:29 | 363 |
| 2026-04-16 | Thursday | 16 | 12,124 | 292 | 175 | 970 | 834 | 18 | 524 | 05:46 | 14:29 | 347 |
| 2026-04-17 | Friday | 16 | 2,896 | 83 | 72 | 565 | 500 | 7 | 137 | 06:17 | 14:29 | 106 |
| all | | | 65,291 | 1,525 | 403 | 1,250 | 1,016 | 45 | 2,760 | | | |

The pickers per day (25, 17, 19, 20, 18, 7) are the `pickers_per_day` of the
submitted configuration. Order numbers carry the creation date: 1,461 orders
were picked on their creation date, 64 on a later day (from Friday 10 April
onto Saturday, from Saturday onto Monday, and from the previous working day).

**Day labels (reviewer 2).** The submitted scripts number the days D1 to D6 in
date order (`make_results_scenarios_multirun.py`, `day_label = D{day+1}`), so
D1 is Saturday 11 April and Monday is D2, not D1. Their focus days are
consistent with this: Scenario 1 on day index 1 (Monday), Scenario 2 on 4
(Thursday), Scenario 4 on 3 and 4 (Wednesday, Thursday). Scenario 3 runs on
index 2 (Tuesday 14 April), while the warehouse dated its list S3 Wednesday
15 April. Any figure or text that counts Monday as day 1, or takes weekdays
from the workbook's pivot (which calls Saturday "Sonntag"), is off by one day.
The manuscript was not checked here.

### Values [r01_values, r01_categories]

| Measure | Median | 5 % | 95 % | Max |
|---|---|---|---|---|
| quantity per line [packages] | 12 | 6 | 72 | 2,800 |
| weight per line [kg] | 9.3 | 4.5 | 55.4 | 1,924 |
| volume per package | 13.8 | 8.0 | 21.4 | 127 |
| lines per order | 41 | 2 | 84 | 143 |
| pallets per order | 2 | 1 | 3 | 9 |

Lines by area: area 1 (aisles 1–6) 30,132, area 7 (aisles 7–8) 33,623, area 10
(aisle 10) 1,536. 97.8 % of lines are on storage level 01.

### From 1,525 to 1,422 to 1,419 orders [r01_paper_filter]

| Step | Orders | Lines |
|---|---|---|
| export | 1,525 | 65,291 |
| drop aisle 10 (`to_canonical`: aisles 1–8) | −101 with only aisle-10 lines | −1,536 |
| drop one article (`to_canonical`) | −2 with only that article | −450 |
| orders of the submitted model | 1,422 | 63,305 |
| drop historic span ≤ 0 (`valid_rows`) | −3 (single-line, start = end) | −3 |
| orders of Figure 8 | 1,419 | |

`valid_rows` also requires a match by `GreedyItemAssignment` and a route without
missing edges; that the three span-0 orders account for the whole difference
means every other order passed both. That part was not rerun (the derived
order stream of the submitted run is not in the checkout). No historic span is
negative, so the 24-hour correction of `build_historic_order_times` never
applies.

## R0.2 Who picks an order

### Pickers, zones and aisles per order [r02_orders_summary]

| Lines per order | Orders | 1 picker | 2 pickers | 3+ pickers | 1 area | 2+ areas | Aisles (mean) | Houses (mean) | Span, median [min] |
|---|---|---|---|---|---|---|---|---|---|
| 1–20 | 265 | 263 | 1 | 1 | 249 | 16 | 1.7 | 7.8 | 5.0 |
| 21–40 | 476 | 473 | 1 | 2 | 451 | 25 | 3.0 | 22.7 | 13.0 |
| 41–60 | 453 | 452 | 1 | 0 | 432 | 21 | 3.7 | 30.0 | 18.9 |
| 61–80 | 235 | 234 | 1 | 0 | 228 | 7 | 4.3 | 34.6 | 28.5 |
| >80 | 96 | 95 | 1 | 0 | 96 | 0 | 5.5 | 36.7 | 47.5 |
| all | 1,525 | 1,517 | 5 | 3 | 1,456 | 69 | 3.3 | 25.0 | 18.1 |

Of the 8 orders with several pickers, 5 have pickers whose intervals overlap
by more than 60 s (parallel) and 3 hand over (sequential); in none does each
picker take one zone. Orders spanning areas are mostly the 56 orders covering
areas 1, 7 and 10 together, 25 of them on Monday 13 April
[r02_order_areas_by_day].

### Store-days [r02_store_days]

| Store-days | 963 |
|---|---|
| with 1 order | 485 |
| with 2 orders | 403 |
| with 3 orders | 70 |
| with 4–6 orders | 5 |
| multi-order: one order per main area | 97 % |
| multi-order: one picker per order | 98.5 % |
| multi-order: orders overlapping in time | 39 % |

Main areas of the multi-order store-days: 1+7 in 368, 1+7+10 in 63, 7+10 in
23, 1+10 in 10. Single-order store-days are area 7 (368), area 1 (115) or
area 10 (2).

### Pickers and zones [r02_picker_zones]

83 of 106 picker-days stay in one area, 6 use two and 17 three; 87 have at
least 95 % of their lines in one area. Over the week 20 of 45 pickers work in
more than one area. Pickers are largely, not strictly, bound to a zone per
day.

### Interleaving [r02_interleaving, r02_concurrency]

| Open orders of a picker | Hours | Share of time with an order open |
|---|---|---|
| 1 | 455.0 | 89.0 % |
| 2 | 41.1 | 8.0 % |
| 3 | 14.2 | 2.8 % |
| 4 | 1.2 | 0.2 % |

A picker has no order open for 232.7 h of the 744 h between first and last
line (R0.3). Of 1,536 picker-order pairs, 1,163 are picked in one uninterrupted
run, 37 in two and 336 in three or more; 57 of 106 picker-days alternate between
orders, and 259 orders start while an earlier order of the same picker is
still open.

**Bearing on the model.** H1 of the plan (order span over several pickers) is
almost void: the order span is a single picker's span for 99.5 % of orders.
What the span does contain is the picker's other open orders (11 % of order
time) and the idle gaps inside the order (R0.3). The decision of Section 9 on
modelling zones and order splitting has this evidence: the WMS already splits
by zone into orders, and the model's unit, one order per picker, matches the
data's unit; what the model lacks is the store-day that ties the zone orders
of one store to one departure.

## R0.3 Picker timelines

Lines of each picker and pick date in time order; the gap before a line is its
start minus the previous line's end [r03_gaps].

| Measure | n | 5 % | 25 % | Median | 75 % | 95 % |
|---|---|---|---|---|---|---|
| line duration [s] | 65,291 | 1 | 8 | 15 | 28 | 71 |
| gap within an order [s] | 58,167 | 0 | 0 | 0 | 0 | 1 |
| gap at an order change [s] | 7,018 | −60 | −16 | −2 | 0 | 371 |

Negative gaps are lines that start before the picker's previous line ended:
4,639 lines, 3,096 by more than 10 s, 75 % of those at an order change
(interleaving, R0.2). 1,160 lines have zero length and 6,080 last 1 s.

### Gaps above a threshold [r03_gap_thresholds]

| Threshold | Gaps within an order | Orders split by them | Gaps at an order change | Share of order changes | Hours in gaps above |
|---|---|---|---|---|---|
| 1 min | 380 | 267 | 1,179 | 16.8 % | 270.8 |
| 2 min | 212 | 179 | 1,067 | 15.2 % | 264.0 |
| 5 min | 75 | 72 | 441 | 6.3 % | 220.4 |
| 10 min | 36 | 34 | 279 | 4.0 % | 199.0 |
| 15 min | 33 | 32 | 243 | 3.5 % | 190.7 |
| 30 min | 18 | 18 | 148 | 2.1 % | 150.3 |

Gaps inside an order are rare at every threshold: above 2 min only 0.36 % of
within-order gaps, splitting 179 orders. The threshold that defines an
observed tour (R1) is to be chosen by the user (Section 9); between 2 and 5 min
the count of within-order gaps falls from 212 to 75 while order changes with a
gap fall from 1,067 to 441.

### Breaks [r03_break_gaps]

Gaps longer than 15 min by the half hour in which they start:

| 05:30 | 06:00 | 06:30 | 07:00 | 07:30 | 08:00 | 08:30 | 09:00 | 09:30 | 10:00 | 10:30 | 11:00 | 11:30 | 12:00 | 12:30 | 13:00 | 13:30 | 14:00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 18 | 21 | 30 | 21 | 19 | 24 | 13 | 20 | 19 | 15 | 26 | 6 | 22 | 8 | 7 | 4 | 1 |

Medians per half hour lie between 16 and 50 min. 47 of 106 picker-days have a long gap
overlapping the configured break 8:00–8:30 and 39 overlapping 11:00–11:30.
Long gaps are spread over the day; whether they are staggered breaks, waiting
for work or other tasks is not recorded in the export.

### Presence and busy time [r03_presence, r03_presence_by_day]

| Measure (per picker-day) | 5 % | 25 % | Median | 75 % | 95 % |
|---|---|---|---|---|---|
| first line [clock] | 05:54 | 05:59 | 06:02 | 06:54 | 10:02 |
| last line [clock] | 11:37 | 12:20 | 14:20 | 14:29 | 14:30 |
| present, first to last line [h] | 4.2 | 5.4 | 8.0 | 8.5 | 8.5 |
| lines | 83 | 356 | 591 | 914 | 1,144 |
| orders | 5 | 9 | 13 | 20 | 26 |
| line time (union) / present | 0.16 | 0.53 | 0.69 | 0.75 | 0.87 |
| active (gaps > 5 min idle) / present | 0.16 | 0.59 | 0.78 | 0.87 | 0.94 |

| Date | Pickers | Present [h] | Line time, union [h] | Share | Active, 5 min [h] | Share |
|---|---|---|---|---|---|---|
| 2026-04-11 | 25 | 132.0 | 78.9 | 0.60 | 88.8 | 0.67 |
| 2026-04-13 | 17 | 126.1 | 72.8 | 0.58 | 82.5 | 0.65 |
| 2026-04-14 | 19 | 154.6 | 89.5 | 0.58 | 101.0 | 0.65 |
| 2026-04-15 | 20 | 159.4 | 111.3 | 0.70 | 124.0 | 0.78 |
| 2026-04-16 | 18 | 133.7 | 89.8 | 0.67 | 101.2 | 0.76 |
| 2026-04-17 | 7 | 38.4 | 22.7 | 0.59 | 26.3 | 0.68 |
| all | 106 picker-days | 744.2 | 465.0 | 0.62 | 523.7 | 0.70 |

Presence is measured from first to last line; time before the first and after
the last line of a shift is not in the data. The summed (not unioned) line
time is 68 % of presence; it double-counts concurrent lines. The paper's
utilisation of 47.75 % is the simulation's active over available picker time
(`make_results_scenarios_multirun.py`, `compute_summary_row`); it is a model
output, not a measurement, and R2.3 compares the two on one definition.

## R0.4 Timestamps [r04_timestamps]

| Check | Result |
|---|---|
| resolution | 1 s, every clock time HH:MM:SS |
| missing start or end | 0 |
| clock range | 05:40:27 to 14:29:59 |
| lines across midnight, end before start | 0 |
| zero-length lines | 1,160 |
| exact duplicates (all fields but LFDNR) | 0 |
| positions with more than one line | 650 (1,337 lines), 544 of them on more than one pallet |
| orders with lines on two booking dates | 147 (295 lines, all booked one day after the pick date) |
| workbook gap column reproduced from its definition | 99.9 % of rows |
| lines starting before the picker's previous line ended | 4,639 (3,096 by > 10 s) |

Positions with several lines are quantities split over pallets, not duplicates.
The 295 rebooked lines are discussed in finding 6; the analysis assigns them to
the pick date (`audit.with_pick_date`). The submitted model takes each line's
booking date (`build_historic_replay_stream`), so the lines of these orders
fall on two simulated days; their effect on the submitted runs is for R2.2.

## R0.5 Stores, trucks and transport units

### Stores and orders [r05_stores_units]

Every order belongs to exactly one store. A store has 1 to 6 orders per day
(485 store-days with one, 403 with two, 75 with three or more), which are its
zone orders (R0.2). 2,757 of 2,760 pallets carry one order; 216 pallets have
lines on two booking dates (the rebooked lines).

### Transport units [r05_pallets_per_day]

The workbook's pivot defines `TU = 2.6 × distinct orders` (cell formulas
`=D*2.6`), `TU per hour = TU / 7.5` and `TU per pallet = TU / pallets`.
Pallets per order range from 1.62 to 2.00 per day, so TU per pallet ranges from
1.30 to 1.61 by construction. A TU is not counted in the export. By pick
date:

| Date | Orders | Pallets | Stores | TU (2.6 per order) | Pallets per order |
|---|---|---|---|---|---|
| 2026-04-11 | 259 | 419 | 137 | 673.4 | 1.62 |
| 2026-04-13 | 275 | 484 | 205 | 715.0 | 1.76 |
| 2026-04-14 | 291 | 547 | 180 | 756.6 | 1.88 |
| 2026-04-15 | 325 | 649 | 194 | 845.0 | 2.00 |
| 2026-04-16 | 292 | 524 | 175 | 759.2 | 1.79 |
| 2026-04-17 | 83 | 137 | 72 | 215.8 | 1.65 |

(The pivot counts by booking date and so reports 837.2, 943.8, 902.2 and 275.6
TU for 14 to 17 April.)

### The disruption lists [r05_disruption_sheets]

| Sheet | Date | TU | Orders | TU / 2.6 | TU / 1.6 | Stores | Orders picked the day before | Lines | Pallets | Completed |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 "120 TUs vor 13 Uhr" | Mon 13.04 | 120 | 48 | 46.2 | 75 | 45 | 0 | 1,817 | 85 | 11:26–13:12 |
| S2 "100 TUs vor 13 Uhr" | Thu 16.04 | 100 | 38 | 38.5 | 62.5 | 37 | 3 | 1,731 | 81 | 11:21–13:07 |
| S3 "6 TUs +" | Wed 15.04 | 6 | template | 2.3 | 3.75 | 1 | | 53 | | |
| S4 "80 TUs vor 8.30 Uhr" | Tue 14.04 | 80 | 33 | 30.8 | 50 | 30 | 7 | 1,812 | 67 | 07:50–08:38 |
| S5 "200 TUs nach 10 Uhr" | Thu 16.04 | 200 | 76 | 76.9 | 125 | 60 | 9 | 3,464 | 165 | 10:02–11:59 |

The warehouse selected whole orders until their number reached TU / 2.6, among
orders completed shortly before the named hour (S1, S2, S4) or after it (S5).
Each list spans 30 to 60 stores. Lists S2, S4 and S5 include orders picked the
previous day whose rebooked lines carry the list's date. S3 is the 53 lines of
one existing store's order, marked in the pivot "Beispiel-Auftrag duplizieren"
(duplicate the example order), with no order number, picker or times.

### The submitted scenarios against the lists

| Paper | Submitted configuration | Warehouse list | Differences |
|---|---|---|---|
| Scenario 1 | Mon (day 1), 7:30, orders due ≥ 13:00 pulled to 10:00, 120 TU | S1 | `TruckDisruption` takes unstarted orders due ≥ 13:00 until 1.6 × orders ≥ 120; Monday has exactly 72 model orders due ≥ 13:00 and 72 × 1.6 = 115 < 120, so it can take all 72, the number reviewer 2 cites (the list: 48) |
| Scenario 2 | Thu (day 4), 10:30, due ≥ 13:00 to 11:00, 120 TU | S2 (100 TU) | TU 120 versus 100; the list is 38 orders of 37 stores; Thursday has 82 model orders due ≥ 13:00 and the event stops at 75 at most; the paper's 50 affected orders are for R2.2 to explain |
| Scenario 3 | Tue (day 2), 8:30, due 10:00, ⌈6 / 1.6⌉ = 4 generated orders of a new store | S3 (Wed) | day; generated orders versus the duplicate of one existing order |
| Scenario 4 | Wed (day 3), 6:30, Thursday orders due exactly 10:00 moved to Wednesday 10:00, 200 TU | S5 | due times are the end of the completion hour, so the hook selects the orders completed 9:00–10:00: exactly 20 model orders on Thursday (the list: 76 orders completed after 10:00) |
| — | not used | S4 | |

The ratios reviewer 2 questions, 120 TU to 72 orders and 200 TU to 20 orders,
are therefore produced by the hooks' selection rules together with the hourly
due buckets, not by the warehouse's data.

### Departures

The export has no truck, route, tour or departure field, and orders complete
in every hour from 6:00 to 14:30 without visible clusters
[r05_completion_hours]. The relation between
store and truck and the departure schedule cannot be derived; R3.1 needs the
departure plan per store from the user.

## Open questions for the user

1. Calibration and validation days (proposal: days 1 to 4) and the gap
   threshold defining an observed tour (R0.3 gives the trade-off).
2. Zones and order splitting (Section 9): the data shows zone orders of one
   store picked by different pickers in parallel. Model the store-day as the
   unit that ties them to one departure, or justify the order as the unit?
3. The departure plan per store (R3.1), and whether the warehouse's lists
   (S1, S2, S3, S5) are to replace the submitted hooks' order selection.
4. The meaning of TYP (E, P, S1E, NP) and LHM_TYP codes, and the unit of
   VOLUMEN_KOLLI.
5. The article master data (dimensions), needed for H2 (pick time by package
   size) and not in the checkout.
