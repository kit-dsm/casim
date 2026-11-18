# IJPE: the historic picking days of a grocery warehouse

## Question

How well does CASIM reproduce the six recorded picking days of the
warehouse's WMS export (Saturday 11 April and Monday 13 to Friday 17 April
2026), first with the recorded decisions (the replay, which isolates the time
model) and then with CASIM's own decisions (P1.3 of `docs/ijpe/plan.md`)? The
scenario is also the base of the disruption scenarios of P2.

## Data source

The data is confidential. The user places the WMS export and `layout.csv` in
`data/` (`data/SOURCE.md`); nothing at order, line or picker level is ever
committed. Without the data the scenario composes and checks but does not
run.

The scenario's `source.py` reads the export with `casim.sources.wms_export`,
builds the warehouse graph of `data/layout.csv` with
`casim.sources.grid_layout` (its cross-aisle edge matches the 15 m
aisle-change distance stated by the warehouse), and builds the state, the
recorded plan and the events with `casim.sources.recorded`. Days are assigned
by timestamp. Lines without a pick node in the layout (aisle 10, one location
in aisle 7) and orders picked by two pickers are left out and counted.

What the source emits for a day (P1.1):

- the layout, the storage of each article at its recorded pick nodes, and one
  picker per recorded picker with the time model of `source.speed`,
  `source.time_per_pick`, `source.tour_setup_time` and, optionally,
  `source.per_unit_s` (seconds per package);
- every order of the day, available at the shift start (5:30);
- presence (`source.presence`): `shift` (default) makes every picker present
  from the shift start; `recorded` makes a picker arrive at the setup start of
  its first recorded tour and leave, without return, after its last recorded
  line. A departure never interrupts a tour; work still queued for a picker
  that has left stays open;
- pauses: in chained runs (`source.pause_threshold_s` in seconds), a recorded
  gap between tours longer than the threshold is a `PickerDeparture` with the
  gap's duration, which begins when the running tour ends. Shorter gaps
  vanish, and a gap inside a tour is not modelled. The sweep
  `days_thresholds` runs 2, 5 and 10 minutes until the threshold is fixed. In
  anchored runs (`source.pause_threshold_s=null`, the default) there are no
  pauses; every tour starts no earlier than its recorded start instead. This
  isolates each tour's time error but uses future recorded starts, so it is a
  diagnostic, not a predictive simulation;
- the recorded plan (`information.recorded_plan`): a picker's lines in start
  order, cut where the picker has no order open; orders whose intervals
  overlap share a tour. The export does not show that this order is the
  physical walking order (`docs/ijpe/replay.md`).

## Decisions

`decisions/recorded.yaml` with `cosy_repo/recorded.yaml`: one decision when
the orders are in commits the recorded plan with `RecordedIA` (item
assignment at the recorded pick nodes), `RecordedTours` (batching),
`GivenSequence` (routing through the recorded line sequence) and
`RecordedPlan` (each tour to its recorded picker in recorded order). The time
model is the submitted one: 2,306 mm/s, 19 s per line, 215 s setup per tour.

Variants:

- `recorded_presence`: recorded presence with the 5 minute pause threshold;
  the decision also sees pickers not yet arrived (`off_shift`);
- the submitted assumptions, one at a time: `greedy_locations` (greedy item
  assignment), `ushape` (U-shape routing of each recorded tour),
  `one_order_per_tour`, `configured_breaks` (8:00 and 11:00); all three
  together in `submitted`;
- diagnostics, none of them a claim about the real process:
  `aisle_head_origin` and `first_pick_origin` (route origins instead of the
  unobserved depot), `end_time_sequence` (lines ordered by completion time),
  `setup_in_gap` (setup before the first recorded line), `interior_fit`,
  `same_aisle_fit`, `fixed_speed_pick_fit` and `end_time_fixed_speed_fit`
  (fitted pick times; their training samples are selected by duration, so a
  fitted speed is not a measured truck speed).

## How to run

```bash
python -m casim run scenarios/ijpe source.day=2026-04-15
python -m casim run scenarios/ijpe variants=recorded_presence source.day=2026-04-15
python -m casim run scenarios/ijpe sweep=days_thresholds
python -m casim run scenarios/ijpe sweep=variants
python -m casim run scenarios/ijpe sweep=anchored --output-dir scenarios/ijpe/outputs/anchored
python -m scripts.ijpe.trace_diagnosis scenarios/ijpe/outputs/anchored
python -m scripts.ijpe.replay
```

## Expected outputs

CASIM's usual outputs and the facts tables `facts/tours.csv`,
`facts/orders.csv` and `facts/picks.csv` go to `outputs/` (ignored by git).
The scripts in `scripts/ijpe/` read them and the export and write aggregate
comparisons to `outputs/`. The baseline keeps one record per day of the
default replay with scalar KPIs, counts and digests only
(`tests/baseline/ijpe_replay__<day>.json`).

## Reference result

The anchored replay of the six days with the submitted time model has an
order-span RMSE of 16.27 min on 1,414 retained orders. Most of this error
comes from long WMS line intervals and gaps inside orders that the export does
not label; correcting the line order, the tour origin or the pick-time
formula changes it by less than 1 min. The analysis and its limits are in
`docs/ijpe/replay.md` and `docs/ijpe/status.md`.
