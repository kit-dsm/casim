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
  line; `arrival` makes a picker arrive as in `recorded` but never leave. A
  departure never interrupts a tour; work still queued for a picker that has
  left stays open;
- pauses: in chained runs (`source.pause_threshold_s` in seconds), a recorded
  gap between tours longer than the threshold is a `PickerDeparture` with the
  gap's duration, which begins when the running tour ends. Shorter gaps
  vanish. With `source.stops_in_tours`, a stop inside a tour over the same
  threshold (a gap between lines, or a line's excess over 19 s) is a timed
  absence too, beginning when the tour ends; otherwise stops inside a tour
  are not modelled. `source.absences` says where the absences come from:
  `recorded` (these pauses and stops), `fitted` (scheduled breaks and
  allowance drawn from a model fitted on the other study days,
  `casim.sources.absence`, with replication seed `source.absence_seed`; the
  roster stays recorded) or a list of `[start_s, end_s]` (fixed breaks for
  every picker). Every absence is a timed `PickerDeparture`. The sweep
  `days_thresholds` compares 2, 5 and 10 minutes. The selected threshold is
  5 minutes; `variants/recorded_presence.yaml` uses it. In
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

- `calibrated`: the calibrated day of the system-level validation and the
  base of the disruption scenarios: recorded batches, U-shape routing and
  FIFO assignment as a pull (`decisions/pull.yaml`: OBRSP makes the
  recorded tours into pick lists, routes and schedules them once and commits
  nothing, ORSP keeps their routes and gives the next one to each picker that
  becomes free), recorded presence, five-minute pauses and stops inside
  tours, setup inside the recorded gaps and 16.26 s per line
  (`scripts/ijpe/line_time.py`); with `decisions=recorded
  cosy_repo=recorded_ushape` each tour is pushed to its recorded picker.
  `python -m scripts.ijpe.system_validation` compares it with the WMS (`docs/ijpe/validation.md`; the run commands are
  in its docstring);
- `recorded_presence`: recorded presence with the 5 minute pause threshold
  (the recorded decision, `decisions/recorded.yaml`, sees every picker,
  also those not yet arrived or away);
- `system_fifo` (P1.3): recorded batches with CASIM's U-shape routing and
  FIFO assignment, recorded presence and the 5 minute pauses, with the
  submitted time model (the diagnostics of `docs/ijpe/validation.md`);
- `submitted`: the submitted model's assumptions, U-shape routing, one order
  per tour and fixed breaks at 8:00 and 11:00;
- `fixed_dispatch`: synthetic store departure slots as due dates.

The diagnostic variants of R1 (route origins, completion-time line order,
fitted pick times, the submitted assumptions one at a time) were removed on
18 November; their findings are in `docs/ijpe/replay.md` and
`docs/ijpe/validation.md`, their files in git history.

## How to run

```bash
python -m casim run scenarios/ijpe source.day=2026-04-15
python -m casim run scenarios/ijpe variants=recorded_presence source.day=2026-04-15
python -m casim run scenarios/ijpe variants=fixed_dispatch source.day=2026-04-15
python -m casim run scenarios/ijpe sweep=days_thresholds
python -m casim run scenarios/ijpe sweep=anchored --output-dir scenarios/ijpe/outputs/anchored
python -m scripts.ijpe.replay
python -m scripts.ijpe.current_state
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
formula changes it by less than 1 min. The current package scope, input,
release rule, picker-day finish errors and matched-tour work errors are in
`docs/ijpe/current_state.md`. Earlier diagnostics and their limits are in
`docs/ijpe/replay.md` and `docs/ijpe/status.md`.
