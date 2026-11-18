# Recorded-replay verification scenario

The smallest replay of a recorded picking day (`docs/ijpe/plan.md`,
R1.3): two pickers, five recorded lines, one interleaved tour, one pause, every
start and end derived by hand. The source is the scenario's own `source.py`
over the library module `casim.sources.recorded`, which the IJPE replay uses
on the warehouse's data.

## The case

A conventional layout of 2 aisles with 4 pick locations, unit edges; the depot
start node (-1, -1) and end node (-2, -1) each join the front of aisle 1 at
(1, 0) with length 1. Speed 1, pick 2 s, setup 5 s, shift start 0. Recorded
lines (picker, order, location, start-end):

| Picker | Order | Location | Recorded |
|---|---|---|---|
| 0 | 1 | (1, 2) | 10–20 |
| 0 | 2 | (1, 3) | 20–24 |
| 0 | 1 | (1, 4) | 24–30 |
| 0 | 3 | (2, 1) | 100–110 |
| 1 | 4 | (2, 2) | 40–50 |

Order 2 lies inside order 1's interval, so orders 1 and 2 form one recorded
tour with the sequence (1, 2), (1, 3), (1, 4). Order 3 is picker 0's second
tour, after a 70 s gap; order 4 is picker 1's only tour. With the pause
threshold of 50 s the gap is a pause from 30 to 100. Both pickers are present
from 0; their first tours have binding recorded starts at 10 and 40. One
decision at t = 0 commits the recorded plan.

## Expected values (derived by hand)

- Picker 0, orders 1 and 2: paused until 10, setup 10–15. Walk (-1,-1) →
  (1,0) → (1,1) → (1,2): 3, pick 18–20; walk 1 to (1,3), pick 21–23; walk 1 to
  (1,4), pick 24–26; back (1,4) → (1,0) → (-2,-1): 5, end 31. Distance 10.
- Picker 1, order 4: paused until 40, setup 40–45. Walk (-1,-1) → (1,0) →
  (2,0) → (2,1) → (2,2): 4, pick 49–51; back 4, end 55. Distance 8.
- Picker 0, order 3: the 70 s departure is emitted at 30 while the first
  tour is still running. It takes effect at 31 and returns the picker at 101;
  setup 101–106, walk 3 to (2,1), pick 109–111, back 3, end 114. Distance 6.

Orders complete with their tours: 1 and 2 at 31, 4 at 55, 3 at 114. Variant
`chained` (threshold 100 s) has no pause between picker 0's tours: order 3's
tour sets up at 31–36, picks 39–41 and ends at 44. Variant `anchored` holds every
tour to its recorded start through the plan instead of breaks, so order 3 starts
at 100 rather than 101. `tests/verification/test_recorded_replay.py`
checks the pick, tour and order tables against these values.

## Running

```bash
python -m casim run scenarios/recorded_replay
python -m casim run scenarios/recorded_replay variants=chained
```
