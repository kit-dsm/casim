# Synthetic fixed departure plan for the IJPE revision

The WMS export contains a store identifier for each order, but no truck,
route, departure time or loading record. The plan below is a fallback
experiment, not a reconstruction of warehouse transport. It answers the
reviewer's feasibility objection by fixing deadlines without using the
observed order completion time. It must be described as synthetic in the
manuscript and response letter.

## Rule fixed for the baseline and every disruption scenario

There are four daily dispatch slots at 10:00, 12:00, 14:00 and 16:00. This
uses the submitted model's two-hour spacing and end of dispatch at 16:00;
10:00 is the first slot so that all stores have at least four hours from the
05:30 shift start. A 30-minute staging lead time makes the picking deadlines
09:30, 11:30, 13:30 and 15:30. The lead time is an assumption, not a measured
value. The schedule has no asserted truck capacity because the export cannot
identify the order-to-truck allocation or a truck's capacity.

The first eight bytes of SHA-256 of the UTF-8 store identifier, interpreted as
an unsigned big-endian integer modulo four, select a slot in the listed order.
Every zone order of the same store uses that slot on every day. The rule reads
the store identifier only. It does not read picking start or end times, order
size, picker assignment, or the scenario outcome. A disrupted store keeps its
original slot except where the disruption explicitly changes its requested
departure. New stores use the same rule. The assignment is deterministic and
reproducible, but the hash is only a neutral allocation in the absence of
routes or delivery windows. The four slots should be called dispatch groups,
not observed trucks.

`python -m scripts.ijpe.departure_fallback` audits the rule on the same
located, single-picker cohort used by the IJPE replay. The table is a
descriptive WMS check, not a simulator result or an independent validation of
the dispatch assumptions. Tardiness is the positive difference between the
last recorded line end and the fixed picking deadline. The script prints only
day aggregates.

| Day, April 2026 | Orders | Late orders | Late share | Mean tardiness per order, min |
|---|---:|---:|---:|---:|
| 11 | 233 | 38 | 16.3% | 11.5 |
| 13 | 259 | 72 | 27.8% | 36.6 |
| 14 | 263 | 63 | 24.0% | 32.3 |
| 15 | 306 | 82 | 26.8% | 31.5 |
| 16 | 272 | 73 | 26.8% | 32.0 |
| 17 | 81 | 42 | 51.9% | 62.1 |

The deadlines are therefore binding in the recorded data and are not feasible
by construction. The high Friday late share remains visible. None of these
figures claims that the warehouse missed real departures; actual departures
were not supplied.

`variants/fixed_dispatch.yaml` puts these deadlines on source orders and uses
the selected five-minute pause threshold. A Friday 17 April smoke run completed
all 81 replay orders and had a simulated on-time rate of 54.3% against the
synthetic deadlines. This is a model result; the recorded Friday late share in
the table above is 51.9%. C5 must make departure accounting consistent in
orders and pallets before the fixed slots become truck departure events and
before departure delays or orders carried are reported. Compare the synthetic
rule with the submitted hourly bucket
and a stated tighter-margin variant using the same orders and simulation
policy. Keep the schedule and store assignment identical in all policy and
disruption runs so that changes in service have a common reference.

## Is there a stable store schedule in the data?

If each store had the same departure every day, its slot could be taken from
the other days (leave one day out) instead of from a hash. The recorded week
does not support this. Store-day completion is the last line end of a store's
orders on that day; aggregates over 1,414 orders and six days:

- The rank correlation of the stores' completion order between two days is
  0.13 to 0.57, mostly 0.2 to 0.5. The store explains about 30 % of the
  variance of its completion time; the time varies by about 2 h within a store.
- A store's typical slot from the other days (median completion, rounded up to
  the next deadline) equals its slot on the held-out day in 32 % of 867
  store-days, against 26 % by chance (76 % within one slot).
- Deadlines from the other days leave many recorded orders late even with a
  margin: the latest completion on the other days plus 60 min leaves 17.9 %
  late and puts 38 % of the deadlines after the shift end; the median plus
  30 min leaves 33 %. Their rank correlation with the recorded pick start is
  0.28 to 0.33.

A store's departure may differ by weekday, which one week cannot show. The
WMS order numbers carry a store sequence that is almost the same every day
(rank correlation 0.6 to 1.0 between days), unrelated to store number, order
size and location and only weakly related to the pick start (0.1 to 0.2);
whether it follows the tour plan is a question for the partner.

## Deadline tightness (P3.2)

Reviewer 2 asks for the sensitivity of the results to the due-time definition,
derived from staging and departure. The hash plan stays the main rule, and its
tightness is varied with `source.due_dates.profile` in
`scenarios/ijpe_week/scenario.yaml`. Both profiles order the stores the same
way (the hash read as a number in [0, 1)), cut them at other points, and keep
the 30 minutes of staging.

| Profile | Share of stores at 10, 12, 14, 16 h | Basis |
|---|---|---|
| loose (main) | equal (hash modulo four) | the submitted two-hour spacing |
| central (sensitivity) | 0.39, 0.26, 0.20, 0.15 | the area's recorded orders done by 09:30, 11:30, 13:30 and later, six days |

The central profile matches the recorded completions only in aggregate; a
store's slot still does not depend on when its orders were picked. A third
profile from the capacity workbook's warehouse-level departures (0.45, 0.42,
0.10, 0.03) was tried and dropped: it leaves a third of the orders late even on
the light days of the simulated week, which contradicts a warehouse that
mostly meets its trucks.

**Fit to the record, not lateness.** Applied to the WMS completion times, the
loose deadlines leave 26.2 % of the orders after their deadline and the
central ones 37.1 %. The warehouse worked to its real departures, which are
not known, and a store's synthetic slot is unrelated to them, so these shares
measure how a plan that is independent of the record fits the record. They
are not the warehouse's lateness and not a benchmark for the simulation.

**Simulated baseline** (week, EDD, seed 0, the roster without separate
breaks; late includes orders still open at the end of the stream):

| Profile | Sat | Mon | Tue | Wed | Thu | Fri | Week |
|---|---:|---:|---:|---:|---:|---:|---:|
| loose | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0.0 % |
| central | 0 % | 0 % | 0 % | 1 % | 1 % | 0 % | 0.5 % |
| bucket (submitted rule) | 17 % | 1 % | 1 % | 6 % | 7 % | 0 % | 5.9 % |

Like the recorded warehouse, the simulated one finishes every day's orders on
the day and meets the synthetic deadlines. Its lines done by 10:00, 12:00 and
14:00 (Monday to Thursday) are 1.06, 0.99 and 1.06 times the recorded ones. An
earlier roster subtracted fixed breaks from picker-hours that already
contained them; it fell behind (0.91 by 12:00), carried 18 orders into
Thursday and 38 into Friday, and made Friday 96 % late. The submitted hourly
bucket leaves each order 0 to 60 minutes after its recorded completion, so
every order the simulation finishes later than the record is late; Saturday's
17 % follows from 12 pickers for a day the record worked with 21 logins and
finished by 12:23. In the revised model the bucket is a reference for the
policy ranking, not a level.

**How results are reported.** Lateness levels say nothing about the real
warehouse. The experiments report paired differences against the baseline of
the same seed and profile (affected and unaffected orders), for deadlines
inside the validity envelope (P1.7), and check whether the ranking of the
policies is the same under loose, central and the submitted hourly bucket.
The staging time is an assumption; if the manuscript's Section 4.1 or the
partner gives one, it replaces `staging_minutes`.
