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
