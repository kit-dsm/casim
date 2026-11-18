# Henn waiting

## Question

How do Henn's (2012) waiting rule and an order-window strategy compare for online order batching
with one picker? The scenario is a study, not a benchmark: its instances differ from the
published ones (see Data source).

## Data source

Sixteen Henn instances in the format of Heßler and Irnich (2022), read by the library source
`hessler_irnich`, one file named by `source.filepath`, with Henn's arrival times from
`data/HennOnline/`. The item positions of these files differ from the published OOBP archive
(between 0 and 6 orders per file match), so results are not comparable with published OOBP
values; the `oobp` scenario runs the archive. `python scripts/fetch_data.py henn_waiting`
restores the files. The picker arrives with the first order; the order stream closes at the last
arrival. The orders have no due dates.

## Decisions

- `decisions/henn_waiting.yaml` (default): one OBRSP decision triggered by order arrival, stream
  closure and an idle picker; it sees the order buffer and its own planned tours, which it may
  rebuild, and commits one tour per decision. `cosy_repo/baseline_waiting_henn.yaml`
  combines greedy item assignment, FCFS batching, S-shape routing, SPT scheduling and Henn's
  waiting rule. `cosy_repo/repair_waiting_henn.yaml` adds incumbent-start batching as a second
  candidate, chosen by the ranker with a change penalty.
- `decisions/order_window.yaml`: waits until five orders are open, without a waiting component.
- `decisions/release_only_example.yaml`: a planning decision that keeps drafts and a second
  decision that commits drafts due within 300 s; an example of separating planning from
  commitment.

## How to run

```bash
python -m casim run scenarios/henn_waiting
python -m casim run scenarios/henn_waiting decisions=order_window
python -m casim run scenarios/henn_waiting cosy_repo=repair_waiting_henn +decisions.OBRSP.change_penalty=10
python -m casim run scenarios/henn_waiting +sweep=waiting_strategy_sweep
python scenarios/henn_waiting/plots.py
```

## Expected outputs

`scenarios/henn_waiting/outputs/run/` (one directory per combination of a sweep) with the
standard outputs. The sweep runs both strategies on fifteen instances; `plots.py` writes to
`outputs/plots/` a summary table (`summary.csv`, `summary_table.tex`), the makespan and the mean
picker utilisation per instance, and the makespan over the number of orders.

## Reference result

There is no published reference for these instances. The recorded regression values
(`tests/baseline/henn__*.json`) are, for the default instance `72s-100-75-0` (100 orders):

| Configuration | Makespan (s) | Mean order completion (s) | Tours | Distance |
|---|---:|---:|---:|---:|
| Henn waiting | 30,668.8 | 12,572.6 | 27 | 11,808 |
| Henn waiting with repair candidate | 29,969.8 | 11,980.6 | 26 | 11,289 |
| Release-only example | 29,015.8 | 10,895.5 | 24 | 10,695 |

The base version of CASIM gave 34,702 s with 25 tours for the Henn waiting configuration; the
differences are explained in `docs/validation.md`.

## Sources

1. K. Heßler and S. Irnich (2022), Modeling and Exact Solution of Picker Routing and Order
   Batching Problems, LM-2022-03.
2. S. Henn (2012), Algorithms for on-line order batching in an order picking warehouse,
   Computers & Operations Research 39(11), 2549-2563.
