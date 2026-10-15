# Phase B report

15 October 2026, casim `dd68452`, ware_ops_algos `0bd0a7e`. Phase B of
`docs/revision_scopes_corrections.md` (Section 8.2) is done. Details and evidence are in
`docs/literature_benchmarks.md` (sections named below), `docs/baseline.md` and
`docs/revision_scopes_status.md`. Work continues with Phase S while this report waits for an answer.

## Open points for the user

1. **L2, journal, high load: a difference from the published values without a demonstrated
   cause (Section 9).** Every journal FIRST cell with n = 90 or 120 is above the published
   value and outside its 95 % interval, by 2.7 to 13.7 % (C/90/45 +13.7 %, C/120/45 +12.8 %).
   At n ≤ 60 all eight FIRST cells lie within ±1.6 % and contain the published value. At
   C/120/45 the picker is busy 883 of 887 minutes, so the gap is in setup plus travel (about
   95 min, roughly 27 % of travel). CASIM's α = 0 and α = 1 results coincide at high load (887
   and 889), while the journal and the working paper publish 787 and 856 for the same class;
   against the working paper the high-load gaps are +3.9 to +7.6 %. This points to instances of
   the journal that differ from the working paper's, which neither paper states. Options:
   (a) accept it as a statistical non-reproduction of the journal's high-load cells and compare
   the high-load regime only with the working paper; (b) let me test candidate readings
   (the printed λ = 0.08625 scaled to other n, a different item-count distribution, the
   aisle-exit geometry) as diagnostics, without changing the protocol; (c) contact the author
   for instances or seeds.
2. **F64** (NearestNeighbourhood reported against walked distance on Kris) stays outstanding by
   your earlier direction.
3. **The decisions of Section 9.1** remain untouched: F46 (`spt_scheduling.yaml` due date),
   F49 (`Resources.get_features()`), F34 (per-picker processing time), the cards' objective
   vocabulary, the grocery scenario (S.4 deletes it unless you decide otherwise), order splitting,
   the scenario command line (Phase S builds only `casim run` and `casim check`), rolling horizon
   on the Henn instances, the stochastic waiting scenario's role, and `Resource.available` /
   `occupied`.

## Register status (Section 7.4)

| # | Benchmark | Kind | Status |
|---|---|---|---|
| L1 | Seven CASOP sets, static | exact | run: 415 rows (B.4, B.4a) |
| L2 | Henn (2012), online FCFS | statistical | reduced validation set run (B.5); the rest deferred and listed |
| L3 | OOBP, Gil-Borrás et al. (2020) | bound | run: 192 runs (B.5) |
| L4 | D'Haen et al. (2023) | bound | deferred with Phase 6 |
| L5 | Walking vs. Waiting, RADAR | reproduction | deferred with Phase 6 (needs the original 4D4L simulator) |
| L6 to L10 | Lorenz; Gil-Borrás 2024 and 2021; van Gils; D'Haen 2024 | bound | not pursued for now, with reasons in the register |

## L1, static sets (B.4, B.4a)

Three comparisons per run ("B.4 full stage", "B.4a"):

1. CASIM against the direct algorithm calls: equal on 310 rows (260 non-RR, 50 RR). The other
   105 rows are Kris NearestNeighbourhood rows whose difference equals the measured F64 gap.
2. Direct calls against CASOP v0.1.0: equal on all 415 rows; ware_ops_algos still computes what
   the published release computes.
3. Against the literature: no CASIM result below a proven optimum. The one direct result below
   a marked optimum (`instances_207_1_small`, 128,700 < 132,420) is F64; CASIM gives 133,020.

F64: across 3,672 Kris NN routes, 202 matrix-leg sums differ from the router's report (71,904
distance units in total), and 566 graph-edge sums differ. The literature gap of Kris total time
is 0.45 to 48.70 % with the walked distance, −2.81 to 48.70 % with the reported one. B.4a: RR now
returns its node sequence; on all 65 RR rows walked distance, graph-edge sum and reported
distance are equal, CASIM equals direct, direct equals CASOP, and the applicability sets of the
seven CASOP data cards equal `main`.

## L3, OOBP (B.5)

192 runs (64 Henn instances under both depots, 64 Albareda instances), kind bound
("B.5 full stage (L3)"). 40 runs are below at least one best-known value (30 completion,
11 turnover, one both) and carry the status **protocol not reproduced**, as you directed; all
192 audit files have no KPI recomputation difference.

## L2, Henn (B.5)

"B.5 L2 validation set": S-shape, FIRST and SAV, 20 replications, journal (α = 0) on 16
classes and working paper (α = 1) on its 8, with 95 % intervals and the run time per run
(23.8 s on average, 5 to 120 s by n).

- F67: without `NbrOrdersCondition` the working-paper C/30/45 FIRST cell drops from 798.8 to
  497.1 min (published 467). The remaining +6.4 % is below a bound: the expected last arrival is
  464.5 min and the last batch needs at least about 3.8 min, so 467 is unreachable under the
  stated arrival process. The journal cells are unaffected (identical traces with and without
  the gate).
- FIRST is Henn's first FCFS batch: shown by the code path and by
  `tests/verification/test_l2_first_selection.py` on all 480 recorded FIRST runs.
- Journal: 8 of 8 FIRST cells with n ≤ 60 match; n = 90 and 120 do not (open point 1).
- Working paper: Henn's published turnover is the arrival-to-completion time (47.1 against 44
  at C/30/45), not the start-based time its text states (about 18). Completion is within ±1.9 %
  at n = 60 and 3.9 to 9.4 % above elsewhere.
- SAV: 246 of 960 runs fail on F62 (no feasible candidate after a `ReplaceSuffix` with orders of
  a cancelled draft); Task 4.13 fixes it, and the SAV cells are to be rerun then.
- The Henn scenario (`henn_waiting` after Phase S) uses the same gate with Henn waiting and is
  likely affected by F67; Task 4.15 lists its baseline change.

## Baseline (7.6, B.7)

The 45 runnable inventory configurations, 415 L1, 192 L3 and 714 L2 runs are recorded with the
checks of B.2. Repeated recordings differ only in the commit fields. Every file is `consistent`
except the known epsilon `overlap` violations of Task 4.14 (51 L1, 118 L3 files) and F62's
unrunnable 61s-80-30-0 configurations. `docs/baseline.md` has the inventory.
