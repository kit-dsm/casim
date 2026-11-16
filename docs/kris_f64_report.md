# F64: Kris layout loading

Status: root cause established on the 243 instances of `KrisSmallDataCorrected` (SHA-256 `cf5ea2a7…`, the archive `scripts/fetch_data.py` fetches). `KrisLargeDataCorrected` is not yet checked. Scripts: `raw.py` (index-based reader), `loader_check.py` (original and corrected loader, all routers), `solutions.py` (published solutions), `exact207.py` (enumeration of `instances_207_1`).

## Findings

The instance files are consistent. Arcs are symmetric except the one-way legs out of the departure depot and into the arrival depot. The `//LocStart` section is complete and equals the undirected shortest-path distances over the arcs, except for the zero distance between the two depots, which share a position. All 1,110 batch distances of the published small solutions (`allSolutions.zip`) are reproduced exactly as the sum of these shortest paths between consecutive pick locations, from the departure to the arrival depot, so this is the reference metric. The published batch completion times equal setup time plus travel time per distance unit times distance plus pick time per order line.

In the files, products of aisle k lie at x = 48k and the intersections of aisle k at x = 48k + 24 (offset 24 in all 243 files). `IBRSPLoader._parse` (ware_ops_pipes `554a153`) moves every intersection to the nearest x at which the instance has a product. The graph and the distance matrix are keyed by coordinates. In an instance where an aisle holds no product, its intersections are moved onto a neighbouring aisle, several vertices become one node, and the matrix keeps the entry of the vertex written last. This happens in 166 of 243 instances, merges 3,270 vertices and leaves 103,278 wrong matrix entries; entries between coordinates that stand for one vertex are all correct. In `instances_31_1`, the products occupy aisles 0 to 3 of 6, so intersections 41/44/47, 42/45/48 and 43/46/49 share x = 144, and the entry (96,1022)→(144,1022) holds the distance from vertex 40 to vertex 49 (144) instead of vertex 43 (48).

Effect on single-order routes (2,916 per router):

| Router | Reported ≠ executed, original loader | Reported too high, original loader | Total reported, original | Total reported, corrected |
|---|---|---|---|---|
| NearestNeighbourhood | 243 | 8 | 8,895,036 | 8,893,980 |
| SShape | 212 | 247 | 12,210,692 | 12,168,740 |
| LargestGap | 97 | 239 | 10,701,860 | 10,635,044 |
| Midpoint | 96 | 239 | 10,717,656 | 10,650,648 |
| Return | 212 | 247 | 12,831,090 | 12,789,138 |

"Executed" is the sum of matrix entries between consecutive route nodes, which CASIM charges. NearestNeighbourhood reports the shortest-path distance of its pick sequence, except on eight routes whose first leg from the depot reaches a merged aisle-0 intersection (96 too long); CASIM's executed distance is wrong on 243 routes. The aisle-based routers read merged intersections as waypoints and report too much. With the corrected loader, reported and executed distance are equal on every route of every router. Batched routes visit more aisles; their error has not been measured. Savings and local-search batching use router distances as costs and can change their batches.

`instances_207_1` is not a case of F64. The minimum total picking time over all capacity-feasible partitions with exact routing is 118,044 without due dates and 132,420 with every order finished by its due date on two pickers, which is the published optimum. CASOP's 128,700 therefore violates a due date. The manuscript already compares Kris against the BKS only on instances with a deadline-feasible pipeline (272 of 295) and treats due dates as soft for the SBS analysis; CASIM's runner and docs do not.

## Fix

The loader has to map vertices to coordinates one to one, apart from vertices at distance zero. The ware_ops_algos heuristics need each aisle to be one vertical line through its pick nodes, so the intersections of aisle k move to the x of aisle k's product column. That x is read from the aisle number in the vertex label: the products' x where aisle k has products, otherwise the intersection's x minus the offset, which the loader derives from aisles that have both and requires to be unique. The loader then checks two invariants and fails if either is violated: no two vertices at positive distance share a coordinate, and every matrix entry equals the file's shortest path between the vertices of the two coordinates. A test runs both invariants on every Kris instance. The product coordinates, and with them the storage locations, stay as in the file.

`LayoutParameters` receives `n_aisles = int(max x)` (264 for six aisles) and `n_pick_locations` equal to a y coordinate (1022). Neither is a count. Whether any Kris-applicable algorithm or card reads them has not been checked; they should be counts or absent.

The loader lives in ware_ops_pipes, which agents may not change. CASIM's `casop` and `kris` sources import it, so CASIM has the same error.

## Changes to results and docs

For the manuscript, Table 3 (Kris row) and the Kris rows of the SBS table depend on the loader and need a rerun with the corrected loader in ware_ops_pipes. The aisle-based routers and every batching method that uses their costs are affected most; pipelines with NN routing much less, but Savings(NN), the Kris SBS for total picking time and makespan, uses NN distances as savings and has not been measured. The large set has to be checked first.

In CASIM, the following text is wrong after this analysis and needs rewriting in place:

- The plan's B.4 bullet and the F64 row of the appendix state that NearestNeighbourhood under-reports and that the walked distance is correct. The cause is the loader; NN's reported distance is the correct one; the fix is in the loader.
- The B.4 stop rule and Section 9 exempt differences "where F64 explains it". With the corrected loader, comparison 1 requires equality on Kris as on every other set, and the exemption is removed.
- The Kris check against the literature compares only solutions that meet every due date, as in the manuscript, and reports infeasible solutions separately. A result below the optimum is a stop only if it is deadline-feasible. The status entry that explains `instances_207_1` by F64 and the Phase B report's gap range of −2.81 to 48.70 % with the reported distance are wrong on this point; negative Kris gaps come from deadline violations.
- The register entry for Kris states the reference metric (shortest paths between consecutive pick locations from the file, verified on all published small solutions) and the objective (total picking time under hard due dates, Briant et al. 2023).
- The 21 Kris `overlap` violations of the L1 baselines are attributed to F64. After the fix, they either disappear or have another cause; the Kris L1 baselines are recorded again.
- The design doc's description of sources states the loader invariant: a layout's coordinates identify its vertices.
