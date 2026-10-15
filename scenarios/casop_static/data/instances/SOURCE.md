# Static CASOP benchmark instances

The 355 files listed in `SHA256SUMS.txt` are the distinct instances selected by
`docs/literature_benchmarks/l1_runlist.csv`. Run
`python scripts/fetch_data.py casop_instances` to download and verify them.
The downloads were checked on 14 October 2026. Source archive SHA-256 values
are pinned in `scripts/fetch_data.py`; this directory commits no instance data.

| Sets | Public source |
|---|---|
| SPRP, SPRP-SS, Bahceci–Öncan, Henn–Wäscher, Muter–Öncan | [Heßler–Irnich Mainz archive](https://download.uni-mainz.de/fb03-logistikmanagement/benchmarks/instances_SPRP-SS_JOBPRP.zip) |
| SPRP-SS | [Heßler SPRP-SS repository](https://github.com/katrinhessler/SPRP-SS) |
| Foodmart | [Foodmart benchmark archive](https://pagesperso.g-scop.grenoble-inp.fr/~cambazah/batching/data/FoodmartData.zip) |
| Kris small and large | [Corrected small](https://pagesperso.g-scop.grenoble-inp.fr/~cambazah/sequencing/data/KrisSmallDataCorrected.zip) and [corrected large](https://pagesperso.g-scop.grenoble-inp.fr/~cambazah/sequencing/data/KrisLargeDataCorrected.zip) archives |

The Henn files retain the uniform or class-based policy in their target
directory. `MuterOencanWG` follows the public ware_ops_pipes preparation:
change the two cell distances from 1 to 1.5 and decode the source text as
Windows-1252 before writing UTF-8. The selected file checksums pin the exact
prepared bytes. Licensing follows each source archive or repository; see
`docs/literature_benchmarks.md` for the benchmark citations and terms.
