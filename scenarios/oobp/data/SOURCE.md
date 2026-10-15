# OOBP instances (Henn-Wäscher and Albareda-Sambola)

Source: the [official OOBP instance archive](https://grafo.etsii.urjc.es/optsicom/oobp/oobp-files/oobp_instances.zip) of the [OOBP benchmark page](https://grafo.etsii.urjc.es/optsicom/oobp.html), first downloaded 2026-10-13, archive SHA-256 `c8c7e1096ef515d373c770a705ad8aeda975d5874cafcefab2fdb4dcea29e20d`. Citation requested by the page: Gil-Borrás, S., Pardo, E. G., Alonso-Ayuso, A., Duarte, A. (2020), GRASP with Variable Neighborhood Descent for the online order batching problem, Journal of Global Optimization 78, 295-325, doi 10.1007/s10898-020-00910-2. Licence: none stated.

The data is not committed. Restore it with

    python scripts/fetch_data.py oobp

which downloads the archive into `.download/`, checks its SHA-256, and extracts the files listed in `SHA256SUMS.txt`, checking each one:

- the 194 Henn entries of `OOBP_instances/henn/` at the top level of this directory: `abc1/`, `abc2/`, `ran1/`, `ran2/` (16 order files, 16 arrival files and 16 settings per group), `README.txt` and `description.pdf`;
- the 193 files of `OOBP_instances/albareda/` below `albareda/`: warehouses `W1` to `W4`, order counts 100, 150, 200 and 250, instances `000`, `030`, `060` and `090`, each with a layout, an order and an arrival file, plus `README.txt`. That is 64 instances. The benchmark page states 80 Albareda instances and order counts from 50; the archive has no 50-order folder, and Gil-Borrás et al. (2020, Section 4.1) and `experiments.xlsx` use 64.

The Henn files were committed in `6eaf5ee` and removed from the tree again under the rule of Section 3.1; the restored files are byte-identical to them. The converted Heßler-Irnich files of the `henn_waiting` scenario are a different representation (F60, `docs/oobp_source_comparison.md`). OOBP reproductions use the archive files and their depot convention.
