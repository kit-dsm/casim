# Original OOBP Henn instances

The files under `abc1/`, `abc2/`, `ran1/`, and `ran2/`, together with `README.txt` and `description.pdf`, are the 194 unchanged Henn entries from the [official OOBP instance archive](https://grafo.etsii.urjc.es/optsicom/oobp/oobp-files/oobp_instances.zip), downloaded 2026-10-13. The archive also contains Albareda instances, which are not part of the Henn experiments and are not copied here. Paths inside the Henn directory are preserved.

Archive SHA-256: `c8c7e1096ef515d373c770a705ad8aeda975d5874cafcefab2fdb4dcea29e20d`.

`SHA256SUMS.txt` gives a SHA-256 checksum for each extracted file. The existing converted Heßler-Irnich files remain in the parent `data/` directory and are used by the existing local Henn scenario. OOBP reproductions must use these archive files and their source depot convention.

## Albareda-Sambola instances (Task B.3)

`albareda/` holds the `OOBP_instances/albareda/` tree of the same archive (SHA-256 above), extracted on 2026-10-14 with paths preserved: four warehouses `W1` to `W4`, order counts 100, 150, 200 and 250, and instances `000`, `030`, `060` and `090`, each with a layout file, an order file and an arrival file, plus the archive's `README.txt`. That is 64 instances (193 files). The benchmark page states 80 Albareda instances and order counts from 50; the archive has no 50-order folder, and Gil-Borrás et al. (2020, Section 4.1) and `experiments.xlsx` use 64.

The benchmark page states no licence for the archive; it asks only that Gil-Borrás et al. (2020) be cited. The Albareda files are therefore kept out of Git (`.gitignore`) and live only in this local directory; `ALBAREDA_SHA256SUMS.txt` lists the SHA-256 of every file so that a re-download can be checked. To restore them, download the archive from the URL above and extract `OOBP_instances/albareda/` here.
