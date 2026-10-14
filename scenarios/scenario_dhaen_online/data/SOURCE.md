# D'Haen, Braekers and Ramaekers (2023): OSF instances and results

Source: OSF project "Integrated scheduling of order picking operations under dynamic order arrivals", <https://osf.io/7r4gn/>, doi [10.17605/OSF.IO/7R4GN](https://doi.org/10.17605/OSF.IO/7R4GN), licence CC BY 4.0 (stated on the OSF project), first downloaded 2026-10-14. Citation: D'Haen, R., Braekers, K., Ramaekers, K. (2023), Integrated scheduling of order picking operations under dynamic order arrivals, International Journal of Production Research 61(10), 3205-3226, doi [10.1080/00207543.2022.2078747](https://doi.org/10.1080/00207543.2022.2078747).

The data is not committed. Restore it with

    python scripts/fetch_data.py dhaen

which downloads each file below into `osf/` (paths below `Instances and results OIBRSP/` preserved) and checks it against `SHA256SUMS.txt`; every SHA-256 equals the hash OSF reports.

| File | OSF download | Bytes |
|---|---|---:|
| `osf/README.txt` | <https://osf.io/download/h4ys6/> | 3,625 |
| `osf/Distance matrix/DistanceMatrix_12SubAisles.txt` | <https://osf.io/download/4p9jn/> | 2,060,948 |
| `osf/Online problem/Instances/Instances.zip` | <https://osf.io/download/5vfs9/> | 27,363,361 |
| `osf/Online problem/Results online problem instances.xlsx` | <https://osf.io/download/d73kr/> | 1,706,117 |
| `osf/Online problem/Parameter tuning/Instances.zip` | <https://osf.io/download/nru89/> | 856,267 |
| `osf/Online problem/Parameter tuning/Results parameter tuning.xlsx` | <https://osf.io/download/nc7gp/> | 516,560 |

Not fetched because of their size; they stay at OSF: `Distance matrix/DistanceMatrix_24SubAisles.txt` (36.0 MB), `DistanceMatrix_36SubAisles.txt` (198.7 MB), `Online problem/Detailed results/Detailed results with order anticipation.zip` (121.9 MB) and `... without order anticipation.zip` (122.1 MB), and the static problem's `Instances.zip` (55.7 MB, the instances of van Gils et al. 2019), `Results ILS.zip` (86.4 MB) and `Results LNS.zip` (86.4 MB).

`Instances.zip` holds 1,920 files: an order list and an order-line list for each of 960 instances (instance numbers 1 to 96, replications 21 to 30). The results workbook has 5,760 rows, one per instance and solver setting (staging lanes 20, 40 or 60 %, forecasting off or on). Distances in the matrix are in decimetres (OSF `README.txt`). The paper is placed by the user in `docs/research/dhaen/` and is not committed.
