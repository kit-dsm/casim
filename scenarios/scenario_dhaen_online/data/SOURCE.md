# D'Haen, Braekers and Ramaekers (2023): OSF instances and results

Source: OSF project "Integrated scheduling of order picking operations under dynamic order arrivals", <https://osf.io/7r4gn/>, doi [10.17605/OSF.IO/7R4GN](https://doi.org/10.17605/OSF.IO/7R4GN), licence CC BY 4.0 (stated on the OSF project). Paper: D'Haen, R., Braekers, K., Ramaekers, K. (2023), Integrated scheduling of order picking operations under dynamic order arrivals, International Journal of Production Research 61(10), 3205-3226, doi [10.1080/00207543.2022.2078747](https://doi.org/10.1080/00207543.2022.2078747).

Downloaded on 2026-10-14 through the OSF API into `osf/`, paths below `Instances and results OIBRSP/` preserved. Every SHA-256 equals the hash OSF reports for the file.

| File | Bytes | SHA-256 |
|---|---:|---|
| `osf/README.txt` | 3,625 | `c95c1b52df544c10747b68a556d90852a6064ab8ba4debacce207534a2b35b33` |
| `osf/Distance matrix/DistanceMatrix_12SubAisles.txt` | 2,060,948 | `83eab8a7781b683276aa00b88184975821aaf02053b618a7b5c1d8b55b721d79` |
| `osf/Online problem/Instances/Instances.zip` | 27,363,361 | `37f0ebaf4d5e955c1feb685e3c12318c96d6447b23b1fc7d71b29fe0aa5f7fbd` |
| `osf/Online problem/Results online problem instances.xlsx` | 1,706,117 | `8cd8174132dbbf262f6f0c4961bf3c9ca3a3af2c37d9e8ab0c86840835c6ccbf` |
| `osf/Online problem/Parameter tuning/Instances.zip` | 856,267 | `b1f5b16d3c589bc00017c209612b73696a2da5f97757c82b195ee148b29cd402` |
| `osf/Online problem/Parameter tuning/Results parameter tuning.xlsx` | 516,560 | `37fff3236247062acd10f758ea4f9bb742721f33220a2ac0e0d4e92e2b95fd4f` |

Not downloaded because of their size; they stay at OSF: `Distance matrix/DistanceMatrix_24SubAisles.txt` (36.0 MB), `DistanceMatrix_36SubAisles.txt` (198.7 MB), `Online problem/Detailed results/Detailed results with order anticipation.zip` (121.9 MB) and `... without order anticipation.zip` (122.1 MB), and the static problem's `Instances.zip` (55.7 MB, the instances of van Gils et al. 2019), `Results ILS.zip` (86.4 MB) and `Results LNS.zip` (86.4 MB).

`Instances.zip` holds 1,920 files: an order list and an order-line list for each of 960 instances (instance numbers 1 to 96, replications 21 to 30). The results workbook has 5,760 rows, one per instance and solver setting (staging lanes 20, 40 or 60 %, forecasting off or on), with columns including `OrderPickTime`, `Tardiness`, `Makespan` and `TotalSolutionTime`. Distances in the matrix are in decimetres (OSF `README.txt`).
