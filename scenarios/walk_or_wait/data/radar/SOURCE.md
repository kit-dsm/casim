# RADAR: Benchmark Instances for Manual Warehouse Order Picking

Source: Barlang, M., Lehmann, T., Furmans, K. (2026), Benchmark Instances for Manual Warehouse Order Picking, KIT RADAR, doi [10.35097/mwsv59v8sk9sqaan](https://doi.org/10.35097/mwsv59v8sk9sqaan), <https://radar.kit.edu/radar/en/dataset/mwsv59v8sk9sqaan>, licence CC BY 4.0, archive dated 2026-04-29, downloaded 2026-10-14. Companion paper: Barlang, Lehmann, Furmans, Walking vs. Waiting: Performance Impact of Waiting Strategies in Manual Warehouses.

The data is not committed. Restore it with

    python scripts/fetch_data.py radar

which downloads the BagIt archive from <https://radar.kit.edu/radar-backend/archives/mwsv59v8sk9sqaan/versions/1/content> (2,002,430,464 bytes, SHA-256 `5d866dae22333c1e0b0dc3277e68dfe5ee8ab7d0267dd9fe3aa60713d9c8b93f`; its MD5 `213b4c49ba3c744ab5a584ac30372f6e` is the archive checksum the RADAR page shows) into `.download/`, opens `data/dataset/Instances.tar` inside it, and extracts the 118 files listed in `SHA256SUMS.txt` into `selected/`, checking each one: the 8 × 16 layout pickle, its random storage assignment, the 100 standard-case order files, the dataset's `Readme.txt` and the result tables under `Results/` (CSV and XLSX). The dataset's `Thumbs.db` files and an Office lock file (`~$summary_by_folder_20250704_230527.xlsx`) are not extracted.

`raw/` may hold an earlier local download of the same dataset in another container (`mwsv59v8sk9sqaan.tar`); the scenario does not use it.
