# CASOP v0.1.0 results

Source: release `v0.1.0` of [kit-dsm/ware_ops_pipes](https://github.com/kit-dsm/ware_ops_pipes/releases/tag/v0.1.0), asset `casop-results-v0.1.0.parquet` (<https://github.com/kit-dsm/ware_ops_pipes/releases/download/v0.1.0/casop-results-v0.1.0.parquet>), downloaded 2026-10-14. It holds CASOP's result per instance and pipeline for the seven benchmark sets, produced by ware_ops_pipes `f04527ba` with ware_ops_algos `042dcff`; its SHA-256 equals the checksum stated in that release's `experiments/evaluation/README.md`. Licence: that of the ware_ops_pipes repository. Pipeline labels are the short names of `experiments/evaluation/01_prepare_pipeline_results.py` at `v0.1.0`; `scripts/l1/pipelines.py` maps them to today's classes.

The file is not committed. Restore it with

    python scripts/fetch_data.py casop

The benchmark instances themselves are read from ware_ops_pipes' `data/instances/`, placed as that repository's `data/instances/README.md` describes.
