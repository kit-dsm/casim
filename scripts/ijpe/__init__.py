"""Data audit and replay analysis of the IJPE case (docs/ijpe/plan.md).

They read the confidential WMS export and CASIM's outputs and write aggregate
tables to ``scenarios/ijpe/outputs/`` (ignored by git).
"""

from pathlib import Path

SCENARIO = Path(__file__).resolve().parents[2] / "scenarios" / "ijpe"
DATA = SCENARIO / "data"
EXPORT = DATA / "wms_export_kw16.xlsx"
CACHE = DATA / "cache"
OUTPUTS = SCENARIO / "outputs"
