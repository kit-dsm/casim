"""Compare a run's KPIs with the published values of a scenario's ``reference.csv``.

``reference.csv`` has the columns ``kpi`` (a key of ``kpis.json``),
``published``, ``kind`` (exact, statistical, bound or reproduction, Section 7.4)
and ``source``. A ``kpi`` that ``kpis.json`` lacks is a function of that name in
the scenario's ``metrics.py``, called with the output directory. Every other column is a selector named by a dotted key of the
composed configuration, for example ``source.filepath``; a row applies to a
run when each of its selectors equals that configuration value.
"""

from __future__ import annotations

import csv
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

METRICS_FILE = "metrics.py"
REQUIRED = ("kpi", "published", "kind", "source")
KINDS = ("exact", "statistical", "bound", "reproduction")


def read_reference(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = [column for column in REQUIRED if rows and column not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for row in rows:
        if row["kind"] not in KINDS:
            raise ValueError(f"{path}: kind {row['kind']!r} not in {KINDS}")
    return rows


def _metrics(scenario_dir: Path):
    path = scenario_dir / METRICS_FILE
    if not path.exists():
        return None
    spec = spec_from_file_location(f"casim_scenario_metrics_{scenario_dir.name}", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kpi_value(kpi: str, kpis: dict, metrics, output_dir: Path):
    """A key of ``kpis.json``, or a function of that name in ``metrics.py``."""
    if kpi in kpis:
        return kpis[kpi]
    if metrics is not None and callable(getattr(metrics, kpi, None)):
        return getattr(metrics, kpi)(output_dir)
    return None


def compare_with_reference(path: Path, cfg: DictConfig, output_dir: Path) -> list[dict]:
    """Write ``comparison.csv`` with the published and the CASIM value per applicable row."""
    kpis = json.loads((output_dir / "kpis.json").read_text(encoding="utf-8"))
    metrics = _metrics(path.parent)
    table = []
    for row in read_reference(path):
        selectors = {key: value for key, value in row.items() if key not in REQUIRED}
        if any(str(OmegaConf.select(cfg, key)) != value for key, value in selectors.items()):
            continue
        casim = kpi_value(row["kpi"], kpis, metrics, output_dir)
        published = float(row["published"])
        table.append({**selectors, "kpi": row["kpi"], "kind": row["kind"],
                      "published": published, "casim": casim,
                      "difference": None if casim is None else casim - published,
                      "source": row["source"]})
    with open(output_dir / "comparison.csv", "w", newline="", encoding="utf-8") as handle:
        if table:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return table
