"""Every tracked scenario follows the one structure and runs its default configuration."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import pytest

from casim.run import compose_scenario, run
from casim.scenario_check import NOT_PORTED, check_all, check_scenario
from tests.verification.kpi_recomputation import recompute_kpis

ROOT = Path(__file__).resolve().parents[1]


def _tracked_scenarios() -> set[str]:
    listing = subprocess.check_output(["git", "ls-files", "scenarios"], cwd=ROOT, text=True)
    return {Path(line).parts[1] for line in listing.splitlines() if len(Path(line).parts) > 2}


PORTED = sorted(_tracked_scenarios() - NOT_PORTED)


def test_check_all_passes_for_tracked_scenarios():
    assert check_all(ROOT, _tracked_scenarios()) == []


def _missing_data(directory: Path) -> list[str]:
    """Listed data files that are not present: not fetched or not placed yet."""
    missing = []
    for listing in (directory / "data").rglob("SHA256SUMS.txt"):
        for line in listing.read_text(encoding="utf-8").split("\n"):
            if line.strip() and not (listing.parent / line.split(maxsplit=1)[1].lstrip("*")).exists():
                missing.append(line.split(maxsplit=1)[1])
    return missing


@pytest.mark.parametrize("name", PORTED)
def test_default_configuration_runs_without_violations(name, tmp_path):
    directory = ROOT / "scenarios" / name
    if _missing_data(directory):
        pytest.skip(f"data of {name} not present; see its data/SOURCE.md")
    cfg = compose_scenario(directory, output_dir=tmp_path)
    cfg.experiment.audit = True
    run(cfg)
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    kpis = json.loads((tmp_path / "kpis.json").read_text(encoding="utf-8"))
    assert audit["violations"] == {"check_plan": {}, "check_progress": {}}
    assert not recompute_kpis(audit, kpis)
    trace = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    assert trace["trace_digest"] == audit["trace_digest"]


def test_minimal_matches_its_hand_derived_values(tmp_path):
    run(compose_scenario(ROOT / "scenarios" / "minimal", output_dir=tmp_path))
    with open(tmp_path / "comparison.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["kpi"] for row in rows} == {
        "total_distance", "makespan", "mean_order_completion_time"}
    assert all(float(row["difference"]) == 0.0 for row in rows)


def test_check_rejects_other_layouts(tmp_path):
    scenario = tmp_path / "bad"
    scenario.mkdir()
    (scenario / "scenario.yaml").write_text("name: bad\nsource: {name: nowhere}\n")
    (scenario / "hooks.py").write_text(
        "from casim.simulation_engine.simulation_engine import SimulationEngine\n")
    (scenario / "config").mkdir()
    findings = check_scenario(scenario)
    assert "missing README.md" in findings
    assert "file hooks.py is not allowed" in findings
    assert any("directory config/" in finding for finding in findings)
    assert any("unknown source 'nowhere'" in finding for finding in findings)
    assert any("runs no simulation loop" in finding for finding in findings)
