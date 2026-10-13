"""Keep decision selection separate from live state and event handling."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "src" / "casim"


def test_decision_engine_imports_no_live_state_or_events():
    forbidden = ("casim.state", "casim.simulation_engine", "casim.events")
    for path in (ROOT / "decision_engine").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert all(not name.startswith(forbidden) for name in names), (path, names)


def test_pipeline_does_not_read_event_causes():
    for path in (ROOT / "pipelines").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            assert not (isinstance(node, ast.Attribute) and node.attr == "cause"
                        and isinstance(node.value, ast.Attribute)
                        and node.value.attr == "context"), (path, node.lineno)
