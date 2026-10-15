"""``casim check --all``: every scenario directory follows the one structure.

A scenario is ``scenarios/<name>/`` with ``README.md``, ``scenario.yaml``,
``data/SOURCE.md`` and ``data/SHA256SUMS.txt``; optionally ``source.py``,
``metrics.py``, ``plots.py`` and ``reference.csv``; and the Hydra groups
``data_card``, ``decisions``, ``cosy_repo``, ``sweep`` and ``variants``. Its
source is a library source or ``local`` (its own ``source.py``), it runs no
simulation loop, every file of every group composes and validates, and every
data file it holds matches its checksum.
"""

from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path

from omegaconf import OmegaConf

from casim.reference import read_reference
from casim.run import REFERENCE_FILE, SCENARIO_FILE, check, compose_scenario
from casim.sources.base import LIBRARY, LOCAL, LOCAL_CLASS

REQUIRED = ("README.md", SCENARIO_FILE, "data/SOURCE.md", "data/SHA256SUMS.txt")
OPTIONAL_FILES = ("source.py", "metrics.py", "plots.py", REFERENCE_FILE)
GROUPS = ("data_card", "decisions", "cosy_repo", "sweep", "variants")
IGNORED = ("data", "outputs", "__pycache__", "__init__.py")
FORBIDDEN_IMPORTS = ("casim.simulation_engine", "casim.runner", "casim.run",
                     "casim.decision_engine")
# Scenarios not yet ported to this structure (Phase S, Task S.4); IJPE stays
# until Task 6b.6. Each port removes its entry.
NOT_PORTED = frozenset({
    "scenario_henn2012",
    "scenario_ijpe",
    "scenario_dhaen_online",  # data of L4 only; deferred with Phase 6
})


def _layout(directory: Path) -> list[str]:
    findings = [f"missing {name}" for name in REQUIRED if not (directory / name).exists()]
    for entry in sorted(directory.iterdir()):
        if entry.name in IGNORED or entry.name in REQUIRED:
            continue
        if entry.is_dir():
            if entry.name not in GROUPS:
                findings.append(f"directory {entry.name}/ is not one of {GROUPS}")
            elif any(item.suffix != ".yaml" for item in entry.iterdir()):
                findings.append(f"{entry.name}/ may contain only .yaml files")
        elif entry.name not in OPTIONAL_FILES:
            findings.append(f"file {entry.name} is not allowed")
    return findings


def _python(directory: Path) -> list[str]:
    findings = []
    for path in sorted(directory.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for module in modules:
                if module.startswith(FORBIDDEN_IMPORTS):
                    findings.append(f"{path.name} imports {module}: a scenario runs no "
                                    "simulation loop of its own")
    return findings


def _source(directory: Path, spec) -> list[str]:
    if spec is None or "name" not in spec:
        return ["scenario.yaml has no source: {name: ...}"]
    name = spec["name"]
    if name == LOCAL:
        path = directory / "source.py"
        if not path.exists():
            return ["source 'local' needs source.py"]
        classes = {node.name: node for node in ast.parse(path.read_text(encoding="utf-8")).body
                   if isinstance(node, ast.ClassDef)}
        if LOCAL_CLASS not in classes:
            return [f"source.py defines no {LOCAL_CLASS}"]
        methods = {item.name for item in classes[LOCAL_CLASS].body
                   if isinstance(item, ast.FunctionDef)}
        return [f"{LOCAL_CLASS} lacks {method}()" for method in ("initial_state", "events")
                if method not in methods]
    if name not in LIBRARY:
        return [f"unknown source {name!r}; allowed: {sorted([*LIBRARY, LOCAL])}"]
    if (directory / "source.py").exists():
        return [f"source.py is present although source {name!r} is a library source"]
    return []


def _checksums(directory: Path) -> list[str]:
    """Every listed data file that is present matches its checksum.

    A dataset may sit in a subdirectory of ``data/`` with its own
    ``SHA256SUMS.txt`` (the format of ``scripts/fetch_data.py``). Listed files
    that are absent have not been fetched yet and are not a finding. That no
    data file is committed is the job of ``.gitignore``, not of this check.
    """
    findings = []
    for listing in sorted((directory / "data").rglob("SHA256SUMS.txt")):
        for line in listing.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            path = listing.parent / name.lstrip("*")
            if path.exists() and sha256(path.read_bytes()).hexdigest() != digest:
                findings.append(f"{path.relative_to(directory).as_posix()} does not match "
                                "its checksum")
    return findings


def _compositions(directory: Path) -> list[list[str]]:
    """The default composition and one per further file of every group."""
    defaults = OmegaConf.load(directory / SCENARIO_FILE).get("defaults", [])
    chosen = {key.removeprefix("optional "): value for item in defaults
              if OmegaConf.is_dict(item) for key, value in item.items()}
    overrides = [[]]
    for group in GROUPS:
        folder = directory / group
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.yaml")):
            if chosen.get(group) == path.stem:
                continue
            prefix = "" if group in chosen else "+"
            overrides.append([f"{prefix}{group}={path.stem}"])
    return overrides


def check_scenario(directory: Path) -> list[str]:
    findings = _layout(directory) + _python(directory) + _checksums(directory)
    if not (directory / SCENARIO_FILE).exists():
        return findings
    findings += _source(directory, OmegaConf.load(directory / SCENARIO_FILE).get("source"))
    if (directory / REFERENCE_FILE).exists():
        try:
            read_reference(directory / REFERENCE_FILE)
        except ValueError as error:
            findings.append(str(error))
    variants = [f"variants={path.stem}" for path in sorted((directory / "variants").glob("*.yaml"))]
    for overrides in _compositions(directory):
        error = _first_error(directory, overrides)
        # A file that fits only some instances (a set's own pipeline) must
        # compose with at least one variant.
        if error is not None and overrides and not overrides[0].startswith("variants="):
            if any(_first_error(directory, [*overrides, variant]) is None for variant in variants):
                continue
        if error is not None:
            findings.append(f"{' '.join(overrides) or 'default'}: {error}")
    return findings


def _first_error(directory: Path, overrides: list[str]) -> str | None:
    try:
        check(compose_scenario(directory, overrides))
    except Exception as error:  # every configuration error is a finding
        return str(error)
    return None


def _has_content(directory: Path) -> bool:
    """Directories holding only caches or outputs are leftovers, not scenarios."""
    return any(path.is_file() and not {"__pycache__", "outputs", "cache"}
               & set(path.relative_to(directory).parts)
               for path in directory.rglob("*"))


def check_all(root: Path, names: set[str] | None = None) -> list[str]:
    """Findings for every scenario directory under ``root/scenarios``, or the named ones."""
    findings = []
    for directory in sorted((root / "scenarios").iterdir()):
        if (not directory.is_dir() or directory.name in NOT_PORTED
                or directory.name.startswith(("_", "."))
                or (names is not None and directory.name not in names)
                or not _has_content(directory)):
            continue
        findings += [f"{directory.name}: {finding}" for finding in check_scenario(directory)]
    return findings
