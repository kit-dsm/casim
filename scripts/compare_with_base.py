"""Compare recorded scenario KPIs; Phase 5 extends this to decision traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(reference: dict, current: dict) -> list[tuple[str, object, object]]:
    return [(key, reference[key], current.get(key))
            for key in reference if current.get(key) != reference[key]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("current", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    differences = compare(reference, current)
    for key in reference:
        print(f"{key}: {reference[key]} -> {current.get(key)}")
    if differences:
        print(f"{len(differences)} KPI differences")
    else:
        print(f"Identical on {len(reference)} reference KPIs")


if __name__ == "__main__":
    main()
