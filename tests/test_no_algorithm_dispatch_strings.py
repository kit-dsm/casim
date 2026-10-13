"""Only subproblem components choose algorithm and waiting-policy classes."""

import re
from pathlib import Path

from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards


SOURCE = Path(__file__).resolve().parents[1] / "src" / "casim"


def test_no_algorithm_or_positioning_names_outside_subproblems():
    names = {name for card in load_packaged_algo_cards()
             for name in (card.algo_name, card.implementation.get("class_name"))
             if name}
    names.discard("Return")  # Ordinary prose, unlike the concrete ReturnRouting name.
    names.update(("center_relocation", "aisle_patrol", "WaitForOne",
                  "WaitForTwo", "WaitForThree", "WaitForFour"))
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(name) for name in sorted(names))
                         + r")\b")
    violations = []
    for path in SOURCE.rglob("*.py"):
        if path.is_relative_to(SOURCE / "pipelines" / "subproblems"):
            continue
        matches = sorted(set(pattern.findall(path.read_text(encoding="utf-8"))))
        if matches:
            violations.append((path.relative_to(SOURCE).as_posix(), matches))
    assert not violations, violations
