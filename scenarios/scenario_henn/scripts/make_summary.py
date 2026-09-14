from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _read_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in root.rglob("summary.csv"):
        if path.name == "matrix_summary.csv":
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def make_summary(root: Path) -> None:
    rows = _read_rows(root)
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    if not completed:
        raise ValueError(f"No completed Henn summaries found below {root}")

    fieldnames = sorted({key for row in completed for key in row})
    with (root / "matrix_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(completed)
    (root / "matrix_summary.json").write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in completed:
        grouped[(row["batching"], row["selection"])].append(row)
    aggregate = [
        {
            "batching": batching,
            "selection": selection,
            "runs": len(group_rows),
            "mean_completion_time_s": _mean(
                [float(row["completion_time_s"]) for row in group_rows]
            ),
            "mean_max_turnover_time_s": _mean(
                [float(row["max_turnover_time_s"]) for row in group_rows]
            ),
        }
        for (batching, selection), group_rows in sorted(grouped.items())
    ]
    selection_completion: dict[str, list[float]] = defaultdict(list)
    selection_turnover: dict[str, list[float]] = defaultdict(list)
    for row in aggregate:
        selection_completion[row["selection"]].append(
            row["mean_completion_time_s"]
        )
        selection_turnover[row["selection"]].append(
            row["mean_max_turnover_time_s"]
        )
    selection_means = {
        name: _mean(values)
        for name, values in selection_completion.items()
    }
    turnover_means = {
        name: _mean(values)
        for name, values in selection_turnover.items()
    }
    completion_trend = (
        selection_means.get("long", float("inf"))
        < selection_means.get("short", float("-inf"))
        and selection_means.get("sav", float("inf"))
        < selection_means.get("short", float("-inf"))
    )
    turnover_trend = (
        turnover_means.get("long", float("inf"))
        < turnover_means.get("short", float("-inf"))
        and turnover_means.get("sav", float("inf"))
        < turnover_means.get("short", float("-inf"))
    )

    by_size: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in completed:
        order_count = int(row["instance_id"].split("_")[-2])
        by_size[(order_count, row["batching"])].append(
            float(row["completion_time_s"])
        )
    batching_by_size = [
        {
            "order_count": order_count,
            "batching": batching,
            "mean_completion_time_s": _mean(values),
        }
        for (order_count, batching), values in sorted(by_size.items())
    ]
    largest_sizes = sorted({row["order_count"] for row in batching_by_size})[-2:]
    stronger_large = []
    for order_count in largest_sizes:
        values = {
            row["batching"]: row["mean_completion_time_s"]
            for row in batching_by_size
            if row["order_count"] == order_count
        }
        if "fcfs" in values:
            stronger_large.extend(
                values[name] < values["fcfs"]
                for name in ("cw_like", "ls")
                if name in values
            )
    stronger_batching_trend = bool(stronger_large) and (
        sum(stronger_large) >= len(stronger_large) / 2
    )
    lines = [
        "# Qualitative Henn comparison",
        "",
        "This W5 matrix is not a numerical reproduction of Henn Tables "
        "7.1–7.4. `cw_like` and `ls` use the current ware-ops "
        "approximations, not exact C&W(ii) and ILS.",
        "",
        "| Batching | Selection | Runs | Mean completion (s) | "
        "Mean max turnover (s) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['batching']} | {row['selection']} | {row['runs']} | "
            f"{row['mean_completion_time_s']:.3f} | "
            f"{row['mean_max_turnover_time_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Completion ranking: "
            + " < ".join(
                f"{row['batching']}/{row['selection']}"
                for row in sorted(
                    aggregate,
                    key=lambda value: value["mean_completion_time_s"],
                )
            ),
            "",
            "Turnover ranking: "
            + " < ".join(
                f"{row['batching']}/{row['selection']}"
                for row in sorted(
                    aggregate,
                    key=lambda value: value["mean_max_turnover_time_s"],
                )
            ),
            "",
            "Directional check: LONG and SAV both have lower means than "
            "SHORT — completion: "
            f"**{'CONSISTENT' if completion_trend else 'NOT CONSISTENT'}**; "
            "turnover: "
            f"**{'CONSISTENT' if turnover_trend else 'NOT CONSISTENT'}**.",
            "",
            "Directional check: C&W-like and LS tend to improve completion "
            "over FCFS for the 80- and 100-order groups: "
            f"**{'CONSISTENT' if stronger_batching_trend else 'NOT CONSISTENT'}**.",
            "",
        ]
    )
    (root / "qualitative_henn_comparison.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    make_summary(args.sweep_dir.resolve())


if __name__ == "__main__":
    main()
