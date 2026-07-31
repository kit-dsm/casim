from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scenarios.scenario_ijpe.schema import (
    COL_DATE,
    COL_ORDER_ID,
    COL_CUSTOMER_ID,
    COL_PICKER_ID,
    COL_ARTICLE_ID,
    COL_QUANTITY,
    COL_AISLE,
    COL_HOUSE,
    COL_PICKLOCATION,
    COL_KOLLI_VOLUME,
    COL_LINE_WEIGHT,
    COL_START_SEC,
    COL_END_SEC,
    COL_ORDER_DATE,
    COL_DUE_DATE,
    COL_LENGTH,
    COL_WIDTH,
    COL_HEIGHT,
    COL_KOLLI_SIZE,
)

DAY_SEC = 86400


GENERATED_ORDER_COLS = [
    COL_DATE,
    COL_ORDER_ID,
    COL_CUSTOMER_ID,
    COL_PICKER_ID,
    COL_ARTICLE_ID,
    COL_QUANTITY,
    COL_LINE_WEIGHT,
    COL_KOLLI_VOLUME,
    COL_AISLE,
    COL_HOUSE,
    COL_PICKLOCATION,
    COL_START_SEC,
    COL_END_SEC,
    COL_ORDER_DATE,
    COL_DUE_DATE,
    COL_LENGTH,
    COL_WIDTH,
    COL_HEIGHT,
    COL_KOLLI_SIZE,
]


def _draw_dist(rng: np.random.Generator, dist: list[dict]):
    values = [x["value"] for x in dist]
    probs = np.array([x["prob"] for x in dist], dtype=float)
    probs = probs / probs.sum()

    return rng.choice(values, p=probs)


def _draw_articles(
    rng: np.random.Generator,
    articles: list[dict],
    n: int,
) -> list[dict]:
    probs = np.array([a["prob"] for a in articles], dtype=float)
    probs = probs / probs.sum()

    idx = rng.choice(
        np.arange(len(articles)),
        size=n,
        replace=n > len(articles),
        p=probs,
    )

    return [articles[i] for i in idx]


def generate_n_orders(
    calibration: dict,
    n_orders: int,
    order_date: float,
    due_date: float,
    customer_id: str,
    order_prefix: str,
    seed: int,
    day_sec: int = DAY_SEC,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    rows = []
    day = int(order_date // day_sec)

    for order_idx in range(1, n_orders + 1):
        order_id = f"{order_prefix}{order_idx:06d}"
        n_lines = int(_draw_dist(rng, calibration["lines_per_order_dist"]))

        articles = _draw_articles(
            rng=rng,
            articles=calibration["articles"],
            n=n_lines,
        )

        for article in articles:
            quantity = int(_draw_dist(rng, article["quantity_dist"]))

            rows.append(
                {
                    COL_DATE: day,
                    COL_ORDER_ID: order_id,
                    COL_CUSTOMER_ID: customer_id,
                    COL_PICKER_ID: None,

                    COL_ARTICLE_ID: article[COL_ARTICLE_ID],
                    COL_QUANTITY: quantity,
                    COL_LINE_WEIGHT: quantity * float(article["unit_weight"]),

                    # Do not multiply by quantity here.
                    # The loader treats this as kolli/article volume.
                    COL_KOLLI_VOLUME: float(article[COL_KOLLI_VOLUME]),

                    COL_AISLE: int(article[COL_AISLE]),
                    COL_HOUSE: int(article[COL_HOUSE]),
                    COL_PICKLOCATION: int(article[COL_PICKLOCATION]),

                    COL_START_SEC: None,
                    COL_END_SEC: None,

                    COL_ORDER_DATE: float(order_date),
                    COL_DUE_DATE: float(due_date),

                    COL_LENGTH: float(article[COL_LENGTH]),
                    COL_WIDTH: float(article[COL_WIDTH]),
                    COL_HEIGHT: float(article[COL_HEIGHT]),
                    COL_KOLLI_SIZE: int(article[COL_KOLLI_SIZE]),
                }
            )

    return pd.DataFrame(rows, columns=GENERATED_ORDER_COLS)


def generate_profile(
    calibration: dict,
    *,
    seed: int,
    days: int,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Generate a deterministic multi-day canonical order stream."""
    rng = np.random.default_rng(seed)
    daily = calibration["daily_order_counts"]
    parts = []
    for day in range(days):
        count = max(1, round(daily[day % len(daily)] * scale))
        due_values = [x["value"] for x in calibration["due_window_sec_dist"]]
        due_probs = np.array(
            [x["prob"] for x in calibration["due_window_sec_dist"]],
            dtype=float,
        )
        due_counts = rng.multinomial(count, due_probs / due_probs.sum())
        for due_index, (due_sec, n_orders) in enumerate(
            zip(due_values, due_counts)
        ):
            if not n_orders:
                continue
            parts.append(generate_n_orders(
                calibration,
                n_orders=int(n_orders),
                order_date=float(day * DAY_SEC + 2 * 3600),
                due_date=float(day * DAY_SEC + due_sec),
                customer_id=f"generated-day-{day + 1}",
                order_prefix=f"G{day + 1:02d}{due_index:02d}-",
                seed=int(rng.integers(0, 2**31 - 1)),
            ))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a seeded canonical IJPE order stream"
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=6)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    generated = generate_profile(
        calibration,
        seed=args.seed,
        days=args.days,
        scale=args.scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
