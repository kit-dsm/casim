from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scenarios.scenario_ijpe.schema import (
    COL_DATE,
    COL_ORDER_ID,
    COL_CUSTOMER_ID,
    COL_ARTICLE_ID,
    COL_QUANTITY,
    COL_LINE_WEIGHT,
    COL_KOLLI_VOLUME,
    COL_AISLE,
    COL_HOUSE,
    COL_PICKLOCATION,
    COL_DUE_DATE,
    COL_LENGTH,
    COL_WIDTH,
    COL_HEIGHT,
    COL_KOLLI_SIZE,
)


DAY_SEC = 86400


def _json_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _dist(values) -> list[dict]:
    counts = (
        pd.Series(values)
        .value_counts(normalize=True)
        .sort_index()
    )

    return [
        {
            "value": _json_value(value),
            "prob": float(prob),
        }
        for value, prob in counts.items()
    ]


def fit_public_generator_calibration(
    orders: pd.DataFrame,
    out_json: Path,
) -> dict:
    """Fit a public generator calibration from the validated historic stream.

    Input:
        order_stream_historic.csv

    The calibration preserves article IDs from the historic stream. This is
    required because generated scenario orders must reference articles that
    exist in the loaded simulation domain.
    """

    order_props = (
        orders
        .groupby(COL_ORDER_ID, sort=False)
        .agg(
            date=(COL_DATE, "first"),
            customer_id=(COL_CUSTOMER_ID, "first"),
            n_lines=(COL_ARTICLE_ID, "size"),
            due_date=(COL_DUE_DATE, "first"),
        )
        .reset_index()
    )

    daily_order_counts = (
        order_props
        .groupby("date")
        .size()
        .to_numpy(dtype=int)
    )

    due_windows = (
        order_props[COL_DUE_DATE]
        .mod(DAY_SEC)
        .round()
        .astype(int)
    )

    customer_freq = (
        order_props[COL_CUSTOMER_ID]
        .astype(str)
        .value_counts(normalize=True)
        .rename_axis("customer_id")
        .reset_index(name="prob")
    )

    customers = [
        {
            "customer_id": str(row.customer_id),
            "prob": float(row.prob),
        }
        for row in customer_freq.itertuples(index=False)
    ]

    article_freq = (
        orders[COL_ARTICLE_ID]
        .astype(str)
        .value_counts(normalize=True)
        .rename_axis(COL_ARTICLE_ID)
        .reset_index(name="prob")
    )

    articles = []

    for article_id, group in orders.groupby(COL_ARTICLE_ID, sort=False):
        article_id = str(article_id)

        loc = (
            group
            .groupby([COL_AISLE, COL_HOUSE, COL_PICKLOCATION])
            .size()
            .sort_values(ascending=False)
            .index[0]
        )

        quantity = group[COL_QUANTITY].astype(int)
        quantity_nonzero = quantity.replace(0, np.nan)

        article_prob = float(
            article_freq.loc[
                article_freq[COL_ARTICLE_ID] == article_id,
                "prob",
            ].iloc[0]
        )

        unit_weight = (
            group[COL_LINE_WEIGHT].astype(float)
            / quantity_nonzero
        ).median()

        articles.append(
            {
                COL_ARTICLE_ID: article_id,
                "prob": article_prob,
                "quantity_dist": _dist(quantity),

                "unit_weight": float(unit_weight),
                COL_KOLLI_VOLUME: float(group[COL_KOLLI_VOLUME].median()),

                COL_AISLE: int(loc[0]),
                COL_HOUSE: int(loc[1]),
                COL_PICKLOCATION: int(loc[2]),

                COL_LENGTH: float(group[COL_LENGTH].median()),
                COL_WIDTH: float(group[COL_WIDTH].median()),
                COL_HEIGHT: float(group[COL_HEIGHT].median()),
                COL_KOLLI_SIZE: int(group[COL_KOLLI_SIZE].median()),
            }
        )

    calibration = {
        "version": 1,
        "daily_order_counts": daily_order_counts.tolist(),
        "due_window_sec_dist": _dist(due_windows),
        "lines_per_order_dist": _dist(order_props["n_lines"].astype(int)),
        "customers": customers,
        "articles": articles,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(calibration, indent=2))

    return calibration


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    data_dir = root / "scenario_ijpe" / "data"

    in_csv = data_dir / "order_stream_historic.csv"
    out_json = data_dir / "calibrated_generator.json"

    orders = pd.read_csv(in_csv)

    fit_public_generator_calibration(
        orders=orders,
        out_json=out_json,
    )

    print(f"Wrote {out_json}")
