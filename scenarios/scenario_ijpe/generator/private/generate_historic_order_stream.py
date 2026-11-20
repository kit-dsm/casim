from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scenarios.scenario_ijpe.generator.private.orders_to_canonical import to_canonical
from scenarios.scenario_ijpe.generator.private.anonymize import anonymize
from scenarios.scenario_ijpe.grocery_retailer_loader_digraph import (
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
)


DAY_SEC = 86400


@dataclass
class ReplayConfig:
    n_days: int | None = 6
    wms_release_sec: int = 2 * 3600
    due_bucket_sec: int = 3600
    clear_historic_times: bool = True


def build_historic_replay_stream(
    picks: pd.DataFrame,
    cfg: ReplayConfig,
) -> pd.DataFrame:
    orders = picks.copy()

    historic_dates = sorted(orders[COL_DATE].unique())

    if cfg.n_days is not None:
        historic_dates = historic_dates[: cfg.n_days]

    orders = orders[orders[COL_DATE].isin(historic_dates)].copy()

    date_to_day = {
        historic_date: day
        for day, historic_date in enumerate(historic_dates)
    }

    order_due_sec = (
        orders
        .groupby(COL_ORDER_ID)[COL_END_SEC]
        .max()
    )

    order_due_sec = (
        np.ceil(order_due_sec / cfg.due_bucket_sec)
        * cfg.due_bucket_sec
    )

    orders = orders.merge(
        order_due_sec.rename("_due_sec"),
        left_on=COL_ORDER_ID,
        right_index=True,
        how="left",
    )

    orders[COL_DATE] = orders[COL_DATE].map(date_to_day).astype(int)

    day_offset = orders[COL_DATE] * DAY_SEC

    orders[COL_ORDER_DATE] = day_offset + cfg.wms_release_sec
    orders[COL_DUE_DATE] = day_offset + orders["_due_sec"]

    orders = orders.drop(columns=["_due_sec"])

    if cfg.clear_historic_times:
        orders[COL_START_SEC] = None
        orders[COL_END_SEC] = None

    return orders


def enrich_orders_with_master(
    orders: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    master_attrs = (
        master[
            [
                COL_ARTICLE_ID,
                COL_LENGTH,
                COL_WIDTH,
                COL_HEIGHT,
                COL_KOLLI_SIZE,
            ]
        ]
        .drop_duplicates(subset=[COL_ARTICLE_ID])
        .copy()
    )

    orders = orders.drop(
        columns=[
            col
            for col in [
                COL_LENGTH,
                COL_WIDTH,
                COL_HEIGHT,
                COL_KOLLI_SIZE,
            ]
            if col in orders.columns
        ],
        errors="ignore",
    )

    return orders.merge(
        master_attrs,
        on=COL_ARTICLE_ID,
        how="left",
        validate="many_to_one",
    )


def _dist(values) -> list[dict]:
    counts = pd.Series(values).value_counts(normalize=True).sort_index()

    return [
        {
            "value": value.item() if hasattr(value, "item") else value,
            "prob": float(prob),
        }
        for value, prob in counts.items()
    ]


def fit_public_generator_calibration(
    orders: pd.DataFrame,
    out_json: Path,
) -> dict:
    order_props = (
        orders
        .groupby(COL_ORDER_ID, sort=False)
        .agg(
            date=(COL_DATE, "first"),
            customer_id=(COL_CUSTOMER_ID, "first"),
            n_lines=(COL_ARTICLE_ID, "size"),
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
        orders
        .drop_duplicates(COL_ORDER_ID)[COL_DUE_DATE]
        .mod(DAY_SEC)
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
                article_freq[COL_ARTICLE_ID] == str(article_id),
                "prob",
            ].iloc[0]
        )

        articles.append(
            {
                COL_ARTICLE_ID: str(article_id),
                "prob": article_prob,
                "quantity_dist": _dist(quantity),
                "unit_weight": float(
                    (group[COL_LINE_WEIGHT] / quantity_nonzero).median()
                ),
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

    picks_xlsx = (
        root
        / "scenario_ijpe"
        / "data"
        / "historic_source"
        / "OrdPick_REWE051_2026_KW16.xlsx"
    )

    master_xlsx = (
        root
        / "scenario_ijpe"
        / "data"
        / "historic_source"
        / "LBH_REWE051_2026_KW16.xlsx"
    )

    out_dir = root / "scenario_ijpe" / "data"
    maps_path = out_dir / "historic_source" / "id_maps_new.json"

    picks, master = to_canonical(
        picks_xlsx,
        master_xlsx,
    )

    picks, master = anonymize(
        picks=picks,
        master=master,
        maps_path=maps_path,
        seed=42,
    )

    orders = build_historic_replay_stream(
        picks=picks,
        cfg=ReplayConfig(
            n_days=6,
            wms_release_sec=2 * 3600,
            due_bucket_sec=3600,
            clear_historic_times=True,
        ),
    )

    orders = enrich_orders_with_master(
        orders=orders,
        master=master,
    )

    # Revision patch: the names the scenario configuration reads.
    out_csv = out_dir / "order_stream_historic.csv"
    orders.to_csv(out_csv, index=False)

    fit_public_generator_calibration(
        orders=orders,
        out_json=out_dir / "calibrated_generator.json",
    )

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_dir / 'calibrated_generator.json'}")