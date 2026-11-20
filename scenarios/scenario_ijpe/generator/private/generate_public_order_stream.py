"""Write the pseudonymised canonical order stream of the IJPE case.

Added after the submission from the unversioned generator folder. It reads the
partner's WMS export and article file (confidential), replaces article, order,
store and picker numbers by pseudonyms (anonymize.py), clears the recorded
times, checks that orders, lines and articles are preserved, and writes
data/order_stream_public.csv. The rows are the recorded order lines: whether
the output may be published is a decision of the partner, not of this script.
Changes to the original are marked "Revision patch".
"""
from __future__ import annotations

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


@dataclass
class ReplayConfig:
    n_days: int | None = None
    seed: int = 42
    wms_release_sec: float = 2 * 3600
    bucket_hr: int = 1
    seconds_per_day: float = 86400.0
    clear_picker_id: bool = True
    clear_historic_times: bool = True
    due_window_mode: str = "ceil_hour"


class CheckFailure(Exception):
    pass


def _check(log, name, ok, detail="", hard=True):
    log.append((ok, hard))
    mark = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"  [{mark}] {name}" + (f": {detail}" if detail and not ok else ""))


def _summarize_checks(log):
    hard_fail = sum(1 for ok, hard in log if not ok and hard)
    soft_fail = sum(1 for ok, hard in log if not ok and not hard)
    passed = sum(1 for ok, _ in log if ok)

    print(f"\nChecks: {passed}/{len(log)} passed ({hard_fail} fail, {soft_fail} warn)")

    if hard_fail:
        raise CheckFailure(f"{hard_fail} hard check(s) failed; CSV not written")


def validate_anonymization(
    picks_in: pd.DataFrame,
    picks_out: pd.DataFrame,
    master_in: pd.DataFrame,
    master_out: pd.DataFrame,
    log,
):
    _check(
        log,
        "row count preserved",
        len(picks_out) == len(picks_in),
        f"{len(picks_in)} -> {len(picks_out)}",
    )

    for col in (COL_ARTICLE_ID, COL_ORDER_ID, COL_CUSTOMER_ID, COL_PICKER_ID):
        n_nan = int(picks_out[col].isna().sum())

        _check(
            log,
            f"{col} fully mapped",
            n_nan == 0,
            f"{n_nan} unmapped; stale id_maps.json?",
        )

        nin = picks_in[col].astype(str).nunique()
        nout = picks_out[col].nunique()

        _check(
            log,
            f"{col} distinct preserved",
            nin == nout,
            f"{nin} -> {nout}",
        )

    dropped = len(master_in) - len(master_out)

    _check(
        log,
        "master articles retained",
        dropped == 0,
        f"{dropped} dropped",
        hard=False,
    )


def _due_window_hour(
    end_sec: pd.Series,
    mode: str,
    bucket_hr: int = 1,
) -> pd.Series:
    bucket_sec = bucket_hr * 3600

    if mode == "floor_hour":
        return (end_sec // bucket_sec).astype(int)

    if mode == "ceil_hour":
        return np.ceil(end_sec / bucket_sec).astype(int)

    if mode == "nearest_hour":
        return np.round(end_sec / bucket_sec).astype(int)

    raise ValueError(
        f"Unknown due_window_mode={mode!r}. "
        "Use 'floor_hour', 'ceil_hour', or 'nearest_hour'."
    )


def build_historic_replay_stream(
    picks: pd.DataFrame,
    cfg: ReplayConfig,
) -> pd.DataFrame:
    """Pass through historic pick rows and only adapt simulation timing."""

    orders = picks.copy()

    historic_dates = sorted(orders[COL_DATE].unique())

    if cfg.n_days is not None:
        historic_dates = historic_dates[: cfg.n_days]

    orders = orders[orders[COL_DATE].isin(historic_dates)].copy()

    date_to_day = {
        historic_date: day
        for day, historic_date in enumerate(historic_dates)
    }

    # Compute one due window per complete historic order before overwriting date
    # and before clearing historic start/end timestamps.
    order_windows = (
        orders
        .groupby(COL_ORDER_ID)[COL_END_SEC]
        .max()
        .reset_index(name="historic_order_end_sec")
    )

    order_windows["due_hour"] = _due_window_hour(
        order_windows["historic_order_end_sec"],
        cfg.due_window_mode,
        bucket_hr=cfg.bucket_hr,
    )

    orders = orders.merge(
        order_windows[[COL_ORDER_ID, "due_hour"]],
        on=COL_ORDER_ID,
        how="left",
    )

    orders[COL_DATE] = orders[COL_DATE].map(date_to_day).astype(int)

    day_offset = orders[COL_DATE] * cfg.seconds_per_day

    orders[COL_ORDER_DATE] = day_offset + cfg.wms_release_sec
    orders[COL_DUE_DATE] = (
        day_offset
        + orders["due_hour"].astype(int) * cfg.bucket_hr * 3600
    )

    if cfg.clear_picker_id and COL_PICKER_ID in orders.columns:
        orders[COL_PICKER_ID] = None

    if cfg.clear_historic_times:
        if COL_START_SEC in orders.columns:
            orders[COL_START_SEC] = None

        if COL_END_SEC in orders.columns:
            orders[COL_END_SEC] = None

    orders = orders.drop(columns=["due_hour"])

    return orders


def enrich_orders_with_master(
    orders: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    orders = orders.copy()
    master = master.copy()

    required_master_cols = [
        COL_ARTICLE_ID,
        COL_LENGTH,
        COL_WIDTH,
        COL_HEIGHT,
        COL_KOLLI_SIZE,
    ]

    master_attrs = (
        master[required_master_cols]
        .drop_duplicates(subset=[COL_ARTICLE_ID])
        .copy()
    )

    cols_to_replace = [
        COL_LENGTH,
        COL_WIDTH,
        COL_HEIGHT,
        COL_KOLLI_SIZE,
    ]

    orders = orders.drop(
        columns=[col for col in cols_to_replace if col in orders.columns],
        errors="ignore",
    )

    enriched = orders.merge(
        master_attrs,
        on=COL_ARTICLE_ID,
        how="left",
        validate="many_to_one",
    )

    return enriched


def validate_replay_stream(
    source_picks: pd.DataFrame,
    orders: pd.DataFrame,
    cfg: ReplayConfig,
    log,
):
    historic_dates = sorted(source_picks[COL_DATE].unique())

    if cfg.n_days is not None:
        historic_dates = historic_dates[: cfg.n_days]

    src = source_picks[source_picks[COL_DATE].isin(historic_dates)].copy()

    _check(
        log,
        "row count preserved",
        len(orders) == len(src),
        f"{len(orders)} vs {len(src)}",
    )

    _check(
        log,
        "order count preserved",
        orders[COL_ORDER_ID].nunique() == src[COL_ORDER_ID].nunique(),
        f"{orders[COL_ORDER_ID].nunique()} vs {src[COL_ORDER_ID].nunique()}",
    )

    _check(
        log,
        "customer count preserved",
        orders[COL_CUSTOMER_ID].nunique() == src[COL_CUSTOMER_ID].nunique(),
        f"{orders[COL_CUSTOMER_ID].nunique()} vs {src[COL_CUSTOMER_ID].nunique()}",
    )

    _check(
        log,
        "article count preserved",
        orders[COL_ARTICLE_ID].nunique() == src[COL_ARTICLE_ID].nunique(),
        f"{orders[COL_ARTICLE_ID].nunique()} vs {src[COL_ARTICLE_ID].nunique()}",
    )

    src_lines_per_order = src.groupby(COL_ORDER_ID).size().sort_index()
    out_lines_per_order = orders.groupby(COL_ORDER_ID).size().sort_index()

    _check(
        log,
        "lines per order preserved",
        src_lines_per_order.equals(out_lines_per_order),
    )

    _check(
        log,
        "dates normalized to 0..n-1",
        sorted(orders[COL_DATE].unique()) == list(range(len(historic_dates))),
        f"{sorted(orders[COL_DATE].unique())}",
    )

    expected_release = (
        orders[COL_DATE] * cfg.seconds_per_day
        + cfg.wms_release_sec
    )

    _check(
        log,
        "release time equals configured WMS release",
        bool((orders[COL_ORDER_DATE] == expected_release).all()),
    )

    _check(
        log,
        "due after release",
        bool((orders[COL_DUE_DATE] > orders[COL_ORDER_DATE]).all()),
    )

    required_cols = [
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
        COL_ORDER_DATE,
        COL_DUE_DATE,
        COL_LENGTH,
        COL_WIDTH,
        COL_HEIGHT,
        COL_KOLLI_SIZE,
    ]

    for col in required_cols:
        if col not in orders.columns:
            _check(
                log,
                f"{col} exists",
                False,
                f"missing column {col}",
            )
            continue

        n_nan = int(orders[col].isna().sum())

        _check(
            log,
            f"no NaN in {col}",
            n_nan == 0,
            f"{n_nan} NaN",
        )


def describe_orders(orders: pd.DataFrame):
    n_days = orders[COL_DATE].nunique()
    n_orders = orders[COL_ORDER_ID].nunique()
    n_lines = len(orders)
    n_articles = orders[COL_ARTICLE_ID].nunique()

    lines_per_order = orders.groupby(COL_ORDER_ID).size()
    qty_per_line = orders[COL_QUANTITY]
    weight_per_order = orders.groupby(COL_ORDER_ID)[COL_LINE_WEIGHT].sum()

    print("=" * 60)
    print(f"Days:                {n_days}")
    print(f"Orders:              {n_orders}")
    print(f"Pick lines:          {n_lines}")
    print(f"Unique articles:     {n_articles}")

    print(
        f"Lines/order:         mean {lines_per_order.mean():.1f}, "
        f"median {lines_per_order.median():.0f}, "
        f"max {lines_per_order.max()}"
    )

    print(
        f"Quantity/line:       mean {qty_per_line.mean():.1f}, "
        f"max {qty_per_line.max()}"
    )

    print(
        f"Weight/order (kg):   mean {weight_per_order.mean():.0f}, "
        f"max {weight_per_order.max():.0f}"
    )

    lines_per_day = orders.groupby(COL_DATE).size().sort_index()

    print("\nPick lines per picking day:")
    for day, n in lines_per_day.items():
        print(f"  day {int(day)}: {int(n)}")

    order_windows = (
        orders
        .drop_duplicates(COL_ORDER_ID)
        .assign(window_sec=lambda d: d[COL_DUE_DATE] % (24 * 3600))
    )

    window_counts = (
        order_windows
        .groupby("window_sec")
        .size()
        .sort_index()
    )

    print("\nOrders per due window across all days:")
    for sec, n in window_counts.items():
        print(f"  {int(sec) // 3600:02d}:00  {n}")


def build_order_stream(
    picks: pd.DataFrame,
    master: pd.DataFrame,
    out_csv: Path,
    maps_path: Path,
    cfg: ReplayConfig,
):
    log = []

    picks_raw = picks
    master_raw = master

    picks, master = anonymize(
        picks=picks,
        master=master,
        maps_path=maps_path,
        seed=cfg.seed,
    )

    print("Anonymization:")
    validate_anonymization(
        picks_in=picks_raw,
        picks_out=picks,
        master_in=master_raw,
        master_out=master,
        log=log,
    )

    print("Replay transformation:")
    orders = build_historic_replay_stream(
        picks=picks,
        cfg=cfg,
    )

    print("Master enrichment:")
    orders = enrich_orders_with_master(
        orders=orders,
        master=master,
    )

    print("Replay stream:")
    validate_replay_stream(
        source_picks=picks,
        orders=orders,
        cfg=cfg,
        log=log,
    )

    _summarize_checks(log)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    orders.to_csv(
        out_csv,
        index=False,
    )

    describe_orders(orders)

    return orders


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "scenario_ijpe" / "data"

    picks_xlsx = (
        out_dir
        / "historic_source"
        / "OrdPick_REWE051_2026_KW16.xlsx"
    )

    master_xlsx = (
        out_dir
        / "historic_source"
        / "LBH_REWE051_2026_KW16.xlsx"
    )

    picks, master = to_canonical(
        picks_xlsx,
        master_xlsx,
    )

    build_order_stream(
        picks=picks,
        master=master,
        # Revision patch: a name of its own, not the submitted stream's.
        out_csv=out_dir / "order_stream_public.csv",
        maps_path=out_dir / "historic_source" / "id_maps_new.json",
        cfg=ReplayConfig(
            n_days=6,
            seed=42,
            wms_release_sec=2 * 3600,
            bucket_hr=1,
            due_window_mode="ceil_hour",
            clear_picker_id=False,
            clear_historic_times=True,
        ),
    )