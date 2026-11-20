"""Convert the WMS workbook and the article file to canonical picks and master data.

Added after the submission: this converter built the submitted order stream
but was not in the repository. Changes to the version used then are marked
"Revision patch"; they only let it read the files as delivered on 9 and
11 June 2026 and keep the excluded article's number out of git.
"""
from hashlib import sha256
from pathlib import Path

import pandas as pd

EXCLUDED_ARTICLE_SHA256 = "5264f5a34ac5ee9a0db1c0482ace924789e5151c6cb90d8695f35d628fc3de2c"

from scenarios.scenario_ijpe.grocery_retailer_loader_digraph import (
    COL_ARTICLE_ID,
    COL_ORDER_ID,
    COL_CUSTOMER_ID,
    COL_PICKER_ID,
    COL_QUANTITY,
    COL_LINE_WEIGHT,
    COL_KOLLI_VOLUME,
    COL_AISLE,
    COL_DATE,
    COL_START_SEC,
    COL_END_SEC,
    COL_LENGTH,
    COL_WIDTH,
    COL_HEIGHT,
    COL_HOUSE,
    COL_KOLLI_SIZE,
    COL_PICKLOCATION,
)

PICKS_CANONICAL_COLS = [
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
]

MASTER_CANONICAL_COLS = [
    COL_ARTICLE_ID,
    COL_LENGTH,
    COL_WIDTH,
    COL_HEIGHT,
    COL_KOLLI_SIZE,
]

FULL_CANONICAL_COLS = PICKS_CANONICAL_COLS + [
    COL_LENGTH,
    COL_WIDTH,
    COL_HEIGHT,
    COL_KOLLI_SIZE,
]


def kolli_size_from_id(article_id: str) -> int:
    return int(str(article_id)[-4:])


def _project_canonical(df: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    """Hard schema boundary: return only the requested canonical columns."""

    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            f"{name} has duplicate columns after renaming: {duplicated}. "
            "Fix the source-column normalization before projection."
        )

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(
            f"{name} is missing required canonical columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.loc[:, columns].copy()

    extra = [col for col in out.columns if col not in columns]
    if extra:
        raise AssertionError(
            f"{name} still contains non-canonical columns after projection: {extra}"
        )

    return out


def to_canonical(
    picks_path: Path,
    master_path: Path,
    return_full: bool = False,
):
    # Revision patch: the first sheet. The submitted export had one sheet,
    # "Tabelle1"; the export of 11 June 2026 has the picks first, as
    # "OP Daten KW16_2026", followed by the disruption lists.
    picks = pd.read_excel(
        picks_path,
        sheet_name=0,
        dtype={
            "PALNR": str,
            "ARTIKELNR": str,
            "PERS_NR": str,
        },
    )

    # Normalize source column names that may already have been preprocessed.
    picks = picks.rename(
        columns={
            "Gang": COL_AISLE,
            "House": COL_HOUSE,
            "Lagerplatz House": "place",
        }
    )

    picks["ARTIKELNR"] = (
        picks["ARTIKELNR"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    if COL_PICKLOCATION not in picks.columns:
        picks[COL_PICKLOCATION] = picks[COL_HOUSE] * 100 + picks["place"]

    picks = picks[picks[COL_AISLE].isin([1, 2, 3, 4, 5, 6, 7, 8])].copy()
    # Revision patch: the one excluded article is matched by its SHA-256
    # to keep its number out of git.
    digests = {a: sha256(a.encode()).hexdigest() for a in picks["ARTIKELNR"].unique()}
    picks = picks[picks["ARTIKELNR"].map(digests) != EXCLUDED_ARTICLE_SHA256].copy()

    picks = picks.rename(
        columns={
            "ARTIKELNR": COL_ARTICLE_ID,
            "AUFTRAGSNR": COL_ORDER_ID,
            "KUNDENNR": COL_CUSTOMER_ID,
            "PERS_NR": COL_PICKER_ID,
            "MENGE_IST": COL_QUANTITY,
            "GEWICHT_SOLL": COL_LINE_WEIGHT,
            "VOLUMEN_KOLLI": COL_KOLLI_VOLUME,
        }
    )

    picks[COL_DATE] = pd.to_datetime(picks["NEU_DATUM"]).dt.date
    picks[COL_START_SEC] = pd.to_timedelta(
        picks["BEGINN_ZEIT"].astype(str)
    ).dt.total_seconds()
    picks[COL_END_SEC] = pd.to_timedelta(
        picks["ENDE_ZEIT"].astype(str)
    ).dt.total_seconds()

    picks[COL_QUANTITY] = picks[COL_QUANTITY].astype(float).round().astype(int)

    picks[COL_LINE_WEIGHT] = (
        picks[COL_LINE_WEIGHT]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    picks[COL_KOLLI_VOLUME] = (
        picks[COL_KOLLI_VOLUME]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    # Hard boundary: after this point, picks contains no raw Excel columns.
    picks = _project_canonical(
        picks,
        PICKS_CANONICAL_COLS,
        name="picks",
    )

    master = pd.read_excel(
        master_path,
        dtype={"ARTIKELNR": str},
    )

    master.columns = (
        master.columns
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )

    master["ARTIKELNR"] = (
        master["ARTIKELNR"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    master = master.rename(
        columns={
            "ARTIKELNR": COL_ARTICLE_ID,
            "Länge": COL_LENGTH,
            "Breite": COL_WIDTH,
            "Höhe": COL_HEIGHT,
            # Revision patch: the article file of 9 June 2026 names the
            # columns with their unit.
            "Länge (mm)": COL_LENGTH,
            "Breite (mm)": COL_WIDTH,
            "Höhe (mm)": COL_HEIGHT,
        }
    )

    master[COL_KOLLI_SIZE] = master[COL_ARTICLE_ID].apply(kolli_size_from_id)

    # Keep one canonical master row per article.
    master = (
        master
        .drop_duplicates(subset=[COL_ARTICLE_ID])
        .copy()
    )

    # Hard boundary: after this point, master contains no raw article-master columns.
    master = _project_canonical(
        master,
        MASTER_CANONICAL_COLS,
        name="master",
    )

    full_df = picks.merge(
        master,
        on=COL_ARTICLE_ID,
        how="left",
        validate="many_to_one",
    )

    # Hard boundary for the optional merged dataframe as well.
    full_df = _project_canonical(
        full_df,
        FULL_CANONICAL_COLS,
        name="full_df",
    )

    if return_full:
        return picks, master, full_df

    return picks, master