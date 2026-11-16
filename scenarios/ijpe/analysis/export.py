"""Read the WMS export of the IJPE case (docs/ijpe_revision_plan.md, Phase R0).

The export is confidential and lives in ``scenarios/ijpe/data/`` (ignored by
git). Reading the workbook needs ``openpyxl``, which is not a casim
dependency; ``convert`` therefore caches its sheets as parquet once:

    uv run --with openpyxl python -m scenarios.ijpe.analysis.export

Everything else reads the parquet cache.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
EXPORT = DATA / "wms_export_kw16.xlsx"
CACHE = DATA / "cache"
LINES_SHEET = "OP Daten KW16_2026"
TU_SHEET = "TE pro h"
# Sheets listing the orders of each disruption the warehouse defined; the
# paper's Scenarios 1 to 4 are derived from S1, S2, S3 and S5 (data_audit.md, R0.5).
DISRUPTION_SHEETS = {
    "S1": "S1 - 13.04. 120 TUs vor 13 Uhr",
    "S2": "S2 - 16.04. 100 TUs vor 13 Uhr",
    "S3": "S3 - 15.04. 6 TUs +",
    "S4": "S4 - 14.04. 80 TUs vor 8.30 Uhr",
    "S5": "S5 - 16.04. 200 TUs nach 10 Uhr",
}

# Raw column -> name used here. Units: quantity in packages (Kolli), weight
# in kg per line, volume in litres per package (data_audit.md, R0.1).
COLUMNS = {
    "NEU_DATUM": "date",
    "LFDNR": "seq",
    "AUFTRAGSNR": "order",
    "AUFTRAGSPOS": "position",
    "ARTIKELNR": "article",
    "Q_PLATZ": "location",
    "BEGINN_ZEIT": "start_clock",
    "ENDE_ZEIT": "end_clock",
    "PERS_NR": "picker",
    "LHM_TYP": "carrier_type",
    "MENGE_IST": "quantity",
    "GEWICHT_SOLL": "weight",
    "VOLUMEN_KOLLI": "package_volume",
    "TYP": "line_type",
    "PALNR": "pallet",
    "KUNDENNR": "customer",
    "area": "area",
    "aisle": "aisle",
    "house": "house",
    "place": "place",
}
GAP_COLUMN_PREFIX = "L"  # "Lücken Auftrag", mis-encoded in the workbook
EXCLUDED_ARTICLE_SHA256 = "5264f5a34ac5ee9a0db1c0482ace924789e5151c6cb90d8695f35d628fc3de2c"


def convert(export: Path = EXPORT, cache: Path = CACHE) -> None:
    """Cache every sheet of the workbook as parquet, all columns as strings."""
    cache.mkdir(parents=True, exist_ok=True)
    workbook = pd.ExcelFile(export)
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(workbook, sheet_name=sheet, dtype=str)
        frame.columns = [str(column) for column in frame.columns]
        frame.to_parquet(cache / f"{_slug(sheet)}.parquet")


def _slug(sheet: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in sheet).strip("_")


def raw(sheet: str, cache: Path = CACHE) -> pd.DataFrame:
    path = cache / f"{_slug(sheet)}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `uv run --with openpyxl python -m "
                                "scenarios.ijpe.analysis.export` first")
    return pd.read_parquet(path)


def lines(cache: Path = CACHE) -> pd.DataFrame:
    """One row per order line, typed, with start and end as timestamps.

    Times in the export are clock times without a date; ``start`` and ``end``
    attach them to ``date``, the booking date. Lines whose clock end precedes
    their clock start keep that (R0.4 counts them); nothing is dropped or
    repaired here. ``order_gap`` is the workbook's column "Lücken Auftrag":
    the time from this row's end to the next row's start when both rows have
    the same picker and order and the gap exceeds 10 s.
    """
    frame = raw(LINES_SHEET, cache)
    # Sheet row order: the workbook's own gap column is a formula over it.
    frame.insert(0, "row", range(len(frame)))
    gap = [column for column in frame.columns if column.startswith(GAP_COLUMN_PREFIX)
           and column.endswith("Auftrag")]
    frame = frame.rename(columns={**COLUMNS, **{column: "order_gap" for column in gap}})
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("seq", "carrier_type", "area", "aisle", "house", "place"):
        frame[column] = frame[column].astype(int)
    for column in ("quantity", "weight", "package_volume"):
        frame[column] = frame[column].str.replace(",", ".", regex=False).astype(float)
    frame["position"] = frame["position"].astype(int)
    frame["start"] = frame["date"] + pd.to_timedelta(frame["start_clock"])
    frame["end"] = frame["date"] + pd.to_timedelta(frame["end_clock"])
    frame["order_gap"] = pd.to_timedelta(frame["order_gap"])
    # Package size code: the last four digits of the article number (the
    # submitted loader's kolli_size_from_id).
    frame["package_code"] = frame["article"].str[-4:].astype(int)
    return frame


def is_excluded_article(articles: pd.Series) -> pd.Series:
    """The one article ``to_canonical`` drops, matched by hash to keep its number out of git."""
    digests = {a: sha256(a.encode()).hexdigest() for a in articles.unique()}
    return articles.map(digests) == EXCLUDED_ARTICLE_SHA256


def paper_lines(frame: pd.DataFrame) -> pd.DataFrame:
    """The lines the submitted model kept (``orders_to_canonical.to_canonical``).

    It keeps aisles 1 to 8 (dropping aisle 10, which is area 10) and drops
    one article.
    """
    return frame[frame["aisle"].between(1, 8) & ~is_excluded_article(frame["article"])]


def disruption_orders(cache: Path = CACHE) -> pd.DataFrame:
    """The orders of the disruption sheets S1, S2, S4, S5: sheet, date, order.

    S3 lists article lines of a new store with neither order number nor times
    (a template order to duplicate); ``disruption_lines`` returns it.
    """
    rows = []
    for key, sheet in DISRUPTION_SHEETS.items():
        if key == "S3":
            continue
        frame = raw(sheet, cache)
        rows.append(pd.DataFrame({"sheet": key, "date": pd.to_datetime(frame["NEU_DATUM"]),
                                  "order": frame["AUFTRAGSNR"]}))
    return pd.concat(rows, ignore_index=True)


def disruption_lines(cache: Path = CACHE) -> pd.DataFrame:
    frame = raw(DISRUPTION_SHEETS["S3"], cache).rename(columns=COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("quantity", "weight", "package_volume"):
        frame[column] = frame[column].str.replace(",", ".", regex=False).astype(float)
    return frame


if __name__ == "__main__":
    convert()
    print(f"cached {EXPORT.name} in {CACHE}")
