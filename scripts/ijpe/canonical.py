"""Write the IJPE data as canonical tables: the only reader of the WMS workbook.

Reads the confidential WMS export once (``scripts.ijpe.wms_export``, the
parquet cache of ``scripts/ijpe/convert.py``) and writes to
``scenarios/ijpe/data/canonical/`` (confidential, ignored by git). The
replay, the fit, the validations and the week (``casim.sources.order_stream``)
read only these tables.

- ``orders.csv``: ``order_id``, ``day`` (the pick date), ``release`` (05:30 of
  the day: orders are known from the start of picking), ``customer`` (store);
- ``lines.csv``: the replayed cohort, one row per line in recorded order:
  ``order_id``, ``article``, ``quantity``, ``x``, ``y`` (pick node), and the
  record: ``picker`` (login), ``start``, ``end`` (on the pick date) and ``row``
  (the sheet row), and ``weight`` (kg, GEWICHT_SOLL) and ``volume`` (m³: the
  line's packages, its weight over the package weight of the article file,
  times the package volume VOLUMEN_KOLLI). The cohort: the six study days by pick date, lines with a
  pick node (aisles 1 to 8 without aisle 10 and one location), orders of one
  login;
- ``records.csv``: every record of the export, also outside the cohort
  (``row``, ``picker``, ``pick_date``, ``start``, ``end``), for the work the fit
  counts outside the replayed lines;
- ``excluded.csv``: per day what the cohort leaves out (``day``, ``item``, ``count``);
- ``lists.csv``: the warehouse's disruption lists S1, S2, S4, S5 (``list``, ``order_id``);
- ``templates/S3.csv``: the template order of list S3, located lines;
- ``layout.csv``: the grid layout.

The fit of line and setup times (``python -m scripts.ijpe.fit_non_picking``)
reads these tables and writes ``data/non_picking/``.

    uv run --with openpyxl python -m scripts.ijpe.canonical

Reading the article file (``lbh_kw16.xlsx``) needs openpyxl. Prints counts only.
"""

from __future__ import annotations

import shutil

import pandas as pd

from casim.sources.grid_layout import GridLayout
from scripts.ijpe.wms_export import (
    day_lines, disruption_lines, disruption_orders, lines as export_lines, with_pick_date,
)
from scenarios.ijpe.source import CANONICAL, STUDY_DAYS
from scripts.ijpe import CACHE, DATA

RELEASE = "05:30"
STAMP = "%Y-%m-%d %H:%M:%S.%f"


def _stamp(column: pd.Series) -> pd.Series:
    return column.dt.strftime(STAMP)


def package_weights() -> pd.Series:
    """kg per package by article number, from the partner's article file."""
    master = pd.read_excel(DATA / "lbh_kw16.xlsx", dtype=str)
    article = master["ARTIKELNR"].str.strip().str.replace(r"\.0$", "", regex=True)
    weight = master["Gewicht (kg)"].str.replace(",", ".", regex=False).astype(float)
    return pd.Series(weight.to_numpy(), index=article.to_numpy())


def volume_m3(frame: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Goods volume per line: packages (weight over package weight) times litres per package.

    MENGE_IST counts single units, not packages (docs/ijpe/scenarios.md, Section B).
    """
    packages = frame["weight"] / frame["article"].map(weights)
    return packages * frame["package_volume"] / 1000.0


def main() -> None:
    weights = package_weights()
    _, locations = GridLayout().build(DATA / "layout.csv")
    nodes = GridLayout.pick_nodes(locations)
    orders, lines, excluded = [], [], []
    for day in STUDY_DAYS:
        frame, left_out = day_lines(day, nodes, CACHE)
        frame = frame.sort_values(["start", "row"])
        firsts = frame.groupby("order", sort=False).first()
        orders.append(pd.DataFrame({"order_id": firsts.index.astype(int), "day": day,
                                    "release": f"{day} {RELEASE}",
                                    "customer": firsts["customer"].to_numpy()}))
        lines.append(pd.DataFrame({
            "order_id": frame["order"].astype(int), "article": frame["article"].astype(int),
            "quantity": frame["quantity"].astype(int),
            "x": [n[0] for n in frame["node"]], "y": [n[1] for n in frame["node"]],
            "picker": frame["picker"], "start": _stamp(frame["start"]),
            "end": _stamp(frame["end"]), "row": frame["row"].astype(int),
            "weight": frame["weight"].to_numpy(), "volume": volume_m3(frame, weights).to_numpy()}))
        excluded += [{"day": day, "item": item, "count": count} for item, count in left_out.items()]
    orders, lines = pd.concat(orders, ignore_index=True), pd.concat(lines, ignore_index=True)
    records = with_pick_date(export_lines(CACHE))
    records = pd.DataFrame({"row": records["row"].astype(int), "picker": records["picker"],
                            "pick_date": records["pick_date"].dt.strftime("%Y-%m-%d"),
                            "start": _stamp(records["start"]), "end": _stamp(records["end"])})
    lists = disruption_orders(CACHE).rename(columns={"sheet": "list", "order": "order_id"})
    lists = lists[["list", "order_id"]].assign(order_id=lambda f: f["order_id"].astype(int))
    template = disruption_lines(CACHE)
    located = [nodes.get((int(a), int(h), int(p))) for a, h, p in
               zip(template["aisle"], template["house"], template["place"])]
    template = pd.DataFrame({"article": template["article"].astype(int),
                             "quantity": template["quantity"].astype(int),
                             "x": [n[0] if n else None for n in located],
                             "y": [n[1] if n else None for n in located]}).dropna()
    (CANONICAL / "templates").mkdir(parents=True, exist_ok=True)
    orders.to_csv(CANONICAL / "orders.csv", index=False)
    lines.to_csv(CANONICAL / "lines.csv", index=False)
    records.to_csv(CANONICAL / "records.csv", index=False)
    pd.DataFrame(excluded).to_csv(CANONICAL / "excluded.csv", index=False)
    lists.to_csv(CANONICAL / "lists.csv", index=False)
    template.to_csv(CANONICAL / "templates" / "S3.csv", index=False)
    shutil.copyfile(DATA / "layout.csv", CANONICAL / "layout.csv")
    print(f"lines without a package weight: {int(lines['volume'].isna().sum())}; "
          f"volume {lines['volume'].sum():.0f} m3 over {lines['order_id'].nunique()} orders")
    print(f"{len(orders)} orders, {len(lines)} lines, {len(records)} records, "
          f"lists {lists['list'].value_counts().to_dict()}, template {len(template)} lines")


if __name__ == "__main__":
    main()
