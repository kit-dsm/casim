"""Write the IJPE week as canonical tables for ``casim.sources.order_stream``.

Reads the confidential WMS export once (the cohort of ``scenarios/ijpe``:
the six study days, located lines, orders of one picker) and writes to
``scenarios/ijpe_week/data/`` (confidential, ignored by git):

- ``orders.csv``: ``order_id``, ``release`` (the day's shift start, 05:30: the
  orders are known from the start of picking), ``customer`` (the store);
- ``lines.csv``: ``order_id``, ``article``, ``quantity``, ``x``, ``y`` (pick node),
  in recorded pick order;
- ``layout.csv``: the grid layout;
- ``lists.csv``: the warehouse's disruption lists S1, S2, S4 and S5
  (``list``, ``order_id``);
- ``templates/S3.csv``: the template order of list S3, located lines;
- ``times.json``: the fit of line and setup times on all six days
  (``data/non_picking/week.json`` of ``scenarios/ijpe``,
  ``python -m scripts.ijpe.fit_non_picking``).

    python -m scripts.ijpe.canonical

Prints counts only.
"""

from __future__ import annotations

import shutil

import pandas as pd

from casim.sources.grid_layout import GridLayout
from casim.sources.wms_export import day_lines, disruption_lines, disruption_orders
from scenarios.ijpe.source import STUDY_DAYS
from scripts.ijpe import CACHE, DATA

TARGET = DATA.parents[1] / "ijpe_week" / "data"
RELEASE = "05:30"


def main() -> None:
    _, locations = GridLayout().build(DATA / "layout.csv")
    nodes = GridLayout.pick_nodes(locations)
    orders, lines = [], []
    for day in STUDY_DAYS:
        frame, _ = day_lines(day, nodes, CACHE)
        frame = frame.sort_values(["start", "row"])
        firsts = frame.groupby("order", sort=False).first()
        orders.append(pd.DataFrame({"order_id": firsts.index.astype(int),
                                    "release": f"{day} {RELEASE}",
                                    "customer": firsts["customer"].to_numpy()}))
        lines.append(pd.DataFrame({"order_id": frame["order"].astype(int),
                                   "article": frame["article"].astype(int),
                                   "quantity": frame["quantity"].astype(int),
                                   "x": [n[0] for n in frame["node"]],
                                   "y": [n[1] for n in frame["node"]]}))
    orders, lines = pd.concat(orders, ignore_index=True), pd.concat(lines, ignore_index=True)
    lists = disruption_orders(CACHE).rename(columns={"sheet": "list", "order": "order_id"})
    lists = lists[["list", "order_id"]].assign(order_id=lambda f: f["order_id"].astype(int))
    template = disruption_lines(CACHE)
    located = [nodes.get((int(a), int(h), int(p))) for a, h, p in
               zip(template["aisle"], template["house"], template["place"])]
    template = pd.DataFrame({"article": template["article"].astype(int),
                             "quantity": template["quantity"].astype(int),
                             "x": [n[0] if n else None for n in located],
                             "y": [n[1] if n else None for n in located]}).dropna()
    (TARGET / "templates").mkdir(parents=True, exist_ok=True)
    orders.to_csv(TARGET / "orders.csv", index=False)
    lines.to_csv(TARGET / "lines.csv", index=False)
    lists.to_csv(TARGET / "lists.csv", index=False)
    template.to_csv(TARGET / "templates" / "S3.csv", index=False)
    shutil.copyfile(DATA / "layout.csv", TARGET / "layout.csv")
    shutil.copyfile(DATA / "non_picking" / "week.json", TARGET / "times.json")
    print(f"{len(orders)} orders, {len(lines)} lines, lists {lists['list'].value_counts().to_dict()}, "
          f"template {len(template)} lines")


if __name__ == "__main__":
    main()
