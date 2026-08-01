from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from scenarios.scenario_ijpe.schema import (
    COL_ARTICLE_ID,
    COL_ORDER_ID,
    COL_CUSTOMER_ID,
    COL_PICKER_ID,
)


def _id_map(values, prefix: str, seed: int) -> dict[str, str]:
    unique = sorted(
        {
            str(v)
            for v in values
            if v is not None and str(v) != ""
        }
    )

    random.Random(seed).shuffle(unique)

    return {
        real: f"{prefix}{i:05d}"
        for i, real in enumerate(unique, 1)
    }


def anonymize(
    picks: pd.DataFrame,
    master: pd.DataFrame,
    maps_path: Path,
    seed: int = 42,
):
    if maps_path.exists():
        maps = json.loads(maps_path.read_text())
    else:
        article_ids = (
            set(picks[COL_ARTICLE_ID].astype(str))
            | set(master[COL_ARTICLE_ID].astype(str))
        )

        maps = {
            "articles": _id_map(article_ids, "A", seed),
            "orders": _id_map(picks[COL_ORDER_ID], "O", seed + 1),
            "customers": _id_map(picks[COL_CUSTOMER_ID], "C", seed + 2),
            "pickers": _id_map(picks[COL_PICKER_ID], "P", seed + 3),
        }

        maps_path.parent.mkdir(parents=True, exist_ok=True)
        maps_path.write_text(json.dumps(maps, indent=2))

    picks = picks.copy()
    master = master.copy()

    picks[COL_ARTICLE_ID] = picks[COL_ARTICLE_ID].astype(str).map(maps["articles"])
    picks[COL_ORDER_ID] = picks[COL_ORDER_ID].astype(str).map(maps["orders"])
    picks[COL_CUSTOMER_ID] = picks[COL_CUSTOMER_ID].astype(str).map(maps["customers"])
    picks[COL_PICKER_ID] = picks[COL_PICKER_ID].astype(str).map(maps["pickers"])

    master[COL_ARTICLE_ID] = master[COL_ARTICLE_ID].astype(str).map(maps["articles"])
    master = master.dropna(subset=[COL_ARTICLE_ID]).copy()

    return picks, master
