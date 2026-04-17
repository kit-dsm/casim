import logging
from pathlib import Path

import pandas as pd

from ware_ops_algos.algorithms import BatchingSolution, PickList, PickPosition

from casim.domain_objects.sim_domain import SimWarehouseDomain

logger = logging.getLogger(__name__)


def parse_sim_time(time_str: str) -> float:
    """HH:MM:SS → seconds since midnight (simulation float time)."""
    h, m, s = map(int, str(time_str).split(":"))
    return float(h * 3600 + m * 60 + s)


def load_pick_history(path: str | Path, date_filter: str = "30.08.2024") -> pd.DataFrame:
    """Read and minimally clean the pick history CSV."""
    df = pd.read_csv(path, sep=";", decimal=",")
    df["ARTIKELNR"] = df["ARTIKELNR"].astype(str).str[:-4]
    df["BEGINN_ZEIT"] = pd.to_datetime(df["BEGINN_ZEIT"], format="%H:%M:%S").dt.time
    df["ENDE_ZEIT"] = pd.to_datetime(df["ENDE_ZEIT"], format="%H:%M:%S").dt.time
    df["MENGE_IST"] = df["MENGE_IST"].astype(float)
    if date_filter is not None:
        df = df[df["NEU_DATUM"] == date_filter]
    return df.reset_index(drop=True)


def build_historic_batching_solution(
    df: pd.DataFrame,
    domain: SimWarehouseDomain,
) -> BatchingSolution:
    pick_lists: list[PickList] = []
    df = df[df["NEU_DATUM"] == "30.08.2024"]
    for auftragsnr, batch_df in df.groupby("AUFTRAGSNR"):
        batch_df = batch_df.sort_values("AUFTRAGSPOS")

        pick_positions = []
        for row in batch_df.itertuples(index=False):
            # resolve raw CSV coords → actual graph node via storage domain
            loc_key = f"{int(row.aisle)}_{row.house}"
            place = str(row.picklocation)
            pick_node = None
            for loc in domain.storage.storage_slots:
                if loc.id == loc_key:
                    for slot in loc.slots:
                        if slot.id == place:
                            pick_node = loc.pick_node
                            break
                    break
            if pick_node is None:
                logger.warning("Cannot resolve pick_node for %s/%s — skipping", loc_key, place)
                continue

            pick_positions.append(PickPosition(
                order_number=int(auftragsnr),
                article_id=row.ARTIKELNR,
                amount=int(row.MENGE_IST),
                pick_node=pick_node,  # actual graph float coordinate
                in_store=int(row.warehouse),
            ))

        if pick_positions:
            pick_lists.append(PickList(
                pick_positions=pick_positions,
                orders=[],
                release=parse_sim_time(str(batch_df["BEGINN_ZEIT"].min())),
                earliest_due_date=None,
            ))

    return BatchingSolution(
        algo_name="historic_wms",
        pick_lists=pick_lists,
        provenance={"source": "historic_csv", "n_batches": len(pick_lists)},
    )