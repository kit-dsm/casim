"""The L2 source: seeded Henn (2009/2012) instances, generated, then read as OOBP files.

The papers describe distributions but do not publish the original random seeds
or individual instances. Generated files use the OOBP Henn text format so the
same archive loader and warehouse geometry can be exercised. Each run writes
its instance below ``data/generated/`` (ignored), reads it with the library
source ``oobp`` and closes the stream at 480 min (``announced``).
"""

from __future__ import annotations

from collections.abc import Iterator
import random
from pathlib import Path

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.sources.oobp import OobpSource

HORIZON_S = 480 * 60


AISLE_WEIGHTS_CLASS = (0.52, 0.12, 0.12, 0.12,
                       0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
AISLE_WEIGHTS_UNIFORM = (0.1,) * 10
BASE_SEED = 20261015


def generate_instance(root: Path, *, storage: str, n_orders: int,
                      capacity: int, replication: int,
                      arrival_mode: str) -> str:
    """Write one instance and return its path relative to ``root/HennOOBP``.

    Both readings use the stated rate ``n_orders / 480`` per minute. The
    ``truncated`` reading counts Poisson arrivals through minute 480; the
    ``conditioned`` reading draws exactly ``n_orders`` arrivals conditional
    on that count, hence uniform order statistics within the period.
    Positions and arrivals use separate streams, shared between capacities.
    """
    if storage not in {"C", "U"} or n_orders not in {30, 60, 90, 120}:
        raise ValueError("unsupported Henn class")
    if capacity not in {45, 75} or not 0 <= replication < 50:
        raise ValueError("unsupported Henn capacity or replication")
    if arrival_mode not in {"truncated", "conditioned"}:
        raise ValueError("arrival_mode must be truncated or conditioned")

    rate = n_orders / 480.0
    seed = BASE_SEED + (0 if storage == "C" else 10000) + n_orders * 100 + replication
    item_rng = random.Random(seed)
    arrival_rng = random.Random(seed + 1000000)
    if arrival_mode == "truncated":
        arrival_minutes = []
        arrival_min = 0.0
        while True:
            arrival_min += arrival_rng.expovariate(rate)
            if arrival_min > 480.0:
                break
            arrival_minutes.append(arrival_min)
    else:
        arrival_minutes = sorted(arrival_rng.uniform(0.0, 480.0)
                                 for _ in range(n_orders))
    stem = f"1s-{n_orders}-{capacity}-{replication}"
    directory = root / "HennOOBP" / storage / arrival_mode / str(n_orders) / str(capacity) / str(replication)
    directory.mkdir(parents=True, exist_ok=True)
    setting = "\n".join((
        "no_aisles_: 10", "no_cells__: 45", "cell_lengt: 1",
        "cell_width: 1.5", "aisle_widt: 2", "dis_ais_wa: 0.5",
        f"m_no_a_p_b: {capacity}",
    )) + "\n"
    (directory / "sett1.txt").write_text(setting, encoding="ascii")

    weights = AISLE_WEIGHTS_CLASS if storage == "C" else AISLE_WEIGHTS_UNIFORM
    orders = []
    arrivals = []
    for order_id, arrival_min in enumerate(arrival_minutes):
        size = item_rng.randint(5, 25)
        items = set()
        while len(items) < size:
            aisle = item_rng.choices(range(10), weights=weights, k=1)[0]
            items.add((aisle * 2 + item_rng.randrange(2), item_rng.randrange(45)))
        orders.append(f"Order {order_id} number of articles {size}")
        orders.extend(f"{index}\tAisle {aisle}\tLocation {cell}"
                      for index, (aisle, cell) in enumerate(sorted(items)))
        arrivals.append(f"{order_id + 1};{arrival_min * 60000:.6f}")
    (directory / f"{stem}.txt").write_text("\n".join(orders) + "\n", encoding="ascii")
    (directory / f"Arrival_Time_{stem}.csv").write_text(
        "\n".join(arrivals) + "\n", encoding="ascii")
    return (directory / f"{stem}.txt").relative_to(root / "HennOOBP").as_posix()


class ScenarioSource:
    """One generated Henn instance; the picker arrives with the first order."""

    def __init__(self, data_dir: Path, *, storage: str, n_orders: int, capacity: int,
                 replication: int, arrival_mode: str = "conditioned"):
        root = Path(data_dir) / "generated"
        relative = generate_instance(root, storage=storage, n_orders=int(n_orders),
                                     capacity=int(capacity), replication=int(replication),
                                     arrival_mode=arrival_mode)
        self.oobp = OobpSource(root / "HennOOBP", relative, format="henn", depot="corner",
                               closure_s=HORIZON_S)

    def initial_state(self) -> SimWarehouseDomain:
        return self.oobp.initial_state()

    def events(self) -> Iterator[Event]:
        return self.oobp.events()
