"""Seeded Henn (2009/2012) instance generator for the L2 statistical benchmark.

The papers describe distributions but do not publish the original random seeds
or individual instances. Generated files use the OOBP Henn text format so the
same archive loader and warehouse geometry can be exercised.
"""

from __future__ import annotations

import random
from pathlib import Path


AISLE_WEIGHTS_CLASS = (0.52, 0.12, 0.12, 0.12,
                       0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
AISLE_WEIGHTS_UNIFORM = (0.1,) * 10
BASE_SEED = 20261015


def generate_instance(root: Path, *, storage: str, n_orders: int,
                      capacity: int, replication: int,
                      printed_rate: bool = False) -> str:
    """Write one instance and return its path relative to ``root/HennOOBP``.

    Arrival rates are per minute. The stated rule is ``n_orders / 480``;
    ``printed_rate`` uses the papers' 0.08625 for the n=30 sensitivity run.
    The same seed and positions are used for both capacities of a class.
    """
    if storage not in {"C", "U"} or n_orders not in {30, 60, 90, 120}:
        raise ValueError("unsupported Henn class")
    if capacity not in {45, 75} or not 0 <= replication < 50:
        raise ValueError("unsupported Henn capacity or replication")
    if printed_rate and n_orders != 30:
        raise ValueError("printed-rate variant exists only for n=30")

    rate = 0.08625 if printed_rate else n_orders / 480.0
    seed = BASE_SEED + (0 if storage == "C" else 10000) + n_orders * 100 + replication
    rng = random.Random(seed)
    variant = "printed" if printed_rate else "stated"
    stem = f"1s-{n_orders}-{capacity}-{replication}"
    directory = root / "HennOOBP" / storage / variant / str(n_orders) / str(capacity) / str(replication)
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
    arrival_s = 0.0
    for order_id in range(n_orders):
        size = rng.randint(5, 25)
        items = set()
        while len(items) < size:
            aisle = rng.choices(range(10), weights=weights, k=1)[0]
            items.add((aisle * 2 + rng.randrange(2), rng.randrange(45)))
        orders.append(f"Order {order_id} number of articles {size}")
        orders.extend(f"{index}\tAisle {aisle}\tLocation {cell}"
                      for index, (aisle, cell) in enumerate(sorted(items)))
        arrival_s += rng.expovariate(rate / 60.0)
        arrivals.append(f"{order_id + 1};{arrival_s * 1000:.6f}")
    (directory / f"{stem}.txt").write_text("\n".join(orders) + "\n", encoding="ascii")
    (directory / f"Arrival_Time_{stem}.csv").write_text(
        "\n".join(arrivals) + "\n", encoding="ascii")
    return (directory / f"{stem}.txt").relative_to(root / "HennOOBP").as_posix()
