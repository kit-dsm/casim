"""Realized process times: what the emulated warehouse does and no decision sees.

The state's warehouse W holds, besides layout, orders and resources, the
truth of the processes the simulator executes. A source may fix in W_0 how
long each order line and each setup of a picker actually take, for example
drawn from distributions with a replication seed. Execution takes the time
from here; planning uses the resource's time model (``tour_setup_time``,
``pick_time``), which is the planner's belief. The decision snapshot never
contains the truth (``StateAdapter``), so no decision plans with realized
times, and every policy of a replication meets the same times (common random
numbers). Without a truth, execution uses the resource's time model, as
planning does.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

LineKey = tuple[int, int, int]   # order, article, amount


@dataclass
class ProcessTruth:
    """Realized seconds of order lines and setups, fixed before the first decision.

    ``line_s``: per (order, article, amount) the seconds of each such line of
    the order, in position order; lines with the same key take them in turn.
    ``setup_s``: per picker the seconds of its first, second, ... setup.
    A line or setup not given here takes the resource's time model.
    """

    line_s: Mapping[LineKey, tuple[float, ...]] = field(default_factory=dict)
    setup_s: Mapping[int, tuple[float, ...]] = field(default_factory=dict)
    _lines_taken: dict[LineKey, int] = field(default_factory=dict, init=False, repr=False)
    _setups_taken: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def pick(self, pick) -> float | None:
        """Seconds of the line ``pick`` (order_number, article_id, amount), or None."""
        key = (int(pick.order_number), int(pick.article_id), int(pick.amount))
        times = self.line_s.get(key)
        if not times:
            return None
        taken = self._lines_taken.get(key, 0)
        self._lines_taken[key] = taken + 1
        return float(times[taken % len(times)])

    def setup(self, picker_id: int) -> float | None:
        """Seconds of the picker's next setup, or None."""
        times = self.setup_s.get(picker_id)
        taken = self._setups_taken.get(picker_id, 0)
        if not times or taken >= len(times):
            return None
        self._setups_taken[picker_id] = taken + 1
        return float(times[taken])


def drawn(fit: Mapping, seed: int, key: int, orders: Iterable, pickers: Iterable[int],
          first_setup_s: float) -> ProcessTruth:
    """The realized time of every order line and of every setup of a picker,
    drawn from a fit of empirical times.

    ``fit``: ``quantity_bounds`` (upper bounds of the quantity classes),
    ``line_s`` (per class the sampled line seconds), ``setup_s`` (sampled
    setup seconds) and optionally ``order_sd``. A line's time is drawn from
    its quantity class with a stream seeded by (seed, key, order) and one draw
    per position; a picker's setups from a stream seeded by (seed, key,
    picker), after a first setup of ``first_setup_s``. The lines of an order
    are slow or fast together: each is multiplied by the order's factor,
    lognormal with mean 1 and standard deviation ``order_sd``, from a stream
    of its own, so the line draws do not change with it. Every policy of a
    replication therefore meets the same times.
    """
    orders = list(orders)
    bounds, pools = fit["quantity_bounds"], fit["line_s"]
    sigma = float(np.sqrt(np.log1p(float(fit.get("order_sd", 0.0)) ** 2)))
    line_s: dict[LineKey, tuple[float, ...]] = {}
    for order in orders:
        uniforms = np.random.default_rng([seed, key, 1, int(order.order_id)]).random(
            len(order.order_positions))
        factor = 1.0 if sigma == 0.0 else float(np.exp(
            np.random.default_rng([seed, key, 3, int(order.order_id)]).normal(
                -sigma ** 2 / 2, sigma)))
        for position, u in zip(order.order_positions, uniforms):
            pool = pools[bisect_left(bounds, position.amount)]
            line = (int(order.order_id), int(position.article_id), int(position.amount))
            line_s[line] = (*line_s.get(line, ()), factor * float(pool[int(u * len(pool))]))
    setups = fit["setup_s"]
    setup_s = {}
    for picker in pickers:
        uniforms = np.random.default_rng([seed, key, 2, int(picker)]).random(len(orders))
        setup_s[picker] = (float(first_setup_s),
                           *(float(setups[int(u * len(setups))]) for u in uniforms))
    return ProcessTruth(line_s=line_s, setup_s=setup_s)
