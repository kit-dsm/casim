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

from collections.abc import Mapping
from dataclasses import dataclass, field

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
