from __future__ import annotations

from collections.abc import Mapping


class OrderCostReward:
    """Exact incremental reward for separable order-age costs.

    For order flow time ``F``, this represents
    ``weight * max(0, F - threshold) ** power``. Threshold zero and power one
    give total flow time; greater powers emphasize the tail; a positive
    threshold represents an SLA duration. Absolute ``due_times`` instead
    represent ``weight * max(0, completion - due) ** power`` and include any
    existing tardiness when an overdue order arrives.
    """

    def __init__(
        self,
        arrivals: Mapping[int, float],
        *,
        power: float = 1.0,
        weights: float | Mapping[int, float] = 1.0,
        thresholds_s: float | Mapping[int, float] | None = None,
        due_times: Mapping[int, float] | None = None,
        normalizer: float = 1.0,
    ):
        if power < 1.0:
            raise ValueError("Order-cost power must be at least one")
        if normalizer <= 0.0:
            raise ValueError("Reward normalizer must be positive")
        self.arrivals = {int(key): float(value) for key, value in arrivals.items()}
        self.power = float(power)
        self.weights = self._values(weights, "weight", positive=True)
        if thresholds_s is not None and due_times is not None:
            raise ValueError("Specify thresholds_s or due_times, not both")
        if due_times is None:
            self.thresholds_s = self._values(
                0.0 if thresholds_s is None else thresholds_s,
                "threshold",
                positive=False,
            )
        else:
            missing = set(self.arrivals) - set(due_times)
            if missing:
                raise ValueError(
                    f"Missing due times for orders {sorted(missing)}"
                )
            self.thresholds_s = {
                order_id: float(due_times[order_id]) - arrival
                for order_id, arrival in self.arrivals.items()
            }
        self.normalizer = float(normalizer)

    def _values(
        self,
        values: float | Mapping[int, float],
        name: str,
        *,
        positive: bool,
    ) -> dict[int, float]:
        if isinstance(values, Mapping):
            missing = set(self.arrivals) - set(values)
            if missing:
                raise ValueError(f"Missing {name}s for orders {sorted(missing)}")
            result = {
                order_id: float(values[order_id]) for order_id in self.arrivals
            }
        else:
            result = {order_id: float(values) for order_id in self.arrivals}
        invalid = {
            order_id: value
            for order_id, value in result.items()
            if (value <= 0.0 if positive else value < 0.0)
        }
        if invalid:
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"Order {name}s must be {qualifier}: {invalid}")
        return result

    def order_cost(self, order_id: int, flow_time_s: float) -> float:
        if order_id not in self.arrivals:
            raise KeyError(f"Unknown order: {order_id}")
        if flow_time_s < 0.0:
            raise ValueError("Flow time cannot be negative")
        excess = max(0.0, float(flow_time_s) - self.thresholds_s[order_id])
        return self.weights[order_id] * excess**self.power

    def accrued_cost(
        self,
        current_time: float,
        completion_times: Mapping[int, float],
    ) -> float:
        """Return cost accrued by completed and currently active orders."""
        now = float(current_time)
        total = 0.0
        for order_id, arrival in self.arrivals.items():
            completion = completion_times.get(order_id)
            if completion is not None and completion < arrival:
                raise ValueError(f"Order {order_id} completed before arrival")
            if now < arrival:
                continue
            end = now if completion is None else min(now, float(completion))
            total += self.order_cost(order_id, max(0.0, end - arrival))
        return total

    def objective(self, completion_times: Mapping[int, float]) -> float:
        """Return the completed episode objective and require every order."""
        missing = set(self.arrivals) - set(completion_times)
        if missing:
            raise ValueError(f"Missing completions for orders {sorted(missing)}")
        return sum(
            self.order_cost(order_id, completion_times[order_id] - arrival)
            for order_id, arrival in self.arrivals.items()
        )

    def incremental_reward(
        self,
        previous_cost: float,
        current_cost: float,
    ) -> float:
        """Return the normalized negative cost increment."""
        increment = float(current_cost) - float(previous_cost)
        if increment < -1e-9:
            raise ValueError("Accrued order cost cannot decrease")
        return -max(0.0, increment) / self.normalizer
