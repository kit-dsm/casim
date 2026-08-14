"""Structured-batching decoders: scores + structured state -> feasible batch.

There is exactly one supported path from actor scores to a decoded action: a
``Decoder.select`` call.  Training, validation, evaluation, checkpoint
evaluation, and audit all go through the same decoder instance owned by the
``StructuredPolicy``.

Two decoders are supported:

* ``knapsack`` -- exact additive-score 0/1 knapsack over per-order scores
  subject to picker capacity.  This is the exact decoder for an additive score
  objective and ignores route cost.
* ``route_aware_greedy`` -- a *greedy* marginal-acceptance decoder that
  approximates the route-aware pricing problem.  It is not an exact
  profitable-SPRP oracle and must not be presented as one.

The route-aware problem being approximated is::

    maximize    sum_i score[i] * x_i - route_cost_weight * route_cost(B)
    subject to  sum_i demand[i] * x_i <= capacity
                B is nonempty (waiting is not supported by this decoder)

where ``route_cost(B)`` is the fixed S-shape route distance of the batch of
orders in ``B`` (the same router used for dispatch), and
``route_cost_weight = 1 / route_cost_scale``.  The greedy decoder sorts orders
by decreasing score and accepts order ``i`` when its marginal contribution
``sum scores(B + i) - route_cost(B + i) / scale - [sum scores(B) - route_cost(B) / scale]``
is positive, subject to capacity.  Tie-breaking is deterministic: decreasing
score, then the original order index.  The batch is always nonempty if any
feasible order exists (waiting is not decoded).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Protocol

import numpy as np

from scenarios.scenario_henn_rl.structured.state import BatchingState

RouteCostFn = Callable[[list[list[tuple[int, int]]], list[int]], float]


@dataclass(frozen=True)
class DecoderResult:
    """Result of decoding one decision."""

    selected_indices: np.ndarray
    objective: float
    route_cost: float


class Decoder(Protocol):
    """Scores + structured state -> feasible batch selection."""

    name: str

    def select(self, scores: np.ndarray, state: BatchingState) -> DecoderResult:
        ...

    def config(self) -> dict:
        """Return a serializable description of this decoder."""
        ...


def knapsack_batch(
    scores: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    *,
    allow_empty: bool,
) -> np.ndarray:
    """Return the exact additive-score capacity-feasible subset."""
    scores = np.asarray(scores, dtype=float)
    demands = np.asarray(demands, dtype=int)
    if scores.ndim != 1 or scores.shape != demands.shape:
        raise ValueError("Scores and demands must be equally sized vectors")
    if np.any(demands <= 0) or np.any(demands > capacity):
        raise ValueError("Every order demand must fit the positive capacity")
    values = np.full(capacity + 1, -np.inf)
    values[0] = 0.0
    selections: list[tuple[int, ...] | None] = [None] * (capacity + 1)
    selections[0] = ()
    for index, (score, demand) in enumerate(zip(scores, demands)):
        for used in range(capacity - int(demand), -1, -1):
            if selections[used] is None:
                continue
            candidate = values[used] + float(score)
            target = used + int(demand)
            if candidate > values[target] + 1e-12:
                values[target] = candidate
                selections[target] = selections[used] + (index,)
    best_capacity = int(np.argmax(values))
    selected = selections[best_capacity] or ()
    if not selected and not allow_empty:
        feasible = np.flatnonzero(demands <= capacity)
        selected = (int(feasible[np.argmax(scores[feasible])]),)
    return np.asarray(selected, dtype=int)


def greedy_route_aware_batch(
    scores: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    order_positions: list[list[tuple[int, int]]],
    route_cost_fn: RouteCostFn,
    route_cost_scale: float,
    *,
    allow_empty: bool,
) -> tuple[np.ndarray, float, float]:
    """Greedy marginal-cost route-aware batch selection.

    Returns ``(selected_indices, objective, route_cost)`` where ``objective``
    is ``sum scores(B) - route_cost(B) / scale``.  This is a greedy
    approximation of the route-aware pricing problem, not an exact oracle.
    """
    scores = np.asarray(scores, dtype=float)
    demands = np.asarray(demands, dtype=int)
    scale = float(route_cost_scale) if route_cost_scale > 0 else 1.0
    n = len(scores)
    if n == 0:
        return np.asarray([], dtype=int), 0.0, 0.0
    order = np.argsort(-scores, kind="stable")
    selected: list[int] = []
    selected_demand = 0
    current_value = 0.0
    current_route_cost = 0.0
    for i in order:
        i = int(i)
        if selected_demand + int(demands[i]) > capacity:
            continue
        candidate = selected + [i]
        new_route_cost = float(route_cost_fn(order_positions, candidate))
        new_value = float(scores[candidate].sum()) - new_route_cost / scale
        marginal = new_value - current_value
        if marginal > 0.0 or (not selected and not allow_empty):
            selected = candidate
            selected_demand += int(demands[i])
            current_value = new_value
            current_route_cost = new_route_cost
    if not selected and not allow_empty:
        feasible = np.flatnonzero(demands <= capacity)
        if feasible.size:
            selected = [int(feasible[np.argmax(scores[feasible])])]
            current_route_cost = float(route_cost_fn(order_positions, selected))
            current_value = float(scores[selected].sum()) - current_route_cost / scale
    return np.asarray(selected, dtype=int), float(current_value), float(current_route_cost)


def brute_force_route_aware_batch(
    scores: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    order_positions: list[list[tuple[int, int]]],
    route_cost_fn: RouteCostFn,
    route_cost_scale: float,
    *,
    allow_empty: bool,
) -> tuple[np.ndarray, float, float]:
    """Exact route-aware optimum by enumeration (tests only, not in RL path).

    Enumerates every capacity-feasible nonempty subset and returns the true
    maximizer of ``sum scores - route_cost / scale``.  Use this only in unit
    tests on small synthetic states to distinguish the greedy decoder from the
    exact optimum.
    """
    scores = np.asarray(scores, dtype=float)
    demands = np.asarray(demands, dtype=int)
    scale = float(route_cost_scale) if route_cost_scale > 0 else 1.0
    n = len(scores)
    best_indices: list[int] = []
    best_value = -np.inf
    best_route_cost = 0.0
    for size in range(0 if allow_empty else 1, n + 1):
        for combo in combinations(range(n), size):
            if demands[list(combo)].sum() > capacity:
                continue
            route_cost = float(route_cost_fn(order_positions, list(combo)))
            value = float(scores[list(combo)].sum()) - route_cost / scale
            if value > best_value + 1e-12:
                best_value = value
                best_indices = list(combo)
                best_route_cost = route_cost
    if not best_indices and not allow_empty:
        feasible = np.flatnonzero(demands <= capacity)
        if feasible.size:
            best_indices = [int(feasible[np.argmax(scores[feasible])])]
            best_route_cost = float(route_cost_fn(order_positions, best_indices))
            best_value = float(scores[best_indices].sum()) - best_route_cost / scale
    return np.asarray(best_indices, dtype=int), float(best_value), float(best_route_cost)


@dataclass(frozen=True)
class KnapsackDecoder:
    """Exact additive-score 0/1 knapsack decoder."""

    allow_empty: bool = False

    @property
    def name(self) -> str:
        return "knapsack"

    def select(self, scores: np.ndarray, state: BatchingState) -> DecoderResult:
        indices = knapsack_batch(
            np.asarray(scores, dtype=float),
            state.demands,
            int(state.capacity),
            allow_empty=self.allow_empty,
        )
        objective = float(np.asarray(scores, dtype=float)[indices].sum())
        return DecoderResult(
            selected_indices=indices, objective=objective, route_cost=0.0
        )

    def config(self) -> dict:
        return {"name": "knapsack", "allow_empty": bool(self.allow_empty)}


@dataclass(frozen=True)
class GreedyRouteAwareDecoder:
    """Greedy route-aware decoder (approximate, not exact pricing).

    The route cost of a candidate batch is evaluated by ``route_cost_fn``, which
    must be the fixed S-shape router used for dispatch.  The route-cost scale is
    read from the state so the decoder itself holds no per-decision data.
    """

    route_cost_fn: RouteCostFn
    allow_empty: bool = False

    @property
    def name(self) -> str:
        return "route_aware_greedy"

    def select(self, scores: np.ndarray, state: BatchingState) -> DecoderResult:
        indices, objective, route_cost = greedy_route_aware_batch(
            np.asarray(scores, dtype=float),
            state.demands,
            int(state.capacity),
            state.order_positions,
            self.route_cost_fn,
            float(state.route_cost_scale),
            allow_empty=self.allow_empty,
        )
        return DecoderResult(
            selected_indices=indices, objective=objective, route_cost=route_cost
        )

    def config(self) -> dict:
        return {"name": "route_aware_greedy", "allow_empty": bool(self.allow_empty)}


_DECODERS = {
    "knapsack": KnapsackDecoder,
    "route_aware_greedy": GreedyRouteAwareDecoder,
}


def make_decoder(
    name: str,
    *,
    route_cost_fn: RouteCostFn | None = None,
    allow_empty: bool = False,
) -> Decoder:
    """Construct a decoder by name.

    Unknown names raise ``ValueError``; selection is never silently coerced to
    knapsack.  ``route_aware_greedy`` requires a ``route_cost_fn``.
    """
    if name not in _DECODERS:
        raise ValueError(
            f"Unknown decoder {name!r}; supported: {sorted(_DECODERS)}"
        )
    if name == "route_aware_greedy":
        if route_cost_fn is None:
            raise ValueError("route_aware_greedy requires a route_cost_fn")
        return GreedyRouteAwareDecoder(route_cost_fn=route_cost_fn, allow_empty=allow_empty)
    return KnapsackDecoder(allow_empty=allow_empty)
