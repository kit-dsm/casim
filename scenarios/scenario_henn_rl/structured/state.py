"""Typed structured-batching decision state.

The structured observation is pure decision data: the visible orders, their
features and demands, the cart capacity, the order pick positions required by
the route-aware decoder, and the stream-closure flag.  It deliberately carries
no reference to the live environment: a state copied into replay must remain
usable after the originating episode has advanced or closed.

Previously this state was an unstructured dictionary that smuggled a bound
``_route_cost_fn`` (a method of the live episode) and a ``_route_cost_scale``
through private keys.  That made replay states depend on the live environment
and broke serialization.  The route-cost callable now lives on the decoder
(owned by the policy), and only the scalar route-cost scale is stored here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

OrderPositions = list[list[tuple[int, int]]]


@dataclass(frozen=True)
class BatchingState:
    """Pure, environment-independent structured-batching decision state.

    Attributes:
        order_ids: visible order identifiers, sorted by (arrival, id).
        features: per-order feature matrix, shape ``(n, feature_count)``.
        demands: per-order item demand, shape ``(n,)``.
        capacity: picker cart capacity in items.
        order_positions: per-order list of ``(aisle, y)`` pick positions used by
            the route-aware decoder to evaluate batch route cost.
        input_closed: whether the order stream is exhausted (no more arrivals).
        feature_schema: name of the feature layout, e.g. ``deadline_v1``.
        route_cost_scale: route-cost normalization scale used by the route-aware
            decoder.  Zero for knapsack states; a positive layout-derived scale
            for route-aware states.
    """

    order_ids: np.ndarray
    features: np.ndarray
    demands: np.ndarray
    capacity: int
    order_positions: OrderPositions
    input_closed: bool
    feature_schema: str
    route_cost_scale: float = 0.0

    def __len__(self) -> int:
        return int(len(self.order_ids))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view used only for JSON diagnostics/records."""
        return {
            "order_ids": self.order_ids.tolist(),
            "features": self.features.tolist(),
            "demands": self.demands.tolist(),
            "capacity": int(self.capacity),
            "order_positions": [list(p) for p in self.order_positions],
            "input_closed": bool(self.input_closed),
            "feature_schema": str(self.feature_schema),
            "route_cost_scale": float(self.route_cost_scale),
        }


def copy_state(state: BatchingState | None) -> BatchingState | None:
    """Return an independent copy of ``state`` safe to store in replay."""
    if state is None:
        return None
    return BatchingState(
        order_ids=state.order_ids.copy(),
        features=state.features.copy(),
        demands=state.demands.copy(),
        capacity=int(state.capacity),
        order_positions=[list(positions) for positions in state.order_positions],
        input_closed=bool(state.input_closed),
        feature_schema=str(state.feature_schema),
        route_cost_scale=float(state.route_cost_scale),
    )


def state_features_tensor(
    state: BatchingState, *, location_features: bool = True
) -> torch.Tensor:
    """Return the per-order feature tensor, optionally masking location columns.

    The location columns (indices 3:6 in the ``deadline_v1`` schema) encode
    absolute pick positions; masking them probes the actor without the
    location inductive bias.
    """
    values = torch.as_tensor(state.features, dtype=torch.float32)
    if not location_features:
        values = values.clone()
        values[:, 3:6] = 0.0
    return values


def action_mask(indices: np.ndarray, size: int) -> torch.Tensor:
    """Return a float indicator vector of ``size`` with ``indices`` set to 1."""
    result = torch.zeros(size, dtype=torch.float32)
    result[torch.as_tensor(np.asarray(indices, dtype=int), dtype=torch.long)] = 1.0
    return result


@dataclass
class Transition:
    """One replay transition over independent copied states.

    Mutable: ``return_to_go`` is assigned after the episode completes and the
    discounted returns are computed.  The embedded ``state``/``next_state``
    are frozen ``BatchingState`` copies, so the transition's mutability does
    not endanger replay-state independence.
    """

    state: BatchingState
    action: torch.Tensor
    reward: float
    next_state: BatchingState | None
    done: bool
    return_to_go: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


def copy_transition(transition: Transition) -> Transition:
    """Return a transition whose states are independent copies."""
    return Transition(
        state=copy_state(transition.state),
        action=transition.action.detach().clone(),
        reward=float(transition.reward),
        next_state=copy_state(transition.next_state),
        done=bool(transition.done),
        return_to_go=float(transition.return_to_go),
        extras=dict(transition.extras),
    )
