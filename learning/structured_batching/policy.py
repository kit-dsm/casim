"""Observation construction, actor/critic networks, and decoding for batching RL.

The runtime path is deliberately direct: make an observation from CASIM's
snapshot and resolved orders, score orders with the actor, decode a feasible
set, and pass those order IDs back to CASIM.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BatchingObservation:
    """The immutable tensor boundary stored in replay."""

    order_ids: np.ndarray
    features: np.ndarray
    demands: np.ndarray
    capacity: int
    order_positions: tuple[tuple[object, ...], ...]
    route_cost_scale: float = 0.0

    def __len__(self):
        return len(self.order_ids)


def make_observation(
    snapshot,
    orders,
    *,
    total_orders: int,
    episode_horizon: float,
    route_aware: bool,
) -> BatchingObservation:
    """Build the existing eight order features from semantic planning data."""
    now = float(snapshot.dynamic_warehouse_info.time or 0.0)
    horizon = max(1.0, float(episode_horizon))
    capacity = int(
        snapshot.resources.resources[0].pick_cart.capacities[0]
        * snapshot.resources.resources[0].pick_cart.n_boxes
    )
    position_scale = max(
        1.0,
        max(
            float(node[1])
            for node in snapshot.layout.layout_network.graph.nodes
            if isinstance(node, tuple) and len(node) > 1
        ),
    )
    features = []
    demands = []
    positions_by_order = []
    for order in orders:
        positions = tuple(order.pick_positions)
        y_positions = [float(position.pick_node[1]) for position in positions]
        demand = sum(int(position.in_store) for position in positions)
        if order.due_date is None:
            raise ValueError("Structured batching requires order due dates")
        demands.append(demand)
        positions_by_order.append(positions)
        features.append(
            [
                max(0.0, now - float(order.order_date or 0.0)) / horizon,
                demand / max(1, capacity),
                len(positions) / max(1, capacity),
                min(y_positions) / position_scale,
                max(y_positions) / position_scale,
                float(np.mean(y_positions)) / position_scale,
                len(orders) / max(1, total_orders),
                float(
                    np.clip(
                        (float(order.due_date) - now) / horizon,
                        -1.0,
                        1.0,
                    )
                ),
            ]
        )
    graph = snapshot.layout.graph_data
    one_pass = (
        graph.dist_pick_locations * (graph.n_pick_locations - 1)
        + 2 * graph.dist_bottom_to_pick_location
    )
    route_scale = (
        float(max(1.0, graph.n_aisles * one_pass)) if route_aware else 0.0
    )
    return BatchingObservation(
        order_ids=np.asarray([int(order.order_id) for order in orders], dtype=np.int64),
        features=np.asarray(features, dtype=np.float32),
        demands=np.asarray(demands, dtype=np.int64),
        capacity=capacity,
        order_positions=tuple(positions_by_order),
        route_cost_scale=route_scale,
    )


def features_tensor(observation, *, location_features=True):
    values = torch.as_tensor(observation.features, dtype=torch.float32)
    if not location_features:
        values = values.clone()
        values[:, 3:6] = 0.0
    return values


class OrderScoreActor(torch.nn.Module):
    """Permutation-equivariant order scores for the structured decoder."""

    def __init__(self, feature_count=8, hidden=32):
        super().__init__()
        self.feature_count = int(feature_count)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_count, hidden),
            torch.nn.ReLU(),
        )
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, features):
        encoded = self.encoder(features)
        context = encoded.mean(dim=0, keepdim=True).expand_as(encoded)
        return self.scorer(torch.cat([encoded, context], dim=1)).flatten()


class StructuredCritic(torch.nn.Module):
    def __init__(self, feature_count=8, hidden=32):
        super().__init__()
        self.order_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_count + 1, hidden),
            torch.nn.ReLU(),
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, features, action):
        encoded = self.order_encoder(
            torch.cat([features, action.to(features.dtype).reshape(-1, 1)], dim=1)
        )
        return self.value(encoded.mean(dim=0)).squeeze()


class PairwiseStructuredCritic(torch.nn.Module):
    """Permutation-invariant critic with selected-order interactions."""

    def __init__(self, feature_count=8, hidden=32):
        super().__init__()
        self.order_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_count, hidden),
            torch.nn.ReLU(),
        )
        self.pair_encoder = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden, hidden),
            torch.nn.ReLU(),
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(3 * hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, features, action):
        encoded = self.order_encoder(features)
        weights = action.to(features.dtype).flatten()
        selected = (encoded * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(1.0)
        selected_encoded = encoded[weights > 0.5]
        if len(selected_encoded) < 2:
            interactions = torch.zeros_like(selected)
        else:
            indices = torch.triu_indices(len(selected_encoded), len(selected_encoded), offset=1)
            pair_inputs = torch.cat(
                [
                    selected_encoded[indices[0]] + selected_encoded[indices[1]],
                    torch.abs(selected_encoded[indices[0]] - selected_encoded[indices[1]]),
                ],
                dim=1,
            )
            interactions = self.pair_encoder(pair_inputs).mean(dim=0)
        return self.value(
            torch.cat([encoded.mean(dim=0), selected, interactions])
        ).squeeze()


CRITIC_TYPES = {
    "mean": StructuredCritic,
    "interaction": PairwiseStructuredCritic,
}


def make_critic(kind, *, feature_count, hidden=32):
    if kind not in CRITIC_TYPES:
        raise ValueError(f"Unknown critic kind: {kind!r}")
    return CRITIC_TYPES[kind](feature_count=feature_count, hidden=hidden)


def knapsack_batch(scores, demands, capacity):
    """Return the exact additive-score capacity-feasible nonempty subset."""
    scores = np.asarray(scores, dtype=float)
    demands = np.asarray(demands, dtype=int)
    if scores.ndim != 1 or scores.shape != demands.shape:
        raise ValueError("Scores and demands must be equally sized vectors")
    if np.any(demands <= 0) or np.any(demands > capacity):
        raise ValueError("Every order demand must fit the positive capacity")
    values = np.full(capacity + 1, -np.inf)
    values[0] = 0.0
    selections = [None] * (capacity + 1)
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
    selected = selections[int(np.argmax(values))] or ()
    if not selected:
        feasible = np.flatnonzero(demands <= capacity)
        selected = (int(feasible[np.argmax(scores[feasible])]),)
    return np.asarray(selected, dtype=int)


def _selected_positions(order_positions, selected_indices):
    return [
        position
        for index in selected_indices
        for position in order_positions[int(index)]
    ]


def greedy_route_aware_batch(
    scores,
    demands,
    capacity,
    order_positions,
    router,
    route_cost_scale,
):
    """Greedily accept positive route-aware marginal contributions."""
    scores = np.asarray(scores, dtype=float)
    demands = np.asarray(demands, dtype=int)
    scale = float(route_cost_scale) if route_cost_scale > 0 else 1.0
    selected = []
    used = 0
    value = 0.0
    route_cost = 0.0
    for index in np.argsort(-scores, kind="stable"):
        index = int(index)
        if used + int(demands[index]) > capacity:
            continue
        candidate = selected + [index]
        candidate_cost = float(
            router.score(_selected_positions(order_positions, candidate))
        )
        candidate_value = float(scores[candidate].sum()) - candidate_cost / scale
        if candidate_value > value or not selected:
            selected = candidate
            used += int(demands[index])
            value = candidate_value
            route_cost = candidate_cost
    if not selected:
        feasible = np.flatnonzero(demands <= capacity)
        selected = [int(feasible[np.argmax(scores[feasible])])]
        route_cost = float(router.score(_selected_positions(order_positions, selected)))
        value = float(scores[selected].sum()) - route_cost / scale
    return np.asarray(selected, dtype=int), float(value), float(route_cost)


def decode_indices(scores, observation, decoder_name, router=None):
    """Turn actor scores into one feasible nonempty batch."""
    if decoder_name == "knapsack":
        return knapsack_batch(scores, observation.demands, observation.capacity)
    if decoder_name == "route_aware_greedy":
        if router is None:
            raise ValueError("route_aware_greedy requires the fixed router")
        return greedy_route_aware_batch(
            scores,
            observation.demands,
            observation.capacity,
            observation.order_positions,
            router,
            observation.route_cost_scale,
        )[0]
    raise ValueError(f"Unknown decoder: {decoder_name!r}")


def action_mask(indices, size):
    result = torch.zeros(size, dtype=torch.float32)
    result[torch.as_tensor(np.asarray(indices, dtype=int), dtype=torch.long)] = 1.0
    return result


def decode_action(scores, observation, decoder_name, router=None):
    values = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else scores
    return action_mask(
        decode_indices(values, observation, decoder_name, router),
        len(values),
    )
