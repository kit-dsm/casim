from __future__ import annotations

import numpy as np
import torch

from scenarios.scenario_henn_rl.structured_environment import knapsack_batch


class OrderScoreActor(torch.nn.Module):
    """Permutation-equivariant order scores for the knapsack oracle."""

    def __init__(self, feature_count: int = 7, hidden: int = 32):
        super().__init__()
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
    """Evaluate a complete variable-size state/action pair."""

    def __init__(self, feature_count: int = 7, hidden: int = 32):
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
        action_column = action.to(features.dtype).reshape(-1, 1)
        encoded = self.order_encoder(
            torch.cat([features, action_column], dim=1)
        )
        return self.value(encoded.mean(dim=0)).squeeze()


class PairwiseStructuredCritic(torch.nn.Module):
    """Critic with explicit permutation-invariant selected-order interactions."""

    def __init__(self, feature_count: int = 7, hidden: int = 32):
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
        selected = (encoded * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(
            1.0
        )
        selected_encoded = encoded[weights > 0.5]
        if len(selected_encoded) < 2:
            interactions = torch.zeros_like(selected)
        else:
            indices = torch.triu_indices(
                len(selected_encoded), len(selected_encoded), offset=1
            )
            pair_inputs = torch.cat(
                [
                    selected_encoded[indices[0]] + selected_encoded[indices[1]],
                    torch.abs(
                        selected_encoded[indices[0]] - selected_encoded[indices[1]]
                    ),
                ],
                dim=1,
            )
            interactions = self.pair_encoder(pair_inputs).mean(dim=0)
        pooled = torch.cat([encoded.mean(dim=0), selected, interactions])
        return self.value(pooled).squeeze()


def _features(state: dict[str, object], *, location_features: bool = True):
    values = torch.as_tensor(state["features"], dtype=torch.float32)
    if not location_features:
        values = values.clone()
        values[:, 3:6] = 0.0
    return values


def _mask(indices: np.ndarray, size: int):
    result = torch.zeros(size, dtype=torch.float32)
    result[torch.as_tensor(indices, dtype=torch.long)] = 1.0
    return result


def decode_scores(scores, state: dict[str, object]):
    indices = knapsack_batch(
        scores.detach().cpu().numpy(),
        state["demands"],
        int(state["capacity"]),
        allow_empty=False,
    )
    return _mask(indices, len(scores))


def structured_candidates(
    actor,
    state: dict[str, object],
    *,
    candidate_count: int,
    sigma: float,
    generator,
    location_features: bool = True,
    deduplicate: bool = False,
) -> list[torch.Tensor]:
    """Generate feasible candidates using the existing SRL perturbations."""
    if candidate_count < 1:
        raise ValueError("candidate_count must include one unperturbed action")
    features = _features(state, location_features=location_features)
    with torch.no_grad():
        scores = actor(features)
        candidates = [decode_scores(scores, state)]
        for _ in range(candidate_count - 1):
            noise = torch.randn(scores.shape, generator=generator)
            candidates.append(decode_scores(scores + sigma * noise, state))
    if not deduplicate:
        return candidates
    distinct = []
    seen = set()
    for action in candidates:
        key = tuple(action.tolist())
        if key not in seen:
            seen.add(key)
            distinct.append(action)
    return distinct


def critic_soft_target(
    actor,
    critic,
    state: dict[str, object],
    *,
    candidate_count: int,
    sigma: float,
    temperature: float,
    generator,
    location_features: bool = True,
    normalize_advantages: bool = False,
):
    """SRL Eq. (4): critic-weighted average of feasible candidates."""
    features = _features(state, location_features=location_features)
    with torch.no_grad():
        actions = structured_candidates(
            actor,
            state,
            candidate_count=candidate_count,
            sigma=sigma,
            generator=generator,
            location_features=location_features,
        )
        values = torch.stack(
            [critic(features, action) for action in actions]
        )
        weights = candidate_weights(
            values,
            temperature=temperature,
            normalize_advantages=normalize_advantages,
        )
        target = sum(
            weight * action for weight, action in zip(weights, actions)
        )
    entropy = float(
        -(weights * weights.clamp_min(1e-12).log()).sum().cpu()
    )
    return target, {
        "candidate_count": candidate_count,
        "unique_candidates": len({tuple(action.tolist()) for action in actions}),
        "candidate_q_spread": float((values.max() - values.min()).cpu()),
        "target_entropy": entropy,
        "normalized_advantages": normalize_advantages,
    }

def candidate_weights(
    values: torch.Tensor,
    *,
    temperature: float,
    normalize_advantages: bool,
) -> torch.Tensor:
    """Return candidate weights, optionally standardized within the state."""
    if temperature <= 0.0:
        raise ValueError("Candidate temperature must be positive")
    logits = values
    if normalize_advantages:
        scale = values.std(unbiased=False).clamp_min(1e-6)
        logits = (values - values.mean()) / scale
    return torch.softmax(logits / temperature, dim=0)


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Return complete return-to-go labels for one finished episode."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("Return discount must be in [0, 1]")
    returns = [0.0] * len(rewards)
    future = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        future = float(rewards[index]) + gamma * future
        returns[index] = future
    return returns


def fenchel_young_loss(
    actor,
    state: dict[str, object],
    target_action,
    *,
    sample_count: int,
    epsilon: float,
    generator,
    location_features: bool = True,
):
    """Monte Carlo estimate of the smoothed FY loss in SRL Eq. (2)."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    features = _features(state, location_features=location_features)
    scores = actor(features)
    terms = []
    decoded = []
    for _ in range(sample_count):
        noise = torch.randn(scores.shape, generator=generator)
        action = decode_scores(scores.detach() + epsilon * noise, state)
        decoded.append(action)
        terms.append(((scores + epsilon * noise) * action).sum())
    loss = torch.stack(terms).mean() - (scores * target_action.detach()).sum()
    return loss, torch.stack(decoded).mean(dim=0)


def _copy_state(state: dict[str, object] | None):
    if state is None:
        return None
    return {
        "order_ids": state["order_ids"].copy(),
        "features": state["features"].copy(),
        "demands": state["demands"].copy(),
        "capacity": int(state["capacity"]),
        "input_closed": bool(state["input_closed"]),
    }
