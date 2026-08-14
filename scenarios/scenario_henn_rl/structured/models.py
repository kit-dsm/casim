from __future__ import annotations

import numpy as np
import torch

from scenarios.scenario_henn_rl.structured.policy import StructuredPolicy
from scenarios.scenario_henn_rl.structured.state import BatchingState


class OrderScoreActor(torch.nn.Module):
    """Permutation-equivariant order scores for the structured decoder."""

    def __init__(self, feature_count: int = 7, hidden: int = 32):
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


CRITIC_TYPES = {
    "mean": StructuredCritic,
    "interaction": PairwiseStructuredCritic,
}


def make_critic(kind: str, *, feature_count: int) -> torch.nn.Module:
    if kind not in CRITIC_TYPES:
        raise ValueError(f"Unknown critic kind: {kind!r}")
    return CRITIC_TYPES[kind](feature_count=feature_count)


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


def structured_candidates(
    policy: StructuredPolicy,
    state: BatchingState,
    *,
    candidate_count: int,
    sigma: float,
    generator: torch.Generator,
    deduplicate: bool = False,
) -> list[torch.Tensor]:
    """Generate feasible candidates using the existing SRL perturbations."""
    if candidate_count < 1:
        raise ValueError("candidate_count must include one unperturbed action")
    with torch.no_grad():
        scores = policy.actor(policy.features(state))
        candidates = [policy.decode(scores, state)]
        for _ in range(candidate_count - 1):
            noise = torch.randn(scores.shape, generator=generator)
            candidates.append(policy.decode(scores + sigma * noise, state))
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
    policy: StructuredPolicy,
    critic: torch.nn.Module,
    state: BatchingState,
    *,
    candidate_count: int,
    sigma: float,
    temperature: float,
    generator: torch.Generator,
    normalize_advantages: bool = False,
    actions: list[torch.Tensor] | None = None,
    values: torch.Tensor | None = None,
):
    """SRL Eq. (4): critic-weighted average of feasible candidates.

    ``actions`` optionally supplies a pre-generated candidate set so the same
    candidates can be weighted under several target configurations; ``values``
    optionally supplies candidate quality directly (e.g. exact rollout Q)
    instead of critic predictions.
    """
    features = policy.features(state)
    with torch.no_grad():
        if actions is None:
            actions = structured_candidates(
                policy,
                state,
                candidate_count=candidate_count,
                sigma=sigma,
                generator=generator,
            )
        if values is not None:
            if len(values) != len(actions):
                raise ValueError("Values count must match the candidate set size")
        else:
            values = torch.stack(
                [critic(features, action) for action in actions]
            )
        weights = candidate_weights(
            values,
            temperature=temperature,
            normalize_advantages=normalize_advantages,
        )
        target = sum(weight * action for weight, action in zip(weights, actions))
        entropy = float(
            -(weights * weights.clamp_min(1e-12).log()).sum().cpu()
        )
    return target, {
        "candidate_count": candidate_count,
        "unique_candidates": len({tuple(action.tolist()) for action in actions}),
        "candidate_q_spread": float((values.max() - values.min()).cpu()),
        "target_entropy": entropy,
        "max_weight": float(weights.max().cpu()),
        "effective_count": float(
            1.0 / (weights.pow(2).sum().clamp_min(1e-12).cpu())
        ),
        "normalized_advantages": normalize_advantages,
    }


def fenchel_young_loss(
    policy: StructuredPolicy,
    state: BatchingState,
    target_action,
    *,
    sample_count: int,
    epsilon: float,
    generator: torch.Generator,
):
    """Monte Carlo estimate of the smoothed FY loss in SRL Eq. (2)."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    features = policy.features(state)
    scores = policy.actor(features)
    terms = []
    decoded = []
    for _ in range(sample_count):
        noise = torch.randn(scores.shape, generator=generator)
        action = policy.decode(scores.detach() + epsilon * noise, state)
        decoded.append(action)
        terms.append(((scores + epsilon * noise) * action).sum())
    loss = torch.stack(terms).mean() - (scores * target_action.detach()).sum()
    return loss, torch.stack(decoded).mean(dim=0)
