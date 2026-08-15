"""Candidate generation, critic targets, returns, and Fenchel-Young loss."""

import torch

from learning.structured_batching.policy import (
    decode_action,
    features_tensor,
)


def candidate_weights(values, *, temperature, normalize_advantages):
    if temperature <= 0.0:
        raise ValueError("Candidate temperature must be positive")
    logits = values
    if normalize_advantages:
        logits = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)
    return torch.softmax(logits / temperature, dim=0)


def complete_returns(rewards):
    """Return undiscounted return-to-go labels for one finite episode."""
    returns = [0.0] * len(rewards)
    future = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        future += float(rewards[index])
        returns[index] = future
    return returns


def structured_candidates(
    actor,
    observation,
    decoder_name,
    router,
    *,
    candidate_count,
    sigma,
    generator,
    location_features=True,
):
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    with torch.no_grad():
        scores = actor(features_tensor(observation, location_features=location_features))
        candidates = [decode_action(scores, observation, decoder_name, router)]
        for _ in range(candidate_count - 1):
            candidates.append(
                decode_action(
                    scores + sigma * torch.randn(scores.shape, generator=generator),
                    observation,
                    decoder_name,
                    router,
                )
            )
    return candidates


def critic_soft_target(
    actor,
    critic,
    observation,
    decoder_name,
    router,
    *,
    candidate_count,
    sigma,
    temperature,
    generator,
    normalize_advantages=False,
    location_features=True,
):
    features = features_tensor(observation, location_features=location_features)
    with torch.no_grad():
        actions = structured_candidates(
            actor,
            observation,
            decoder_name,
            router,
            candidate_count=candidate_count,
            sigma=sigma,
            generator=generator,
            location_features=location_features,
        )
        values = torch.stack([critic(features, action) for action in actions])
        weights = candidate_weights(
            values,
            temperature=temperature,
            normalize_advantages=normalize_advantages,
        )
        target = sum(weight * action for weight, action in zip(weights, actions))
    return target, {
        "unique_candidates": len({tuple(action.tolist()) for action in actions}),
        "candidate_q_spread": float((values.max() - values.min()).cpu()),
    }


def fenchel_young_loss(
    actor,
    observation,
    target_action,
    decoder_name,
    router,
    *,
    sample_count,
    epsilon,
    generator,
    location_features=True,
):
    features = features_tensor(observation, location_features=location_features)
    scores = actor(features)
    terms, decoded = [], []
    for _ in range(sample_count):
        noise = torch.randn(scores.shape, generator=generator)
        action = decode_action(
            scores.detach() + epsilon * noise,
            observation,
            decoder_name,
            router,
        )
        decoded.append(action)
        terms.append(((scores + epsilon * noise) * action).sum())
    loss = torch.stack(terms).mean() - (scores * target_action.detach()).sum()
    return loss, torch.stack(decoded).mean(dim=0)
