from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from scenarios.scenario_henn_rl.counterfactuals import _counterfactual_examples
from scenarios.scenario_henn_rl.structured_policy import (
    PairwiseStructuredCritic,
    StructuredCritic,
    _features,
    candidate_weights,
    fenchel_young_loss,
)


def _critic_memorization_metrics(critic, examples) -> dict[str, float | None]:
    correlations = []
    top_matches = []
    pairwise_matches = []
    losses = []
    with torch.no_grad():
        for state, actions, target in examples:
            features = _features(state)
            predicted = torch.stack(
                [critic(features, action) for action in actions]
            )
            losses.append(
                float(torch.nn.functional.smooth_l1_loss(predicted, target))
            )
            correlation = spearmanr(
                predicted.cpu().numpy(), target.cpu().numpy()
            ).statistic
            if not np.isnan(correlation):
                correlations.append(float(correlation))
            top_matches.append(int(predicted.argmax() == target.argmax()))
            differences = target[:, None] - target[None, :]
            predicted_differences = predicted[:, None] - predicted[None, :]
            keep = torch.triu(torch.ones_like(differences, dtype=torch.bool), 1)
            keep &= differences.abs() > 1e-8
            pairwise_matches.append(
                float(
                    (
                        torch.sign(predicted_differences[keep])
                        == torch.sign(differences[keep])
                    ).float().mean()
                )
            )
    return {
        "mean_loss": float(np.mean(losses)),
        "mean_spearman": float(np.mean(correlations)) if correlations else None,
        "top_accuracy": float(np.mean(top_matches)),
        "pairwise_accuracy": float(np.mean(pairwise_matches)),
    }

def overfit_counterfactual_critic(
    actor,
    audit: dict[str, object],
    *,
    epochs: int,
    learning_rate: float,
    loss_kind: str,
    seed: int,
    interaction_aware: bool = False,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Test whether the existing critic can memorize exact candidate ranks."""
    if epochs < 1:
        raise ValueError("Memorization epochs must be positive")
    if loss_kind not in {"regression", "pairwise_ranking", "pairwise_top"}:
        raise ValueError(f"Unknown critic memorization loss: {loss_kind!r}")
    examples = _counterfactual_examples(actor, audit)
    torch.manual_seed(seed)
    critic = (
        PairwiseStructuredCritic() if interaction_aware else StructuredCritic()
    )
    optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    checkpoints = sorted({0, epochs, epochs // 20, epochs // 4, epochs // 2})
    history = []
    for epoch in range(epochs + 1):
        if epoch in checkpoints:
            history.append(
                {"epoch": epoch, **_critic_memorization_metrics(critic, examples)}
            )
        if epoch == epochs:
            break
        losses = []
        for state, actions, target in examples:
            features = _features(state)
            predicted = torch.stack(
                [critic(features, action) for action in actions]
            )
            if loss_kind == "regression":
                losses.append(
                    torch.nn.functional.smooth_l1_loss(predicted, target)
                )
                continue
            pair_indices = torch.triu_indices(len(target), len(target), offset=1)
            true_differences = target[pair_indices[0]] - target[pair_indices[1]]
            keep = true_differences.abs() > 1e-8
            predicted_differences = (
                predicted[pair_indices[0]] - predicted[pair_indices[1]]
            )[keep]
            labels = (true_differences[keep] > 0).to(predicted.dtype)
            losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    predicted_differences, labels
                )
            )
            if loss_kind == "pairwise_top":
                losses[-1] = losses[-1] + torch.nn.functional.cross_entropy(
                    predicted.unsqueeze(0), target.argmax().reshape(1)
                )
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = history[-1]
    return critic, {
        "loss": loss_kind,
        "critic": "pairwise_interaction" if interaction_aware else "mean_pooled",
        "states": len(examples),
        "candidates": sum(len(actions) for _, actions, _ in examples),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "history": history,
        "final": final,
    }


def calibrate_from_counterfactual_audit(
    actor,
    critic,
    audit: dict[str, object],
    *,
    critic_epochs: int,
    actor_epochs: int,
    critic_learning_rate: float,
    actor_learning_rate: float,
    epsilon: float,
    fy_samples: int,
    seed: int,
) -> dict[str, object]:
    """Fit the existing critic and actor on exact training-only branches."""
    if critic_epochs < 1 or actor_epochs < 1:
        raise ValueError("Counterfactual calibration epochs must be positive")
    examples = _counterfactual_examples(actor, audit)

    before = _critic_memorization_metrics(critic, examples)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=critic_learning_rate
    )
    for _ in range(critic_epochs):
        for state, actions, target in examples:
            features = _features(state)
            predicted = torch.stack(
                [critic(features, action) for action in actions]
            )
            loss = torch.nn.functional.smooth_l1_loss(predicted, target)
            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()
    after_critic = _critic_memorization_metrics(critic, examples)

    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=actor_learning_rate
    )
    generator = torch.Generator().manual_seed(seed)
    actor_losses = []
    for _ in range(actor_epochs):
        for state, actions, _ in examples:
            features = _features(state)
            with torch.no_grad():
                values = torch.stack(
                    [critic(features, action) for action in actions]
                )
                weights = candidate_weights(
                    values,
                    temperature=1.0,
                    normalize_advantages=True,
                )
                target_action = sum(
                    weight * action
                    for weight, action in zip(weights, actions)
                )
            loss, _ = fenchel_young_loss(
                actor,
                state,
                target_action,
                sample_count=fy_samples,
                epsilon=epsilon,
                generator=generator,
            )
            actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()
            actor_losses.append(float(loss.detach()))
    return {
        "states": len(examples),
        "candidates": sum(len(actions) for _, actions, _ in examples),
        "target": "state_standardized_exact_counterfactual_return",
        "critic_epochs": critic_epochs,
        "actor_epochs": actor_epochs,
        "critic_before": before,
        "critic_after": after_critic,
        "final_actor_loss": actor_losses[-1],
    }
