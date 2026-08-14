"""Structured policy: the single entry point for selecting a batch.

``StructuredPolicy`` composes the order-score actor, the feature
transformation, and the configured decoder.  Training, validation, evaluation,
checkpoint evaluation, and audit all select actions through this abstraction so
that actor and decoder cannot be mixed and matched by accident.

The policy does not own the critic, replay buffer, simulator, or experiment
configuration; it only maps a structured state to a feasible action.
"""

from __future__ import annotations

import numpy as np
import torch

from scenarios.scenario_henn_rl.structured.decoders import Decoder
from scenarios.scenario_henn_rl.structured.state import (
    action_mask,
    state_features_tensor,
    BatchingState,
)


class StructuredPolicy:
    """Actor + feature transformation + decoder -> feasible batch action."""

    def __init__(
        self,
        actor: torch.nn.Module,
        decoder: Decoder,
        *,
        location_features: bool = True,
    ):
        self.actor = actor
        self.decoder = decoder
        self.location_features = bool(location_features)

    @property
    def decoder_name(self) -> str:
        return self.decoder.name

    def features(self, state: BatchingState) -> torch.Tensor:
        return state_features_tensor(state, location_features=self.location_features)

    def decode(self, scores: torch.Tensor | np.ndarray, state: BatchingState) -> torch.Tensor:
        """Return the feasible-action indicator vector for ``scores``.

        Decoding never carries a gradient: the decoder operates on a detached
        numpy copy of the scores.  Gradients flow through the score terms in
        the structured losses, not through the discrete action.
        """
        if isinstance(scores, torch.Tensor):
            scores_np = scores.detach().cpu().numpy()
        else:
            scores_np = np.asarray(scores, dtype=float)
        size = len(scores_np)
        indices = self.decoder.select(scores_np, state).selected_indices
        return action_mask(indices, size)

    def select_action(
        self,
        state: BatchingState,
        *,
        sigma: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return the feasible-action indicator vector for ``state``."""
        with torch.no_grad():
            scores = self.actor(self.features(state))
            if sigma > 0.0 and generator is not None:
                scores = scores + sigma * torch.randn(
                    scores.shape, generator=generator
                )
        return self.decode(scores, state)

    def select_indices(
        self,
        state: BatchingState,
        *,
        sigma: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> np.ndarray:
        """Return the selected order indices for ``state``."""
        with torch.no_grad():
            scores = self.actor(self.features(state))
            if sigma > 0.0 and generator is not None:
                scores = scores + sigma * torch.randn(
                    scores.shape, generator=generator
                )
        return self.decoder.select(scores.detach().cpu().numpy(), state).selected_indices
