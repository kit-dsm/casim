"""Versioned structured-workbench checkpoint I/O and compatibility checks.

A full checkpoint reproduces the policy: feature schema, actor and critic
architecture and state, decoder type and parameters, route-cost scaling,
reward/objective configuration, and the data configuration identity.  The same
loadable schema is used for every advertised checkpoint file.  Evaluation and
audit reconstruct the exact actor-decoder combination from the checkpoint and
reject incompatible objective, feature schema, data, or decoder configuration
instead of silently evaluating a different policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from scenarios.scenario_henn_rl.structured.models import (
    CRITIC_TYPES,
    OrderScoreActor,
)

CHECKPOINT_FORMAT_VERSION = 2

# Data-spec fields that define the policy's training distribution and feature
# scaling.  Split sizes (train/validation/test instance counts) are excluded:
# evaluating a held-out split of a different size is legitimate and does not
# change the policy.
_POLICY_DATA_FIELDS = (
    "seed",
    "base_instance_id",
    "order_counts",
    "cart_capacity",
    "location_policy",
    "min_order_items",
    "max_order_items",
    "orders_per_four_hours",
    "minimum_lead_hours",
    "cutoff_cadence_hours",
    "travel_speed",
    "time_per_pick",
    "tour_setup_time",
)


def _policy_data_identity(data_spec: Any) -> tuple:
    spec = dict(data_spec)
    return tuple(spec.get(field) for field in _POLICY_DATA_FIELDS)


def build_checkpoint(
    *,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    critic_kind: str,
    feature_schema: str,
    feature_count: int,
    actor_hidden: int,
    decoder_config: dict,
    objective: dict,
    objective_scale: float | None,
    data_spec: dict,
    selected_episode: int,
    route_cost_scale: float = 0.0,
) -> dict:
    """Assemble a versioned, self-describing checkpoint dict."""
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "feature_schema": str(feature_schema),
        "feature_count": int(feature_count),
        "actor_hidden": int(actor_hidden),
        "critic_kind": str(critic_kind),
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "decoder": {k: v for k, v in dict(decoder_config).items()},
        "objective": dict(objective),
        "objective_scale": objective_scale,
        "data": dict(data_spec),
        "selected_episode": int(selected_episode),
        "route_cost_scale": float(route_cost_scale),
    }


def save_checkpoint(path: str | Path, checkpoint: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_workbench_checkpoint(path: str | Path):
    """Load a versioned workbench checkpoint and validate its schema."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a structured-workbench checkpoint")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version: "
            f"{checkpoint.get('format_version')!r}; expected "
            f"{CHECKPOINT_FORMAT_VERSION}"
        )
    if checkpoint.get("feature_schema") != "deadline_v1":
        raise ValueError(
            f"Unsupported feature schema: {checkpoint.get('feature_schema')!r}"
        )
    if "decoder" not in checkpoint:
        raise ValueError("Checkpoint is missing the decoder configuration")
    feature_count = int(checkpoint["feature_count"])
    actor = OrderScoreActor(
        feature_count=feature_count,
        hidden=int(checkpoint["actor_hidden"]),
    )
    critic_kind = str(checkpoint["critic_kind"])
    if critic_kind not in CRITIC_TYPES:
        raise ValueError(f"Unknown checkpoint critic: {critic_kind!r}")
    critic = CRITIC_TYPES[critic_kind](feature_count=feature_count)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    actor.eval()
    critic.eval()
    return actor, critic, checkpoint


def checkpoint_decoder_name(checkpoint: dict) -> str:
    return str(checkpoint["decoder"]["name"])


def checkpoint_route_cost_scale(checkpoint: dict) -> float:
    return float(checkpoint.get("route_cost_scale", 0.0))


def assert_policy_compatible(
    checkpoint: dict,
    *,
    objective: dict | None = None,
    data_spec: dict | None = None,
    decoder_name: str | None = None,
    feature_schema: str | None = None,
) -> None:
    """Reject a command whose policy-defining configuration is incompatible.

    Evaluation and audit must reconstruct the exact actor-decoder combination
    stored in the checkpoint; an incompatible objective, data configuration,
    decoder, or feature schema is rejected explicitly rather than silently
    evaluating a different policy.
    """
    cp_decoder = checkpoint_decoder_name(checkpoint)
    if decoder_name is not None and decoder_name != cp_decoder:
        raise ValueError(
            f"Checkpoint decoder {cp_decoder!r} is incompatible with the "
            f"requested decoder {decoder_name!r}; evaluate one policy under "
            f"one decoder"
        )
    if feature_schema is not None and checkpoint.get("feature_schema") != feature_schema:
        raise ValueError(
            f"Checkpoint feature schema {checkpoint.get('feature_schema')!r} "
            f"is incompatible with the requested schema {feature_schema!r}"
        )
    if objective is not None:
        cp_objective = dict(checkpoint.get("objective", {}))
        requested = dict(objective)
        if cp_objective and requested and cp_objective != requested:
            raise ValueError(
                "Checkpoint objective is incompatible with the requested "
                f"objective: {cp_objective} vs {requested}"
            )
    if data_spec is not None:
        cp_identity = _policy_data_identity(checkpoint.get("data", {}))
        requested_identity = _policy_data_identity(data_spec)
        if cp_identity and requested_identity and cp_identity != requested_identity:
            raise ValueError(
                "Checkpoint data configuration is incompatible with the "
                "requested data configuration"
            )


def assert_route_cost_scale_matches(
    checkpoint: dict, route_cost_scale: float, *, tol: float = 1e-6
) -> None:
    """Verify the live route-cost scale equals the checkpoint scale."""
    expected = checkpoint_route_cost_scale(checkpoint)
    if checkpoint_decoder_name(checkpoint) == "route_aware_greedy" and abs(
        expected - float(route_cost_scale)
    ) > tol:
        raise ValueError(
            f"Route-cost scale mismatch: checkpoint {expected} vs live "
            f"{float(route_cost_scale)}; the layout/data configuration differs "
            f"from the checkpoint"
        )
