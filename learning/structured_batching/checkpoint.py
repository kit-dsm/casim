"""Versioned checkpoint I/O and compatibility checks for Structured-RL."""

from pathlib import Path

import torch

from learning.structured_batching.policy import (
    CRITIC_TYPES,
    OrderScoreActor,
    make_critic,
)

CHECKPOINT_FORMAT_VERSION = 4

_DATA_IDENTITY_FIELDS = (
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


def build_checkpoint(
    *,
    actor,
    critic,
    critic_kind,
    feature_count,
    actor_hidden,
    decoder_name,
    data_spec,
    selected_episode,
    route_cost_scale=0.0,
    objective_scale=1.0,
):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "feature_count": int(feature_count),
        "actor_hidden": int(actor_hidden),
        "critic_kind": str(critic_kind),
        "critic_hidden": int(critic.order_encoder[0].out_features),
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "decoder": str(decoder_name),
        "data": dict(data_spec),
        "selected_episode": int(selected_episode),
        "route_cost_scale": float(route_cost_scale),
        "objective_scale": float(objective_scale),
    }


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError("Unsupported Structured-RL checkpoint")
    feature_count = int(checkpoint["feature_count"])
    actor = OrderScoreActor(feature_count, int(checkpoint["actor_hidden"]))
    critic = make_critic(
        str(checkpoint["critic_kind"]),
        feature_count=feature_count,
        hidden=int(checkpoint["critic_hidden"]),
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    actor.eval()
    critic.eval()
    return actor, critic, checkpoint


def assert_checkpoint_compatible(
    checkpoint,
    *,
    decoder_name,
    route_cost_scale,
    feature_count=None,
    objective_scale=None,
):
    """Reject only real structural or semantic incompatibilities.

    The evaluation dataset does not need to match the training dataset; only
    the observation schema, actor/critic architecture, decoder semantics,
    and reward normalization must be compatible.
    """
    if str(checkpoint["decoder"]) != str(decoder_name):
        raise ValueError("Checkpoint decoder does not match the requested decoder")
    if feature_count is not None and int(checkpoint["feature_count"]) != int(feature_count):
        raise ValueError("Checkpoint feature count does not match the observation schema")
    if decoder_name == "route_aware_greedy" and abs(
        float(checkpoint.get("route_cost_scale", 0.0)) - float(route_cost_scale)
    ) > 1e-6:
        raise ValueError("Checkpoint route scale does not match the live layout")
    if objective_scale is not None:
        saved = float(checkpoint.get("objective_scale", 1.0))
        if abs(saved - float(objective_scale)) > 1e-6 * max(1.0, abs(float(objective_scale))):
            raise ValueError(
                "Checkpoint reward normalization scale does not match the live scale"
            )
