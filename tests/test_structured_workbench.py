from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from scenarios.scenario_henn_rl.structured.data import (
    GeneratedHennDataLoader,
    generated_instance_splits,
    generated_manifest,
)


def _spec(**overrides):
    value = OmegaConf.to_container(
        OmegaConf.load(
            "scenarios/scenario_henn_rl/config/data/henn_lorenz.yaml"
        ),
        resolve=True,
    )
    value.update(overrides)
    return value


def _orders(domain):
    return [
        (
            float(order.order_date),
            float(order.due_date),
            tuple(position.article_id for position in order.order_positions),
        )
        for order in domain.orders.orders
    ]


def test_generated_splits_are_balanced_disjoint_and_reconstructable():
    spec = _spec()
    first = generated_instance_splits(spec)
    second = generated_instance_splits(copy.deepcopy(spec))
    assert first == second
    assert {key: len(value) for key, value in first.items()} == {
        "train": 128,
        "validation": 32,
        "test": 32,
    }
    assert len(set().union(*map(set, first.values()))) == 192
    assert generated_manifest(spec)["splits"] == first
    for values in first.values():
        assert {int(value.split("_")[2]) for value in values} == {
            40,
            60,
            80,
            100,
        }


@pytest.mark.parametrize("action", ["generate", "train", "evaluate", "audit"])
@pytest.mark.parametrize("objective", ["flow", "sla"])
@pytest.mark.parametrize("decoder", ["knapsack", "route_aware_greedy"])
def test_workbench_hydra_groups_compose(action, objective, decoder):
    from hydra import compose, initialize_config_dir

    config_dir = str(
        Path("scenarios/scenario_henn_rl/config").resolve()
    )
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="structured",
            overrides=[
                f"experiment={action}",
                f"objective={objective}",
                f"decoder={decoder}",
            ],
        )
    assert cfg.experiment.action == action
    assert cfg.objective.gamma == 1.0
    assert cfg.decoder.name == decoder
    assert cfg.progress is True
    assert "decoder" not in cfg.model


def test_generated_deadlines_follow_hourly_four_hour_cutoff_rule():
    spec = _spec()
    instance_id = generated_instance_splits(spec)["train"][0]
    domain = GeneratedHennDataLoader(spec).load(instance_id)
    for order in domain.orders.orders:
        allowance = float(order.due_date) - float(order.order_date)
        assert 4 * 3600 <= allowance < 5 * 3600
        assert float(order.due_date) % 3600 == pytest.approx(0.0)


def test_generation_factor_changes_preserve_unrelated_draws():
    base = _spec()
    instance_id = generated_instance_splits(base)["train"][0]
    original = _orders(GeneratedHennDataLoader(base).load(instance_id))
    faster = _orders(
        GeneratedHennDataLoader(
            _spec(orders_per_four_hours=110)
        ).load(instance_id)
    )
    assert [len(row[2]) for row in original] == [len(row[2]) for row in faster]
    assert [row[2] for row in original] == [row[2] for row in faster]
    assert [row[0] for row in original] != [row[0] for row in faster]


def test_generated_loads_do_not_share_mutable_episode_objects():
    spec = _spec()
    instance_id = generated_instance_splits(spec)["train"][0]
    loader = GeneratedHennDataLoader(spec)
    left = loader.load(instance_id)
    right = loader.load(instance_id)
    assert left.orders.orders[0] is not right.orders.orders[0]
    assert left.resources.resources[0] is not right.resources.resources[0]
    assert left.storage.locations[0] is not right.storage.locations[0]
    left.orders.orders[0].due_date = -1
    assert right.orders.orders[0].due_date > 0


def test_generated_episode_exposes_deadline_slack_and_exact_tardiness_reward():
    from scenarios.scenario_henn_rl.structured.environment import (
        StructuredBatchingEpisode,
    )

    spec = _spec(train_instances=4, validation_instances=4, test_instances=4)
    instance_id = generated_instance_splits(spec)["train"][0]
    episode = StructuredBatchingEpisode(
        [instance_id],
        data_loader=GeneratedHennDataLoader(spec),
        use_order_due_dates=True,
        include_due_slack=True,
    )
    state = episode.reset(instance_id=instance_id)
    assert state.features.shape[1] == 8
    assert state.feature_schema == "deadline_v1"
    episode_return = 0.0
    done = False
    while not done:
        indices = episode.oracle_action(
            torch.ones(len(state.order_ids)).numpy(), state
        )
        state, reward, done, _, info = episode.step(indices)
        episode_return += reward
    assert episode_return == pytest.approx(
        -info["objective_cost"] / episode.env.reward_normalizer,
        abs=1e-10,
    )
    episode.close()


def test_workbench_checkpoint_rejects_legacy_feature_schema(tmp_path):
    from scenarios.scenario_henn_rl.structured.checkpoint import (
        load_workbench_checkpoint,
    )

    path = tmp_path / "legacy.pt"
    torch.save({"format_version": 2, "feature_schema": "legacy_v1", "decoder": {"name": "knapsack"}}, path)
    with pytest.raises(ValueError, match="Unsupported feature schema"):
        load_workbench_checkpoint(path)


def test_supported_entry_does_not_import_studies():
    code = """
import sys
import scenarios.scenario_henn_rl.experiment_structured_batching
assert not any(name.startswith('scenarios.scenario_henn_rl.studies') for name in sys.modules)
"""
    import subprocess

    subprocess.run([sys.executable, "-c", code], check=True)


def test_wandb_tracking_is_optional_and_preserves_requested_mode(monkeypatch):
    from scenarios.scenario_henn_rl.structured.tracking import (
        log_artifact,
        log_evaluation,
        metric_callback,
        start_tracking,
    )

    assert start_tracking({"enabled": False}, {}) is None
    captured = {}

    class Run:
        def __init__(self):
            self.rows = []

        def log(self, value):
            self.rows.append(value)

        def log_artifact(self, value):
            self.artifact = value

    class Artifact:
        def __init__(self, *args, **kwargs):
            self.files = []

        def add_file(self, path, name):
            self.files.append((path, name))

    run = Run()
    fake = SimpleNamespace(
        init=lambda **kwargs: captured.update(kwargs) or run,
        Table=lambda **kwargs: ("table", kwargs),
        Artifact=Artifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    selected = start_tracking(
        {
            "enabled": True,
            "mode": "offline",
            "project": "test",
            "entity": None,
            "group": None,
            "name": None,
        },
        {"seed": 11},
    )
    assert selected is run
    assert captured["mode"] == "offline"
    metric_callback(run)("train", 2, {"return": -1.0})
    log_evaluation(
        run,
        "validation",
        {
            "mean_objective_per_order": 1.0,
            "episodes": [{"instance_id": "one"}],
        },
    )
    assert run.rows[0]["train/episode"] == 2
    assert "validation/instances" in run.rows[-1]
    artifact_file = Path(__file__)
    log_artifact(run, artifact_file.parent, [artifact_file])
    assert run.artifact.files[0][1] == artifact_file.name
