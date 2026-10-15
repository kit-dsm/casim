from pathlib import Path

import gymnasium as gym

from casim.envs.base_env import BaseEnv
from casim.run import compose_scenario, prepare
from casim.sources.base import release

MINIMAL = Path(__file__).resolve().parents[1] / "scenarios" / "minimal"


class DummyRLEnv(BaseEnv):
    def _configure_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(1)

    def _configure_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(1)


def test_dummy_rl_env_without_learnable_problems_finishes_at_reset(tmp_path):
    prepared = prepare(compose_scenario(MINIMAL, output_dir=tmp_path))
    env = DummyRLEnv(prepared.sim, prepared.engine, [release(prepared.source)])
    _, info = env.reset()
    assert info["done_at_reset"] is True
    assert env.action_space is not None
