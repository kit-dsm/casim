import os
import shutil
import unittest
from pathlib import Path

import gymnasium as gym
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from casim.envs.base_env import BaseEnv
from scenarios.experiment_commons import (
    load_and_flatten_data_card,
    setup_scenario,
    setup_decision_engine,
)

TEST_DIR = Path(__file__).parent
os.environ["PROJECT_ROOT"] = TEST_DIR.as_posix()


class DummyRLEnv(BaseEnv):
    def _configure_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(1)

    def _configure_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(1)


class TestDummyRLEnv(unittest.TestCase):
    def setUp(self):
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        self.tmp_dir = TEST_DIR / "tmp_output"
        (self.tmp_dir / "event_logs").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _load_cfg(self, config_name="test_henn_online_config", overrides=None):
        with initialize(version_base=None, config_path="./config"):
            return compose(config_name=config_name, overrides=overrides or [])

    def test_dummy_rl_env(self):
        def add_orders_hook(sim, domain):
            orders = domain.orders.orders
            for order in orders:
                sim.add_order(order)

        cfg = self._load_cfg()
        datacard = load_and_flatten_data_card(cfg.data_card)
        sim = setup_scenario(cfg)
        decision_engine = setup_decision_engine(cfg, datacard)

        reset_hooks = [add_orders_hook]
        env = DummyRLEnv(sim, decision_engine, reset_hooks)
        dynamic_warehouse_state, info = env.reset()
        assert info["done_at_reset"], True
        self.assertIsNotNone(env.action_space)

        # dynamic_warehouse_state, _ = env.reset()
        # done = False
        # while not done:
        #     solution = env.sim.decision_engine.get_solution(dynamic_warehouse_state)
        #     obs, reward, done, truncated, info = env.step(solution)


if __name__ == "__main__":
    unittest.main()