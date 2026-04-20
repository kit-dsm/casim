import logging

import gymnasium as gym
import numpy as np

from ware_ops_algos.algorithms import AlgorithmSolution

from casim.decision_engine.decision_engine import DecisionEngine
from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.simulation_engine import SimulationEngine

logger = logging.getLogger(__name__)


class BaseEnv(gym.Env):
    def __init__(self, sim: SimulationEngine,
                 decision_engine: DecisionEngine,
                 reset_hooks=None):
        super().__init__()
        self.sim = sim
        self.decision_engine = decision_engine
        self.reset_hooks = reset_hooks or []

        self.observation_space = self._configure_observation_space()
        self.action_space = self._configure_action_space()
        self.dynamic_warehouse_state_prev: SimWarehouseDomain | None = None
        self.learnable_problems = []

    def _configure_action_space(self) -> gym.spaces.space:
        ...

    def _configure_observation_space(self) -> gym.spaces.space:
        ...

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(hooks=self.reset_hooks)

        done, domain = self._run_until_learnable_or_done()
        self.dynamic_warehouse_state_prev = domain

        if done:
            return self._terminal_obs(), {"done_at_reset": True}

        return self._get_obs(domain), {}

    def step(self, action: int):
        solution = self._action_to_solution(action)
        events = self._solution_to_events(solution)
        self.sim.step(events, problem_class=self.learnable_problems[0], solution=solution)

        # run forward, solving non-learnable problems internally
        done, domain = self._run_until_learnable_or_done()

        self.dynamic_warehouse_state_prev = domain
        if done:
            return self._terminal_obs(), self._compute_reward(True), True, False, {}

        obs = self._get_obs(domain)
        reward = self._compute_reward(False)
        return obs, reward, False, False, {}

    def _run_until_learnable_or_done(self):
        while True:
            done, domain = self.sim.run()
            if done:
                return True, None
            if domain.problem_class in self.learnable_problems:
                return False, domain
            # Non-learnable: solve with decision engine and keep going
            events_to_add, solution = self.decision_engine.on_trigger(domain)
            self.sim.step(events_to_add, domain.problem_class, solution)

    def _action_to_solution(self, action: int | AlgorithmSolution):
        ...

    def _solution_to_events(self, solution: AlgorithmSolution) -> list[Event]:
        ...

    def _compute_reward(self, done: bool):
        if done:
            return 1
        else:
            return 0

    def _get_obs(self, dynamic_warehouse_state: SimWarehouseDomain):
        ...

    def _terminal_obs(self):
        return np.zeros((self.observation_space.shape), dtype=np.float32)

    def get_action_mask(self):
        ...

