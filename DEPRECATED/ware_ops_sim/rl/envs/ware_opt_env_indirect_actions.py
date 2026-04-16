from typing import Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ware_ops_algos.algorithms import AlgorithmSolution, CombinedRoutingSolution

from ware_ops_algos.domain_models import Order
from ware_ops_sim.sim import SimWarehouseDomain, WarehouseSimulation


class WarehouseEnvIndirect(gym.Env):

    def __init__(self, sim_factory: Callable[[], WarehouseSimulation], reset_hooks=None):
        super().__init__()
        self.sim_factory = sim_factory
        self.reset_hooks = reset_hooks or []
        self.info = None

        self.max_orders = 10
        self.n_features = 8

        self.sim: WarehouseSimulation = self.sim_factory()
        self.observation_space = spaces.Box(
            low=-1, high=np.inf, shape=(self.max_orders, self.n_features), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.sim.control.get_execution().pipelines))
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim.reset(hooks=self.reset_hooks)
        done, info = self.sim.run()
        self.info = info

        observation = self._get_obs()
        return observation, {"dynamic_info": info}

    def step(self, action):
        print("Action", action)
        # solution = self._action_to_solution(action)
        execution = self.sim.control.get_execution(self.info.problem_class)
        solution = execution.solve(self.info, action)
        problem = self.info.problem_class
        self.sim.step_sim(solution, problem)
        done, info = self.sim.run()

        # TODO Have a look at context / info vs observation and how dynamic_info fits into that

        obs = self._get_obs()
        # print("obs", obs)
        reward = self._compute_reward(solution)
        print("Reward", reward)
        context = {"dynamic_info": info} if info else {}
        return obs, reward, done, False, context

    def _compute_reward(self, solution: AlgorithmSolution):
        if isinstance(solution, CombinedRoutingSolution):
            distance = solution.routes[0].distance
            return - distance / 100000
        else:
            return - self.sim.state.tracker.average_tour_makespan

    def _encode_order(self, order: Order, domain: SimWarehouseDomain) -> np.ndarray:
        current_time = domain.warehouse_info.time
        article_location_mapping = domain.storage.article_location_mapping

        positions = order.order_positions

        n_positions = len(positions)
        total_amount = sum(p.amount for p in positions)

        locations = [article_location_mapping[p.article_id][0] for p in positions]  # TODO FIX DUPLICATE ENTRIES
        aisles = [loc.x for loc in locations]
        positions_y = [loc.y for loc in locations]

        centroid_x = np.mean(aisles)
        centroid_y = np.mean(positions_y)
        aisle_span = max(aisles) - min(aisles)
        position_span = max(positions_y) - min(positions_y)
        n_unique_aisles = len(set(aisles))
        # time_until_due = order.due_date - current_time

        return np.array([
            n_positions,
            total_amount,
            centroid_x,
            centroid_y,
            aisle_span,
            position_span,
            n_unique_aisles,
        ], dtype=np.float32)

    def _get_obs(self):
        domain: SimWarehouseDomain = self.info
        orders = domain.orders.orders

        order_obs = np.zeros((self.max_orders, self.n_features), dtype=np.float32)
        for i, order in enumerate(orders[:self.max_orders]):
            features = self._encode_order(order, domain)
            order_obs[i] = np.append(features, 1.0)

        return order_obs

    def get_action_mask(self):
        mask_1 = np.ones(self.action_space.n, dtype=bool)
        print("mask", mask_1)
        return mask_1

