from typing import Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ware_ops_algos.algorithms import OrderSelectionSolution, AlgorithmSolution, WarehouseOrder, \
    PickListSelectionSolution
from ware_ops_algos.domain_models import Order, ResourceType

from ware_ops_sim.sim import SimWarehouseDomain, WarehouseSimulation

from ware_ops_sim.sim.events.events import Event, BaseTourEvent, PickerArrival, OrderSelectionDone


class WarehouseEnvBatching(gym.Env):

    def __init__(self, sim_factory: Callable[[], WarehouseSimulation], reset_hooks=None):
        super().__init__()
        self.sim_factory = sim_factory
        self.reset_hooks = reset_hooks or []
        self.info = None

        self.max_orders = 363
        self.n_order_features = 8
        self.n_context_features = 2
        self.n_features = self.n_context_features + self.n_order_features

        self.sim: WarehouseSimulation | None = None
        self.observation_space = spaces.Box(
            low=-1, high=np.inf, shape=(self.max_orders, self.n_features), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.max_orders)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim = self.sim_factory()
        self.sim.reset(hooks=self.reset_hooks)

        done, info = self.sim.run()
        self.info = info

        observation = self._get_obs()
        return observation, {"dynamic_info": info}

    # @dataclass
    # class BatchObject:
    #     batch_id: int
    #     orders: list[WarehouseOrder]
    #
    # @dataclass
    # class BatchingSolution(AlgorithmSolution):
    #     batches: list[BatchObject] | None = None
    #     # pick_lists: list[list[PickPosition]] = None
    #     pick_lists: list[PickList] = None

    def step(self, action):
        # print("Action", action)
        if action == 0:
            # Indicate batch release -> release the last batch
            solution = self._action_to_solution(action)

        solution = self._action_to_solution(action)
        problem = self.info.problem_class
        self.sim.step_sim(solution, problem)
        done, info = self.sim.run()
        self.info = info
        # TODO Have a look at context / info vs observation and how dynamic_info fits into that

        obs = self._get_obs()
        # print("obs", obs)
        reward = self._compute_reward(done)
        # print("Reward", reward)
        context = {"dynamic_info": info} if info else {}
        return obs, reward, done, False, {}

    def _compute_reward(self, done: bool):
        if done:
            return - self.sim.state.tracker.final_makespan
        else:
            return - self.sim.state.tracker.average_tour_makespan
        # return - self.sim.state.tracker.average_tour_makespan

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
        if self.info is None:
            return np.zeros((self.max_orders, self.n_features), dtype=np.float32)

        orders = domain.orders.orders
        domain_type = domain.resources.resources[0].tpe
        tpe_one_hot = None
        if domain_type == ResourceType.HUMAN:
            tpe_one_hot = 0
        elif domain_type == ResourceType.COBOT:
            tpe_one_hot = 1
        global_obs = np.array([
            len(domain.warehouse_info.active_tours),
            tpe_one_hot
            # domain.resources.resources[0].current_location[0],
            # domain.resources.resources[0].current_location[1]
        ], dtype=np.float32)

        order_obs = np.zeros((self.max_orders, self.n_order_features), dtype=np.float32)
        if orders:
            for i, order in enumerate(orders[:self.max_orders]):
                features = self._encode_order(order, domain)
                order_obs[i] = np.append(features, 1.0)
        global_tiled = np.tile(global_obs, (self.max_orders, 1))

        return np.concatenate([order_obs, global_tiled], axis=1)

    def _action_to_solution(self, action: AlgorithmSolution | int):
        if isinstance(action, AlgorithmSolution):
            return action

        dynamic_info: SimWarehouseDomain = self.info
        # print(len(dynamic_info.orders.orders))
        # selected = dynamic_info.orders.orders[action]
        selected = dynamic_info.warehouse_info.buffered_pick_lists[action]
        # print("RL selected", selected)
        return PickListSelectionSolution(selected_pick_lists=[selected])

    def get_action_mask(self):
        ob = self.sim.state.order_manager.get_pick_list_buffer()
        len_buffer = len(ob)

        mask = np.ones(self.max_orders, dtype=bool)
        mask[len_buffer:] = 0
        return mask

