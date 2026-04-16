import logging
from typing import Callable, Type

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ware_ops_algos.algorithms import OrderSelectionSolution, AlgorithmSolution, WarehouseOrder, \
    PickListSelectionSolution, CombinedRoutingSolution, SchedulingSolution, BatchingSolution
from ware_ops_algos.domain_models import Order, ResourceType

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.decision_events import SequencingDone, RoutingDone, PickListDone
from casim.simulation_engine import SimulationEngine
from ware_ops_sim.sim.events.events import PickListSelectionDone

logger = logging.getLogger(__name__)


class BatchSelectionEnv(gym.Env):
    def __init__(self, sim: SimulationEngine, reset_hooks=None):
        super().__init__()
        self.sim = sim
        self.reset_hooks = reset_hooks or []

        self.max_orders = 363
        self.n_order_features = 8
        self.n_context_features = 2
        self.n_features = self.n_context_features + self.n_order_features

        # self.sim: SimulationEngine | None = None
        self.observation_space = spaces.Box(
            low=-1, high=np.inf, shape=(self.max_orders, self.n_features), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.max_orders)
        self.dynamic_warehouse_state_prev: SimWarehouseDomain | None = None
        self.learnable_problems = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(hooks=self.reset_hooks)

        done, domain = self._run_until_learnable_or_done()
        self.dynamic_warehouse_state_prev = domain

        if done:
            return self._terminal_obs(), {"done_at_reset": True}

        return self._get_obs(domain), {}

    def step(self, action: int | AlgorithmSolution):
        solution = self._action_to_solution(action)
        events = self._solution_to_events(solution)
        self.sim.step(events)

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
            events_to_add = self.sim.decision_engine.on_trigger(domain)
            self.sim.step(events_to_add)

    def _action_to_solution(self, action: int | AlgorithmSolution):
        if isinstance(action, int):
            dynamic_warehouse_state = self.dynamic_warehouse_state_prev
            selected = dynamic_warehouse_state.dynamic_warehouse_info.buffered_pick_lists[action]
            return PickListSelectionSolution(selected_pick_lists=[selected])
        elif isinstance(action, AlgorithmSolution):
            return action
        else:
            raise ValueError

    def _solution_to_events(self, solution: AlgorithmSolution):
        logger.info("Solution type", type(solution))
        sim_time = self.dynamic_warehouse_state_prev.dynamic_warehouse_info.time

        if isinstance(solution, PickListSelectionSolution):
            events_to_return = self._batch_selection_to_events(solution, sim_time)

        elif isinstance(solution, CombinedRoutingSolution):
            events_to_return = self._routes_to_events(solution, sim_time)

        elif isinstance(solution, SchedulingSolution):
            events_to_return = self._schedules_to_events(solution, sim_time)

        elif isinstance(solution, BatchingSolution):
            events_to_return = self._batches_to_events(solution, sim_time)

        else:
            raise Exception("Not a known solution", type(solution))

        return events_to_return

    def _schedules_to_events(self, sequencing_sol: SchedulingSolution, finish_time):
        """
        Turn sequencing solution into TourStart events.
        """
        events_to_return = []
        # jobs = [sequencing_sol.jobs[0]]
        jobs = sequencing_sol.jobs # TODO How to window?
        assignments = sorted(jobs, key=lambda a: (a.picker_id, a.start_time))
        for a in assignments:
            events_to_return.append(SequencingDone(finish_time, a))
        return events_to_return

    def _routes_to_events(self, routing_solution: CombinedRoutingSolution, finish_time) -> list[RoutingDone]:
        events_to_return = []
        routes = routing_solution.routes
        for r in routes:
            events_to_return.append(RoutingDone(finish_time, r))
        return events_to_return

    def _batches_to_events(self, batching_solution: BatchingSolution, finish_time):
        events_to_return = []
        pls = batching_solution.pick_lists
        for pl in pls:
            events_to_return.append(PickListDone(finish_time, pl))
        return events_to_return

    def _batch_selection_to_events(self, batch_selection_sol, finish_time):
        return PickListSelectionDone(
            time=finish_time,
            selected_order=batch_selection_sol.selected_pick_lists[0]
        )

    def _compute_reward(self, done: bool):
        if done:
            return - self.sim.state.tracker.final_makespan
        else:
            return - self.sim.state.tracker.average_tour_makespan


    def _encode_order(self, order: Order, domain: SimWarehouseDomain) -> np.ndarray:
        current_time = domain.dynamic_warehouse_info.time
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

    def _get_obs(self, dynamic_warehouse_state: SimWarehouseDomain):
        orders = dynamic_warehouse_state.orders.orders
        order_obs = np.zeros((self.max_orders, self.n_order_features), dtype=np.float32)
        if orders:
            for i, order in enumerate(orders[:self.max_orders]):
                features = self._encode_order(order, dynamic_warehouse_state)
                order_obs[i] = np.append(features, 1.0)
        return order_obs

    def _terminal_obs(self):
        return np.zeros((self.max_orders, self.n_features), dtype=np.float32)

    def get_action_mask(self):
        ob = self.sim.state.order_manager.get_pick_list_buffer()
        len_buffer = len(ob)

        mask = np.ones(self.max_orders, dtype=bool)
        mask[len_buffer:] = 0
        return mask

