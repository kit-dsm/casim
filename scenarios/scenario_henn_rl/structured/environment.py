from __future__ import annotations

import time

import numpy as np
from ware_ops_algos.algorithms import (
    BatchObject,
    CombinedRoutingSolution,
)
from ware_ops_algos.algorithms.batching.batching_utils import CapacityChecker

from scenarios.scenario_henn.algorithm import (
    HennWakeUp,
    decide_henn,
    route_service_time,
)
from scenarios.scenario_henn_rl.environment import ReleaseTimingEnv
from scenarios.scenario_henn_rl.structured.decoders import (
    Decoder,
    make_decoder,
)
from scenarios.scenario_henn_rl.structured.state import BatchingState


class StructuredBatchingEpisode:
    """Variable-order-set Henn episode for a structured batching policy.

    The episode exposes the current structured decision state, validates and
    executes a selected batch, advances CASIM, and returns reward and episode
    information.  It does not decide which learned decoder is active: the
    configured decoder is constructed once and shared with the structured
    policy so training, validation, evaluation, and audit use one path.
    """

    def __init__(
        self,
        instance_ids: list[str] | tuple[str, ...],
        *,
        reward_power: float = 1.0,
        sla_threshold_s: float = 0.0,
        objective_scale: float | None = None,
        data_loader=None,
        use_order_due_dates: bool = False,
        include_due_slack: bool = False,
        decoder: str = "knapsack",
    ):
        self.env = ReleaseTimingEnv(
            instance_ids,
            reward_power=reward_power,
            sla_threshold_s=sla_threshold_s,
            objective_scale=objective_scale,
            data_loader=data_loader,
            use_order_due_dates=use_order_due_dates,
        )
        self.include_due_slack = bool(include_due_slack)
        self.feature_schema = "deadline_v1" if include_due_slack else "legacy_v1"
        self.oracle_time_s = 0.0
        self.single_services: dict[int, float] = {}
        self.single_distances: dict[int, float] = {}
        self._route_cost_scale: float | None = None
        self._decoder_name = str(decoder)
        self._decoder: Decoder = make_decoder(
            self._decoder_name,
            route_cost_fn=self.route_cost
            if self._decoder_name == "route_aware_greedy"
            else None,
        )

    @property
    def decoder(self) -> Decoder:
        """The configured decoder shared with the structured policy."""
        return self._decoder

    @property
    def decoder_name(self) -> str:
        return self._decoder_name

    def reset(self, *, seed=0, instance_id: str | None = None):
        options = None if instance_id is None else {"instance_id": instance_id}
        self.env.reset(seed=seed, options=options)
        self.oracle_time_s = 0.0
        self.single_services = {}
        self.single_distances = {}
        return self.state()

    def route_cost(self, order_positions, selected_indices) -> float:
        """S-shape route distance for the batch of orders at ``selected_indices``."""
        return float(self.env.route_distance(order_positions, selected_indices))

    def _ensure_route_cost_scale(self, planning) -> float:
        if self._route_cost_scale is not None:
            return self._route_cost_scale
        graph = planning.layout.graph_data
        one_pass = (
            graph.dist_pick_locations * (graph.n_pick_locations - 1)
            + 2 * graph.dist_bottom_to_pick_location
        )
        self._route_cost_scale = float(max(1.0, graph.n_aisles * one_pass))
        return self._route_cost_scale

    def state(self) -> BatchingState:
        planning, orders = self.env.resolved_buffer()
        picker = planning.resources.resources[0]
        capacity = int(
            picker.pick_cart.capacities[0] * picker.pick_cart.n_boxes
        )
        now = float(self.env.simulation.state.current_time)
        horizon = self.env.episode_horizon
        position_scale = max(
            1.0,
            max(
                float(node[1])
                for node in planning.layout.layout_network.graph.nodes
                if isinstance(node, tuple) and len(node) > 1
            ),
        )
        features = []
        demands = []
        order_positions = []
        for order in orders:
            positions = [
                float(position.pick_node[1])
                for position in order.pick_positions
            ]
            demand = sum(
                int(position.in_store) for position in order.pick_positions
            )
            demands.append(demand)
            order_positions.append(
                [
                    (int(p.pick_node[0]), int(p.pick_node[1]))
                    for p in order.pick_positions
                ]
            )
            order_features = [
                max(0.0, now - float(order.order_date or 0.0)) / horizon,
                demand / max(1, capacity),
                len(order.pick_positions) / max(1, capacity),
                min(positions) / position_scale,
                max(positions) / position_scale,
                float(np.mean(positions)) / position_scale,
                len(orders) / max(1, len(self.env.arrivals)),
            ]
            if self.include_due_slack:
                if order.due_date is None:
                    raise ValueError("Deadline features require order due dates")
                order_features.append(
                    float(
                        np.clip(
                            (float(order.due_date) - now) / horizon, -1.0, 1.0
                        )
                    )
                )
            features.append(order_features)
        route_cost_scale = (
            self._ensure_route_cost_scale(planning)
            if self._decoder_name == "route_aware_greedy"
            else 0.0
        )
        return BatchingState(
            order_ids=np.asarray(
                [int(order.order_id) for order in orders], dtype=np.int64
            ),
            features=np.asarray(features, dtype=np.float32),
            demands=np.asarray(demands, dtype=np.int64),
            capacity=capacity,
            order_positions=order_positions,
            input_closed=bool(self.env.simulation.state.input_closed),
            feature_schema=self.feature_schema,
            route_cost_scale=route_cost_scale,
        )

    def oracle_action(self, scores: np.ndarray, state: BatchingState):
        """Delegate to the configured decoder (the same path as the policy)."""
        started = time.perf_counter()
        result = self._decoder.select(np.asarray(scores, dtype=float), state)
        self.oracle_time_s += time.perf_counter() - started
        return result.selected_indices

    def candidate_operational_features(
        self, order_ids: list[int]
    ) -> dict[str, float]:
        """Measure one feasible candidate with the configured Henn pipeline.

        Diagnostic-only; not on the supported training path.
        """
        planning, orders = self.env.resolved_buffer()
        requested = set(map(int, order_ids))
        selected = [order for order in orders if int(order.order_id) in requested]
        if not selected or len(selected) != len(requested):
            raise ValueError("Candidate contains unavailable order IDs")
        picker = planning.resources.resources[0]
        checker = CapacityChecker(picker.pick_cart, planning.articles)
        if not checker.orders_fit(selected):
            raise ValueError("Candidate exceeds picker cart capacity")

        batch = BatchObject(batch_id=0, orders=selected)
        routing = self.env.route(batch.pick_positions)
        routing.route.batch = batch
        now = float(self.env.simulation.state.current_time)
        ages = np.asarray(
            [max(0.0, now - float(order.order_date or 0.0)) for order in orders]
        )
        selected_mask = np.asarray(
            [int(order.order_id) in requested for order in orders], dtype=bool
        )
        demands = np.asarray(
            [
                sum(int(position.in_store) for position in order.pick_positions)
                for order in orders
            ],
            dtype=float,
        )
        positions = list(batch.pick_positions)
        aisles = {float(position.pick_node[0]) for position in positions}
        x_values = [float(position.pick_node[0]) for position in positions]
        y_values = [float(position.pick_node[1]) for position in positions]
        selected_ages = ages[selected_mask]
        remaining_ages = ages[~selected_mask]
        selected_demand = float(demands[selected_mask].sum())
        capacity = float(
            picker.pick_cart.capacities[0] * picker.pick_cart.n_boxes
        )

        single_distance = 0.0
        for order in selected:
            order_id = int(order.order_id)
            if order_id not in self.single_distances:
                single = self.env.route(order.pick_positions)
                self.single_distances[order_id] = float(single.route.distance)
            single_distance += self.single_distances[order_id]
        return {
            "route_distance": float(routing.route.distance),
            "route_service_time_s": float(route_service_time(routing.route, picker)),
            "route_saving": single_distance - float(routing.route.distance),
            "selected_orders": float(len(selected)),
            "selected_demand": selected_demand,
            "capacity_fill": selected_demand / max(1.0, capacity),
            "selected_pick_positions": float(len(positions)),
            "selected_age_sum_s": float(selected_ages.sum()),
            "selected_age_mean_s": float(selected_ages.mean()),
            "selected_age_max_s": float(selected_ages.max()),
            "remaining_orders": float((~selected_mask).sum()),
            "remaining_demand": float(demands[~selected_mask].sum()),
            "remaining_age_sum_s": float(remaining_ages.sum()),
            "remaining_age_mean_s": float(
                remaining_ages.mean() if len(remaining_ages) else 0.0
            ),
            "remaining_age_max_s": float(
                remaining_ages.max() if len(remaining_ages) else 0.0
            ),
            "aisles_visited": float(len(aisles)),
            "aisle_span": max(x_values) - min(x_values),
            "position_span": max(y_values) - min(y_values),
            "visible_orders": float(len(orders)),
            "visible_demand": float(demands.sum()),
            "episode_progress": now / max(1.0, self.env.episode_horizon),
        }

    def existing_policy_action(
        self,
        state: BatchingState,
        *,
        batching: str,
        waiting_policy: str = "fill_or_age",
        selector: str = "short",
        fill_threshold: float = 0.75,
        max_age_s: float = 300.0,
        time_limit_s: float = 1.0,
    ) -> np.ndarray:
        """Return the next subset selected by a configured Henn policy."""
        decision = self.existing_policy_decision(
            batching=batching,
            waiting_policy=waiting_policy,
            selector=selector,
            fill_threshold=fill_threshold,
            max_age_s=max_age_s,
            time_limit_s=time_limit_s,
        )
        if decision.action == "wait":
            return np.asarray([], dtype=int)
        selected = decision.details["selected_order_ids"]
        if selected and isinstance(selected[0], list):
            selected = selected[0]
        index_by_order = {
            int(order_id): index
            for index, order_id in enumerate(state.order_ids.tolist())
        }
        return np.asarray(
            [index_by_order[int(order_id)] for order_id in selected],
            dtype=int,
        )

    def existing_policy_decision(
        self,
        *,
        batching: str,
        waiting_policy: str,
        selector: str,
        fill_threshold: float = 0.75,
        max_age_s: float = 300.0,
        time_limit_s: float = 1.0,
    ):
        """Build existing candidates and delegate release to ``decide_henn``."""
        planning, orders = self.env.resolved_buffer()
        solution = self.env.build_batches(
            orders,
            batching,
            articles=planning.articles,
            time_limit_s=time_limit_s,
        )
        routes = []
        for batch in solution.batches:
            routing = self.env.route(batch.pick_positions)
            routing.route.batch = batch
            routes.append(routing.route)
        if waiting_policy == "henn_4_1" or selector == "sav":
            picker = planning.resources.resources[0]
            for order in orders:
                if order.order_id in self.single_services:
                    continue
                batch = BatchObject(batch_id=order.order_id, orders=[order])
                routing = self.env.route(batch.pick_positions)
                routing.route.batch = batch
                self.single_services[order.order_id] = route_service_time(
                    routing.route, picker
                )
        candidate = CombinedRoutingSolution(
            algo_name=f"Existing_{batching}",
            execution_time=float(solution.execution_time),
            routes=routes,
            objective_value=sum(float(route.distance) for route in routes),
        )
        return decide_henn(
            candidate=candidate,
            snapshot=planning,
            current_time=float(self.env.simulation.state.current_time),
            next_arrival=self.env._next_arrival(),
            stream_exhausted=bool(self.env.simulation.state.input_closed),
            selector=selector,
            single_services=self.single_services,
            waiting_policy=waiting_policy,
            fill_threshold=fill_threshold,
            max_age_s=max_age_s,
        )

    def step_existing_policy(self, decision):
        """Execute a complete existing-policy decision, including final waves."""
        if decision.action == "wait":
            if decision.wait_until is not None:
                self.env.simulation.add_event(HennWakeUp(decision.wait_until))
            return (*self.step(np.asarray([], dtype=int)), decision)
        self.env.dispatch_actions += 1
        problem_class = self.env.snapshot.problem_class
        self.env.commit_solution(decision.solution, problem_class)
        terminated = self.env._advance()
        reward = self.env.reward_since_previous()
        info = self.env.info(False)
        info["oracle_time_s"] = self.oracle_time_s
        next_state = None if terminated else self.state()
        return next_state, reward, terminated, False, info, decision

    def step(self, selected_indices: np.ndarray):
        state = self.state()
        selected_indices = np.asarray(selected_indices, dtype=int)
        if selected_indices.size == 0:
            observation, reward, terminated, truncated, info = self.env.step(0)
            del observation
        else:
            if np.any(selected_indices < 0) or np.any(
                selected_indices >= len(state.order_ids)
            ):
                raise ValueError("Structured action index is out of bounds")
            order_ids = state.order_ids[selected_indices].tolist()
            self.env.dispatch_actions += 1
            self.env.dispatch_order_ids(order_ids)
            terminated = self.env._advance()
            reward = self.env.reward_since_previous()
            truncated = False
            info = self.env.info(False)
        info["oracle_time_s"] = self.oracle_time_s
        next_state = None if terminated else self.state()
        return next_state, reward, terminated, truncated, info

    def close(self):
        self.env.close()
