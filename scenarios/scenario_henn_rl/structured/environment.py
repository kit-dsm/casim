from __future__ import annotations

import time

import numpy as np
from ware_ops_algos.algorithms import (
    BatchObject,
    ClarkAndWrightBatching,
    CombinedRoutingSolution,
    FifoBatching,
    LocalSearchBatching,
)
from ware_ops_algos.algorithms.routing.routing import SShapeRouting
from ware_ops_algos.algorithms.batching.batching_utils import CapacityChecker

from scenarios.scenario_henn.algorithm import (
    HennWakeUp,
    decide_henn,
    route_service_time,
)
from scenarios.scenario_henn_rl.environment import ReleaseTimingEnv

def knapsack_batch(
    scores: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    *,
    allow_empty: bool,
) -> np.ndarray:
    """Return the exact additive-score capacity-feasible subset."""
    scores = np.asarray(scores, dtype=float)
    demands = np.asarray(demands, dtype=int)
    if scores.ndim != 1 or scores.shape != demands.shape:
        raise ValueError("Scores and demands must be equally sized vectors")
    if np.any(demands <= 0) or np.any(demands > capacity):
        raise ValueError("Every order demand must fit the positive capacity")
    values = np.full(capacity + 1, -np.inf)
    values[0] = 0.0
    selections: list[tuple[int, ...] | None] = [None] * (capacity + 1)
    selections[0] = ()
    for index, (score, demand) in enumerate(zip(scores, demands)):
        for used in range(capacity - int(demand), -1, -1):
            if selections[used] is None:
                continue
            candidate = values[used] + float(score)
            target = used + int(demand)
            if candidate > values[target] + 1e-12:
                values[target] = candidate
                selections[target] = selections[used] + (index,)
    best_capacity = int(np.argmax(values))
    selected = selections[best_capacity] or ()
    if not selected and not allow_empty:
        feasible = np.flatnonzero(demands <= capacity)
        selected = (int(feasible[np.argmax(scores[feasible])]),)
    return np.asarray(selected, dtype=int)


def route_aware_batch(
    scores: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    order_positions: list[list[tuple[int, int]]],
    route_cost_fn,
    route_cost_scale: float,
    *,
    allow_empty: bool,
) -> np.ndarray:
    """Greedy marginal-cost route-aware batch selection.

    Orders are considered in decreasing-score order and added to the batch
    when their marginal contribution ``theta_i - [c(B+{i}) - c(B)] / scale``
    is positive — the profitable-SPRP pricing logic applied greedily.  The
    route cost ``c(B)`` is evaluated by ``route_cost_fn`` (the S-shape router
    already used for dispatch), so the decoder optimises the same non-additive
    route-distance term the simulator measures.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return np.asarray([], dtype=int)
    order = np.argsort(-scores)
    selected: list[int] = []
    selected_demand = 0
    current_cost = 0.0
    current_value = 0.0
    for i in order:
        i = int(i)
        if selected_demand + int(demands[i]) > capacity:
            continue
        candidate = selected + [i]
        new_cost = route_cost_fn(order_positions, candidate)
        new_value = float(scores[i]) + sum(scores[j] for j in selected) - new_cost / route_cost_scale
        marginal = new_value - current_value
        if marginal > 0 or (not selected and not allow_empty):
            selected = candidate
            selected_demand += int(demands[i])
            current_cost = new_cost
            current_value = new_value
    if not selected and not allow_empty:
        feasible = np.flatnonzero(demands <= capacity)
        if feasible.size:
            selected = [int(feasible[np.argmax(scores[feasible])])]
    return np.asarray(selected, dtype=int)


class StructuredBatchingEpisode:
    """Variable-order-set Henn episode for a structured batching policy."""

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
        self.decoder = str(decoder)
        self.oracle_time_s = 0.0
        self.single_services: dict[int, float] = {}
        self.single_distances: dict[int, float] = {}
        self._route_cost_scale = None

    def reset(self, *, seed=0, instance_id: str | None = None):
        options = None if instance_id is None else {"instance_id": instance_id}
        self.env.reset(seed=seed, options=options)
        self.oracle_time_s = 0.0
        self.single_services = {}
        self.single_distances = {}
        return self.state()

    def _route_cost(self, order_positions_list, order_idx) -> float:
        """S-shape route distance for the batch of orders at ``order_idx``."""
        positions = []
        for i in order_idx:
            positions.extend(order_positions_list[i])
        if not positions:
            return 0.0
        from ware_ops_algos.algorithms.routing.routing import PickPosition
        pick_positions = [
            PickPosition(order_number=0, article_id=-1, amount=1,
                         pick_node=(int(a), int(y)), in_store=1)
            for (a, y) in positions
        ]
        return float(self.env._router.solve(pick_positions).route.distance)

    def _ensure_route_cost_scale(self, planning):
        if self._route_cost_scale is not None:
            return self._route_cost_scale
        graph = planning.layout.graph_data
        one_pass = (
            graph.dist_pick_locations * (graph.n_pick_locations - 1)
            + 2 * graph.dist_bottom_to_pick_location
        )
        self._route_cost_scale = float(max(1.0, graph.n_aisles * one_pass))
        return self._route_cost_scale

    def state(self) -> dict[str, object]:
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
                    float(np.clip((float(order.due_date) - now) / horizon, -1.0, 1.0))
                )
            features.append(order_features)
        state = {
            "order_ids": np.asarray(
                [int(order.order_id) for order in orders], dtype=np.int64
            ),
            "features": np.asarray(features, dtype=np.float32),
            "demands": np.asarray(demands, dtype=np.int64),
            "capacity": capacity,
            "input_closed": bool(self.env.simulation.state.input_closed),
            "feature_schema": (
                "deadline_v1" if self.include_due_slack else "legacy_v1"
            ),
            "decoder": self.decoder,
            "order_positions": order_positions,
        }
        if self.decoder == "route_aware":
            state["_route_cost_fn"] = self._route_cost
            state["_route_cost_scale"] = self._ensure_route_cost_scale(planning)
        return state

    def oracle_action(self, scores: np.ndarray, state: dict[str, object]):
        started = time.perf_counter()
        indices = knapsack_batch(
            scores,
            state["demands"],
            int(state["capacity"]),
            allow_empty=False,
        )
        self.oracle_time_s += time.perf_counter() - started
        return indices

    def candidate_operational_features(
        self, order_ids: list[int]
    ) -> dict[str, float]:
        """Measure one feasible candidate with the configured Henn pipeline."""
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
        routing = self.env._router.solve(batch.pick_positions)
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
                single = self.env._router.solve(order.pick_positions)
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
        state: dict[str, object],
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
            for index, order_id in enumerate(state["order_ids"])
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
        if batching == "fifo":
            solution = self.env._batcher.solve(orders)
        elif batching == "cw":
            solution = ClarkAndWrightBatching(
                pick_cart=planning.resources.resources[0].pick_cart,
                articles=planning.articles,
                routing_class=SShapeRouting,
                routing_class_kwargs=self.env._routing_kwargs,
            ).solve(orders)
        elif batching == "ls":
            solution = LocalSearchBatching(
                pick_cart=planning.resources.resources[0].pick_cart,
                articles=planning.articles,
                routing_class=SShapeRouting,
                routing_class_kwargs=self.env._routing_kwargs,
                start_batching_class=FifoBatching,
                time_limit=time_limit_s,
            ).solve(orders)
        else:
            raise ValueError(f"Unknown existing batching policy: {batching}")
        routes = []
        for batch in solution.batches:
            routing = self.env._router.solve(batch.pick_positions)
            routing.route.batch = batch
            routes.append(routing.route)
        if waiting_policy == "henn_4_1" or selector == "sav":
            picker = planning.resources.resources[0]
            for order in orders:
                if order.order_id in self.single_services:
                    continue
                batch = BatchObject(batch_id=order.order_id, orders=[order])
                routing = self.env._router.solve(batch.pick_positions)
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
                selected_indices >= len(state["order_ids"])
            ):
                raise ValueError("Structured action index is out of bounds")
            order_ids = state["order_ids"][selected_indices].tolist()
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
