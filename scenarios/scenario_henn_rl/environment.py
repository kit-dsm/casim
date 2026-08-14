from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ware_ops_algos.algorithms import (
    BatchObject,
    ClarkAndWrightBatching,
    CombinedRoutingSolution,
    GreedyItemAssignment,
    LocalSearchBatching,
)
from ware_ops_algos.algorithms.batching.batching import FifoBatching
from ware_ops_algos.algorithms.batching.batching_utils import CapacityChecker
from ware_ops_algos.algorithms.routing.routing import SShapeRouting
from ware_ops_algos.domain_models import (
    OrdersDomain,
    OrderType,
    Resources,
    ResourceType,
    WarehouseInfoType,
)

from casim.domain_objects.sim_domain import DynamicInfo, SimWarehouseDomain
from casim.events.decision_events import SequencingDone
from casim.events.operational_events import (
    FlushRemainingOrders,
    OrderArrival,
    PickerIdle,
)
from casim.simulation_engine.simulation_engine import (
    NbrOrdersCondition,
    NbrPickersCondition,
    SimulationEngine,
)
from casim.simulation_engine.state_adapter import StateAdapter
from scenarios.scenario_henn.algorithm import (
    _order_routes,
    _schedule,
    decide_henn,
)
from scenarios.scenario_henn.loader import HennDataLoader
from scenarios.scenario_henn.scenario_specific_hooks import (
    add_henn_wakeup_trigger_hook,
    add_orders_hook,
)
from scenarios.scenario_henn_rl.rewards import OrderCostReward


ROOT = Path(__file__).parents[2].resolve()
HENN_DIR = ROOT / "scenarios" / "scenario_henn"


class ReleaseTimingAdapter(StateAdapter):
    """Cheap trigger view plus a detached fixed-solver projection."""

    planning_features = ("buffered_orders", "available_resources")

    @staticmethod
    def _available_resources(state, trigger=None, *, detached: bool):
        resources = Resources(
            state.resources.tpe,
            deepcopy(state.resources.resources)
            if detached
            else state.resources.resources,
        )
        available = [
            resource
            for resource in sorted(
                resources.resources, key=lambda resource: resource.id
            )
            if state.available_for_planning(resource.id)
        ]
        trigger_picker_id = getattr(trigger, "picker_id", None)
        if trigger_picker_id is not None:
            matching = [
                resource
                for resource in available
                if resource.id == int(trigger_picker_id)
            ]
            if matching:
                available = matching
        return Resources(ResourceType.HUMAN, available)

    @staticmethod
    def _domain(state, problem, orders, resources, storage):
        return SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=state.layout_manager.layout,
            orders=OrdersDomain(OrderType.STANDARD, orders),
            resources=resources,
            articles=state.articles,
            storage=storage,
            dynamic_warehouse_info=DynamicInfo(
                tpe=WarehouseInfoType.ONLINE,
                time=state.current_time,
                done=state.done_flag,
            ),
        )

    def transform_state(self, state, problem, trigger=None):
        orders = sorted(
            state.order_manager.get_order_buffer(),
            key=lambda order: (
                order.order_date if order.order_date is not None else 0.0,
                order.order_id,
            ),
        )
        return self._domain(
            state,
            problem,
            orders,
            self._available_resources(state, trigger, detached=False),
            storage=None,
        )

    def solver_snapshot(self, state, problem):
        """Return detached mutable inputs for the fixed solver chain."""
        orders = sorted(
            state.order_manager.get_order_buffer(),
            key=lambda order: (
                order.order_date if order.order_date is not None else 0.0,
                order.order_id,
            ),
        )
        return self._domain(
            state,
            problem,
            orders,
            self._available_resources(state, detached=True),
            state.storage_manager.planning_snapshot(),
        )


def _setup_simulation(instance_id: str, data_loader=None) -> SimulationEngine:
    loader_kwargs = {"instance_id": instance_id}
    if data_loader is None:
        data_loader = HennDataLoader(
            instances_dir=HENN_DIR,
            aisle_end_offset=1.5,
            depot_half_span=2.5,
            travel_speed=0.8,
            time_per_pick=10.0,
            tour_setup_time=180.0,
        )
        loader_kwargs["manifest_path"] = "reproduction/manifest.yaml"
    return SimulationEngine(
        state_adapters={"OBRP": ReleaseTimingAdapter()},
        triggers_map={
            OrderArrival: "OBRP",
            PickerIdle: "OBRP",
            FlushRemainingOrders: "OBRP",
        },
        conditions_map={
            "OBRP": [NbrPickersCondition(1), NbrOrdersCondition(1)]
        },
        loader_kwargs=loader_kwargs,
        data_loader=data_loader,
        event_loggers=[],
        completion_mode="drain",
        horizon_time=None,
        show_progress=False,
    )


class ReleaseTimingEnv(gym.Env):
    """Choose whether to wait or dispatch a fixed FCFS/S-shape candidate."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance_ids: list[str] | tuple[str, ...],
        *,
        reward_power: float = 1.0,
        sla_threshold_s: float = 0.0,
        objective_scale: float | None = None,
        data_loader=None,
        use_order_due_dates: bool = False,
    ):
        super().__init__()
        if not instance_ids:
            raise ValueError("ReleaseTimingEnv requires at least one instance")
        self.instance_ids = tuple(instance_ids)
        self.reward_power = float(reward_power)
        self.sla_threshold_s = float(sla_threshold_s)
        self.objective_scale = (
            None if objective_scale is None else float(objective_scale)
        )
        self.use_order_due_dates = bool(use_order_due_dates)
        self.instance_id = self.instance_ids[0]
        self.simulation = _setup_simulation(self.instance_id, data_loader)
        self.release_adapter = self.simulation.state_adapters["OBRP"]
        self._batcher = None
        self._router = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        self.snapshot = None
        self.arrivals: dict[int, float] = {}
        self.due_times: dict[int, float] = {}
        self.episode_horizon = 1.0
        self.previous_accrued_cost = 0.0
        self.reward_model: OrderCostReward | None = None
        self.wait_actions = 0
        self.dispatch_actions = 0
        self.forced_dispatches = 0
        self.wait_time_s = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        requested = (options or {}).get("instance_id")
        if requested is None:
            requested = self.instance_ids[
                int(self.np_random.integers(len(self.instance_ids)))
            ]
        if requested not in self.instance_ids:
            raise ValueError(f"Instance {requested} is not configured")
        self.instance_id = str(requested)
        self.simulation.loader_kwargs["instance_id"] = self.instance_id
        domain = self.simulation.reset(
            hooks=[add_orders_hook, add_henn_wakeup_trigger_hook("OBRP")]
        )
        self.arrivals = {
            int(order.order_id): float(order.order_date or 0.0)
            for order in domain.orders.orders
        }
        self.due_times = {
            int(order.order_id): float(order.due_date)
            for order in domain.orders.orders
            if order.due_date is not None
        }
        if self.use_order_due_dates and len(self.due_times) != len(self.arrivals):
            raise ValueError("Every generated order must have a due date")
        last_arrival = max(self.arrivals.values(), default=0.0)
        self.episode_horizon = max(1.0, last_arrival + 24 * 60 * 60)
        self.reward_model = OrderCostReward(
            self.arrivals,
            power=self.reward_power,
            thresholds_s=(None if self.use_order_due_dates else self.sla_threshold_s),
            due_times=(self.due_times if self.use_order_due_dates else None),
            weights=1.0,
            normalizer=self._configured_normalizer(),
        )
        self.previous_accrued_cost = 0.0
        self.wait_actions = 0
        self.dispatch_actions = 0
        self.forced_dispatches = 0
        self.wait_time_s = 0.0
        self._configure_fixed_pipeline(domain)
        done = self._advance()
        if done:
            raise RuntimeError("ReleaseTimingEnv completed during reset")
        self.previous_accrued_cost = self.accrued_objective_cost()
        return self.observation(), self.info(False)

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid release action: {action}")
        forced = action == 0 and self.simulation.state.input_closed
        if forced:
            self.forced_dispatches += 1
            action = 1
        started_at = float(self.simulation.state.current_time)
        if action == 0:
            self.wait_actions += 1
        else:
            self.dispatch_actions += 1
            self._dispatch()
        terminated = self._advance()
        if action == 0:
            self.wait_time_s += max(
                0.0, float(self.simulation.state.current_time) - started_at
            )
        reward = self.reward_since_previous()
        observation = (
            np.zeros(6, dtype=np.float32)
            if terminated
            else self.observation()
        )
        return observation, float(reward), terminated, False, self.info(forced)

    @property
    def reward_normalizer(self) -> float:
        if self.reward_model is None:
            return max(1.0, len(self.arrivals) * self.episode_horizon)
        return self.reward_model.normalizer

    def _configured_normalizer(self) -> float:
        """Return the per-instance objective scale used by the reward model."""
        if self.objective_scale is not None:
            return max(1.0, len(self.arrivals)) * self.objective_scale
        return max(
            1.0, len(self.arrivals) * self.episode_horizon**self.reward_power
        )

    def reward_since_previous(self) -> float:
        """Accrue and return the configured objective-derived reward."""
        if self.reward_model is None:
            raise RuntimeError("Reward model is unavailable before reset")
        accrued = self.accrued_objective_cost()
        reward = self.reward_model.incremental_reward(
            self.previous_accrued_cost, accrued
        )
        self.previous_accrued_cost = accrued
        return reward

    def _advance(self) -> bool:
        done, snapshot = self.simulation.run()
        self.snapshot = snapshot
        return bool(done)

    def _dispatch(self) -> None:
        if self.snapshot is None:
            raise RuntimeError("Dispatch requested without a decision snapshot")
        planning = self.release_adapter.solver_snapshot(
            self.simulation.state, self.snapshot.problem_class
        )
        candidate = self._solve_fixed(planning)
        routes = _order_routes(
            list(candidate.routes),
            "first",
            planning.resources.resources[0],
            {},
        )
        if not self.simulation.state.input_closed:
            routes = routes[:1]
        solution = _schedule(
            routes,
            planning,
            float(self.simulation.state.current_time),
            "first",
        )
        self.simulation.step(
            [SequencingDone(self.simulation.state.current_time, solution)],
            planning.problem_class,
            solution,
        )

    def configured_policy_action(
        self,
        *,
        waiting_policy: str,
        fill_threshold: float = 0.75,
        max_age_s: float = 300.0,
    ) -> int:
        """Evaluate a configured Henn policy for the fixed pipeline."""
        if self.snapshot is None:
            raise RuntimeError("No decision snapshot is available")
        planning = self.release_adapter.solver_snapshot(
            self.simulation.state, self.snapshot.problem_class
        )
        decision = decide_henn(
            candidate=self._solve_fixed(planning),
            snapshot=planning,
            current_time=float(self.simulation.state.current_time),
            next_arrival=self._next_arrival(),
            stream_exhausted=bool(self.simulation.state.input_closed),
            selector="first",
            single_services={},
            waiting_policy=waiting_policy,
            fill_threshold=fill_threshold,
            max_age_s=max_age_s,
        )
        return int(decision.action == "dispatch")

    def resolved_buffer(self):
        """Return detached, storage-resolved visible orders for batching."""
        if self.snapshot is None:
            raise RuntimeError("No decision snapshot is available")
        planning = self.release_adapter.solver_snapshot(
            self.simulation.state, self.snapshot.problem_class
        )
        assignment = GreedyItemAssignment(planning.storage).solve(
            planning.orders.orders
        )
        return planning, assignment.resolved_orders

    def dispatch_order_ids(self, order_ids: list[int]) -> None:
        """Route and commit one explicitly selected feasible batch."""
        planning, orders = self.resolved_buffer()
        requested = set(map(int, order_ids))
        selected = [order for order in orders if order.order_id in requested]
        if not selected or len(selected) != len(requested):
            raise ValueError("Structured batch contains unavailable order IDs")
        picker = planning.resources.resources[0]
        checker = CapacityChecker(picker.pick_cart, planning.articles)
        if not checker.orders_fit(selected):
            raise ValueError("Structured batch exceeds picker cart capacity")
        batch = BatchObject(batch_id=0, orders=selected)
        routing = self.route(batch.pick_positions)
        routing.route.batch = batch
        solution = _schedule(
            [routing.route],
            planning,
            float(self.simulation.state.current_time),
            "first",
        )
        self.simulation.step(
            [SequencingDone(self.simulation.state.current_time, solution)],
            planning.problem_class,
            solution,
        )

    def commit_solution(self, solution, problem_class: str) -> None:
        """Commit an existing scheduling solution without rebuilding it."""
        self.simulation.step(
            [SequencingDone(self.simulation.state.current_time, solution)],
            problem_class,
            solution,
        )

    def route(self, pick_positions):
        """Route a list of pick positions with the fixed S-shape router."""
        if self._router is None:
            raise RuntimeError("Router is not configured before reset")
        return self._router.solve(pick_positions)

    def route_distance(
        self, order_positions: list[list[tuple[int, int]]], selected_indices
    ) -> float:
        """Return the fixed S-shape distance for the batch at ``selected_indices``.

        ``order_positions`` is the per-order list of ``(aisle, y)`` pick
        positions stored in the structured state; the selected orders' positions
        are flattened into synthetic pick positions and routed.  This is the
        pure route-cost term used by the route-aware decoder and the same router
        used for dispatch.
        """
        if self._router is None:
            raise RuntimeError("Router is not configured before reset")
        positions = []
        for index in selected_indices:
            positions.extend(order_positions[int(index)])
        if not positions:
            return 0.0
        from ware_ops_algos.algorithms.routing.routing import PickPosition

        pick_positions = [
            PickPosition(
                order_number=0,
                article_id=-1,
                amount=1,
                pick_node=(int(aisle), int(y)),
                in_store=1,
            )
            for (aisle, y) in positions
        ]
        return float(self._router.solve(pick_positions).route.distance)

    def build_batches(self, orders, batching: str, *, articles, time_limit_s: float = 1.0):
        """Build batches with the configured fixed pipeline batching algorithm.

        ``articles`` is the planning-snapshot articles used for capacity
        checking; the routing configuration is owned by the environment.
        """
        if self._batcher is None or self._routing_kwargs is None:
            raise RuntimeError("Pipeline is not configured before reset")
        pick_cart = self._routing_kwargs["picker"][0].pick_cart
        if batching == "fifo":
            return self._batcher.solve(orders)
        if batching == "cw":
            return ClarkAndWrightBatching(
                pick_cart=pick_cart,
                articles=articles,
                routing_class=SShapeRouting,
                routing_class_kwargs=self._routing_kwargs,
            ).solve(orders)
        if batching == "ls":
            return LocalSearchBatching(
                pick_cart=pick_cart,
                articles=articles,
                routing_class=SShapeRouting,
                routing_class_kwargs=self._routing_kwargs,
                start_batching_class=FifoBatching,
                time_limit=time_limit_s,
            ).solve(orders)
        raise ValueError(f"Unknown existing batching policy: {batching}")

    def _configure_fixed_pipeline(self, domain) -> None:
        picker = domain.resources.resources[0]
        network = domain.layout.layout_network
        nodes = list(network.graph.nodes)
        self._batcher = FifoBatching(
            pick_cart=picker.pick_cart, articles=domain.articles
        )
        self._routing_kwargs = {
            "start_node": network.start_node,
            "end_node": network.end_node,
            "closest_node_to_start": network.closest_node_to_start,
            "min_aisle_position": network.min_aisle_position,
            "max_aisle_position": network.max_aisle_position,
            "distance_matrix": network.distance_matrix,
            "predecessor_matrix": network.predecessor_matrix,
            "picker": domain.resources.resources,
            "gen_tour": True,
            "gen_item_sequence": True,
            "node_list": network.node_list,
            "node_to_idx": {
                node: index for index, node in enumerate(nodes)
            },
            "idx_to_node": {
                index: node for index, node in enumerate(nodes)
            },
        }
        self._router = SShapeRouting(**self._routing_kwargs)

    def _solve_fixed(self, snapshot) -> CombinedRoutingSolution:
        assignment = GreedyItemAssignment(snapshot.storage).solve(
            snapshot.orders.orders
        )
        batching = self._batcher.solve(assignment.resolved_orders)
        routes = []
        execution_time = assignment.execution_time + batching.execution_time
        for batch in batching.batches:
            routing = self._router.solve(batch.pick_positions)
            routing.route.batch = batch
            routes.append(routing.route)
            execution_time += routing.execution_time
        return CombinedRoutingSolution(
            algo_name="FixedGreedyIAFifoSShape",
            execution_time=execution_time,
            routes=routes,
            objective_value=sum(route.distance for route in routes),
        )

    def completion_times(self) -> dict[int, float]:
        result = {}
        for tour in self.simulation.state.tracker.completed_tours:
            for order_id in tour[3]:
                result[int(order_id)] = float(tour[2])
        return result

    def accrued_flow_time(self) -> float:
        """Return raw accrued total flow time, independent of reward power."""
        now = float(self.simulation.state.current_time)
        completions = self.completion_times()
        return sum(
            max(0.0, min(now, completions.get(order_id, now)) - arrival)
            for order_id, arrival in self.arrivals.items()
            if now >= arrival
        )

    def accrued_objective_cost(self) -> float:
        if self.reward_model is None:
            raise RuntimeError("Reward model is unavailable before reset")
        return self.reward_model.accrued_cost(
            float(self.simulation.state.current_time),
            self.completion_times(),
        )

    def _next_arrival(self) -> float | None:
        values = [
            float(event.time)
            for event in self.simulation.events
            if isinstance(event, OrderArrival)
            and not getattr(event, "cancelled", False)
        ]
        return min(values) if values else None

    def observation(self):
        state = self.simulation.state
        now = float(state.current_time)
        buffered = state.order_manager.get_order_buffer()
        oldest_age = max(
            (now - float(order.order_date or 0.0) for order in buffered),
            default=0.0,
        )
        next_arrival = self._next_arrival()
        until_next = (
            self.episode_horizon
            if next_arrival is None
            else max(0.0, next_arrival - now)
        )
        picker = self.snapshot.resources.resources[0]
        capacity = max(1.0, float(picker.capacity or 1.0))
        visible_items = sum(
            sum(int(position.amount) for position in order.order_positions)
            for order in buffered
        )
        values = np.array(
            [
                now / self.episode_horizon,
                len(buffered) / max(1, len(self.arrivals)),
                oldest_age / self.episode_horizon,
                until_next / self.episode_horizon,
                visible_items / capacity,
                self.accrued_flow_time() / self.reward_normalizer,
            ],
            dtype=np.float32,
        )
        return np.clip(values, 0.0, 1.0)

    def info(self, forced_dispatch: bool) -> dict:
        tracker = self.simulation.state.tracker
        completions = self.completion_times()
        flow_times = {
            order_id: completion - self.arrivals[order_id]
            for order_id, completion in completions.items()
        }
        tardiness = {
            order_id: max(0.0, completion - self.due_times[order_id])
            for order_id, completion in completions.items()
            if order_id in self.due_times
        }
        return {
            "instance_id": self.instance_id,
            "forced_dispatch": bool(forced_dispatch),
            "total_flow_time": float(self.accrued_flow_time()),
            "objective_cost": float(self.accrued_objective_cost()),
            "reward_power": self.reward_power,
            "total_distance": float(sum(tracker.distance_by_picker.values())),
            "completed_tours": len(tracker.completed_tours),
            "wait_actions": self.wait_actions,
            "dispatch_actions": self.dispatch_actions,
            "forced_dispatches": self.forced_dispatches,
            "wait_time_s": self.wait_time_s,
            "pipeline_controlled_by_agent": False,
            "flow_times_by_order": flow_times,
            "tardiness_by_order": tardiness,
            "violated_orders": sum(value > 0.0 for value in tardiness.values()),
        }
