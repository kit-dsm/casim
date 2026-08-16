"""Single concrete StateAdapter driven by a small exposure vocabulary.

Configuration lives in YAML; domain selection lives here.  The adapter
projects a :class:`SimWarehouseDomain` snapshot for one decision problem
from a closed vocabulary of order, batch, resource, and active-tour
exposures.  ``projected_features()`` is derived from the same exposure so
there is one source of truth for what the snapshot actually contains.
"""

import copy

import numpy as np
import pandas as pd

from ware_ops_algos.algorithms import (
    BatchObject,
    NodeType,
    RouteNode,
    WarehouseOrder,
)
from ware_ops_algos.domain_models import (
    OrdersDomain,
    OrderType,
    Resources,
    ResourceType,
    WarehouseInfoType,
)

from casim.domain_objects.sim_domain import SimWarehouseDomain, DynamicInfo
from casim.state import State


def _orders_snapshot(orders) -> OrdersDomain:
    return OrdersDomain(
        tpe=OrderType.STANDARD,
        orders=copy.deepcopy(list(orders)),
    )


def _batches_snapshot(batches):
    return copy.deepcopy(list(batches))


def _resources_snapshot(resources) -> Resources:
    return Resources(
        ResourceType.HUMAN,
        copy.deepcopy(list(resources)),
    )


def _planning_domain(
    state: State,
    problem: str,
    *,
    layout,
    orders: OrdersDomain,
    resources: Resources,
    warehouse_info: DynamicInfo,
) -> SimWarehouseDomain:
    return SimWarehouseDomain(
        problem_class=problem,
        objective=state.active_objective,
        layout=layout,
        orders=orders,
        resources=resources,
        articles=state.articles,
        storage=state.storage_manager.planning_snapshot(),
        dynamic_warehouse_info=warehouse_info,
    )


def _sort_orders(orders):
    return sorted(
        orders,
        key=lambda order: (
            float(order.order_date or 0.0),
            int(order.order_id),
        ),
    )


def _sort_batches(batches):
    return sorted(
        batches,
        key=lambda batch: (batch.earliest_due_date, batch.batch_id),
    )


def _apply_due_horizon(batches, now: float, due_horizon_s):
    if due_horizon_s is None:
        return batches
    due_limit = float(now) + float(due_horizon_s)
    return [b for b in batches if b.earliest_due_date <= due_limit]


def _apply_limit(items, limit):
    if limit is None:
        return items
    return items[: int(limit)]


def _select_resources(state: State, source: str):
    if source == "all":
        return sorted(
            state.resources.resources, key=lambda r: r.id
        )
    if source == "available":
        return [
            r for r in sorted(state.resources.resources, key=lambda r: r.id)
            if r.available
        ]
    if source == "nonactive":
        return [
            r for r in sorted(state.resources.resources, key=lambda r: r.id)
            if r.available and not r.occupied
        ]
    if source == "dispatchable":
        return [
            r for r in sorted(state.resources.resources, key=lambda r: r.id)
            if state.is_dispatchable(r.id)
        ]
    raise ValueError(f"Unknown resource source: {source!r}")


def _apply_trigger_scope(resources, trigger):
    picker_id = getattr(trigger, "picker_id", None)
    if picker_id is None:
        return resources
    matching = [r for r in resources if r.id == int(picker_id)]
    return matching


# ───────────────────────── Active-tour projection helpers ──────────────


def _position(tour, time: float) -> tuple[RouteNode, float | None]:
    if not tour.is_travelling:
        return tour.current_node(), None
    duration = tour.edge_arrives_at - tour.edge_started_at
    progress = 1.0 if duration <= 0 else (time - tour.edge_started_at) / duration
    progress = min(1.0, max(0.0, progress))
    ox, oy = tour.edge_origin.position
    dx, dy = tour.edge_destination.position
    return (
        RouteNode(
            (ox + (dx - ox) * progress, oy + (dy - oy) * progress),
            NodeType.ROUTE,
        ),
        progress,
    )


def _residual_batch(tour) -> BatchObject:
    remaining = list(tour.remaining_picks)
    orders = []
    for order in tour.batch.orders:
        positions = []
        for position in order.pick_positions:
            if position in remaining:
                positions.append(position)
                remaining.remove(position)
        orders.append(
            WarehouseOrder(
                order_id=order.order_id,
                parent_order_id=order.parent_order_id,
                due_date=order.due_date,
                order_date=order.order_date,
                pick_positions=tuple(positions),
            )
        )
    if remaining:
        raise ValueError("Residual picks cannot be mapped to active orders")
    return BatchObject(
        tour.batch.batch_id,
        orders,
        bin_assignments={
            cart_bin.bin_id: tuple(sorted(cart_bin.order_ids))
            for cart_bin in getattr(tour, "cart_bins", [])
        },
    )


def _layout_with_origin(layout, tour, origin, progress):
    projected = copy.copy(layout)
    network = copy.copy(layout.layout_network)
    if progress is None or origin.position in layout.layout_network.graph:
        network.start_node = origin.position
        network.closest_node_to_start = origin.position
        projected.layout_network = network
        return projected
    graph = layout.layout_network.graph.copy()
    graph.add_node(origin.position, pos=origin.position, type="route_origin")
    graph.add_edge(
        origin.position,
        tour.edge_destination.position,
        weight=float(tour.edge_distance) * (1.0 - progress),
    )
    if not graph.is_directed():
        graph.add_edge(
            origin.position,
            tour.edge_origin.position,
            weight=float(tour.edge_distance) * progress,
        )
    base_nodes = list(network.distance_matrix.index)
    nodes = [*base_nodes, origin.position]
    size = len(base_nodes)
    base_distances = network.distance_matrix.to_numpy(dtype=float, copy=False)
    distances = np.full((size + 1, size + 1), np.inf)
    distances[:size, :size] = base_distances
    distances[size, size] = 0.0

    predecessors = np.full((size + 1, size + 1), -9999, dtype=int)
    predecessors[:size, :size] = network.predecessor_matrix
    origin_idx = base_nodes.index(tour.edge_origin.position)
    destination_idx = base_nodes.index(tour.edge_destination.position)
    travelled = float(tour.edge_distance) * progress
    remaining = float(tour.edge_distance) * (1.0 - progress)

    if graph.is_directed():
        forward = remaining + base_distances[destination_idx]
        first_endpoint = np.full(size, destination_idx, dtype=int)
    else:
        via_origin = travelled + base_distances[origin_idx]
        via_destination = remaining + base_distances[destination_idx]
        choose_origin = via_origin <= via_destination
        forward = np.where(choose_origin, via_origin, via_destination)
        first_endpoint = np.where(choose_origin, origin_idx, destination_idx)
        distances[:size, size] = forward
        predecessors[:size, size] = first_endpoint

    distances[size, :size] = forward
    for target_idx, endpoint_idx in enumerate(first_endpoint):
        if not np.isfinite(forward[target_idx]):
            continue
        predecessors[size, target_idx] = (
            size if target_idx == endpoint_idx
            else network.predecessor_matrix[endpoint_idx, target_idx]
        )
    network.graph = graph
    network.node_list = nodes
    network.distance_matrix = pd.DataFrame(
        distances, index=nodes, columns=nodes
    )
    network.predecessor_matrix = np.asarray(predecessors)
    network.start_node = origin.position
    network.closest_node_to_start = origin.position
    projected.layout_network = network
    return projected


# ─────────────────────────── StateAdapter ──────────────────────────────


class StateAdapter:
    """Project one decision problem's planning snapshot from operational state.

    The adapter is configured by a small exposure mapping (orders, batches,
    resources, active_tour) parsed from the engine YAML in :mod:`casim.setup`.
    """

    def __init__(
        self,
        *,
        orders: dict | None = None,
        batches: dict | None = None,
        resources: dict | None = None,
        active_tour: dict | None = None,
    ):
        self.orders_cfg = orders
        self.batches_cfg = batches
        self.resources_cfg = resources or {"source": "all"}
        self.active_tour_cfg = active_tour

    # ── orders ──

    def _selected_orders(self, state: State):
        cfg = self.orders_cfg
        if cfg is None:
            return []
        source = str(cfg.get("source", "buffered"))
        if source != "buffered":
            raise ValueError(f"Unknown order source: {source!r}")
        orders = _sort_orders(state.order_manager.get_order_buffer())
        return _apply_limit(orders, cfg.get("limit"))

    # ── batches ──

    def _selected_batches(self, state: State):
        cfg = self.batches_cfg
        if cfg is None:
            return [], []
        source = str(cfg.get("source", "buffered"))
        now = state.current_time
        due_horizon_s = cfg.get("due_horizon_s")
        limit = cfg.get("limit")
        if source == "buffered":
            batches = _apply_due_horizon(
                state.order_manager.get_pick_list_buffer(), now, due_horizon_s
            )
            batches = _sort_batches(batches)
            return _apply_limit(batches, limit), []
        if source == "buffered_and_replannable":
            eligible_batches = _apply_due_horizon(
                state.order_manager.get_pick_list_buffer(), now, due_horizon_s
            )
            replannable = []
            for tour in state.tour_manager.replannable_tours():
                if _apply_due_horizon([tour.batch], now, due_horizon_s):
                    replannable.append(tour)
                    eligible_batches.append(tour.batch)
            batches = _sort_batches(eligible_batches)
            batches = _apply_limit(batches, limit)
            if limit is not None:
                included = {b.batch_id for b in batches}
                replannable = [
                    t for t in replannable if t.batch.batch_id in included
                ]
            return batches, replannable
        raise ValueError(f"Unknown batch source: {source!r}")

    # ── resources ──

    def _selected_resources(self, state: State, trigger):
        cfg = self.resources_cfg
        source = str(cfg.get("source", "all"))
        scope = str(cfg.get("scope", ""))
        resources = _select_resources(state, source)
        if scope == "trigger_if_present":
            resources = _apply_trigger_scope(resources, trigger)
        return resources

    def _ready_times(self, state: State, resources):
        for resource in resources:
            resource.available_at = state.tour_manager.picker_ready_at(
                resource.id, state.current_time
            )

    # ── active-tour projection ──

    def _project_active_tour(self, state: State, trigger):
        cfg = self.active_tour_cfg
        if cfg is None:
            return None
        if trigger is None:
            raise ValueError("Active-tour projection requires its trigger")
        source = str(cfg.get("source", "residual"))
        if source != "residual":
            raise ValueError(f"Unknown active_tour source: {source!r}")
        congestion_penalty = float(cfg.get("congestion_penalty", 0.0))
        picker_id = int(trigger.picker_id)
        tour = state.tour_manager.get_tour(int(trigger.tour_id))
        if (
            tour.assigned_resource != picker_id
            or tour.route_version != int(trigger.route_version)
            or state.tour_manager.get_active_tour_for_picker(picker_id)
            is not tour
        ):
            raise ValueError(
                "Intervention trigger no longer matches active state"
            )
        origin, progress = _position(tour, state.current_time)
        picker = _resources_snapshot(
            [state.get_resource(picker_id)]
        ).resources[0]
        picker.current_location = origin
        residual_batch = _residual_batch(tour)
        layout = _layout_with_origin(
            state.layout_manager.planning_layout(congestion_penalty),
            tour,
            origin,
            progress,
        )
        return tour, picker, residual_batch, layout, origin, progress

    # ── main projection ──

    def transform_state(
        self, state: State, problem: str, trigger=None
    ) -> SimWarehouseDomain:
        active = self._project_active_tour(state, trigger)
        if active is not None:
            return self._active_tour_snapshot(state, problem, trigger, active)

        orders = _orders_snapshot(self._selected_orders(state))
        batches, replannable = self._selected_batches(state)
        resources = _resources_snapshot(self._selected_resources(state, trigger))
        self._ready_times(state, resources.resources)
        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            buffered_batches=_batches_snapshot(batches),
            replannable_tours=copy.deepcopy(replannable),
            done=state.done_flag,
            n_staged_pallets=state.n_staged_pallets,
            is_break=state.is_break,
        )
        return _planning_domain(
            state,
            problem,
            layout=state.layout_manager.layout,
            orders=orders,
            resources=resources,
            warehouse_info=warehouse_info,
        )

    def _active_tour_snapshot(self, state, problem, trigger, active):
        tour, picker, residual_batch, layout, origin, progress = active
        congestion_penalty = float(
            self.active_tour_cfg.get("congestion_penalty", 0.0)
        )
        orders_cfg = self.orders_cfg
        if orders_cfg is not None and str(orders_cfg.get("source")) == "buffered":
            orders = _orders_snapshot(self._selected_orders(state))
        else:
            orders = OrdersDomain(tpe=OrderType.STANDARD, orders=[])
        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            current_picker=picker,
            buffered_batches=[residual_batch],
            done=state.done_flag,
            n_staged_pallets=state.n_staged_pallets,
            active_tour_id=tour.tour_id,
            route_version=tour.route_version,
            intervention_resumes_execution=bool(trigger.resumes_execution),
            origin_type="edge" if progress is not None else "node",
            edge_origin=(
                tour.edge_origin.position if progress is not None else None
            ),
            edge_destination=(
                tour.edge_destination.position if progress is not None else None
            ),
            edge_progress=progress,
            cart_bin_order_ids=state.tour_manager.bin_order_ids(tour.tour_id),
            locked_bin_ids=tuple(
                sorted(state.tour_manager.locked_bin_ids(tour.tour_id))
            ),
        )
        return _planning_domain(
            state,
            problem,
            layout=layout,
            orders=orders,
            resources=Resources(ResourceType.HUMAN, [picker]),
            warehouse_info=warehouse_info,
        )

    # ── projected features ──

    def projected_features(self) -> tuple[str, ...]:
        features: list[str] = []
        if self.orders_cfg is not None:
            features.append("buffered_orders")
        if self.batches_cfg is not None:
            source = str(self.batches_cfg.get("source", "buffered"))
            features.append("buffered_batches")
            if source == "buffered_and_replannable":
                features.append("replannable_unstarted_work")
        res_source = str(self.resources_cfg.get("source", "all"))
        if res_source == "dispatchable":
            features.append("available_resources")
        elif res_source == "nonactive":
            features.append("available_resources")
            features.append("resource_ready_times")
        elif res_source == "available":
            features.append("available_resources")
            features.append("resource_ready_times")
        elif res_source == "all":
            pass
        if self.active_tour_cfg is not None:
            features.extend(
                [
                    "active_residual_tour",
                    "residual_picks",
                    "arbitrary_origin_node",
                    "arbitrary_origin_edge",
                    "active_residual_batch",
                    "bin_ownership",
                    "locked_bins",
                    "residual_capacity",
                ]
            )
            if self.orders_cfg is not None:
                features.append("buffered_orders")
            if float(self.active_tour_cfg.get("congestion_penalty", 0.0)) > 0:
                features.append("congestion_cost_snapshot")
        return tuple(features)
