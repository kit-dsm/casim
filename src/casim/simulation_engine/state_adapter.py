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
from casim.domain_objects.tour_model import TourStates
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


class StateAdapter:
    planning_features: tuple[str, ...] = ()

    def transform_state(self, state: State, problem: str, trigger=None):
        raise NotImplementedError

    def projected_features(self) -> tuple[str, ...]:
        return tuple(self.planning_features)


class OrderWindowAdapter(StateAdapter):
    planning_features = ("buffered_orders", "available_resources")

    def __init__(
        self,
        max_orders: int | None = None,
        max_pickers: int | None = None,
    ):
        self.max_orders = max_orders
        self.max_pickers = max_pickers

    def transform_state(self, state: State, problem: str, trigger=None):
        selected_orders = sorted(
            state.order_manager.get_order_buffer(),
            key=lambda order: (
                order.order_date if order.order_date is not None else 0.0,
                order.order_id,
            ),
        )
        if self.max_orders is not None:
            selected_orders = selected_orders[: self.max_orders]

        orders = _orders_snapshot(selected_orders)

        layout = state.layout_manager.layout

        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_batches=None,
            active_tours=None,
            done=state.done_flag,
            n_staged_pallets=0
        )
        selected_resources = [
            resource
            for resource in sorted(
                state.resources.resources,
                key=lambda value: value.id,
            )
            if state.available_for_planning(resource.id)
        ]
        trigger_picker_id = getattr(trigger, "picker_id", None)
        if trigger_picker_id is not None:
            matching = [
                resource
                for resource in selected_resources
                if resource.id == int(trigger_picker_id)
            ]
            if matching:
                selected_resources = matching
        if self.max_pickers is not None:
            selected_resources = selected_resources[: self.max_pickers]

        dynamic_resources = _resources_snapshot(selected_resources)

        return _planning_domain(
            state,
            problem,
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            warehouse_info=warehouse_info,
        )


class ActiveTourRoutingAdapter(StateAdapter):
    """Project and commit one active tour's residual ORP."""

    planning_features = (
        "active_residual_tour",
        "residual_picks",
        "arbitrary_origin_node",
        "arbitrary_origin_edge",
        "active_residual_batch",
        "bin_ownership",
        "locked_bins",
        "residual_capacity",
        "buffered_orders",
    )

    def __init__(self, congestion_penalty: float = 0.0):
        self.congestion_penalty = float(congestion_penalty)

    def projected_features(self) -> tuple[str, ...]:
        features = list(self.planning_features)
        if self.congestion_penalty > 0:
            features.append("congestion_cost_snapshot")
        return tuple(features)

    @staticmethod
    def _position(tour, time: float) -> tuple[RouteNode, float | None]:
        if not tour.is_travelling:
            return tour.current_node(), None
        duration = tour.edge_arrives_at - tour.edge_started_at
        progress = 1.0 if duration <= 0 else (
            (time - tour.edge_started_at) / duration
        )
        progress = min(1.0, max(0.0, progress))
        ox, oy = tour.edge_origin.position
        dx, dy = tour.edge_destination.position
        return (
            RouteNode(
                (
                    ox + (dx - ox) * progress,
                    oy + (dy - oy) * progress,
                ),
                NodeType.ROUTE,
            ),
            progress,
        )

    @staticmethod
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

    @staticmethod
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
        base_distances = network.distance_matrix.to_numpy(
            dtype=float,
            copy=False,
        )
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
            forward = np.where(
                choose_origin,
                via_origin,
                via_destination,
            )
            first_endpoint = np.where(
                choose_origin,
                origin_idx,
                destination_idx,
            )
            distances[:size, size] = forward
            predecessors[:size, size] = first_endpoint

        distances[size, :size] = forward
        for target_idx, endpoint_idx in enumerate(first_endpoint):
            if not np.isfinite(forward[target_idx]):
                continue
            predecessors[size, target_idx] = (
                size
                if target_idx == endpoint_idx
                else network.predecessor_matrix[endpoint_idx, target_idx]
            )
        network.graph = graph
        network.node_list = nodes
        network.distance_matrix = pd.DataFrame(
            distances,
            index=nodes,
            columns=nodes,
        )
        network.predecessor_matrix = np.asarray(predecessors)
        network.start_node = origin.position
        network.closest_node_to_start = origin.position
        projected.layout_network = network
        return projected

    def transform_state(self, state: State, problem: str, trigger=None):
        if trigger is None:
            raise ValueError("Active-tour projection requires its trigger")
        picker_id = int(trigger.picker_id)
        tour = state.tour_manager.get_tour(int(trigger.tour_id))
        if (
            tour.assigned_resource != picker_id
            or tour.route_version != int(trigger.route_version)
            or state.tour_manager.get_active_tour_for_picker(picker_id)
            is not tour
        ):
            raise ValueError("Intervention trigger no longer matches active state")
        origin, progress = self._position(tour, state.current_time)
        picker = _resources_snapshot(
            [state.get_resource(picker_id)],
        ).resources[0]
        picker.current_location = origin
        residual_batch = self._residual_batch(tour)
        layout = self._layout_with_origin(
            state.layout_manager.planning_layout(self.congestion_penalty),
            tour,
            origin,
            progress,
        )
        return _planning_domain(
            state,
            problem,
            layout=layout,
            orders=_orders_snapshot(state.order_manager.get_order_buffer()),
            resources=Resources(ResourceType.HUMAN, [picker]),
            warehouse_info=DynamicInfo(
                tpe=WarehouseInfoType.ONLINE,
                time=state.current_time,
                current_picker=picker,
                buffered_batches=[residual_batch],
                done=state.done_flag,
                active_tour_id=tour.tour_id,
                route_version=tour.route_version,
                intervention_resumes_execution=bool(
                    trigger.resumes_execution
                ),
                origin_type="edge" if progress is not None else "node",
                edge_origin=(
                    tour.edge_origin.position if progress is not None else None
                ),
                edge_destination=(
                    tour.edge_destination.position
                    if progress is not None else None
                ),
                edge_progress=progress,
                cart_bin_order_ids=state.tour_manager.bin_order_ids(
                    tour.tour_id
                ),
                locked_bin_ids=tuple(
                    sorted(
                        state.tour_manager.locked_bin_ids(tour.tour_id)
                    )
                ),
            ),
        )


class ORSPAdapter(StateAdapter):
    planning_features = (
        "buffered_batches",
        "available_resources",
        "existing_tours",
        "resource_ready_times",
    )

    def __init__(
        self,
        max_batches: int | None = None,
        due_horizon_s: float | None = None,
    ):
        self.max_batches = max_batches
        self.due_horizon_s = due_horizon_s

    def _window_batches(self, batches, now: float):
        batches = sorted(
            batches,
            key=lambda batch: (batch.earliest_due_date, batch.batch_id),
        )
        if self.due_horizon_s is not None:
            due_limit = now + float(self.due_horizon_s)
            batches = [
                batch
                for batch in batches
                if batch.earliest_due_date <= due_limit
            ]
        if self.max_batches is not None:
            batches = batches[: self.max_batches]
        return batches

    def transform_state(self, state: State, problem: str, trigger=None):
        layout = state.layout_manager.layout
        selected_batches = self._window_batches(
            state.order_manager.get_pick_list_buffer(),
            state.current_time,
        )
        buffered_pls = _batches_snapshot(selected_batches)
        active_or_scheduled_tours = []
        all_tours = state.tour_manager.all_tours
        for tour_id, tour in all_tours.items():
            if tour.status in [TourStates.STARTED,
                               TourStates.SCHEDULED,
                               TourStates.ASSIGNED]:
                active_or_scheduled_tours.append(copy.deepcopy(tour))
        n_staged_pallets = state.n_staged_pallets
        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_batches=buffered_pls,
            active_tours=active_or_scheduled_tours,
            done=state.done_flag,
            n_staged_pallets=n_staged_pallets,
            is_break=state.is_break
        )
        selected_resources = [
            resource
            for resource in state.resources.resources
            if not resource.occupied and resource.available
        ]
        dynamic_resources = _resources_snapshot(selected_resources)
        for resource in dynamic_resources.resources:
            resource.available_at = state.tour_manager.picker_ready_at(
                resource.id,
                state.current_time,
            )

        return _planning_domain(
            state,
            problem,
            layout=layout,
            orders=OrdersDomain(tpe=OrderType.STANDARD, orders=[]),
            resources=dynamic_resources,
            warehouse_info=warehouse_info,
        )


class ReORSPAdapter(StateAdapter):
    planning_features = (
        "buffered_batches",
        "available_resources",
        "existing_tours",
        "replannable_unstarted_work",
        "resource_ready_times",
    )

    def __init__(
        self,
        max_batches: int | None = None,
        due_horizon_s: float | None = None,
    ):
        self.max_batches = max_batches
        self.due_horizon_s = due_horizon_s

    def _eligible(self, batch, now: float) -> bool:
        return (
            self.due_horizon_s is None
            or batch.earliest_due_date
            <= now + float(self.due_horizon_s)
        )

    def transform_state(self, state: State, problem: str, trigger=None):
        layout = state.layout_manager.layout
        candidate_batches = [
            batch
            for batch in state.order_manager.get_pick_list_buffer()
            if self._eligible(batch, state.current_time)
        ]
        replannable_tours = []
        all_tours = state.tour_manager.all_tours
        for tour in all_tours.values():
            if tour.status not in [TourStates.STARTED,
                                   TourStates.DONE,
                                   TourStates.CANCELLED,
                                   TourStates.PENDING]:
                if self._eligible(tour.batch, state.current_time):
                    replannable_tours.append(tour)
                    candidate_batches.append(tour.batch)

        candidate_batches.sort(
            key=lambda batch: (batch.earliest_due_date, batch.batch_id)
        )
        if self.max_batches is not None:
            candidate_batches = candidate_batches[: self.max_batches]
            included_ids = {
                batch.batch_id for batch in candidate_batches
            }
            replannable_tours = [
                tour
                for tour in replannable_tours
                if tour.batch.batch_id in included_ids
            ]
        buffered_pls = _batches_snapshot(candidate_batches)
        scheduled_tours = copy.deepcopy(replannable_tours)

        n_staged_pallets = state.n_staged_pallets
        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_batches=buffered_pls,
            replannable_tours=scheduled_tours,
            done=state.done_flag,
            n_staged_pallets=n_staged_pallets,
            is_break=state.is_break
        )
        selected_resources = [
            resource
            for resource in state.resources.resources
            if resource.available
        ]
        dynamic_resources = _resources_snapshot(selected_resources)
        for resource in dynamic_resources.resources:
            resource.available_at = state.tour_manager.picker_ready_at(
                resource.id,
                state.current_time,
            )

        return _planning_domain(
            state,
            problem,
            layout=layout,
            orders=OrdersDomain(tpe=OrderType.STANDARD, orders=[]),
            resources=dynamic_resources,
            warehouse_info=warehouse_info,
        )


# State adapters are projections only; decision process events own commitment.
class OBPAdapter(StateAdapter):
    planning_features = ("buffered_orders",)

    def transform_state(self, state: State, problem: str, trigger=None):
        orders = _orders_snapshot(state.order_manager.get_order_buffer())

        layout = state.layout_manager.layout
        warehouse_info = DynamicInfo(
            tpe=WarehouseInfoType.ONLINE,
            time=state.current_time,
            congestion_rate=None,
            current_picker=None,
            buffered_batches=None,
            active_tours=None,
            done=state.done_flag,
            n_staged_pallets=0
        )
        dynamic_resources = _resources_snapshot(
            state.resources.resources,
        )

        return _planning_domain(
            state,
            problem,
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            warehouse_info=warehouse_info,
        )
