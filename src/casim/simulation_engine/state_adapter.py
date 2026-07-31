import copy

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall

from ware_ops_algos.algorithms import (
    BatchObject,
    NodeType,
    RouteNode,
    WarehouseOrder,
)
from ware_ops_algos.domain_models import Resources, WarehouseInfoType, ResourceType, OrdersDomain, OrderType

from casim.domain_objects.sim_domain import SimWarehouseDomain, DynamicInfo
from casim.domain_objects.tour_model import TourStates
from casim.state import State


class StateAdapter:
    planning_features: tuple[str, ...] = ()

    def __init__(self):
        pass

    def transform_state(self, state: State, problem: str, trigger=None):
        pass

    def projected_features(self) -> tuple[str, ...]:
        return tuple(self.planning_features)

class OrderWindowAdapter(StateAdapter):
    planning_features = ("buffered_orders", "available_resources")

    def __init__(
        self,
        max_orders: int | None = None,
        max_pickers: int | None = None,
    ):
        super().__init__()
        self.max_orders = max_orders
        self.max_pickers = max_pickers

    def transform_state(self, state: State, problem: str, trigger=None):
        buffered_orders = sorted(
            state.order_manager.planning_order_buffer(),
            key=lambda order: (
                order.order_date if order.order_date is not None else 0.0,
                order.order_id,
            ),
        )
        if self.max_orders is not None:
            buffered_orders = buffered_orders[: self.max_orders]

        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)

        layout = state.layout_manager.get_layout()

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
        resources = state.resource_manager.planning_snapshot()
        dynamic_resources_list = []
        for r in sorted(resources.resources, key=lambda resource: resource.id):
            if state.available_for_planning(r.id):
                dynamic_resources_list.append(r)
        trigger_picker_id = getattr(trigger, "picker_id", None)
        if trigger_picker_id is not None:
            matching = [
                resource
                for resource in dynamic_resources_list
                if resource.id == int(trigger_picker_id)
            ]
            if matching:
                dynamic_resources_list = matching
        if self.max_pickers is not None:
            dynamic_resources_list = dynamic_resources_list[: self.max_pickers]

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )
        return dynamic_information
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
        nodes = list(graph.nodes)
        adjacency = nx.to_scipy_sparse_array(
            graph,
            nodelist=nodes,
            weight="weight",
            dtype=float,
        )
        distances, predecessors = floyd_warshall(
            adjacency,
            directed=graph.is_directed(),
            return_predecessors=True,
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
        picker = state.resource_manager.planning_snapshot(
            {picker_id}
        ).resources[0]
        picker.current_location = origin
        residual_batch = self._residual_batch(tour)
        layout = self._layout_with_origin(
            state.layout_manager.planning_layout(self.congestion_penalty),
            tour,
            origin,
            progress,
        )
        return SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=OrdersDomain(
                tpe=OrderType.STANDARD,
                orders=state.order_manager.planning_order_buffer(),
            ),
            resources=Resources(ResourceType.HUMAN, [picker]),
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=DynamicInfo(
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
        super().__init__()
        self.orders = None
        self.selected_picker = None
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
        layout = state.layout_manager.get_layout()
        buffered_pls = self._window_batches(
            state.order_manager.planning_pick_list_buffer(),
            state.current_time,
        )
        active_or_scheduled_tours = []
        all_tours = state.tour_manager.all_tours
        for tour_id, tour in all_tours.items():
            if tour.status in [TourStates.STARTED,
                               TourStates.SCHEDULED,
                               TourStates.ASSIGNED]:
                active_or_scheduled_tours.append(copy.deepcopy(tour))
        n_staged_pallets = 0
        if state.dock_manager is not None:
            n_staged_pallets = state.dock_manager.n_staged_pallets
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
        resources = state.resource_manager.planning_snapshot()
        dynamic_resources_list = []
        for r in resources.resources:
            if not r.occupied and r.available:
                r.available_at = state.tour_manager.picker_ready_at(
                    r.id,
                    state.current_time,
                )
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=OrdersDomain(tpe=OrderType.STANDARD, orders=[]),
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information
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
        super().__init__()
        self.orders = None
        self.selected_picker = None
        self.max_batches = max_batches
        self.due_horizon_s = due_horizon_s

    def _eligible(self, batch, now: float) -> bool:
        return (
            self.due_horizon_s is None
            or batch.earliest_due_date
            <= now + float(self.due_horizon_s)
        )

    def transform_state(self, state: State, problem: str, trigger=None):
        layout = state.layout_manager.get_layout()
        buffered_pls = sorted(
            state.order_manager.planning_pick_list_buffer(),
            key=lambda batch: (batch.earliest_due_date, batch.batch_id),
        )
        scheduled_tours = []
        batches = []
        all_tours = state.tour_manager.all_tours
        for tour_id, tour in all_tours.items():  # collect all unstarted unfinished tours, STARTED and PENDING are executed as planned
            if tour.status not in [TourStates.STARTED,
                                   TourStates.DONE,
                                   TourStates.CANCELLED,
                                   TourStates.PENDING]:
                if self._eligible(tour.batch, state.current_time):
                    scheduled_tours.append(copy.deepcopy(tour))
                    buffered_pls.append(copy.deepcopy(tour.batch))

        buffered_pls = [
            batch
            for batch in buffered_pls
            if self._eligible(batch, state.current_time)
        ]
        buffered_pls.sort(
            key=lambda batch: (batch.earliest_due_date, batch.batch_id)
        )
        if self.max_batches is not None:
            buffered_pls = buffered_pls[: self.max_batches]
            included_ids = {batch.batch_id for batch in buffered_pls}
            scheduled_tours = [
                tour
                for tour in scheduled_tours
                if tour.batch.batch_id in included_ids
            ]

        n_staged_pallets = 0
        if state.dock_manager is not None:
            n_staged_pallets = state.dock_manager.n_staged_pallets
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
        resources = state.resource_manager.planning_snapshot()
        dynamic_resources_list = []
        for r in resources.resources:
            if r.available:
                r.available_at = state.tour_manager.picker_ready_at(
                    r.id,
                    state.current_time,
                )
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=OrdersDomain(tpe=OrderType.STANDARD, orders=[]),
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information
# State adapters are projections only; decision process events own commitment.
class OBPAdapter(StateAdapter):
    planning_features = ("buffered_orders",)

    def __init__(self):
        super().__init__()
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: State, problem: str, trigger=None):
        buffered_orders = state.order_manager.planning_order_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)

        layout = state.layout_manager.get_layout()
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
        resources = state.resource_manager.planning_snapshot()
        dynamic_resources_list = []
        for r in resources.resources:
            # if not r.occupied:
            dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information
# Adapters intentionally project state; process events perform commitment.
class OSBPAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: State, problem: str, trigger=None):
        buffered_orders = state.order_manager.planning_order_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)

        layout = state.layout_manager.get_layout()
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
        resources = state.resource_manager.planning_snapshot()
        dynamic_resources_list = []
        for r in resources.resources:
            # if not r.occupied:
            dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information
class ReOSBPAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.orders = None

    def transform_state(self, state: State, problem: str, trigger=None):
        buffered_orders = state.order_manager.planning_order_buffer()
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=buffered_orders)

        layout = state.layout_manager.get_layout()
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
        resources = state.resource_manager.planning_snapshot()
        dynamic_resources_list = []
        for r in resources.resources:
            # if not r.occupied:
            if r.available:
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information
class RLORSPAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: State, problem: str, trigger=None):
        layout = state.layout_manager.get_layout()
        buffered_pls = state.order_manager.planning_pick_list_buffer()[:1]
        # sorted_pls = sorted(buffered_pls, key=lambda o: o.due_date)

        active_or_scheduled_tours = []
        all_tours = state.tour_manager.all_tours
        for tour_id, tour in all_tours.items():
            if tour.status in [TourStates.STARTED,
                               TourStates.SCHEDULED,
                               TourStates.ASSIGNED]:
                active_or_scheduled_tours.append(copy.deepcopy(tour))
        n_staged_pallets = 0
        if state.dock_manager is not None:
            n_staged_pallets = state.dock_manager.n_staged_pallets
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
        resources = state.resource_manager.planning_snapshot()
        dynamic_resources_list = []
        for r in resources.resources:
            if not r.occupied:
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=OrdersDomain(tpe=OrderType.STANDARD, orders=[]),
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information

class RLOSBPAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        self.orders = None
        self.selected_picker = None

    def transform_state(self, state: State, problem: str, trigger=None):
        buffered_orders = state.order_manager.planning_order_buffer()
        sorted_orders = sorted(buffered_orders, key=lambda o: o.due_date)
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=sorted_orders[:1])

        layout = state.layout_manager.get_layout()
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
        resources = state.resource_manager.planning_snapshot()
        dynamic_resources_list = []
        for r in resources.resources:
            if not r.occupied:
                dynamic_resources_list.append(r)

        dynamic_resources = Resources(ResourceType.HUMAN, dynamic_resources_list)

        dynamic_information = SimWarehouseDomain(
            problem_class=problem,
            objective=state.active_objective,
            layout=layout,
            orders=orders,
            resources=dynamic_resources,
            articles=state.storage_manager.get_articles(),
            storage=state.get_storage(),
            dynamic_warehouse_info=warehouse_info
        )

        return dynamic_information

