from ware_ops_algos.algorithms import Route, RouteNode, NodeType, WarehouseOrder, BatchObject, ScheduledJob
from ware_ops_algos.domain_models import LayoutData, Articles, StorageLocations, Resources, WarehouseInfo, PlannerInformation, Dock

from .order_manager import OrderManager
from .resource_manager import ResourceManager
from .tour_manager import TourManager
from .layout_manager import LayoutManager
from .storage_manager import StorageManager
from .dock_manager import DockManager
from ..trackers import ExperimentTracker


class State:
    """
    Holds all mutable simulation data.
    """
    _SHARED_FIELDS = {'_layout', '_articles', '_storage', '_resources'}

    def __init__(self,
                 layout: LayoutData,
                 articles: Articles,
                 storage: StorageLocations,
                 resources: Resources,
                 active_objective,
                 warehouse_info: WarehouseInfo | None = None,
                 information: PlannerInformation | None = None,
                 dock: Dock | None = None):
        # time is a float (simulation time units)
        self.current_time: float = 0.0
        self.current_picker_id = None
        self.break_duration: float = 0.0
        self.resource_manager = ResourceManager(resources=resources)
        self.storage_manager = StorageManager(articles=articles,
                                              storage=storage)
        self.order_manager = OrderManager()
        self.tour_manager = TourManager()
        self.layout_manager = LayoutManager(layout=layout)
        self.dock_manager = DockManager(dock)

        self._layout = layout
        self._articles = articles
        self._storage = storage
        self._resources = resources
        self.tracker = ExperimentTracker(
            n_pickers=len(resources.resources),
            )
        self.statistics = []
        self.done_flag = False
        self.decision_versions: dict[str, int] = {}
        self.information_version = 0
        self.last_solved_information_version: dict[str, int] = {}
        self.considered_active_orders: set[tuple[int, int]] = set()
        self.decision_considered_active_orders: dict[str, set[tuple[int, int]]] = {}
        self.decision_waiting: dict[str, bool] = {}
        self.decision_reconsider_at: dict[str, float | None] = {}
        self.tour_source: dict[int, str] = {}
        self.is_break: bool = False
        self.active_objective = active_objective
        self.warehouse_info = warehouse_info
        self.information = information

    def apply_all(self, decision, commands):
        """Apply one decision's prevalidated commands and advance its version."""
        events = apply(self, tuple(commands))
        self.decision_versions[decision] = self.decision_versions.get(decision, 0) + 1
        return events


    def get_storage(self) -> StorageLocations:
        return self._storage

    def available_for_planning(self, picker_id: int) -> bool:
        """Picker is usable by the planner only if not occupied and not reserved by queued tours."""
        res = self.resource_manager.get_resource(picker_id)
        return res.available and not res.occupied and not self.tour_manager.has_future_tours(picker_id)

    def add_statistic(self, picker_id: int,
                      time_value: float,
                      order_id: int) -> None:
        self.statistics.append([picker_id, time_value, order_id])

    def add_selected_order_to_planning_state(self,
                                             order: WarehouseOrder,
                                             picker_id: int | None = None):
        self.tour_manager.add_selected_order(order,
                                             picker_id)

    def add_ia_to_planning_state(self, resolved_orders):
        # TODO What to do here?
        pass

    def add_pick_list_to_planning_state(self, pick_list: BatchObject) -> None:
        self.order_manager.add_pick_list_to_buffer(pick_list)

    def add_selected_pick_list_to_planning_state(self, pick_list: BatchObject,
                                                 picker_id: int) -> None:
        self.order_manager.add_selected_pick_list(pick_list,
                                                  picker_id)

    # -> All of these functions fill the TourExecution Dataclass
    # which is maintained by the tour manager
    def add_route_to_planning_state(self, route: Route, picker_id: int | None = None) -> None:
        tour_id = self.tour_manager.create_tour(route)
        assert picker_id >= 0
        if picker_id is not None:
            self.tour_manager.assign_tour(tour_id, picker_id)

    def add_sequencing_to_planning_state(self, scheduled_job: ScheduledJob) -> int:
        picker = self.resource_manager.get_resource(scheduled_job.picker_id)
        cart = picker.pick_cart
        order_ids = sorted(scheduled_job.order_numbers)
        bins = {}
        if cart is not None:
            if cart.box_can_mix_orders:
                bins = {order_id: 0 for order_id in order_ids}
            else:
                bins = {order_id: index for index, order_id in enumerate(order_ids)}
        tour_id = self.tour_manager.create_tour(scheduled_job.job.route, scheduled_job.job.processing_time)
        self.tour_manager.assign_tour(tour_id, scheduled_job.picker_id)
        self.tour_manager.schedule_tour(tour_id,
                                        scheduled_job.start_time,
                                        scheduled_job.end_time)
        self.tour_manager.get_tour(tour_id).cart_bins = bins
        return tour_id

    def commit_active_route(self, route: Route, admitted_order_ids: tuple[int, ...]) -> int:
        """Apply a routed admission decision to the running tour."""
        tour = self.tour_manager.get_tour(route.batch.tour_id)
        picker_id = tour.assigned_resource
        origin = tour.position_at(self.current_time)
        tour.executed_prefix = list(tour.pinned_prefix(
            tour.edge_destination is not None))
        picker = self.resource_manager.get_resource(picker_id)
        if tour.edge_destination is not None:
            duration = tour.edge_end_time - tour.edge_start_time
            fraction = 1.0 if duration == 0 else min(1.0, max(0.0,
                (self.current_time - tour.edge_start_time) / duration))
            self.tracker.on_travel(picker_id=picker_id, distance=tour.edge_distance * fraction)
        self.order_manager.clear_order_buffer_by_ids(admitted_order_ids)
        tour.order_numbers = sorted(set(tour.order_numbers) | set(admitted_order_ids))
        known = {order.order_id: order for order in tour.batch.orders}
        known.update((order.order_id, order) for order in route.batch.orders)
        tour.batch = BatchObject(batch_id=tour.batch.batch_id,
                                 orders=[known[order_id] for order_id in tour.order_numbers])
        tour.route = route
        tour.annotated_route = list(route.annotated_route)
        tour.remaining_picks = list(route.batch.pick_positions)
        occupied_bins = set(tour.cart_bins.values())
        free_bins = (index for index in range(picker.pick_cart.n_boxes)
                     if index not in occupied_bins)
        tour.cart_bins.update({order_id: (0 if picker.pick_cart.box_can_mix_orders
                                           else next(free_bins))
                               for order_id in sorted(admitted_order_ids)})
        tour.cursor = 0
        tour.routing_origin = route.routing_origin
        tour.edge_origin = tour.edge_destination = None
        tour.edge_start_time = tour.edge_end_time = tour.edge_distance = None
        tour.route_version += 1
        tour.end_time_planned = (self.current_time + route.distance / picker.speed
                                 + len(tour.remaining_picks) * picker.time_per_pick)
        self.resource_manager.update_resource_location(
            picker_id, RouteNode(origin, NodeType.ROUTE)
        )
        return tour.route_version


from .apply import apply

