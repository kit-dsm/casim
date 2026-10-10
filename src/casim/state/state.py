import copy
from copy import deepcopy

from ware_ops_algos.algorithms import Route, RouteNode, NodeType, Job, WarehouseOrder, BatchObject, ScheduledJob, WaitingSolution, SchedulingSolution, BatchingSolution, CombinedRoutingSolution
from ware_ops_algos.domain_models import LayoutData, Articles, StorageLocations, Resources, WarehouseInfo, PlannerInformation, Order, OrderPosition

from .order_manager import OrderManager
from .resource_manager import ResourceManager
from .tour_manager import TourManager
from .layout_manager import LayoutManager
from .storage_manager import StorageManager
from casim.domain_objects.tour_model import TourStates
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
                 information: PlannerInformation | None = None):
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

        self._layout = layout
        self._articles = articles
        self._storage = storage
        self._resources = resources
        self.tracker = ExperimentTracker(
            n_pickers=len(resources.resources),
            )
        self.statistics = []
        self.done_flag = False
        self.wait_version = 0
        self.considered_active_orders: set[tuple[int, int]] = set()
        self.is_break: bool = False
        self.active_objective = active_objective
        self.warehouse_info = warehouse_info
        self.information = information


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
        # route: Route = deepcopy(route)
        tour_id = self.tour_manager.create_tour(route)
        assert picker_id >= 0
        if picker_id is not None:
            self.tour_manager.assign_tour(tour_id, picker_id)
            # print(f"created tour {tour_id} for picker {picker_id}")
            # print("check resource", self.tour_manager.get_tour(tour_id).assigned_resource)

        # print("Tour created:", self.tour_manager.get_tour(tour_id))
        # self.tour_manager.assign_tour(tour_id, sequencing.picker_id)

    def add_sequencing_to_planning_state(self, scheduled_job: ScheduledJob) -> None:
        picker = self.resource_manager.get_resource(scheduled_job.picker_id)
        cart = picker.pick_cart
        order_ids = sorted(scheduled_job.order_numbers)
        bins = {}
        if cart is not None:
            if cart.box_can_mix_orders:
                bins = {order_id: 0 for order_id in order_ids}
            else:
                bins = {order_id: index for index, order_id in enumerate(order_ids)}
        # tour_id = self.tour_manager.create_tour(deepcopy(scheduled_job.job.route))
        tour_id = self.tour_manager.create_tour(scheduled_job.job.route, scheduled_job.job.processing_time)
        self.tour_manager.assign_tour(tour_id, scheduled_job.picker_id)
        self.tour_manager.schedule_tour(tour_id,
                                        scheduled_job.start_time,
                                        scheduled_job.end_time)
        self.tour_manager.get_tour(tour_id).cart_bins = bins

    def commit_waiting(self, solution: WaitingSolution) -> None:
        """Apply the waiting policy's release decision."""
        self.wait_version += 1
        if solution.action != "release":
            return
        for job in solution.jobs:
            self.add_sequencing_to_planning_state(job)
        self.order_manager.clear_order_buffer_by_ids(
            {oid for job in solution.jobs for oid in job.order_numbers}
        )

    def commit_solution(self, problem_class: str, solution) -> None:
        """Apply an ordinary CoSy solution; adapters never mutate live state."""
        if isinstance(solution, WaitingSolution):
            self.commit_waiting(solution)
            return
        if isinstance(solution, SchedulingSolution):
            if problem_class == "RORSP":
                for tour in self.tour_manager.all_tours.values():
                    if tour.status not in {TourStates.STARTED, TourStates.DONE,
                                           TourStates.CANCELLED, TourStates.PENDING}:
                        tour.status = TourStates.CANCELLED
                        if tour.assigned_resource is not None:
                            self.tour_manager.remove_canceled_tour_for_picker(
                                tour.assigned_resource, tour.tour_id)
            for job in solution.jobs:
                self.add_sequencing_to_planning_state(job)
            if problem_class in {"ORSP", "RORSP", "RLORSP"}:
                self.order_manager.clear_pick_list_buffer(
                    [job.job.route.batch for job in solution.jobs]
                )
            else:
                self.order_manager.clear_order_buffer_by_ids(
                    {oid for job in solution.jobs for oid in job.order_numbers}
                )
            return
        if isinstance(solution, BatchingSolution):
            split = problem_class in {"OSBP", "ReOSBP", "RLOSBP"}
            if split:
                ids = {order.parent_order_id or order.order_id
                       for batch in solution.batches for order in batch.orders}
            else:
                ids = {order.order_id for batch in solution.batches
                       for order in batch.orders}
            self.order_manager.clear_order_buffer_by_ids(ids)
            for batch in solution.batches:
                self.add_pick_list_to_planning_state(batch)
            if split:
                for batch in solution.batches:
                    for order in batch.orders:
                        for pick in order.pick_positions:
                            child = Order(
                                order_id=order.order_id,
                                parent_order_id=order.parent_order_id,
                                order_date=order.order_date,
                                due_date=order.due_date,
                                order_positions=[OrderPosition(
                                    order_number=order.order_id,
                                    article_id=pick.article_id,
                                    article_name=pick.article_id,
                                    amount=pick.amount,
                                )],
                            )
                            self.order_manager.add_order_to_buffer(child)
                            self.order_manager.clear_order_buffer([child])
            return
        if isinstance(solution, CombinedRoutingSolution):
            return
        if solution is not None:
            raise TypeError(f"No commit operation for {type(solution)}")

    def commit_active_route(self, route: Route, admitted_order_ids: tuple[int, ...]) -> int:
        """Apply a routed admission decision to the running tour."""
        tour = self.tour_manager.get_tour(route.batch.tour_id)
        picker_id = tour.assigned_resource
        origin = tour.position_at(self.current_time)
        picker = self.resource_manager.get_resource(picker_id)
        if tour.edge_destination is not None:
            duration = tour.edge_end_time - tour.edge_start_time
            fraction = 1.0 if duration == 0 else min(1.0, max(0.0,
                (self.current_time - tour.edge_start_time) / duration))
            self.tracker.on_travel(picker_id=picker_id, distance=tour.edge_distance * fraction)
        self.order_manager.clear_order_buffer_by_ids(admitted_order_ids)
        tour.order_numbers = sorted(set(tour.order_numbers) | set(admitted_order_ids))
        tour.batch = route.batch
        tour.annotated_route = list(route.annotated_route)
        tour.remaining_picks = list(route.batch.pick_positions)
        occupied_bins = set(tour.cart_bins.values())
        free_bins = (index for index in range(picker.pick_cart.n_boxes)
                     if index not in occupied_bins)
        tour.cart_bins.update({order_id: next(free_bins)
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

