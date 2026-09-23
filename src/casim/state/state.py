import copy
from collections import Counter
from copy import deepcopy

from ware_ops_algos.algorithms import Route, RouteNode, NodeType, Job, WarehouseOrder, BatchObject, ScheduledJob, WaitingSolution, SchedulingSolution, BatchingSolution, CombinedRoutingSolution
from ware_ops_algos.domain_models import LayoutData, Articles, StorageLocations, Resources, WarehouseInfo, Order, OrderPosition

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
                 warehouse_info: WarehouseInfo | None = None):
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


    def get_storage(self) -> StorageLocations:
        return self._storage

    def available_for_planning(self, picker_id: int) -> bool:
        """Picker is usable by the planner only if not occupied and not reserved by queued tours."""
        res = self.resource_manager.get_resource(picker_id)
        return (not res.occupied) and (not self.tour_manager.has_future_tours(picker_id))

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
                available_bins = cart.n_boxes or len(cart.capacities or [])
                if len(order_ids) > available_bins:
                    raise ValueError("Committed route exceeds cart bins")
                bins = {order_id: index for index, order_id in enumerate(order_ids)}
        # tour_id = self.tour_manager.create_tour(deepcopy(scheduled_job.job.route))
        tour_id = self.tour_manager.create_tour(scheduled_job.job.route, scheduled_job.job.processing_time)
        self.tour_manager.assign_tour(tour_id, scheduled_job.picker_id)
        self.tour_manager.schedule_tour(tour_id,
                                        scheduled_job.start_time,
                                        scheduled_job.end_time)
        self.tour_manager.get_tour(tour_id).cart_bins = bins

    def commit_waiting(self, solution: WaitingSolution) -> None:
        """Validate a release before creating tours and process events."""
        if solution.action != "release":
            if solution.action != "wait":
                raise ValueError("Unknown waiting action")
            if solution.reconsider_at is not None and solution.reconsider_at <= self.current_time:
                raise ValueError("Waiting reconsideration time must be in the future")
            self.wait_version += 1
            return
        released_ids = [oid for job in solution.jobs for oid in job.order_numbers]
        order_ids = set(released_ids)
        available = {order.order_id for order in self.order_manager.get_order_buffer()}
        if not order_ids <= available:
            raise ValueError(f"Release contains unavailable orders: {order_ids - available}")
        if len(released_ids) != len(order_ids):
            raise ValueError("Release assigns an order to more than one job")
        for job in solution.jobs:
            picker = self.resource_manager.get_resource(job.picker_id)
            cart = picker.pick_cart
            if cart is not None and not cart.box_can_mix_orders:
                capacity = cart.n_boxes or len(cart.capacities or [])
                if len(job.order_numbers) > capacity:
                    raise ValueError("Released route exceeds cart bins")
        self.wait_version += 1
        for job in solution.jobs:
            self.add_sequencing_to_planning_state(job)
        self.order_manager.clear_order_buffer_by_ids(order_ids)

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

    def commit_active_route(self, route: Route, tour_id: int,
                            expected_version: int, picker_id: int) -> int:
        """Replace only an active tour's unserved route after validating it."""
        tour = self.tour_manager.get_tour(tour_id)
        if (tour.status != TourStates.STARTED or tour.assigned_resource != picker_id
                or tour.route_version != expected_version or tour.picking_until is not None):
            raise ValueError("Active route changed before commitment")
        if route.batch is None or not route.annotated_route or route.routing_origin is None:
            raise ValueError("Active replacement needs a routed residual batch and origin")
        origin = tour.position_at(self.current_time)
        if any(abs(a - b) > 1e-8 for a, b in zip(route.routing_origin.position, origin)):
            raise ValueError("Replacement does not start at the picker's actual position")
        if route.annotated_route[0].position != route.routing_origin.position:
            raise ValueError("Replacement route has a different first node")
        if route.annotated_route[-1].position != self.layout_manager.get_layout().graph_data.end_location:
            raise ValueError("Replacement route must end at the depot")
        if (route.routing_origin.edge_destination is not None and
                (len(route.annotated_route) < 2 or
                 route.annotated_route[1].position != route.routing_origin.edge_destination)):
            raise ValueError("Replacement must finish the current edge first")
        candidate = list(route.batch.pick_positions)
        identity = lambda p: (p.order_number, p.article_id, p.pick_node, p.amount)
        if Counter(map(identity, tour.remaining_picks)) - Counter(map(identity, candidate)):
            raise ValueError("Replacement drops an uncompleted pick")
        if Counter(map(identity, tour.completed_picks)) & Counter(map(identity, candidate)):
            raise ValueError("Replacement repeats a completed pick")
        old_ids = set(tour.order_numbers)
        candidate_ids = set(route.batch.order_numbers)
        new_ids = candidate_ids - old_ids
        visible = {order.order_id for order in self.order_manager.get_order_buffer()}
        if not new_ids or not new_ids <= visible:
            raise ValueError("Replacement must contain newly visible orders")
        picker = self.resource_manager.get_resource(picker_id)
        bins = dict(tour.cart_bins)
        cart = picker.pick_cart
        if cart is not None:
            capacity = cart.n_boxes or len(cart.capacities or [])
            if cart.box_can_mix_orders:
                bins.update({order_id: 0 for order_id in new_ids})
            else:
                empty = [index for index in range(capacity) if index not in bins.values()]
                if len(new_ids) > len(empty):
                    raise ValueError("No empty cart bin for inserted order")
                bins.update(zip(sorted(new_ids), empty))

        if tour.edge_destination is not None:
            duration = tour.edge_end_time - tour.edge_start_time
            fraction = 1.0 if duration == 0 else min(1.0, max(0.0,
                (self.current_time - tour.edge_start_time) / duration))
            self.tracker.on_travel(picker_id=picker_id, distance=tour.edge_distance * fraction)
        self.order_manager.clear_order_buffer_by_ids(new_ids)
        tour.order_numbers = sorted(old_ids | candidate_ids)
        tour.batch = route.batch
        tour.annotated_route = list(route.annotated_route)
        tour.remaining_picks = candidate
        tour.cart_bins = bins
        tour.cursor = 0
        tour.routing_origin = route.routing_origin
        tour.edge_origin = tour.edge_destination = None
        tour.edge_start_time = tour.edge_end_time = tour.edge_distance = None
        tour.route_version += 1
        tour.end_time_planned = (self.current_time + route.distance / picker.speed
                                 + len(candidate) * picker.time_per_pick)
        self.resource_manager.update_resource_location(
            picker_id, RouteNode(origin, NodeType.ROUTE)
        )
        return tour.route_version

