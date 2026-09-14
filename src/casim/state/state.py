import copy
from copy import deepcopy

from ware_ops_algos.algorithms import (
    BatchObject,
    NodeType,
    Route,
    RouteNode,
    ScheduledJob,
    WarehouseOrder,
)
from ware_ops_algos.domain_models import LayoutData, Articles, StorageLocations, Resources

from .order_manager import OrderManager
from .resource_manager import ResourceManager
from .tour_manager import TourManager
from .layout_manager import LayoutManager
from .storage_manager import StorageManager
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
                 active_objective):
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
        self.input_closed = False
        self.is_break: bool = False
        self.active_objective = active_objective
        self.active_batch_insertion_enabled = False


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
        # tour_id = self.tour_manager.create_tour(deepcopy(scheduled_job.job.route))
        tour_id = self.tour_manager.create_tour(scheduled_job.job.route, scheduled_job.job.processing_time)
        self.tour_manager.assign_tour(tour_id, scheduled_job.picker_id)
        self.tour_manager.schedule_tour(tour_id,
                                        scheduled_job.start_time,
                                        scheduled_job.end_time)

    def request_arrival_intervention(self):
        """Request replanning of the first active tour at a safe boundary."""
        if not self.active_batch_insertion_enabled:
            return None
        for tour in sorted(
            self.tour_manager.active_tours(),
            key=lambda value: value.tour_id,
        ):
            if tour.replan_requested or tour.intervention_event_pending:
                continue
            tour.replan_requested = True
            if tour.is_travelling or tour.is_picking:
                return None
            tour.intervention_event_pending = True
            return tour.tour_id, tour.assigned_resource, tour.route_version
        return None

    def tour_event_is_stale(
        self, tour_id: int, route_version: int | None
    ) -> bool:
        if route_version is None:
            return False
        tour = self.tour_manager.all_tours.get(tour_id)
        return tour is None or tour.route_version != route_version

    def accept_intervention_request(
        self, tour_id: int, route_version: int
    ) -> bool:
        tour = self.tour_manager.get_tour(tour_id)
        return (
            tour.route_version == route_version
            and tour.status.value == "started"
            and bool(self.order_manager.get_order_buffer())
        )

    def resume_active_tour(self, tour_id: int) -> int:
        tour = self.tour_manager.get_tour(tour_id)
        tour.replan_requested = False
        tour.intervention_event_pending = False
        return tour.route_version

    def commit_active_plan(
        self,
        tour_id: int,
        route: Route,
        *,
        picker_id: int,
        expected_version: int,
    ) -> int:
        """Validate and atomically replace a node-boundary route suffix."""
        tour = self.tour_manager.get_tour(tour_id)
        if tour.route_version != expected_version:
            return tour.route_version
        if (
            tour.assigned_resource != picker_id
            or self.tour_manager.get_active_tour_for_picker(picker_id) is not tour
        ):
            raise ValueError("Replacement target is not the picker's active tour")
        picker_position = self.resource_manager.get_resource(picker_id).current_location
        if not isinstance(picker_position, RouteNode):
            picker_position = RouteNode(picker_position, NodeType.ROUTE)

        inserted_ids = set(route.batch.order_numbers) - set(tour.order_numbers)
        missing = inserted_ids - {
            order.order_id for order in self.order_manager.get_order_buffer()
        }
        if missing:
            raise ValueError(
                f"Insertion contains orders no longer buffered: {sorted(missing)}"
            )
        updated = self.tour_manager.replace_active_route(
            tour_id,
            route,
            picker_position,
        )
        self.resource_manager.update_resource_location(
            picker_id, updated.annotated_route[0]
        )
        self.order_manager.clear_order_buffer_by_ids(sorted(inserted_ids))
        return updated.route_version

