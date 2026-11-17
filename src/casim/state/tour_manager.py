import logging
from collections import defaultdict

from ware_ops_algos.algorithms import Route, NodeType, WarehouseOrder

from ware_ops_algos.planning.lifecycle import TOUR_TRANSITIONS, TourStates

from casim.domain_objects.tour_model import TourPlanningState

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TourManager:
    def __init__(self):
        self._tour_counter: int = 0
        self._selected_orders: dict[int, WarehouseOrder | None] = {}  # order selected for the next picker to be routed
        self.all_tours: dict[int, TourPlanningState] = {}  # tour_id -> TourExecution
        self.assignable_tours: dict[int, TourPlanningState] = {}
        self._unassigned_tour_ids: set[int] = set()
        # picker_id -> its pending or started tour, or None
        self._active_picker_tour: dict[int, int | None] = {}
        # picker_id -> its committed (scheduled) tours in planned start order:
        # the resource's queue. A planned draft is never in a queue.
        self._picker_tour_queues: dict[int, list[int]] = defaultdict(list)

        self._picker_history: dict[int, list[int]] = defaultdict(list)

    def advance(self, tour_id: int, to: TourStates) -> None:
        """Make one legal lifecycle transition."""
        tour = self.get_tour(tour_id)
        source = TourStates(tour.status)
        if to not in TOUR_TRANSITIONS[source]:
            raise ValueError(f"Tour {tour_id}: {source.value} cannot become {to.value}")
        picker_id = tour.assigned_resource
        if (to is TourStates.PENDING and picker_id is not None
                and self._active_picker_tour.get(picker_id) is not None):
            raise ValueError(f"Picker {picker_id} already has tour "
                             f"{self._active_picker_tour[picker_id]}")
        tour.status = to
        # Committing enters the tour into its resource's queue; pending takes it
        # out as the resource's current tour; cancelling a committed tour takes
        # it out of the queue; done frees the resource. A tour without a
        # resource is in no queue.
        if picker_id is None:
            return
        if to is TourStates.SCHEDULED:
            self._enqueue(tour_id)
        elif to is TourStates.PENDING:
            self._picker_tour_queues[picker_id].remove(tour_id)
            self._active_picker_tour[picker_id] = tour_id
        elif to is TourStates.CANCELLED and source is TourStates.SCHEDULED:
            self._picker_tour_queues[picker_id].remove(tour_id)
        elif to is TourStates.DONE:
            self._active_picker_tour[picker_id] = None

    def _enqueue(self, tour_id: int) -> None:
        """Insert a committed tour into its resource's queue by planned start."""
        queue = self._picker_tour_queues[self.get_tour(tour_id).assigned_resource]
        queue.append(tour_id)
        queue.sort(key=lambda tid: (self.get_tour(tid).planned_start
                                    if self.get_tour(tid).planned_start is not None
                                    else float("inf")))

    def create_tour(self, route_plan: Route, processing_time: float) -> int:
        """
        Create a new TourExecution from a routing plan for a given picker.
        Requires at least a route.

        Returns: tour_id
        """
        self._tour_counter += 1
        tour_id = self._tour_counter
        new_tour = TourPlanningState(
            tour_id=tour_id,
            order_numbers=list(route_plan.batch.order_numbers),
            original_route=route_plan,
            route=route_plan,
            batch=route_plan.batch,
            status=TourStates.PLANNED,
            annotated_route=route_plan.annotated_route,
            processing_time=processing_time,
            remaining_picks=list(route_plan.batch.pick_positions),
        )

        pick_nodes = [n for n in route_plan.annotated_route if n.node_type == NodeType.PICK]
        assert len(pick_nodes) == len(route_plan.item_sequence), print(pick_nodes)

        self.all_tours[tour_id] = new_tour
        self._unassigned_tour_ids.add(tour_id)
        return tour_id

    def assign_tour(self, tour_id: int, picker_id: int):
        """
        Assign an existing tour to a picker.
        The tour_id is removed from the unassigned_tour_ids set and added to the
        corresponding picker -> tours queue.
        """
        tour = self.get_tour(tour_id)
        assert tour_id in self._unassigned_tour_ids
        self._unassigned_tour_ids.remove(tour_id)
        # The tour enters the resource's queue when it is committed.
        tour.assigned_resource = picker_id

    def schedule_tour(self, tour_id: int,
                      start_time: float,
                      end_time: float,
                      earliest_start: float | None = None):
        tour = self.get_tour(tour_id)
        tour.planned_start = start_time
        tour.planned_end = end_time
        tour.earliest_start = earliest_start
        if tour.status is TourStates.SCHEDULED:
            self._picker_tour_queues[tour.assigned_resource].remove(tour_id)
            self._enqueue(tour_id)

    def start_tour(self, tour_id: int, time: float):
        """
        Starts an assigned or scheduled tour and pulls and removes it
        from the picker_tours_queue to the active picker tour view.
        """
        tour = self.get_tour(tour_id)
        if tour.status is not TourStates.PENDING:
            raise ValueError(f"Tour {tour_id} must be pending to start, got {tour.status}")
        # The plan keeps its start; the running tour's planned end is its forecast.
        tour.actual_start = time
        tour.planned_end = time + tour.processing_time
        self.advance(tour_id, TourStates.STARTED)

    def finish_tour(self, tour_id: int, time: float):
        tour = self.get_tour(tour_id)
        picker_id = tour.assigned_resource
        assert tour.status == TourStates.STARTED
        active_tour_id = self._active_picker_tour[picker_id]
        self.advance(tour_id, TourStates.DONE)
        tour.actual_end = time
        # The forecast is resolved: a done tour is planned to end when it ended.
        tour.planned_end = time

        assert active_tour_id == tour_id, (f"Missmatch in active tour on tour finish: "
                                           f"active tour id should be {tour_id} but is {active_tour_id}")
        self._active_picker_tour[picker_id] = None
        self._picker_history[picker_id].append(tour_id)

    def queue(self, picker_id: int) -> tuple[int, ...]:
        """The resource's committed tours in order."""
        return tuple(self._picker_tour_queues[picker_id])

    def current_tour(self, picker_id: int) -> int | None:
        """The resource's pending or started tour."""
        return self._active_picker_tour.get(picker_id)

    def get_tour(self, tour_id: int) -> TourPlanningState:
        return self.all_tours[tour_id]

    def has_tour(self, tour_id: int) -> bool:
        return tour_id in self.all_tours

    def cancel_tour(self, tour_id: int) -> TourPlanningState:
        tour = self.get_tour(tour_id)
        self.advance(tour_id, TourStates.CANCELLED)
        return tour

    def replace_planned_route(self, tour_id: int, route, added_order_ids=()) -> None:
        tour = self.get_tour(tour_id)
        if tour.status not in (TourStates.PLANNED, TourStates.SCHEDULED):
            raise ValueError(f"Tour {tour_id} cannot replace a planned route")
        tour.route = route
        tour.batch = route.batch
        tour.annotated_route = list(route.annotated_route)
        tour.remaining_picks = list(route.batch.pick_positions)
        tour.order_numbers = sorted(set(tour.order_numbers) | set(added_order_ids))
        tour.route_version += 1

    def reschedule(self, tour_id: int, picker_id: int,
                   start_time: float, end_time: float,
                   earliest_start: float | None = None) -> None:
        tour = self.get_tour(tour_id)
        if tour.assigned_resource != picker_id and tour.status is TourStates.SCHEDULED:
            self._picker_tour_queues[tour.assigned_resource].remove(tour_id)
            self._picker_tour_queues[picker_id].append(tour_id)
        tour.assigned_resource = picker_id
        self.schedule_tour(tour_id, start_time, end_time, earliest_start)

    def get_all_tours(self) -> tuple[TourPlanningState, ...]:
        """Return tours in stable creation order for projection."""
        return tuple(self.all_tours[tour_id] for tour_id in sorted(self.all_tours))

    def tour_status(self, tour_id: int) -> TourStates:
        """Return the status owned by a tour's execution record."""
        return TourStates(self.get_tour(tour_id).status)

    def advance_cursor(self, tour_id: int) -> None:
        tour = self.get_tour(tour_id)
        tour.cursor += 1


    # def mark_pick_positions_fulfilled_at(self, tour_id: int, node: Node) -> None:
    #     tour = self.get_tour(tour_id)
    #     for pp in tour.batch.pick_positions:
    #         if pp.pick_node == node:
    #             pp.fulfilled = True

    def add_selected_order(self, order: WarehouseOrder, picker_id: int):
        self._selected_orders[picker_id] = order

    def get_selected_order(self, picker_id: int):
        order = self._selected_orders[picker_id]
        self._selected_orders[picker_id] = None
        return order


