from collections import deque, defaultdict, Counter
from copy import deepcopy

from ware_ops_algos.algorithms import Route, PickList, RouteNode, NodeType, TourPlanningState, TourStates, Node, \
    WarehouseOrder
from ware_ops_sim.sim import logger


class TourManager:
    def __init__(self):
        self._tour_counter: int = 0
        self._selected_orders: dict[int, WarehouseOrder | None] = {}  # order selected for the next picker to be routed
        self.all_tours: dict[int, TourPlanningState] = {}  # tour_id -> TourExecution
        self.assignable_tours: dict[int, TourPlanningState] = {}
        self._unassigned_tour_ids: set[int] = set()
        self._active_picker_tour: dict[int, int | None] = {}  # Maps picker_id -> current active tour_id (or None)
        self._picker_tour_queues: dict[int, list[int]] = defaultdict(list)
        # self._scheduled_tours: dict[int, deque[int]] = defaultdict(deque)  # FUTURE WORK: Picker ID -> Queue of Tour IDs

        self._picker_history: dict[int, list[int]] = defaultdict(list)

    def create_tour(self, route_plan: Route) -> int:
        """
        Create a new TourExecution from a routing plan for a given picker.
        Requires at least a route.

        Returns: tour_id
        """
        self._tour_counter += 1
        tour_id = self._tour_counter
        # print(type(route_plan))
        new_tour = TourPlanningState(
            tour_id=tour_id,
            order_numbers=list(route_plan.pick_list.order_numbers),
            original_route=route_plan,
            pick_list=route_plan.pick_list,
            status=TourStates.PLANNED,
            annotated_route=route_plan.annotated_route
        )

        pick_nodes = [n for n in route_plan.annotated_route if n.node_type == NodeType.PICK]
        assert len(pick_nodes) == len(route_plan.item_sequence), print(pick_nodes)
        # logger.info(f"Pick Positions {new_tour.pick_positions}, {new_tour.picks_left}")

        self.all_tours[tour_id] = new_tour
        # self.assignable_tours[tour_id] = new_tour
        self._unassigned_tour_ids.add(tour_id)
        return tour_id

    def assign_tour(self, tour_id: int, picker_id: int):
        """
        Assign an existing tour to a picker.
        The tour_id is removed from the unassigned_tour_ids set and added to the
        corresponding picker -> tours queue.
        """
        tour = self.get_tour(tour_id)
        # print(self._unassigned_tour_ids)
        assert tour_id in self._unassigned_tour_ids
        self._unassigned_tour_ids.remove(tour_id)

        tour.assigned_resource = picker_id
        tour.status = TourStates.ASSIGNED

        self._picker_tour_queues[picker_id].append(tour_id)

    def schedule_tour(self, tour_id: int,
                      start_time: float,
                      end_time: float):
        tour = self.get_tour(tour_id)
        tour.start_time = start_time
        tour.end_time_planned = end_time
        tour.status = TourStates.SCHEDULED
        queues = self._picker_tour_queues
        picker_id = tour.assigned_resource
        assert tour_id in queues[tour.assigned_resource]

        # Maintain time-sorted queue of tours
        queue = queues[picker_id]
        queue.sort(key=lambda tid: self.get_tour(tid).start_time if self.get_tour(tid).start_time is not None else float('inf'))
        logger.info(f"After scheduling tour {tour_id} tour manager has the following tours scheduled: ")
        for t_id in queue:
            t = self.get_tour(t_id)
            logger.info(f"Tour {t_id} with start time: {t.start_time} end time: {t.end_time_planned}")

    def start_tour(self, tour_id: int, time: float):
        """
        Starts an assigned or scheduled tour and pulls and removes it
        from the picker_tours_queue to the active picker tour view.
        """
        tour = self.get_tour(tour_id)
        status = tour.status

        # todo seperate between planned and actual start time
        tour.start_time = time
        # clean up queues based on planning state
        # Tour needs to be at least assigned -> Not picker neutral
        assert status in [TourStates.ASSIGNED, TourStates.SCHEDULED, TourStates.PENDING], (
            ValueError(f"Tour should be {TourStates.ASSIGNED} or"
                       f" {TourStates.SCHEDULED} not {status}"))
        assert tour_id not in self._unassigned_tour_ids


        picker_id = tour.assigned_resource
        # Based on this check, tour should be in picker_tours_queue
        picker_tour_queue = self._picker_tour_queues[picker_id]
        logger.info(f"Pre Tour Start: picker {picker_id} has the following tours scheduled: ")
        for t_id in picker_tour_queue:
            t = self.get_tour(t_id)
            logger.info(f"Tour {t_id} with start time: {t.start_time} end time: {t.end_time_planned}")

        picker_tour_queue.remove(tour_id)
        if picker_id not in self._active_picker_tour.keys():
            self._active_picker_tour[picker_id] = tour_id
        else:
            assert self._active_picker_tour[picker_id] is None, ValueError(f"Picker {picker_id} has active tour,"
                                                                   f" {self._active_picker_tour[picker_id]}")
            self._active_picker_tour[picker_id] = tour_id
        tour.status = TourStates.STARTED

    def finish_tour(self, tour_id: int, time: float):
        tour = self.get_tour(tour_id)
        picker_id = tour.assigned_resource
        assert tour.status == TourStates.STARTED
        active_tour_id = self._active_picker_tour[picker_id]
        tour.status = TourStates.DONE
        tour.end_time = time
        tour.annotated_route = None
        # tour.original_route = None

        assert active_tour_id == tour_id, (f"Missmatch in active tour on tour finish: "
                                           f"active tour id should be {tour_id} but is {active_tour_id}")
        self._active_picker_tour[picker_id] = None
        self._picker_history[picker_id].append(tour_id)

    def get_next_tour_for_picker(self, picker_id: int):
        picker_tour_queue = self._picker_tour_queues[picker_id]
        if picker_tour_queue:
            return picker_tour_queue[0]
        else:
            return None

    def has_future_tours(self, picker_id: int) -> bool:
        """True if picker has any queued tour (including one that hasn't started yet)."""
        future_tour = self.get_next_tour_for_picker(picker_id)
        return bool(future_tour)

    def get_tour(self, tour_id: int) -> TourPlanningState:
        return self.all_tours[tour_id]

    def advance_cursor(self, tour_id: int) -> None:
        tour = self.get_tour(tour_id)
        tour.cursor += 1

    # def pop_next_pick_if_here(self, tour_id: int, node: Node) -> bool:
    #     """
    #     If the next planned pick equals `node`, pop it and return True, else False.
    #     """
    #     tour = self.get_tour(tour_id)
    #     if tour.picks_left and tour.picks_left[0] == node:
    #         tour.picks_left.popleft()
    #         return True
    #     return False

    # def pop_next_pick(self, tour_id: int, node: Node) -> bool:
    #     """
    #     If the next planned pick equals `node`, pop it and return True, else False.
    #     """
    #     tour = self.get_tour(tour_id)
    #     if tour.picks_left and tour.picks_left[0] == node:
    #         tour.picks_left.popleft()
    #         return True

    def mark_pick_positions_fulfilled_at(self, tour_id: int, node: Node) -> None:
        tour = self.get_tour(tour_id)
        for pp in tour.pick_list.pick_positions:
            if pp.pick_node == node:
                pp.fulfilled = True

    def add_selected_order(self, order: WarehouseOrder, picker_id: int):
        self._selected_orders[picker_id] = order

    def get_selected_order(self, picker_id: int):
        order = self._selected_orders[picker_id]
        self._selected_orders[picker_id] = None
        return order


