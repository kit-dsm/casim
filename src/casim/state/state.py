from copy import deepcopy

from ware_ops_algos.algorithms import (
    BatchObject,
    BatchingSolution,
    NodeType,
    PickPosition,
    Route,
    RouteNode,
    ScheduledJob,
    SchedulingSolution,
    WarehouseOrder,
)
from ware_ops_algos.domain_models import (
    Articles,
    LayoutData,
    Order,
    OrderPosition,
    Resources,
    StorageLocations,
)

from .order_manager import OrderManager
from .tour_manager import TourManager
from ..domain_objects.tour_model import TourStates
from .layout_manager import LayoutManager
from .storage_manager import StorageManager
from ..trackers import ExperimentTracker


def _clone_route_plan(route: Route) -> Route:
    """Detach mutable plan ownership without recopying immutable values."""
    batch = route.batch
    detached_batch = None
    if batch is not None:
        detached_batch = BatchObject(
            batch_id=batch.batch_id,
            orders=[
                WarehouseOrder(
                    order_id=order.order_id,
                    parent_order_id=order.parent_order_id,
                    due_date=order.due_date,
                    order_date=order.order_date,
                    pick_positions=tuple(order.pick_positions),
                )
                for order in batch.orders
            ],
            bin_assignments={
                int(bin_id): tuple(order_ids)
                for bin_id, order_ids in batch.bin_assignments.items()
            },
        )
    return Route(
        distance=route.distance,
        route=None if route.route is None else list(route.route),
        item_sequence=(
            None
            if route.item_sequence is None
            else list(route.item_sequence)
        ),
        batch=detached_batch,
        annotated_route=(
            None
            if route.annotated_route is None
            else list(route.annotated_route)
        ),
        service_time=route.service_time,
    )


class State:
    """
    Holds all mutable simulation data.
    """
    def __init__(self,
                 layout: LayoutData,
                 articles: Articles,
                 storage: StorageLocations,
                 resources: Resources,
                 active_objective):
        self.current_time: float = 0.0
        self.break_until: float = 0.0
        self.resources = resources
        self.articles = articles
        self._resources_by_id = {
            resource.id: resource for resource in resources.resources
        }
        if len(self._resources_by_id) != len(resources.resources):
            raise ValueError("Resource IDs must be unique")
        self.storage_manager = StorageManager(storage=storage)
        self.order_manager = OrderManager()
        self.tour_manager = TourManager()
        self.layout_manager = LayoutManager(layout=layout)

        self.tracker = ExperimentTracker(
            n_pickers=len(resources.resources),
            )
        for resource in resources.resources:
            if resource.available:
                self.tracker.on_availability_change(
                    resource.id,
                    True,
                    self.current_time,
                )
        self.done_flag = False
        self.input_closed = False
        self.completion_reason: str | None = None
        self.is_break: bool = False
        self.active_objective = active_objective
        self.intervention_enabled = False
        self.active_batch_insertion_enabled = False
        self.dock_capacity: int | None = None
        self.n_staged_pallets = 0

    def unfinished_work(self) -> dict[str, object]:
        """Return the cross-manager work that prevents drain completion."""
        buffered_order_ids = sorted(
            order.order_id
            for order in self.order_manager.get_order_buffer()
        )
        buffered_batch_order_ids = sorted(
            {
                order.order_id
                for batch in self.order_manager.get_pick_list_buffer()
                for order in batch.orders
            }
        )
        nonterminal_tours = {
            int(tour_id): {
                "status": (
                    tour.status.value
                    if isinstance(tour.status, TourStates)
                    else str(tour.status)
                ),
                "order_ids": sorted(tour.order_numbers),
            }
            for tour_id, tour in self.tour_manager.all_tours.items()
            if tour.status not in (TourStates.DONE, TourStates.CANCELLED)
        }
        tour_order_ids = {
            order_id
            for tour in nonterminal_tours.values()
            for order_id in tour["order_ids"]
        }
        completed_ids = {
            order.order_id
            for order in self.order_manager.completed_orders
        }
        owned_ids = (
            set(buffered_batch_order_ids)
            | tour_order_ids
            | completed_ids
        )
        orphaned_order_ids = sorted(
            self.order_manager.order_history_ids() - owned_ids
        )
        occupied_picker_ids = sorted(
            resource.id
            for resource in self.resources.resources
            if resource.occupied
        )
        reservation_tour_ids = list(
            self.storage_manager.reservation_tour_ids()
        )
        return {
            "buffered_order_ids": buffered_order_ids,
            "buffered_batch_order_ids": buffered_batch_order_ids,
            "nonterminal_tours": nonterminal_tours,
            "reservation_tour_ids": reservation_tour_ids,
            "occupied_picker_ids": occupied_picker_ids,
            "orphaned_order_ids": orphaned_order_ids,
        }

    def has_unfinished_work(self) -> bool:
        work = self.unfinished_work()
        return any(
            (
                work["buffered_order_ids"],
                work["buffered_batch_order_ids"],
                work["nonterminal_tours"],
                work["reservation_tour_ids"],
                work["occupied_picker_ids"],
                work["orphaned_order_ids"],
            )
        )

    def get_resource(self, picker_id: int):
        try:
            return self._resources_by_id[picker_id]
        except KeyError as exc:
            raise ValueError(f"Unknown picker ID {picker_id}") from exc

    def register_order(self, order: Order) -> int:
        return self.order_manager.register_order(order)

    def release_order(self, order_id, release_version: int) -> Order | None:
        return self.order_manager.release_order(order_id, release_version)

    def receive_order(self, order: Order) -> None:
        self.order_manager.add_order_to_buffer(order)

    def reschedule_unreleased_orders(
        self,
        *,
        from_due: float,
        new_due: float,
        new_release: float,
        max_orders: int | None = None,
    ) -> list[tuple[object, float, int]]:
        shifted = self.order_manager.reschedule_unreleased_orders(
            from_due=from_due,
            new_due=new_due,
            new_release=new_release,
            max_orders=max_orders,
        )
        for order_id, _, _ in shifted:
            self.tracker.on_volume_shift(order_id, from_due, new_due)
        return shifted

    def configure_dock(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("Dock capacity cannot be negative")
        self.dock_capacity = int(capacity)
        self.n_staged_pallets = 0

    def release_dock(self, quantity: int | None) -> int:
        if self.dock_capacity is None:
            return 0
        released = (
            self.n_staged_pallets
            if quantity is None
            else min(self.n_staged_pallets, max(0, int(quantity)))
        )
        self.n_staged_pallets -= released
        return released

    def depart_truck(self, time: float, capacity: int | None) -> list[int]:
        delayed = []
        expected = []
        for tour in self.tour_manager.all_tours.values():
            if (
                tour.status == TourStates.STARTED
                and tour.batch.earliest_due_date == time
            ):
                delayed.append(tour.tour_id)
                expected.append(tour.end_time_planned)
        self.tracker.on_truck_departure_delays(time, delayed, expected)
        self.tracker.on_truck_departure(
            time,
            self.n_staged_pallets if capacity is None else capacity,
        )
        self.release_dock(capacity)
        return [
            picker.id
            for picker in self.resources.resources
            if not picker.occupied
        ]

    def picker_should_query(self, picker_id: int) -> bool:
        return (
            self.tour_manager.has_future_tours(picker_id)
            and self.tour_manager.get_active_tour_for_picker(picker_id) is None
        )

    def start_shift(self, time: float) -> None:
        self.is_break = False
        self.break_until = float(time)

    def close_input(self) -> None:
        self.input_closed = True

    def start_break(self, time: float, duration: float) -> float:
        self.is_break = True
        self.break_until = max(self.break_until, float(time + duration))
        return self.break_until

    def finish_break(self, time: float) -> list[int]:
        if time < self.break_until:
            return []
        self.is_break = False
        return [
            picker.id
            for picker in self.resources.resources
            if picker.available
            and not picker.occupied
            and self.tour_manager.has_future_tours(picker.id)
        ]

    def set_picker_availability(
        self,
        picker_id: int,
        available: bool,
        time: float,
    ) -> None:
        picker = self.get_resource(picker_id)
        if bool(picker.available) == bool(available):
            return
        picker.available = bool(available)
        self.tracker.on_availability_change(
            picker_id,
            available,
            time,
        )

    def available_for_planning(self, picker_id: int) -> bool:
        """Picker is usable by the planner only if not occupied and not reserved by queued tours."""
        res = self.get_resource(picker_id)
        return (
            res.available
            and not res.occupied
            and not self.tour_manager.has_future_tours(picker_id)
        )

    def query_picker_tour(
        self,
        picker_id: int,
        time: float,
    ) -> tuple[str, int | None, float]:
        """Reserve the next tour start or mark the picker idle."""
        picker = self.get_resource(picker_id)
        if not picker.available:
            picker.occupied = False
            return "none", None, float(time)
        if self.is_break:
            return "none", None, float(time)
        tour_id = self.tour_manager.get_next_tour_for_picker(picker_id)
        if tour_id is None:
            picker.occupied = False
            self.tracker.on_idle_start(picker_id, time)
            return "idle", None, float(time)
        tour = self.tour_manager.get_tour(tour_id)
        start_time = max(float(time), float(tour.start_time or time))
        if picker.tour_setup_time:
            start_time = max(
                start_time,
                float(time + picker.tour_setup_time),
            )
        tour.status = TourStates.PENDING
        return "start", tour_id, start_time

    def request_arrival_intervention(
        self,
        time: float,
    ) -> tuple[int, int, int] | None:
        """Select one active cart with a free bin for deterministic insertion."""
        if not (
            self.intervention_enabled
            and self.active_batch_insertion_enabled
        ):
            return None
        candidates = sorted(
            self.tour_manager.active_tours(),
            key=lambda tour: (tour.assigned_resource, tour.tour_id),
        )
        for tour in candidates:
            if (
                not self.tour_manager.empty_bin_ids(tour.tour_id)
                or tour.replan_requested
                or tour.intervention_event_pending
            ):
                continue
            tour.replan_requested = True
            if tour.is_picking:
                return None
            if (
                tour.is_travelling
                and tour.edge_arrives_at is not None
                and time >= tour.edge_arrives_at
            ):
                return None
            tour.intervention_event_pending = True
            return (
                int(tour.tour_id),
                int(tour.assigned_resource),
                int(tour.route_version),
            )
        return None

    def accept_intervention_request(
        self,
        tour_id: int,
        route_version: int,
    ) -> bool:
        tour = self.tour_manager.get_tour(tour_id)
        if (
            tour.route_version != route_version
            or tour.status != TourStates.STARTED
        ):
            return False
        if (
            self.active_batch_insertion_enabled
            and not self.order_manager.get_order_buffer()
        ):
            self.clear_intervention_request(tour_id)
            return False
        return True

    def commit_batching_solution(
        self,
        solution: BatchingSolution,
    ) -> list[int]:
        """Move selected raw orders into the batch buffer atomically."""
        batches = deepcopy(solution.batches)
        if not batches:
            return []
        warehouse_orders = [
            order for batch in batches for order in batch.orders
        ]
        order_ids = [order.order_id for order in warehouse_orders]
        if len(order_ids) != len(set(order_ids)):
            raise ValueError(
                "A batching commitment contains an order more than once"
            )
        parent_ids = {
            order.parent_order_id or order.order_id
            for order in warehouse_orders
        }
        missing = parent_ids - self.order_manager.buffered_order_ids()
        if missing:
            raise ValueError(
                "Batching commitment contains orders that are no longer "
                f"buffered: {sorted(missing)}"
            )

        self.order_manager.commit_order_ids(sorted(parent_ids))
        for batch in batches:
            self.order_manager.add_pick_list_to_buffer(batch)

        # Split-order children need their own audit entry because tour
        # completion is recorded against the child identity.
        for order in warehouse_orders:
            if order.parent_order_id is None:
                continue
            self.order_manager.add_order_to_history(
                Order(
                    order_id=order.order_id,
                    parent_order_id=order.parent_order_id,
                    order_date=order.order_date,
                    due_date=order.due_date,
                    order_positions=[
                        OrderPosition(
                            order_number=order.order_id,
                            article_id=pick.article_id,
                            article_name=pick.article_name,
                            amount=pick.amount,
                        )
                        for pick in order.pick_positions
                    ],
                )
            )
        return [batch.batch_id for batch in batches]

    def _find_buffered_batch(
        self,
        batch: BatchObject,
    ) -> BatchObject | None:
        matches = [
            candidate
            for candidate in self.order_manager.get_pick_list_buffer()
            if candidate.order_numbers == batch.order_numbers
        ]
        if len(matches) > 1:
            raise ValueError(
                "Batch buffer contains duplicate order membership for "
                f"{sorted(batch.order_numbers)}"
            )
        return matches[0] if matches else None

    def commit_scheduling_solution(
        self,
        solution: SchedulingSolution,
        replace_tour_ids: tuple[int, ...] = (),
    ) -> list[int]:
        """Commit selected jobs and optionally replace projected future tours."""
        jobs = list(solution.jobs)
        if not jobs and not replace_tour_ids:
            return []
        selected_order_ids = [
            order_id for job in jobs for order_id in job.order_numbers
        ]
        if len(selected_order_ids) != len(set(selected_order_ids)):
            raise ValueError(
                "A scheduling commitment contains an order more than once"
            )

        replace_ids = tuple(int(value) for value in replace_tour_ids)
        replaceable = {}
        for tour_id in replace_ids:
            tour = self.tour_manager.get_tour(tour_id)
            if tour.status in (
                TourStates.STARTED,
                TourStates.PENDING,
                TourStates.DONE,
                TourStates.CANCELLED,
            ):
                raise ValueError(
                    f"Tour {tour_id} is no longer replannable ({tour.status})"
                )
            replaceable[tour.batch.order_numbers] = tour

        raw_ids = self.order_manager.buffered_order_ids()
        buffered_batches = []
        raw_to_commit: set[int] = set()
        for job in jobs:
            route = job.job.route
            if route is None or route.batch is None:
                raise ValueError(
                    "Scheduled job has no executable route and batch"
                )
            picker = self.get_resource(int(job.picker_id))
            if not picker.available:
                raise ValueError(
                    f"Picker {picker.id} is unavailable for committed work"
                )
            buffered = self._find_buffered_batch(route.batch)
            if buffered is not None:
                buffered_batches.append(buffered)
                continue
            batch_ids = set(route.batch.order_numbers)
            if batch_ids.issubset(raw_ids):
                raw_to_commit.update(batch_ids)
                continue
            if route.batch.order_numbers in replaceable:
                continue
            raise ValueError(
                "Scheduled batch is neither buffered raw work, a buffered "
                "batch, nor part of the projected replacement set: "
                f"{sorted(batch_ids)}"
            )

        # Replacement is the less common path.  Snapshot the three affected
        # managers so an unexpected reservation or route failure cannot leave
        # a partially replaced schedule.
        previous = None
        if replace_ids:
            previous = (
                deepcopy(self.order_manager),
                deepcopy(self.tour_manager),
                deepcopy(self.storage_manager),
            )
            for tour_id in replace_ids:
                tour = self.tour_manager.get_tour(tour_id)
                batch = deepcopy(tour.batch)
                self.cancel_tour(tour_id)
                self.order_manager.add_pick_list_to_buffer(batch)

        created: list[int] = []
        try:
            for job in jobs:
                created.append(self._create_scheduled_tour(job))
            committed_batches = []
            for job in jobs:
                buffered = self._find_buffered_batch(job.job.route.batch)
                if buffered is not None:
                    committed_batches.append(buffered)
            self.order_manager.clear_pick_list_buffer(committed_batches)
            if raw_to_commit:
                self.order_manager.commit_order_ids(sorted(raw_to_commit))
        except Exception:
            if previous is not None:
                (
                    self.order_manager,
                    self.tour_manager,
                    self.storage_manager,
                ) = previous
            else:
                for tour_id in reversed(created):
                    self.storage_manager.release_reservation(tour_id)
                    self.tour_manager.discard_tour(tour_id)
            raise
        return created

    def commit_scheduling_decision(
        self,
        solution: SchedulingSolution,
        replace_tour_ids: tuple[int, ...] = (),
    ) -> list[int]:
        self.commit_scheduling_solution(solution, replace_tour_ids)
        return [
            picker_id
            for picker_id in sorted({int(job.picker_id) for job in solution.jobs})
            if self.tour_manager.get_active_tour_for_picker(picker_id) is None
        ]

    def _create_scheduled_tour(self, scheduled_job: ScheduledJob) -> int:
        route = scheduled_job.job.route
        if route is None or route.batch is None:
            raise ValueError("Scheduled job has no executable route and batch")
        picker_id = int(scheduled_job.picker_id)
        picker = self.get_resource(picker_id)
        tour_id = self.tour_manager.create_tour(
            _clone_route_plan(route),
            scheduled_job.job.processing_time,
        )
        try:
            self.tour_manager.initialize_cart_bins(
                tour_id,
                picker.pick_cart,
            )
            tour = self.tour_manager.get_tour(tour_id)
            self.storage_manager.reserve(tour_id, tour.remaining_picks)
            self.tour_manager.assign_tour(tour_id, picker_id)
            self.tour_manager.schedule_tour(
                tour_id,
                scheduled_job.start_time,
                scheduled_job.end_time,
            )
        except Exception:
            self.storage_manager.release_reservation(tour_id)
            self.tour_manager.discard_tour(tour_id)
            raise
        return tour_id

    def confirm_pick(
        self,
        tour_id: int,
        current_position: RouteNode,
    ) -> PickPosition:
        pick = self.tour_manager.next_pick(tour_id, current_position)
        self.storage_manager.confirm_pick(tour_id, pick)
        self.tour_manager.confirm_next_pick(tour_id, pick)
        return pick

    def start_tour(self, tour_id: int, time: float) -> bool:
        tour = self.tour_manager.get_tour(tour_id)
        picker = self.get_resource(tour.assigned_resource)
        if not picker.available:
            return False
        if not tour.annotated_route:
            raise ValueError(f"Tour {tour_id} has no executable route")
        self.tracker.on_idle_end(picker.id, time)
        picker.current_location = tour.annotated_route[0]
        self.tour_manager.start_tour(tour_id, time)
        picker.occupied = True
        self.tracker.on_tour_start(time)
        return True

    def tour_target(self, tour_id: int) -> tuple[int, int]:
        tour = self.tour_manager.get_tour(tour_id)
        return int(tour.assigned_resource), int(tour.route_version)

    def tour_event_is_stale(
        self,
        tour_id: int,
        route_version: int | None,
    ) -> bool:
        return (
            route_version is not None
            and self.tour_manager.get_tour(tour_id).route_version
            != route_version
        )

    @staticmethod
    def _tour_token(tour) -> tuple[int, int, int]:
        return (
            int(tour.assigned_resource),
            int(tour.tour_id),
            int(tour.route_version),
        )

    def request_next_traversal(
        self,
        tour_id: int,
        time: float,
    ) -> tuple[str, float | None]:
        tour = self.tour_manager.get_tour(tour_id)
        if tour.status != TourStates.STARTED:
            return "none", None
        if tour.at_end():
            return "end", float(time)
        picker = self.get_resource(tour.assigned_resource)
        origin = tour.current_node()
        destination = tour.next_node()
        already_waiting = tour.waiting_for == "edge_or_zone"
        if not self.layout_manager.request_travel(
            origin.position,
            destination.position,
            self._tour_token(tour),
        ):
            tour.waiting_for = "edge_or_zone"
            if (
                self.intervention_enabled
                and not already_waiting
                and not tour.intervention_event_pending
            ):
                tour.replan_requested = True
                tour.intervention_event_pending = True
                return "intervention", float(time)
            return "wait", None
        tour.waiting_for = None
        if tour.first_leg_distance_override is not None:
            distance = float(tour.first_leg_distance_override)
            tour.first_leg_distance_override = None
        else:
            distance = self.layout_manager.get_distance(origin, destination)
        arrival = float(time + distance / picker.speed)
        tour.edge_origin = origin
        tour.edge_destination = destination
        tour.edge_distance = distance
        tour.edge_started_at = float(time)
        tour.edge_arrives_at = arrival
        return "arrival", arrival

    def confirm_node_arrival(
        self,
        tour_id: int,
        time: float,
    ) -> tuple[str, float | None, list[tuple[int, int, int]]]:
        tour = self.tour_manager.get_tour(tour_id)
        picker = self.get_resource(tour.assigned_resource)
        awakened: list[tuple[int, int, int]] = []
        if tour.edge_destination is not None:
            awakened = self.layout_manager.release_travel(
                self._tour_token(tour)
            )
            destination = tour.edge_destination
            self.tour_manager.advance_cursor(tour_id)
            picker.current_location = destination
            tour.executed_route_prefix.append(destination)
            self.tracker.on_travel(
                picker_id=picker.id,
                distance=float(tour.edge_distance or 0.0),
            )
            tour.edge_origin = None
            tour.edge_destination = None
            tour.edge_distance = None
            tour.edge_started_at = None
            tour.edge_arrives_at = None
        if tour.at_end():
            return "end", float(time), awakened
        if tour.replan_requested:
            tour.intervention_event_pending = True
            return "intervention", float(time), awakened
        here = picker.current_location
        if (
            tour.remaining_picks
            and tour.remaining_picks[0].pick_node == here.position
        ):
            already_waiting = tour.waiting_for == "pick_location"
            if not self.layout_manager.request_pick(
                here.position,
                self._tour_token(tour),
            ):
                tour.waiting_for = "pick_location"
                if (
                    self.intervention_enabled
                    and not already_waiting
                    and not tour.intervention_event_pending
                ):
                    tour.replan_requested = True
                    tour.intervention_event_pending = True
                    return "intervention", float(time), awakened
                return "wait", None, awakened
            tour.waiting_for = None
            finish = float(time + picker.time_per_pick)
            tour.pick_started_at = float(time)
            tour.pick_ends_at = finish
            return "pick", finish, awakened
        return "travel", float(time), awakened

    def confirm_pick_operation(
        self,
        tour_id: int,
        pick_start: float,
        time: float,
    ) -> tuple[str, list[tuple[int, int, int]]]:
        tour = self.tour_manager.get_tour(tour_id)
        picker = self.get_resource(tour.assigned_resource)
        completed = self.confirm_pick(tour_id, picker.current_location)
        awakened = self.layout_manager.release_pick(self._tour_token(tour))
        self.tracker.on_pick_end(
            tour_id=tour_id,
            picker_id=picker.id,
            order_id=completed.order_number,
            item_id=completed.article_id,
            start_time=pick_start,
            end_time=time,
        )
        if tour.at_end():
            return "end", awakened
        if tour.replan_requested:
            tour.intervention_event_pending = True
            return "intervention", awakened
        return "travel", awakened

    def complete_tour(
        self,
        tour_id: int,
        time: float,
    ) -> tuple[int, list[tuple[int, int, int]]]:
        tour = self.tour_manager.get_tour(tour_id)
        picker_id = int(tour.assigned_resource)
        end = self.layout_manager.layout.graph_data.end_location
        if (
            self.get_resource(picker_id).current_location.position
            != end
        ):
            raise ValueError(f"Tour {tour_id} cannot complete away from {end}")
        awakened = self.layout_manager.release_all(
            (picker_id, int(tour_id), int(tour.route_version))
        )
        self.storage_manager.assert_reservation_empty(tour_id)
        on_time = []
        delayed = []
        for order_id in tour.order_numbers:
            order = self.order_manager.get_order_from_history(order_id)
            if order.due_date is None or order.due_date >= time:
                on_time.append(order_id)
            else:
                delayed.append(order_id)
        self.tour_manager.finish_tour(tour_id, time)
        self.order_manager.mark_orders_completed(tour.order_numbers)
        self.get_resource(picker_id).occupied = False
        if self.dock_capacity is not None:
            if self.n_staged_pallets + 1 > self.dock_capacity:
                raise ValueError("Dock capacity exceeded")
            self.n_staged_pallets += 1
        n_pallets_dock = self.n_staged_pallets
        self.tracker.on_tour_end(
            tour.tour_id,
            tour.start_time,
            time,
            tour.order_numbers,
            picker_id,
            on_time,
            delayed,
            n_pallets_dock,
            len(tour.completed_picks),
        )
        return picker_id, awakened

    def cancel_tour(
        self,
        tour_id: int,
    ) -> list[tuple[int, int, int]]:
        tour = self.tour_manager.get_tour(tour_id)
        awakened = []
        if tour.assigned_resource is not None:
            awakened = self.layout_manager.release_all(
                (
                    int(tour.assigned_resource),
                    int(tour_id),
                    int(tour.route_version),
                )
            )
        self.storage_manager.release_reservation(tour_id)
        self.tour_manager.cancel_tour(tour_id)
        return awakened

    def clear_intervention_request(self, tour_id: int) -> None:
        tour = self.tour_manager.get_tour(tour_id)
        tour.replan_requested = False
        tour.intervention_event_pending = False

    def cancel_unstarted_tours(
        self,
        tour_ids: list[int] | tuple[int, ...],
    ) -> list[BatchObject]:
        """Cancel selected unstarted work and return its batches to planning."""
        tours = [self.tour_manager.get_tour(int(value)) for value in tour_ids]
        invalid = [
            tour.tour_id
            for tour in tours
            if tour.status
            not in (
                TourStates.PLANNED,
                TourStates.ASSIGNED,
                TourStates.SCHEDULED,
            )
        ]
        if invalid:
            raise ValueError(f"Tours are not replannable: {invalid}")
        batches = [deepcopy(tour.batch) for tour in tours]
        for tour in tours:
            self.cancel_tour(tour.tour_id)
        for batch in batches:
            self.order_manager.add_pick_list_to_buffer(batch)
        return batches

    def disrupt_unstarted_work(
        self,
        *,
        from_due: float,
        new_due: float,
        max_orders: int | None = None,
    ) -> list[int]:
        """Move due dates for eligible buffered and unstarted work."""
        candidates = {}
        for order in self.order_manager.get_order_buffer():
            if order.due_date is not None and order.due_date >= from_due:
                candidates[order.order_id] = order
        for batch in self.order_manager.get_pick_list_buffer():
            for order in batch.orders:
                if order.due_date is not None and order.due_date >= from_due:
                    candidates[order.order_id] = order
        replannable_tours = []
        for tour in self.tour_manager.all_tours.values():
            if tour.status not in (
                TourStates.PLANNED,
                TourStates.ASSIGNED,
                TourStates.SCHEDULED,
            ):
                continue
            eligible = [
                order
                for order in tour.batch.orders
                if order.due_date is not None and order.due_date >= from_due
            ]
            if eligible:
                replannable_tours.append(tour.tour_id)
                for order in eligible:
                    candidates[order.order_id] = order
        selected_ids = [
            order.order_id
            for order in sorted(
                candidates.values(),
                key=lambda value: (value.due_date, value.order_id),
            )
        ]
        if max_orders is not None:
            selected_ids = selected_ids[: int(max_orders)]
        selected = set(selected_ids)
        affected_tours = [
            tour_id
            for tour_id in replannable_tours
            if any(
                order.order_id in selected
                for order in self.tour_manager.get_tour(tour_id).batch.orders
            )
        ]
        if affected_tours:
            self.cancel_unstarted_tours(affected_tours)
        operational_orders = [
            *self.order_manager.get_order_buffer(),
            *[
                order
                for batch in self.order_manager.get_pick_list_buffer()
                for order in batch.orders
            ],
            *[
                order
                for tour in self.tour_manager.all_tours.values()
                for order in tour.batch.orders
            ],
        ]
        for order in operational_orders:
            if order.order_id in selected:
                order.due_date = float(new_due)
        for order_id in selected_ids:
            if order_id in self.order_manager.order_history_ids():
                self.order_manager.get_order_from_history(order_id).due_date = (
                    float(new_due)
                )
            self.tracker.on_volume_shift(order_id, from_due, new_due)
        return selected_ids

    def commit_active_plan(
        self,
        tour_id: int,
        route: Route,
        *,
        picker_id: int,
        expected_version: int,
        time: float,
        resumes_execution: bool,
    ) -> tuple[str, int]:
        """Validate and atomically replace one active unexecuted suffix."""
        tour = self.tour_manager.get_tour(tour_id)
        if tour.route_version != expected_version:
            return "none", int(tour.route_version)
        if (
            tour.assigned_resource != picker_id
            or self.tour_manager.get_active_tour_for_picker(picker_id) is not tour
        ):
            raise ValueError("Replacement target is not the picker's active tour")
        depot = self.layout_manager.layout.graph_data.end_location
        if not route.annotated_route or route.annotated_route[-1].position != depot:
            raise ValueError("Replacement route must return to the depot")

        current = self.get_resource(picker_id).current_location
        if not isinstance(current, RouteNode):
            current = RouteNode(current, NodeType.ROUTE)
        origin_type = "node"
        travelled = 0.0
        interrupted_position = None
        first_leg_distance = None
        if tour.is_travelling:
            origin_type = "edge"
            duration = tour.edge_arrives_at - tour.edge_started_at
            progress = 1.0 if duration <= 0 else (
                (time - tour.edge_started_at) / duration
            )
            progress = min(1.0, max(0.0, progress))
            travelled = float(tour.edge_distance) * progress
            remaining = float(tour.edge_distance) - travelled
            ox, oy = tour.edge_origin.position
            dx, dy = tour.edge_destination.position
            current = RouteNode(
                (ox + (dx - ox) * progress, oy + (dy - oy) * progress),
                NodeType.ROUTE,
            )
            interrupted_position = current
            if len(route.annotated_route) > 1:
                next_position = route.annotated_route[1].position
                if next_position == tour.edge_origin.position:
                    first_leg_distance = travelled
                elif next_position == tour.edge_destination.position:
                    first_leg_distance = remaining
                else:
                    raise ValueError(
                        "Mid-edge replacement must first reach an edge endpoint"
                    )

        old_distance = sum(
            float(self.layout_manager.get_distance(left, right))
            for left, right in zip(
                tour.annotated_route[tour.cursor:],
                tour.annotated_route[tour.cursor + 1:],
            )
            if left.position in self.layout_manager.layout.layout_network.graph
            and right.position in self.layout_manager.layout.layout_network.graph
        )
        if origin_type == "edge":
            old_distance -= travelled
        old_ids = set(tour.batch.order_numbers)
        inserted_ids = sorted(set(route.batch.order_numbers) - old_ids)
        owners_before = [
            list(owners) for owners in self.tour_manager.bin_order_ids(tour_id)
        ]
        allow_insertions = self.active_batch_insertion_enabled
        validated_replacement = self.tour_manager.validate_active_route(
            tour_id,
            route,
            current,
            allow_insertions=allow_insertions,
        )
        replacement = [
            (node.position, node.node_type) for node in route.annotated_route
        ]
        current_suffix = [
            (node.position, node.node_type)
            for node in tour.annotated_route[
                tour.cursor + 1 if origin_type == "edge" else tour.cursor:
            ]
        ]
        unchanged = (
            replacement[1:] == current_suffix
            if origin_type == "edge"
            else replacement == current_suffix
        )
        replaced = bool(inserted_ids or not unchanged)
        old_version = int(tour.route_version)
        if replaced:
            self._install_active_plan(
                tour_id,
                route,
                interrupted_position=interrupted_position,
                travelled_distance=travelled,
                first_leg_distance=first_leg_distance,
                validated_replacement=validated_replacement,
            )
        else:
            self.clear_intervention_request(tour_id)
        self.tracker.on_intervention({
            "time": float(time),
            "picker_id": int(picker_id),
            "tour_id": int(tour_id),
            "old_version": old_version,
            "new_version": int(tour.route_version),
            "origin_type": origin_type,
            "old_residual_distance": old_distance,
            "new_residual_distance": float(route.distance),
            "bin_owners_before": owners_before,
            "bin_owners_after": [
                list(owners) for owners in self.tour_manager.bin_order_ids(tour_id)
            ],
            "inserted_order_ids": inserted_ids,
            "replaced": replaced,
        })
        if not replaced and not resumes_execution:
            return "none", int(tour.route_version)
        if tour.at_end():
            return "none", int(tour.route_version)
        if (
            tour.remaining_picks
            and tour.remaining_picks[0].pick_node == tour.current_node().position
        ):
            return "node", int(tour.route_version)
        return "travel", int(tour.route_version)

    def _install_active_plan(
        self,
        tour_id: int,
        route: Route,
        *,
        interrupted_position: RouteNode | None = None,
        travelled_distance: float = 0.0,
        first_leg_distance: float | None = None,
        validated_replacement,
    ):
        tour = self.tour_manager.get_tour(tour_id)
        old_version = int(tour.route_version)
        old_ids = set(tour.batch.order_numbers)
        inserted_ids = set(route.batch.order_numbers) - old_ids
        buffered_ids = {
            order.order_id
            for order in self.order_manager.get_order_buffer()
        }
        if not inserted_ids.issubset(buffered_ids):
            raise ValueError(
                "Inserted orders must still be present in the order buffer"
            )
        inserted_buffer_orders = [
            order
            for order in self.order_manager.get_order_buffer()
            if order.order_id in inserted_ids
        ]
        inserted_picks = [
            pick
            for order in route.batch.orders
            if order.order_id in inserted_ids
            for pick in order.pick_positions
        ]

        previous_reservation = self.storage_manager.reservation_for(tour_id)
        fields = (
            "batch",
            "order_numbers",
            "cart_bins",
            "route_version",
            "annotated_route",
            "cursor",
            "remaining_picks",
            "edge_origin",
            "edge_destination",
            "edge_distance",
            "edge_started_at",
            "edge_arrives_at",
            "replan_requested",
            "intervention_event_pending",
            "executed_route_prefix",
            "first_leg_distance_override",
            "waiting_for",
        )
        previous_tour_state = {
            name: deepcopy(getattr(tour, name))
            for name in fields
        }
        try:
            if inserted_picks:
                self.storage_manager.extend_reservation(
                    tour_id,
                    inserted_picks,
                )
            updated = self.tour_manager.replace_active_route(
                tour_id,
                route,
                validated_replacement=validated_replacement,
            )
            self.layout_manager.replace_token(
                (
                    int(tour.assigned_resource),
                    int(tour_id),
                    old_version,
                ),
                (
                    int(tour.assigned_resource),
                    int(tour_id),
                    int(updated.route_version),
                ),
            )
            if interrupted_position is not None:
                self.get_resource(
                    tour.assigned_resource
                ).current_location = interrupted_position
                if not updated.executed_route_prefix or (
                    updated.executed_route_prefix[-1].position
                    != interrupted_position.position
                ):
                    updated.executed_route_prefix.append(interrupted_position)
            updated.first_leg_distance_override = first_leg_distance
            if inserted_ids:
                self.order_manager.commit_order_ids(
                    sorted(inserted_ids)
                )
            if travelled_distance:
                self.tracker.on_travel(
                    int(tour.assigned_resource),
                    float(travelled_distance),
                )
            return updated
        except Exception:
            self.storage_manager.restore_reservation(
                tour_id,
                previous_reservation,
            )
            for name, value in previous_tour_state.items():
                setattr(tour, name, value)
            self.order_manager.restore_buffered_orders(
                inserted_buffer_orders
            )
            raise

