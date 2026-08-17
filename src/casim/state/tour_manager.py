import logging
import copy
from collections import defaultdict

from ware_ops_algos.algorithms import (
    NodeType,
    PickPosition,
    Route,
    RouteNode,
    WarehouseOrder,
)
from ware_ops_algos.domain_models import DimensionType, PickCart

from casim.domain_objects import (
    CartBinState,
    TourPlanningState,
    TourStates,
)

logger = logging.getLogger(__name__)


class TourManager:
    def __init__(self):
        self._tour_counter: int = 0
        self.all_tours: dict[int, TourPlanningState] = {}  # tour_id -> TourExecution
        self.assignable_tours: dict[int, TourPlanningState] = {}
        self._unassigned_tour_ids: set[int] = set()
        self._active_picker_tour: dict[int, int | None] = {}  # Maps picker_id -> current active tour_id (or None)
        self._picker_tour_queues: dict[int, list[int]] = defaultdict(list)
        self._picker_history: dict[int, list[int]] = defaultdict(list)

    def create_tour(
        self,
        route_plan: Route,
        processing_time: float = 0.0,
    ) -> int:
        """
        Create a new TourExecution from a routing plan for a given picker.
        Requires at least a route.

        Returns: tour_id
        """
        self._tour_counter += 1
        tour_id = self._tour_counter
        annotated_route = list(route_plan.annotated_route or [])
        remaining_picks = self._ordered_pick_positions(route_plan)
        new_tour = TourPlanningState(
            tour_id=tour_id,
            order_numbers=list(route_plan.batch.order_numbers),
            original_route=route_plan,
            batch=route_plan.batch,
            status=TourStates.PLANNED,
            annotated_route=annotated_route,
            processing_time=processing_time,
            executed_route_prefix=(
                [annotated_route[0]] if annotated_route else []
            ),
            remaining_picks=remaining_picks,
        )

        pick_nodes = [n for n in route_plan.annotated_route if n.node_type == NodeType.PICK]
        assert len(pick_nodes) == len(route_plan.item_sequence), print(pick_nodes)

        self.all_tours[tour_id] = new_tour
        self._unassigned_tour_ids.add(tour_id)
        return tour_id

    def discard_tour(self, tour_id: int) -> None:
        """Remove a tour that failed before it became executable."""
        tour = self.all_tours.pop(tour_id)
        self._unassigned_tour_ids.discard(tour_id)
        if tour.assigned_resource is not None:
            queue = self._picker_tour_queues[tour.assigned_resource]
            if tour_id in queue:
                queue.remove(tour_id)

    def initialize_cart_bins(
        self,
        tour_id: int,
        pick_cart: PickCart | None,
    ) -> None:
        tour = self.get_tour(tour_id)
        if pick_cart is None or pick_cart.n_boxes is None:
            tour.cart_bins = []
            tour.cart_bin_semantics_supported = False
            tour.cart_bin_compatibility_reason = (
                "picker has no cart with stable bin identities"
            )
            return

        dimensions = list(pick_cart.dimensions or [])
        dimension_count = int(
            pick_cart.n_dimension
            if pick_cart.n_dimension is not None
            else len(dimensions)
        )
        bins = [
            CartBinState(
                bin_id=bin_id,
                planned_load=[0.0] * dimension_count,
                picked_load=[0.0] * dimension_count,
            )
            for bin_id in range(int(pick_cart.n_boxes))
        ]
        supported = (
            not pick_cart.box_can_mix_orders
            and dimensions == [DimensionType.ORDERS]
            and dimension_count == 1
            and bool(pick_cart.capacities)
            and float(pick_cart.capacities[0]) >= 1.0
        )
        tour.cart_bins = bins
        tour.cart_bin_semantics_supported = supported
        if not supported:
            if pick_cart.box_can_mix_orders:
                reason = "mixed-order bins are not supported"
            elif dimensions != [DimensionType.ORDERS]:
                reason = (
                    "only one DimensionType.ORDERS cart dimension is "
                    "supported"
                )
            elif not pick_cart.capacities or pick_cart.capacities[0] < 1:
                reason = "each order bin must hold one complete order"
            else:
                reason = "cart dimension metadata is inconsistent"
            tour.cart_bin_compatibility_reason = reason
            return
        tour.cart_bin_compatibility_reason = None

        order_ids = [
            order.order_id
            for order in sorted(
                tour.batch.orders,
                key=lambda order: (
                    order.order_date if order.order_date is not None else 0.0,
                    order.order_id,
                ),
            )
        ]
        assignments = {
            int(bin_id): tuple(int(value) for value in owners)
            for bin_id, owners in tour.batch.bin_assignments.items()
        }
        if not assignments:
            if len(order_ids) > len(bins):
                raise ValueError(
                    "Active batch requires more order bins than the cart has"
                )
            assignments = {
                bin_id: (order_id,)
                for bin_id, order_id in enumerate(order_ids)
            }
        assigned = []
        for bin_id, owners in assignments.items():
            if bin_id < 0 or bin_id >= len(bins):
                raise ValueError(f"Unknown cart bin {bin_id}")
            if len(owners) > 1:
                raise ValueError(
                    "Mixed-order bin assignments are not supported"
                )
            if owners:
                bins[bin_id].order_ids = {owners[0]}
                bins[bin_id].planned_load[0] = 1.0
                assigned.append(owners[0])
        if sorted(assigned) != sorted(order_ids):
            raise ValueError(
                "Every active-batch order must own exactly one cart bin"
            )
        tour.batch.bin_assignments = {
            cart_bin.bin_id: tuple(sorted(cart_bin.order_ids))
            for cart_bin in bins
        }

    def bin_order_ids(self, tour_id: int) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(sorted(cart_bin.order_ids))
            for cart_bin in self.get_tour(tour_id).cart_bins
        )

    def locked_bin_ids(self, tour_id: int) -> frozenset[int]:
        return frozenset(
            cart_bin.bin_id
            for cart_bin in self.get_tour(tour_id).cart_bins
            if cart_bin.locked
        )

    def empty_bin_ids(self, tour_id: int) -> tuple[int, ...]:
        return tuple(
            cart_bin.bin_id
            for cart_bin in self.get_tour(tour_id).cart_bins
            if not cart_bin.order_ids
            and not cart_bin.locked
            and not any(cart_bin.planned_load)
            and not any(cart_bin.picked_load)
        )

    @staticmethod
    def _ordered_pick_positions(route: Route) -> list[PickPosition]:
        positions_by_node: dict[tuple, list[PickPosition]] = defaultdict(list)
        for position in route.batch.pick_positions:
            positions_by_node[position.pick_node].append(position)
        for values in positions_by_node.values():
            values.sort(
                key=lambda value: (
                    value.order_number,
                    value.article_id,
                    value.amount,
                    value.in_store,
                )
            )

        ordered: list[PickPosition] = []
        executable_pick_nodes = [
            node.position
            for node in route.annotated_route or []
            if node.node_type == NodeType.PICK
        ]
        for position in executable_pick_nodes:
            candidates = positions_by_node.get(position, [])
            if not candidates:
                raise ValueError(
                    f"Route pick {position} has no matching PickPosition"
                )
            ordered.append(candidates.pop(0))
        if any(positions_by_node.values()):
            raise ValueError("Route omits one or more batch pick positions")
        return ordered

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
        logger.debug("After scheduling tour %d, picker %d queue: %s", tour_id, picker_id, [self.get_tour(tid).start_time for tid in queue])

    def start_tour(self, tour_id: int, time: float):
        """
        Starts an assigned or scheduled tour and pulls and removes it
        from the picker_tours_queue to the active picker tour view.
        """
        tour = self.get_tour(tour_id)
        status = tour.status

        # todo seperate between planned and actual start time
        tour.start_time = time
        tour.end_time_planned = time + tour.processing_time
        # clean up queues based on planning state
        # Tour needs to be at least assigned -> Not picker neutral
        assert status in [TourStates.ASSIGNED, TourStates.SCHEDULED, TourStates.PENDING], (
            ValueError(f"Tour should be {TourStates.ASSIGNED} or"
                       f" {TourStates.SCHEDULED} not {status}"))
        assert tour_id not in self._unassigned_tour_ids


        picker_id = tour.assigned_resource
        # Based on this check, tour should be in picker_tours_queue
        picker_tour_queue = self._picker_tour_queues[picker_id]
        logger.debug("Pre Tour Start: picker %d queue: %s", picker_id, list(picker_tour_queue))

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

    def active_tours(self) -> list[TourPlanningState]:
        return [
            self.all_tours[tour_id]
            for tour_id in self._active_picker_tour.values()
            if tour_id is not None
        ]

    def replannable_tours(self) -> list[TourPlanningState]:
        return [
            tour for tour in self.all_tours.values()
            if tour.is_replannable
        ]

    def get_active_tour_for_picker(
        self,
        picker_id: int,
    ) -> TourPlanningState | None:
        tour_id = self._active_picker_tour.get(picker_id)
        return self.all_tours.get(tour_id) if tour_id is not None else None

    def replace_active_route(
        self,
        tour_id: int,
        route: Route,
        *,
        validated_replacement,
    ) -> TourPlanningState:
        tour = self.get_tour(tour_id)
        replacement_picks, inserted_orders, replacement_bins = (
            validated_replacement
        )

        if inserted_orders:
            tour.batch.orders.extend(inserted_orders)
        if replacement_bins is not None:
            tour.cart_bins = replacement_bins
            tour.batch.bin_assignments = {
                cart_bin.bin_id: tuple(sorted(cart_bin.order_ids))
                for cart_bin in replacement_bins
            }
        tour.order_numbers = list(tour.batch.order_numbers)
        tour.route_version += 1
        tour.annotated_route = list(route.annotated_route)
        tour.cursor = 0
        tour.remaining_picks = replacement_picks
        tour.edge_origin = None
        tour.edge_destination = None
        tour.edge_distance = None
        tour.edge_started_at = None
        tour.edge_arrives_at = None
        tour.replan_requested = False
        tour.intervention_event_pending = False
        tour.waiting_for = None
        return tour

    def validate_active_route(
        self,
        tour_id: int,
        route: Route,
        current_position: RouteNode,
        *,
        allow_insertions: bool = False,
    ) -> tuple[
        list[PickPosition],
        list[WarehouseOrder],
        list[CartBinState] | None,
    ]:
        tour = self.get_tour(tour_id)
        if tour.status != TourStates.STARTED:
            raise ValueError(f"Tour {tour_id} is not active")
        old_ids = set(tour.batch.order_numbers)
        new_ids = set(route.batch.order_numbers)
        if allow_insertions:
            if not old_ids.issubset(new_ids):
                raise ValueError(
                    "Active-plan replacement removed an existing batch order"
                )
        elif new_ids != old_ids:
            raise ValueError("Routing-only intervention changed active batch")
        if not route.annotated_route:
            raise ValueError("Replacement route has no executable nodes")
        if route.annotated_route[0].position != current_position.position:
            raise ValueError(
                "Replacement route does not start at the actual picker position"
            )

        replacement_picks = self._ordered_pick_positions(route)
        inserted_orders = [
            copy.deepcopy(order)
            for order in route.batch.orders
            if order.order_id not in old_ids
        ]
        expected = sorted(
            [
                *tour.remaining_picks,
                *[
                    pick
                    for order in inserted_orders
                    for pick in order.pick_positions
                ],
            ],
            key=lambda value: (
                value.order_number,
                value.article_id,
                value.pick_node,
                value.amount,
            ),
        )
        actual = sorted(
            replacement_picks,
            key=lambda value: (
                value.order_number,
                value.article_id,
                value.pick_node,
                value.amount,
            ),
        )
        if actual != expected:
            raise ValueError("Replacement route changed the remaining picks")

        replacement_bins = None
        if inserted_orders:
            if not allow_insertions:
                raise ValueError("Routing-only intervention inserted orders")
            replacement_bins = self._validate_bin_assignments(
                tour,
                route.batch.bin_assignments,
                {order.order_id for order in inserted_orders},
            )
        return replacement_picks, inserted_orders, replacement_bins

    @staticmethod
    def _validate_bin_assignments(
        tour: TourPlanningState,
        assignments: dict[int, tuple[int, ...]],
        inserted_order_ids: set[int],
    ) -> list[CartBinState]:
        if not tour.cart_bin_semantics_supported:
            raise ValueError(
                "Active-batch insertion is incompatible: "
                f"{tour.cart_bin_compatibility_reason}"
            )
        bins = copy.deepcopy(tour.cart_bins)
        if set(assignments) != {cart_bin.bin_id for cart_bin in bins}:
            raise ValueError(
                "Replacement must provide an assignment for every cart bin"
            )
        assigned_insertions: set[int] = set()
        for cart_bin in bins:
            owners = tuple(int(value) for value in assignments[cart_bin.bin_id])
            if len(owners) > 1:
                raise ValueError("Mixed-order bins are not supported")
            old_owners = set(cart_bin.order_ids)
            new_owners = set(owners)
            if old_owners:
                if new_owners != old_owners:
                    raise ValueError(
                        f"Owned cart bin {cart_bin.bin_id} cannot be reused"
                    )
                continue
            if not new_owners:
                continue
            if cart_bin.locked or any(cart_bin.planned_load) or any(
                cart_bin.picked_load
            ):
                raise ValueError(
                    f"Cart bin {cart_bin.bin_id} is not genuinely empty"
                )
            order_id = next(iter(new_owners))
            if order_id not in inserted_order_ids:
                raise ValueError(
                    "Only newly inserted orders may claim an empty bin"
                )
            cart_bin.order_ids = {order_id}
            cart_bin.planned_load[0] = 1.0
            assigned_insertions.add(order_id)
        if assigned_insertions != inserted_order_ids:
            raise ValueError(
                "Every inserted order must own one previously empty bin"
            )
        return bins

    def next_pick(
        self,
        tour_id: int,
        current_position: RouteNode,
    ) -> PickPosition:
        tour = self.get_tour(tour_id)
        if not tour.remaining_picks:
            raise ValueError(f"Tour {tour_id} has no remaining pick")
        pick = tour.remaining_picks[0]
        if pick.pick_node != current_position.position:
            raise ValueError(
                f"Pick identity {pick.pick_node} does not match "
                f"{current_position}; remaining="
                f"{[value.pick_node for value in tour.remaining_picks]}, "
                f"route_suffix="
                f"{[node.position for node in tour.annotated_route[tour.cursor:]]}"
            )
        if tour.cart_bin_semantics_supported:
            owners = [
                cart_bin.bin_id
                for cart_bin in tour.cart_bins
                if pick.order_number in cart_bin.order_ids
            ]
            if len(owners) != 1:
                raise ValueError(
                    f"Order {pick.order_number} does not own exactly one bin"
                )
        return pick

    def confirm_next_pick(
        self,
        tour_id: int,
        pick: PickPosition,
    ) -> None:
        tour = self.get_tour(tour_id)
        if not tour.remaining_picks or tour.remaining_picks[0] != pick:
            raise ValueError("Confirmed pick is not the tour's next pick")
        tour.remaining_picks.pop(0)
        tour.completed_picks.append(pick)
        tour.pick_started_at = None
        tour.pick_ends_at = None
        if tour.cart_bin_semantics_supported:
            matching = [
                cart_bin
                for cart_bin in tour.cart_bins
                if pick.order_number in cart_bin.order_ids
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"Order {pick.order_number} does not own exactly one bin"
                )
            matching[0].locked = True
            matching[0].picked_load[0] = 1.0

    def cancel_tour(self, tour_id: int) -> None:
        tour = self.get_tour(tour_id)
        if tour.status in (TourStates.DONE, TourStates.CANCELLED):
            return
        tour.status = TourStates.CANCELLED
        self._unassigned_tour_ids.discard(tour_id)
        picker_id = tour.assigned_resource
        if picker_id is not None:
            queue = self._picker_tour_queues[picker_id]
            if tour_id in queue:
                queue.remove(tour_id)
            if self._active_picker_tour.get(picker_id) == tour_id:
                self._active_picker_tour[picker_id] = None

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

    def picker_ready_at(self, picker_id: int, now: float) -> float:
        """Earliest planning time after active or pending locked work."""
        ready = float(now)
        active = self.get_active_tour_for_picker(picker_id)
        if active is not None and active.end_time_planned is not None:
            ready = max(ready, float(active.end_time_planned))
        for tour_id in self._picker_tour_queues[picker_id]:
            tour = self.get_tour(tour_id)
            if (
                tour.status == TourStates.PENDING
                and tour.end_time_planned is not None
            ):
                ready = max(ready, float(tour.end_time_planned))
        return ready

    def mark_pending(self, tour_id: int) -> None:
        """Reserve a queued tour as the next one to start for its picker."""
        self.get_tour(tour_id).status = TourStates.PENDING

    def get_tour(self, tour_id: int) -> TourPlanningState:
        return self.all_tours[tour_id]

    def advance_cursor(self, tour_id: int) -> None:
        tour = self.get_tour(tour_id)
        tour.cursor += 1

