from collections import deque
from dataclasses import field, dataclass
from enum import Enum
from typing import Deque, Optional

# from tests.scratch_cbr import TourPlanningState
from ware_ops_algos.algorithms import RouteNode, Route, BatchObject, PickPosition, RoutingOrigin

Node = tuple[float, float]


class TourStates(str, Enum):
    PLANNED = "planned"  # PickList is generated
    ASSIGNED = "assigned"  # Is assigned to a picker
    SCHEDULED = "scheduled"  # Is scheduled for a point in time
    PENDING = "pending"
    STARTED = "started"  # Tour has started picking
    DONE = "done"  # Tour is done
    CANCELLED = "cancelled"


@dataclass
class TourPlanningState:
    """
    Thin wrapper around a Route object.
    Keeps track of the planning state for a single tour.

    - route_nodes / pick_sequence are copies from the plan (immutable intent).
    - cursor / picks_left / version are the mutable execution state.
    - original_route is kept only for debugging/inspection (do not mutate).
    """
    tour_id: int

    # original plan (copied from Route)
    order_numbers: list[int]
    original_route: Route
    batch: BatchObject
    annotated_route: list[RouteNode]

    assigned_resource: Optional[int] = None
    start_time: Optional[float] = None
    processing_time: Optional[float] = None
    # planning_plan: Optional[TourPlanningState] = None
    end_time: Optional[float] = None
    end_time_planned: Optional[float] = None
    # execution state, mutable during picking
    cursor: int = 0                 # index into route_nodes
    status: str = TourStates.PLANNED
    route_version: int = 0
    remaining_picks: list[PickPosition] = field(default_factory=list)
    completed_picks: list[PickPosition] = field(default_factory=list)
    edge_origin: RouteNode | None = None
    edge_destination: RouteNode | None = None
    edge_start_time: float | None = None
    edge_end_time: float | None = None
    edge_distance: float | None = None
    picking_until: float | None = None
    cart_bins: dict[int, int] = field(default_factory=dict)
    routing_origin: RoutingOrigin | None = None

    def position_at(self, time: float) -> tuple[float, float]:
        if self.edge_origin is None:
            return self.current_node().position
        duration = self.edge_end_time - self.edge_start_time
        fraction = 1.0 if duration == 0 else min(1.0, max(0.0, (time - self.edge_start_time) / duration))
        start, end = self.edge_origin.position, self.edge_destination.position
        return (start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]))

    def current_node(self) -> RouteNode:
        return self.annotated_route[self.cursor]

    def at_end(self) -> bool:
        """True if cursor is on the final node (typically the depot)."""
        return self.cursor >= len(self.annotated_route) - 1

    def next_node(self) -> RouteNode:
        return self.annotated_route[self.cursor + 1]
