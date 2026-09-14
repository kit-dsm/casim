from dataclasses import field, dataclass
from enum import Enum
from typing import Optional

# from tests.scratch_cbr import TourPlanningState
from ware_ops_algos.algorithms import (
    BatchObject,
    PickPosition,
    Route,
    RouteNode,
)

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
    - cursor / remaining_picks / route_version are mutable execution state.
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
    status: TourStates = TourStates.PLANNED
    route_version: int = 0
    executed_route_prefix: list[RouteNode] = field(default_factory=list)
    remaining_picks: list[PickPosition] = field(default_factory=list)
    completed_picks: list[PickPosition] = field(default_factory=list)

    # Travel is committed only on NodeArrival. These fields describe the
    # physical edge occupied between TravelEvent and NodeArrival.
    edge_origin: RouteNode | None = None
    edge_destination: RouteNode | None = None
    edge_distance: float | None = None
    edge_started_at: float | None = None
    edge_arrives_at: float | None = None
    pick_started_at: float | None = None
    pick_ends_at: float | None = None
    replan_requested: bool = False
    intervention_event_pending: bool = False

    def current_node(self) -> RouteNode:
        return self.annotated_route[self.cursor]

    def at_end(self) -> bool:
        """True if cursor is on the final node (typically the depot)."""
        return self.cursor >= len(self.annotated_route) - 1

    def next_node(self) -> RouteNode:
        return self.annotated_route[self.cursor + 1]

    @property
    def is_travelling(self) -> bool:
        return self.edge_arrives_at is not None

    @property
    def is_picking(self) -> bool:
        return self.pick_ends_at is not None

    @property
    def is_replannable(self) -> bool:
        return self.status in {
            TourStates.PLANNED,
            TourStates.ASSIGNED,
            TourStates.SCHEDULED,
        }

    @property
    def is_terminal(self) -> bool:
        return self.status in {TourStates.DONE, TourStates.CANCELLED}
