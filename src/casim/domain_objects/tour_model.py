from collections import deque
from dataclasses import field, dataclass
from enum import Enum
from typing import Deque, Optional

from ware_ops_algos.algorithms import RouteNode, Route, PickList

Node = tuple[float, float]


class TourStates(str, Enum):
    PLANNED = "planned"  # PickList is generated
    ASSIGNED = "assigned"  # Is assigned to a picker
    SCHEDULED = "scheduled"  # Is scheduled for a point in time
    PENDING = "pending"
    STARTED = "started"  # Tour has started picking
    DONE = "done"  # Tour is done


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
    pick_list: PickList
    # pick_nodes: list[Node]
    annotated_route: list[RouteNode]

    assigned_resource: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    end_time_planned: Optional[float] = None
    # execution state, mutable during picking
    cursor: int = 0                 # index into route_nodes
    picks_left: Deque[Node] = field(default_factory=deque)
    open_pick_positions: list = field(default_factory=list)
    status: str = TourStates.PLANNED

    def current_node(self) -> RouteNode:
        return self.annotated_route[self.cursor]

    def at_end(self) -> bool:
        """True if cursor is on the final node (typically the depot)."""
        return self.cursor >= len(self.annotated_route) - 1

    def next_node(self) -> RouteNode:
        return self.annotated_route[self.cursor + 1]