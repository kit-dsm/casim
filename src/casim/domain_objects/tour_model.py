from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ware_ops_algos.planning.lifecycle import TourStates

if TYPE_CHECKING:
    from ware_ops_algos.algorithms.algorithm_interfaces import (
        BatchObject, PickPosition, Route, RouteNode, RoutingOrigin,
    )


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
    route: Route
    batch: BatchObject
    annotated_route: list[RouteNode]
    executed_prefix: list[RouteNode] = field(default_factory=list)

    assigned_resource: Optional[int] = None
    processing_time: Optional[float] = None
    # Times: the binding earliest start (optional); the planned start and end,
    # which the simulator projects from the resource's queue and which later
    # decisions read; the actual start (of the route, after the setup) and end.
    earliest_start: Optional[float] = None
    planned_start: Optional[float] = None
    planned_end: Optional[float] = None
    actual_start: Optional[float] = None
    dispatched_at: Optional[float] = None   # when the tour became pending (setup began)
    actual_end: Optional[float] = None
    # execution state, mutable during picking
    cursor: int = 0                 # index into route_nodes
    status: str = TourStates.PLANNED
    route_version: int = 0
    remaining_picks: list[PickPosition] = field(default_factory=list)
    completed_picks: list[PickPosition] = field(default_factory=list)
    # The leg in progress: the edges one TravelEvent walks, from the cursor
    # (one edge, or with travel: pick every edge up to the next pick node or
    # the route's end). edge_* describe the whole leg; leg_ends holds the
    # arrival time at each of its nodes and leg_distances each edge's length.
    edge_origin: RouteNode | None = None
    edge_destination: RouteNode | None = None
    edge_start_time: float | None = None
    edge_end_time: float | None = None
    edge_distance: float | None = None
    leg_ends: list[float] = field(default_factory=list)
    leg_distances: list[float] = field(default_factory=list)
    picking_until: float | None = None
    cart_bins: dict[int, int] = field(default_factory=dict)
    routing_origin: RoutingOrigin | None = None

    def edge_at(self, time: float) -> tuple | None:
        """The edge of the leg walked at ``time``, as edge-level travel has it.

        Returns (offset from the cursor, origin, destination, start, end,
        distance), or None outside a leg. At an inner node's arrival time the
        next edge has begun, as when edge travel handles the arrival and the
        next travel at that time before any decision.
        """
        if self.edge_origin is None:
            return None
        start, last = self.edge_start_time, len(self.leg_ends) - 1
        for offset, (end, distance) in enumerate(zip(self.leg_ends, self.leg_distances)):
            if time < end or offset == last:
                node = self.cursor + offset
                return (offset, self.annotated_route[node], self.annotated_route[node + 1],
                        start, end, distance)
            start = end

    def position_at(self, time: float) -> tuple[float, float]:
        edge = self.edge_at(time)
        if edge is None:
            return self.current_node().position
        _, origin, destination, start_time, end_time, _ = edge
        duration = end_time - start_time
        fraction = 1.0 if duration == 0 else min(1.0, max(0.0, (time - start_time) / duration))
        start, end = origin.position, destination.position
        return (start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]))

    def current_node(self) -> RouteNode:
        return self.annotated_route[self.cursor]

    def at_end(self) -> bool:
        """True if cursor is on the final node (typically the depot)."""
        return self.cursor >= len(self.annotated_route) - 1

    def next_node(self) -> RouteNode:
        return self.annotated_route[self.cursor + 1]

    def pinned_prefix(self, edge_in_progress: bool,
                      offset: int = 0) -> tuple[RouteNode, ...]:
        """Return all walked nodes across segment-relative route replacements.

        ``offset`` counts the edges of the leg already walked (``edge_at``).
        """
        segment = self.annotated_route[:self.cursor + offset + int(edge_in_progress)]
        if (self.executed_prefix and segment
                and self.executed_prefix[-1] == segment[0]):
            segment = segment[1:]
        return (*self.executed_prefix, *segment)


Node = tuple[float, float]
