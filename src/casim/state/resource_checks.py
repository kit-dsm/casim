"""The resource checks of Section 7.2: dispatch, presence and projection.

They read live state, after the last event of a timestamp, and return
``(code, picker_id, message)`` findings; the audit counts them like the plan
invariants. A transition table for ``ResourceStates`` would add little: the
states are derived and almost every change between snapshots is legitimate,
so the checks constrain the coupling of resources and tours instead.
"""

from __future__ import annotations

from math import isclose
from typing import TYPE_CHECKING

from ware_ops_algos.planning.lifecycle import TourStates

if TYPE_CHECKING:
    from casim.state.state import State

TOLERANCE = 1e-6


def check_resources(state: State) -> list[tuple[str, int, str]]:
    """Findings at the end of the current timestamp."""
    resources, tours = state.resource_manager, state.tour_manager
    now = state.current_time
    findings = []
    running: dict[int, list[int]] = {}
    for tour in tours.get_all_tours():
        if tour.status in (TourStates.PENDING, TourStates.STARTED):
            running.setdefault(tour.assigned_resource, []).append(tour.tour_id)
        if (tour.dispatched_at == now and tour.earliest_start is not None
                and tour.dispatched_at < tour.earliest_start - TOLERANCE):
            findings.append(("early_dispatch", tour.assigned_resource,
                             f"Tour {tour.tour_id} became pending at {now}, "
                             f"before its earliest start {tour.earliest_start}"))
    for resource in resources.get_resources().resources:
        picker_id = resource.id
        if len(running.get(picker_id, ())) > 1:
            findings.append(("concurrent_tours", picker_id,
                             f"Picker {picker_id} has tours {running[picker_id]} at once"))
        current = tours.current_tour(picker_id)
        if (current is not None and tours.get_tour(current).dispatched_at == now
                and not resources.is_present(picker_id)):
            findings.append(("absent_dispatch", picker_id,
                             f"Tour {current} became pending while picker {picker_id} is absent"))
        queue = tours.queue(picker_id)
        if (queue and current is None and resources.is_present(picker_id)
                and not resources.has_pending_departure(picker_id)):
            first = tours.get_tour(queue[0])
            # Reached exactly, as Dispatch decides it; a start a few ulps later is queued.
            if first.earliest_start is None or first.earliest_start <= now:
                findings.append(("missed_dispatch", picker_id,
                                 f"Picker {picker_id} is free but tour {first.tour_id} waits"))
        free_at = resources.free_at(picker_id, now, tours)
        if free_at is None:
            continue
        for tour_id in queue:
            tour = tours.get_tour(tour_id)
            start = free_at if tour.earliest_start is None else max(free_at, tour.earliest_start)
            if not isclose(tour.planned_start, start, abs_tol=TOLERANCE):
                findings.append(("projection", picker_id,
                                 f"Tour {tour_id} is planned at {tour.planned_start}, "
                                 f"its projection is {start}"))
                break
            free_at = tour.planned_end
    return findings


def check_time_balance(state: State) -> list[tuple[str, int, str]]:
    """Per resource, idle, busy and absent time sum to the simulated horizon."""
    tracker = state.tracker
    findings = []
    horizon = tracker.horizon
    for resource in state.resource_manager.get_resources().resources:
        picker_id = resource.id
        total = (tracker.idle_time_by_picker.get(picker_id, 0.0)
                 + tracker.busy_time_by_picker.get(picker_id, 0.0)
                 + tracker.absent_time_by_picker.get(picker_id, 0.0))
        if not isclose(total, horizon, abs_tol=TOLERANCE):
            findings.append(("time_balance", picker_id,
                             f"Picker {picker_id}: idle, busy and absent time sum to "
                             f"{total}, the horizon is {horizon}"))
    return findings
