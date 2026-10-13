"""Convert algorithm solutions to commands against the snapshot they saw."""

from __future__ import annotations

from ware_ops_algos.algorithms.algorithm_interfaces import (
    ActiveTourRoutingSolution, BatchingSolution, CombinedRoutingSolution,
    SchedulingSolution, PositioningSolution, WaitingSolution,
)
from ware_ops_algos.domain_models.lifecycle import Accepts

from .commands import (
    CancelTour, Command, CreatePickList, CreateTour, InsertOrders,
    Reassign, RecordConsidered, ReleaseTour, ReplaceSuffix, Reschedule, Wait,
)


def _same_route(view, route) -> bool:
    return (view.route is not None
            and tuple(view.route.annotated_route) == tuple(route.annotated_route)
            and view.route.distance == route.distance
            and tuple(view.route.batch.order_numbers) == tuple(route.batch.order_numbers))


def _plan_commands(jobs, snapshot, released_jobs=(), modes=None) -> tuple[Command, ...]:
    """Match a complete candidate plan to selected drafts and emit its delta.

    Positive order overlap wins; ties prefer the smaller incumbent tour ID and
    then the earlier candidate. A candidate that drops incumbent orders is
    rebuilt because a draft suffix replacement may only extend membership.
    """
    version = snapshot.context.version
    modes = modes or {}
    drafts = [tour for tour in snapshot.plan.tours
              if tour.accepts == Accepts.REBUILD]
    jobs = tuple(jobs)
    release_keys = {(tuple(job.order_numbers), job.picker_id)
                    for job in released_jobs}
    pairs = sorted(
        ((-max(len(set(view.batch.order_numbers) & set(job.order_numbers)),
                int(not view.batch.order_numbers and not job.order_numbers
                    and view.picker_id == job.picker_id
                    and view.positioning_mode == modes.get(job.job.job_id))),
          view.tour_id, index, view)
         for view in drafts for index, job in enumerate(jobs)),
        key=lambda item: (item[0], item[1], item[2]),
    )
    matches = {}
    matched_tours = set()
    for negative_overlap, tour_id, index, view in pairs:
        if negative_overlap == 0 or tour_id in matched_tours or index in matches:
            continue
        if not set(view.batch.order_numbers).issubset(set(jobs[index].order_numbers)):
            continue
        matches[index] = view
        matched_tours.add(tour_id)
    commands: list[Command] = [CancelTour(
        expected_decision_version=version, tour_id=view.tour_id,
        expected_route_version=view.route_version,
    ) for view in drafts if view.tour_id not in matched_tours]
    created_indices = {}
    route_versions = {}
    for index, job in enumerate(jobs):
        view = matches.get(index)
        if view is None:
            created_indices[index] = len(created_indices)
            commands.append(CreateTour(expected_decision_version=version, job=job,
                                       positioning_mode=modes.get(job.job.job_id)))
            continue
        new_orders = tuple(sorted(set(job.order_numbers)
                                  - set(view.batch.order_numbers)))
        route_changed = not _same_route(view, job.job.route)
        route_versions[index] = view.route_version + int(route_changed)
        if route_changed:
            if new_orders:
                commands.append(InsertOrders(
                    expected_decision_version=version, tour_id=view.tour_id,
                    order_ids=new_orders,
                    expected_route_version=view.route_version,
                    expected_cursor=view.cursor,
                ))
            commands.append(ReplaceSuffix(
                expected_decision_version=version, tour_id=view.tour_id,
                route=job.job.route, expected_route_version=view.route_version,
                expected_cursor=view.cursor,
            ))
        if view.picker_id != job.picker_id:
            commands.append(Reassign(
                expected_decision_version=version, tour_id=view.tour_id,
                picker_id=job.picker_id,
                expected_route_version=route_versions[index],
            ))
        if view.start_time != job.start_time or view.end_time != job.end_time:
            commands.append(Reschedule(
                expected_decision_version=version, tour_id=view.tour_id,
                start_time=job.start_time, end_time=job.end_time,
                expected_route_version=route_versions[index],
            ))
    for index, job in enumerate(jobs):
        if (tuple(job.order_numbers), job.picker_id) not in release_keys:
            continue
        if index in matches:
            view = matches[index]
            commands.append(ReleaseTour(
                expected_decision_version=version, tour_id=view.tour_id,
                expected_route_version=route_versions[index],
            ))
        else:
            commands.append(ReleaseTour(
                expected_decision_version=version,
                creation_index=created_indices[index],
            ))
    return tuple(commands)


def to_commands(solution, snapshot, candidate_plan=None,
                released_jobs=None) -> tuple[Command, ...]:
    """Create a minimal typed intent from a typed solver result.

    Existing algorithms return their established solution types. Matching and
    command validation belong to CASIM, not to those algorithms.
    """
    version = snapshot.context.version if snapshot.context is not None else 0
    if isinstance(solution, WaitingSolution):
        candidates = tuple(candidate_plan.jobs if candidate_plan is not None
                           else solution.jobs)
        commands = list(_plan_commands(
            candidates, snapshot,
            released_jobs=solution.jobs if solution.action == "release" else (),
        ))
        if solution.action == "wait":
            commands.append(Wait(expected_decision_version=version,
                                 reconsider_at=solution.reconsider_at))
        return tuple(commands)

    if isinstance(solution, ActiveTourRoutingSolution):
        views = {tour.tour_id: tour for tour in snapshot.plan.tours}
        admitted: dict[int, list[int]] = {}
        for tour_id, order_id in solution.assignments:
            admitted.setdefault(tour_id, []).append(order_id)
        commands: list[Command] = []
        for route in solution.routes:
            tour_id = route.batch.tour_id
            view = views[tour_id]
            commands.append(InsertOrders(
                expected_decision_version=version, tour_id=tour_id,
                order_ids=tuple(admitted.get(tour_id, ())),
                expected_route_version=view.route_version,
                expected_cursor=view.cursor,
            ))
            commands.append(ReplaceSuffix(
                expected_decision_version=version, tour_id=tour_id,
                route=route, expected_route_version=view.route_version,
                expected_cursor=view.cursor,
            ))
        commands.append(RecordConsidered(
            expected_decision_version=version,
            pairs=tuple(solution.considered_pairs),
        ))
        return tuple(commands)

    if isinstance(solution, SchedulingSolution):
        return _plan_commands(solution.jobs, snapshot,
                              released_jobs=(solution.jobs if released_jobs is None
                                             else released_jobs),
                              modes=(solution.modes if isinstance(solution, PositioningSolution)
                                     else None))

    if isinstance(solution, BatchingSolution):
        return tuple(CreatePickList(expected_decision_version=version, batch=batch)
                     for batch in solution.batches)

    if isinstance(solution, CombinedRoutingSolution) or solution is None:
        return ()
    raise TypeError(f"No command conversion for {type(solution).__name__}")
