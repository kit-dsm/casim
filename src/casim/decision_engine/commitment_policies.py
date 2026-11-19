"""Select visible planned tours to make executable."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite

from ware_ops_algos.planning.commands import (
    CancelTour, Command, CommitTour, CreateTour, Reschedule,
)
from ware_ops_algos.planning.lifecycle import TourStates

from casim.domain_objects.sim_domain import SimWarehouseDomain


@dataclass(frozen=True)
class _Planned:
    tour_id: int | None
    creation_index: int | None
    picker_id: int
    start_time: float

    def commit(self) -> CommitTour:
        return CommitTour(tour_id=self.tour_id,
                          creation_index=self.creation_index)


def _planned_after(commands: tuple[Command, ...],
                   snapshot: SimWarehouseDomain) -> tuple[_Planned, ...]:
    plan = snapshot.dynamic_warehouse_info.plan
    visible_pickers = {resource.id for resource in snapshot.resources.resources}
    planned: dict[int, _Planned] = {
        tour.tour_id: _Planned(tour.tour_id, None, tour.picker_id,
                              tour.start_time)
        for tour in plan.tours
        if tour.status is TourStates.PLANNED and tour.picker_id is not None
        and tour.start_time is not None
    }
    created: list[_Planned] = []
    for command in commands:
        if isinstance(command, CancelTour):
            planned.pop(command.tour_id, None)
        elif isinstance(command, Reschedule) and command.tour_id in planned:
            planned[command.tour_id] = _Planned(
                command.tour_id, None, command.picker_id, command.start_time)
        elif isinstance(command, CreateTour):
            created.append(_Planned(None, len(created), command.job.picker_id,
                                    command.job.start_time))
    choices = (*planned.values(), *created)
    return tuple(sorted((tour for tour in choices
                         if tour.picker_id in visible_pickers),
                        key=lambda tour: (tour.start_time, tour.picker_id,
                                          tour.tour_id if tour.tour_id is not None
                                          else float("inf"),
                                          tour.creation_index or 0)))


class CommitmentPolicy(ABC):
    @abstractmethod
    def apply(self, commands: tuple[Command, ...],
              state_snapshot: SimWarehouseDomain
              ) -> tuple[tuple[CommitTour, ...], float | None]:
        """Return commitments and an optional time to reconsider."""


@dataclass(frozen=True)
class CommitAllPolicy(CommitmentPolicy):
    __pydantic_config__ = {"extra": "forbid"}

    def apply(self, commands, state_snapshot):
        if state_snapshot.dynamic_warehouse_info.memory.waiting:
            return (), None
        return tuple(tour.commit() for tour in
                     _planned_after(commands, state_snapshot)), None


@dataclass(frozen=True)
class PlanOnlyPolicy(CommitmentPolicy):
    """Commit nothing: the decision only plans, and a later decision commits.

    For example, a decision that batches, routes and schedules once leaves
    planned tours with their routes, and a pull decision commits one of them
    to each picker that becomes free.
    """

    __pydantic_config__ = {"extra": "forbid"}

    def apply(self, commands, state_snapshot):
        return (), None


@dataclass(frozen=True)
class SchedulingCommitmentPolicy(CommitmentPolicy):
    """Commit the first ``n_jobs`` planned tours by planned start.

    With ``per_resource`` the first ``n_jobs`` tours of each resource: every
    resource the decision plans for gets its next tours, also when several
    become idle at the same time (a decision runs once per timestamp, so a
    resource left without a tour then would wait for the next trigger).
    """

    n_jobs: int = 1
    per_resource: bool = False
    __pydantic_config__ = {"extra": "forbid"}

    def __post_init__(self):
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")

    def apply(self, commands, state_snapshot):
        if state_snapshot.dynamic_warehouse_info.memory.waiting:
            return (), None
        planned = _planned_after(commands, state_snapshot)
        if not self.per_resource:
            return tuple(tour.commit() for tour in planned[:self.n_jobs]), None
        taken: dict[int, int] = {}
        commits = []
        for tour in planned:
            if taken.get(tour.picker_id, 0) < self.n_jobs:
                taken[tour.picker_id] = taken.get(tour.picker_id, 0) + 1
                commits.append(tour.commit())
        return tuple(commits), None


@dataclass(frozen=True)
class WindowCommitmentPolicy(CommitmentPolicy):
    within_s: float
    __pydantic_config__ = {"extra": "forbid"}

    def __post_init__(self):
        if not isfinite(self.within_s) or self.within_s < 0:
            raise ValueError("within_s must be finite and nonnegative")

    def apply(self, commands, state_snapshot):
        if state_snapshot.dynamic_warehouse_info.memory.waiting:
            return (), None
        plan = state_snapshot.dynamic_warehouse_info.plan
        cutoff = plan.time + self.within_s
        planned = _planned_after(commands, state_snapshot)
        commits = tuple(tour.commit() for tour in planned
                        if tour.start_time <= cutoff)
        future = [tour.start_time - self.within_s for tour in planned
                  if tour.start_time > cutoff]
        return commits, min(future) if future else None
