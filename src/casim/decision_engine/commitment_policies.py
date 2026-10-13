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
    visible_pickers = {resource.id for resource in plan.resources}
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


class CommitAllPolicy(CommitmentPolicy):
    def apply(self, commands, state_snapshot):
        if state_snapshot.dynamic_warehouse_info.memory.waiting:
            return (), None
        return tuple(tour.commit() for tour in
                     _planned_after(commands, state_snapshot)), None


class SchedulingCommitmentPolicy(CommitmentPolicy):
    def __init__(self, n_jobs: int = 1):
        if n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
        self.n_jobs = n_jobs

    def apply(self, commands, state_snapshot):
        if state_snapshot.dynamic_warehouse_info.memory.waiting:
            return (), None
        return tuple(tour.commit() for tour in
                     _planned_after(commands, state_snapshot)[:self.n_jobs]), None


class WindowCommitmentPolicy(CommitmentPolicy):
    def __init__(self, within_s: float):
        if not isfinite(within_s) or within_s < 0:
            raise ValueError("within_s must be finite and nonnegative")
        self.within_s = within_s

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
