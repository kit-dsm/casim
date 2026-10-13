"""Temporary simulator-only positioning edits until Task 4.4."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch

from ware_ops_algos.algorithms import PositioningSolution
from ware_ops_algos.planning.commands import CreateTour
from ware_ops_algos.planning.lifecycle import TourStates


@dataclass(frozen=True)
class RecallPositioning:
    tour_id: int


@dataclass(frozen=True)
class MarkPositioning:
    creation_index: int
    mode: str


def closure_commands(snapshot) -> tuple[RecallPositioning, ...]:
    dynamic = snapshot.dynamic_warehouse_info
    if not dynamic.context.input_closed or dynamic.plan.order_buffer:
        return ()
    return tuple(RecallPositioning(tour.tour_id) for tour in dynamic.plan.tours
                 if tour.status is TourStates.STARTED
                 and tour.positioning_mode == "center_relocation"
                 and not tour.batch.order_numbers)


@singledispatch
def positioning_commands(solution, commands) -> tuple[MarkPositioning, ...]:
    return ()


@positioning_commands.register
def _(solution: PositioningSolution, commands) -> tuple[MarkPositioning, ...]:
    created = [command for command in commands if isinstance(command, CreateTour)]
    return tuple(MarkPositioning(index, solution.modes[command.job.job.job_id])
                 for index, command in enumerate(created)
                 if command.job.job.job_id in solution.modes)
