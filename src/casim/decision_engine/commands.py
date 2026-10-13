"""Typed changes that a warehouse decision may make to the live plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ware_ops_algos.algorithms.algorithm_interfaces import (
        BatchObject, Route, ScheduledJob,
    )


class ViolationCode(str, Enum):
    STALE = "stale"
    SCOPE = "scope"
    PHYSICS = "physics"
    ORDER_NOT_FREE = "order_not_free"


class CommandViolation(ValueError):
    def __init__(self, code: ViolationCode, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, kw_only=True)
class Command:
    expected_decision_version: int


@dataclass(frozen=True, kw_only=True)
class CreatePickList(Command):
    batch: BatchObject


@dataclass(frozen=True, kw_only=True)
class CreateTour(Command):
    job: ScheduledJob
    positioning_mode: str | None = None


@dataclass(frozen=True, kw_only=True)
class CancelTour(Command):
    tour_id: int
    expected_route_version: int


@dataclass(frozen=True, kw_only=True)
class InsertOrders(Command):
    tour_id: int
    order_ids: tuple[int, ...]
    expected_route_version: int
    expected_cursor: int


@dataclass(frozen=True, kw_only=True)
class ReplaceSuffix(Command):
    tour_id: int
    route: Route
    expected_route_version: int
    expected_cursor: int


@dataclass(frozen=True, kw_only=True)
class Reschedule(Command):
    tour_id: int
    start_time: float
    end_time: float
    expected_route_version: int


@dataclass(frozen=True, kw_only=True)
class Reassign(Command):
    tour_id: int
    picker_id: int
    expected_route_version: int


@dataclass(frozen=True, kw_only=True)
class ReleaseTour(Command):
    tour_id: int | None = None
    expected_route_version: int = 0
    creation_index: int | None = None


@dataclass(frozen=True, kw_only=True)
class Wait(Command):
    reconsider_at: float | None
    reason: str = "wait"


@dataclass(frozen=True, kw_only=True)
class RecordConsidered(Command):
    pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True, kw_only=True)
class RecallPositioning(Command):
    tour_id: int
    expected_route_version: int
    expected_cursor: int
