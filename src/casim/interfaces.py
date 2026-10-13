"""Small contracts between simulation, projection and solving."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ware_ops_algos.algorithms import AlgorithmSolution
from ware_ops_algos.domain_models import SimWarehouseDomain

if TYPE_CHECKING:
    from casim.config.schema import RevisionScope, SelectionConfig
    from casim.state import State


class DecisionSolver(Protocol):
    def solve(self, snapshot: SimWarehouseDomain, action=None, evaluate=None
              ) -> tuple[AlgorithmSolution, str, float, tuple | None]: ...


class StateProjector(Protocol):
    selection: SelectionConfig
    scope: RevisionScope

    def transform_state(self, state: State, decision: str,
                        problem_class: str | None = None
                        ) -> SimWarehouseDomain: ...
