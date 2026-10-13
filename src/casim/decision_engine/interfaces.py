"""Contracts between decision solving and state projection."""

from __future__ import annotations

from typing import Protocol

from ware_ops_algos.algorithms import AlgorithmSolution
from casim.domain_objects.sim_domain import SimWarehouseDomain

class DecisionSolver(Protocol):
    def solve(self, state_snapshot: SimWarehouseDomain) -> dict[str, AlgorithmSolution]: ...
