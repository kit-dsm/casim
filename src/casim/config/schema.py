"""Strict user configuration for named warehouse decisions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ware_ops_algos.planning.lifecycle import PHYSICAL, TourStates
from ware_ops_algos.planning.scope import RevisionScope, StateInformation
from ware_ops_algos.planning.validation import check_scope

from casim.decision_engine.commitment_policies import (
    CommitAllPolicy, SchedulingCommitmentPolicy, WindowCommitmentPolicy,
)
from casim.pipelines.taxonomy import TAXONOMY


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class PeriodicTrigger(StrictModel):
    every_s: float = Field(gt=0)


_POLICIES = {
    f"{policy.__module__}.{policy.__name__}": policy
    for policy in (CommitAllPolicy, SchedulingCommitmentPolicy, WindowCommitmentPolicy)
}
Policy = CommitAllPolicy | SchedulingCommitmentPolicy | WindowCommitmentPolicy


class DecisionConfig(StrictModel):
    problem_class: str
    state_information: StateInformation
    scope: RevisionScope
    commitment_policy: Policy | None = None
    change_penalty: float = Field(default=0.0, ge=0)
    precedence: int | None = Field(default=None, ge=0)
    solver: dict[str, Any] | None = None
    triggers: tuple[str | PeriodicTrigger, ...] = ()
    conditions: tuple[dict[str, Any], ...] = ()

    @field_validator("problem_class")
    @classmethod
    def known_problem_class(cls, value: str) -> str:
        if value not in TAXONOMY:
            raise ValueError(f"problem_class {value!r} must be one of {sorted(TAXONOMY)}")
        return value

    @field_validator("commitment_policy", mode="before")
    @classmethod
    def parse_policy(cls, value):
        if value is None or isinstance(value, tuple(_POLICIES.values())):
            return value
        if not isinstance(value, dict):
            raise ValueError("commitment_policy must have a _target_ and its parameters")
        target = value.get("_target_")
        if target not in _POLICIES:
            raise ValueError(f"commitment_policy._target_ {target!r} must be one of {sorted(_POLICIES)}")
        policy = _POLICIES[target]
        try:
            return policy(**{key: parameter for key, parameter in value.items()
                             if key != "_target_"})
        except TypeError as error:
            raise ValueError(f"Invalid commitment_policy parameters: {error}") from error

    @model_validator(mode="after")
    def valid_scope_and_solver(self):
        violations = check_scope(self.state_information, self.scope)
        if violations:
            status = violations[0]
            raise ValueError(
                f"state_information.tours {status.value}: scope gives "
                f"{self.scope.accepts(status).value}, PHYSICAL allows "
                f"{PHYSICAL[status].value}"
            )
        if self.solver is None and TourStates.PLANNED not in self.state_information.tours:
            raise ValueError("A solver-free decision must expose planned tours")
        if self.solver is not None:
            if "output" in self.solver or "solution_ranker" in self.solver:
                raise ValueError("solver.output and solver.solution_ranker are removed")
            live_only = (self.state_information.orders
                         and TourStates.STARTED in self.state_information.tours
                         and TourStates.PLANNED not in self.state_information.tours
                         and TourStates.SCHEDULED not in self.state_information.tours)
            if (self.problem_class in {"ORSP", "OBRSP"} and not live_only
                    and self.commitment_policy is None):
                raise ValueError("commitment_policy is required for a scheduling decision")
        if (TourStates.PLANNED in self.state_information.tours
                and self.commitment_policy is None):
            raise ValueError("commitment_policy is required when planned tours are exposed")
        return self


class DecisionDefinitions(StrictModel):
    decisions: dict[str, DecisionConfig]
