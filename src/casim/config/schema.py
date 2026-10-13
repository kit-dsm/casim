"""The user-facing lifecycle selection and revision-scope vocabulary."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ware_ops_algos.domain_models.lifecycle import (
    Accepts, OrderStates, PickListStates, ResourceStates, TimeAttribute,
    TourStates, PHYSICAL, accepts_for,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WindowConfig(StrictModel):
    by: TimeAttribute
    horizon_s: float = Field(gt=0)

    @model_validator(mode="after")
    def supported_attribute(self) -> "WindowConfig":
        if self.by not in (TimeAttribute.DUE_DATE, TimeAttribute.START_TIME):
            raise ValueError("window.by must be due_date or start_time")
        return self


class SelectionConfig(StrictModel):
    orders: tuple[OrderStates, ...] = ()
    pick_lists: tuple[PickListStates, ...] = ()
    tours: tuple[TourStates, ...] = ()
    resources: tuple[ResourceStates, ...] = ()
    window: WindowConfig | None = None

    @model_validator(mode="after")
    def one_form_per_object(self) -> "SelectionConfig":
        if set(self.orders) - {OrderStates.BUFFERED}:
            raise ValueError("selection.orders selects buffered orders only")
        if set(self.pick_lists) - {PickListStates.BUFFERED}:
            raise ValueError("selection.pick_lists selects buffered pick lists only")
        if TourStates.CANCELLED in self.tours:
            raise ValueError("cancelled tours cannot be selected")
        if len(set(self.tours)) != len(self.tours):
            raise ValueError("selection.tours contains a duplicate status")
        return self


class RevisionScope(StrictModel):
    freeze_from: TourStates = TourStates.PLANNED
    frozen_accepts: Accepts = Accepts.NONE

    @model_validator(mode="after")
    def valid_cut(self) -> "RevisionScope":
        if self.freeze_from not in (
            TourStates.PLANNED, TourStates.ASSIGNED, TourStates.SCHEDULED,
            TourStates.RELEASED, TourStates.PENDING,
        ):
            raise ValueError("scope.freeze_from must be planned through pending")
        if self.frozen_accepts == Accepts.REBUILD:
            raise ValueError("scope.frozen_accepts cannot be rebuild")
        return self


class PeriodicTrigger(StrictModel):
    every_s: float = Field(gt=0)


class DecisionConfig(StrictModel):
    """One named decision owns projection, solve, and commitment settings."""

    solver: dict | None = None
    commitment: "CommitmentConfig" = Field(default_factory=lambda: CommitmentConfig())
    waiting: "WaitingConfig" = Field(default_factory=lambda: WaitingConfig())
    release_gates: "ReleaseGates" = Field(default_factory=lambda: ReleaseGates())
    selection: SelectionConfig
    scope: RevisionScope = Field(default_factory=RevisionScope)
    triggers: tuple[str | PeriodicTrigger, ...] = ()
    conditions: tuple[dict, ...] = ()
    precedence: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def scope_within_physics(self) -> "DecisionConfig":
        if self.solver is None and (self.selection.orders or self.selection.pick_lists
                                    or TourStates.SCHEDULED not in self.selection.tours):
            raise ValueError("A decision without a solver selects scheduled tours only for release")
        for status in self.selection.tours:
            allowed = accepts_for(status, self.scope.freeze_from,
                                  self.scope.frozen_accepts)
            if allowed not in PHYSICAL[status]:
                raise ValueError(
                    f"scope gives {status.value} tours {allowed.value}, "
                    "which the physical matrix disallows"
                )
        return self


class CommitmentConfig(StrictModel):
    rule: Literal["release_all", "release_next_per_picker", "release_within"] = "release_all"
    within_s: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def check_window(self):
        if (self.rule == "release_within") != (self.within_s is not None):
            raise ValueError("within_s is used exactly with release_within")
        return self


class WaitingConfig(StrictModel):
    policy: Literal["no_waiting", "order_count",
                    "henn", "analytic_stochastic"] = "no_waiting"
    k: int | None = Field(default=None, ge=1)
    target_batch_size_orders: int = Field(default=4, ge=2)
    min_orders: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def check_count(self):
        if (self.policy == "order_count") != (self.k is not None):
            raise ValueError("k is used exactly with order_count")
        return self


class ReleaseGates(StrictModel):
    outside_break: bool = False
    max_staged_pallets: int | None = Field(default=None, ge=0)


class DecisionDefinitions(StrictModel):
    """The decisions section of a composed experiment configuration."""

    on_violation: Literal["reject", "drop", "repair"] = "reject"
    decisions: dict[str, DecisionConfig]
