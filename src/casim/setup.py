"""Resolve named decisions for the simulation and decision engines."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from ware_ops_algos.domain_algo_mapper.domain_algo_mapper import DomainAlgorithmMapper
from ware_ops_algos.planning.decision_features import decision_features
from ware_ops_algos.planning.lifecycle import Accepts

from casim.config.schema import DecisionConfig, DecisionDefinitions, PeriodicTrigger
from casim.decision_engine.commitment_policies import CommitmentPolicy
from casim.decision_engine.solution_ranker import SolutionRanker
from casim.pipelines.pipeline_runner import CoSySolver
from casim.pipelines.taxonomy import TAXONOMY, variant
from casim.simulation_engine.conditions import Condition
from casim.simulation_engine.state_adapter import StateAdapter


@dataclass(frozen=True)
class DecisionSetup:
    name: str
    config: DecisionConfig
    solver: CoSySolver | None
    ranker: SolutionRanker | None
    commitment_policy: CommitmentPolicy | None
    adapter: StateAdapter
    conditions: tuple[Condition, ...]
    trigger_types: tuple[type, ...]
    pipeline_count: int
    decision_features: dict
    subproblems: tuple[str, ...]
    exclusions: tuple[dict, ...]
    write_set: tuple[str, ...]
    period_s: float | None = None

    @property
    def problem_class(self) -> str:
        return self.config.problem_class


def _event_type(name: str) -> type:
    if "." in name:
        return get_class(name)
    for module in ("casim.events.operational_events", "casim.events.decision_events"):
        try:
            return get_class(f"{module}.{name}")
        except (ImportError, AttributeError, ValueError):
            pass
    raise ValueError(f"Unknown trigger event {name!r}")


def _write_set(config: DecisionConfig) -> set[str]:
    info = config.state_information
    writes = set()
    if info.orders:
        writes.add("orders")
    if info.pick_lists:
        writes.add("pick_lists")
    writes.update(f"tours[{status.value}]" for status in info.tours
                  if config.scope.accepts(status) is not Accepts.NONE)
    return writes


def resolve(cfg: DictConfig, data_card) -> dict[str, DecisionSetup]:
    if "decisions" not in cfg:
        raise ValueError("Configuration has no decisions group")
    definitions = DecisionDefinitions.model_validate({
        "decisions": OmegaConf.to_container(cfg.decisions, resolve=True),
    })
    setups = {}
    mapper = DomainAlgorithmMapper(TAXONOMY)
    for name, raw in cfg.decisions.items():
        model = definitions.decisions[name]
        features = decision_features(model.state_information, model.scope)
        stages = variant(model.problem_class, features)
        if (model.solver is not None and not (
                model.state_information.orders or model.state_information.pick_lists
                or any(model.scope.accepts(status) is not Accepts.NONE
                       for status in model.state_information.tours))):
            raise ValueError(f"{name}: solver has no exposed work to solve")
        solver = (instantiate(
            raw.solver, problem_class=model.problem_class,
            instances_dir=Path(cfg.instances_base),
            cache_dir=Path(cfg.cache_base) / cfg.data_card.name,
            output_dir=cfg.experiment.output_dir,
            instance_name=cfg.experiment.instance_name,
            verbose=True, luigi_cfg=cfg.luigi,
        ) if model.solver is not None else None)
        card = deepcopy(data_card)
        card.problem_class = model.problem_class
        card.decision_features = {"type": "decision", "features": features}
        configured = ({path.rsplit(".", 1)[-1]
                       for path in raw.solver.repo.components}
                      if solver is not None else set())
        exclusions = []
        for algorithm in solver.algorithm_cards if solver is not None else ():
            component = algorithm.implementation.get("component_name", algorithm.algo_name)
            if component not in configured:
                continue
            reasons = list(mapper.rejection_reasons(algorithm, card))
            if algorithm.problem_type not in stages:
                reasons.append(f"problem type {algorithm.problem_type!r} not in variant {stages}")
            if reasons:
                exclusions.append({"algorithm": algorithm.algo_name,
                                   "reasons": tuple(dict.fromkeys(reasons))})
        if solver is not None:
            try:
                solver.build_pipelines(card)
            except ValueError as error:
                if exclusions:
                    details = "; ".join(
                        f"{entry['algorithm']}: {', '.join(entry['reasons'])}"
                        for entry in exclusions)
                    raise ValueError(f"{name}: configured cards excluded: {details}") from error
                raise
            if solver.output == "routing" and features["routing_origin"] != "live":
                raise ValueError(f"{name}: routing-only decisions create no executable tours")
            if solver.output == "batching" and len(solver.pipelines) > 1:
                raise ValueError(f"{name}: multiple batching candidates have no objective")
        trigger_types = tuple(_event_type(event) for event in model.triggers
                              if isinstance(event, str))
        periods = [event.every_s for event in model.triggers
                   if isinstance(event, PeriodicTrigger)]
        if len(periods) > 1:
            raise ValueError(f"{name}: only one every_s trigger is allowed")
        setups[name] = DecisionSetup(
            name=name, config=model, solver=solver,
            ranker=(SolutionRanker(data_card.objective, model.change_penalty)
                    if solver is not None else None),
            commitment_policy=model.commitment_policy,
            adapter=StateAdapter(model.state_information, model.scope),
            conditions=tuple(instantiate(condition) for condition in raw.get("conditions", ())),
            trigger_types=trigger_types,
            pipeline_count=len(solver.pipelines) if solver is not None else 0,
            decision_features=features, subproblems=stages,
            exclusions=tuple(exclusions), write_set=tuple(sorted(_write_set(model))),
            period_s=periods[0] if periods else None,
        )
    names = list(setups)
    for index, left_name in enumerate(names):
        left = setups[left_name]
        for right_name in names[index + 1:]:
            right = setups[right_name]
            shared = set(left.write_set) & set(right.write_set)
            if shared and left.config.precedence == right.config.precedence:
                raise ValueError(f"{left_name} and {right_name} both write {sorted(shared)}; "
                                 "give them distinct precedence values")
    return setups
