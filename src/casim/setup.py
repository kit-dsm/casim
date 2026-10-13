"""Resolve named decisions once for both the solver and simulation engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from casim.config.schema import (
    CommitmentConfig, DecisionConfig, DecisionDefinitions, PeriodicTrigger,
    WaitingConfig,
)
from ware_ops_algos.domain_models.lifecycle import Accepts, accepts_for
from ware_ops_algos.domain_algo_mapper.domain_algo_mapper import DomainAlgorithmMapper
from ware_ops_algos.taxonomy.taxonomy import TAXONOMY
from casim.pipelines.pipeline_runner import CoSySolver
from casim.simulation_engine.conditions import Condition
from casim.simulation_engine.state_adapter import StateAdapter


@dataclass(frozen=True)
class DecisionSetup:
    name: str
    problem_class: str
    config: DecisionConfig
    solver: CoSySolver | None
    commitment: CommitmentConfig
    waiting: WaitingConfig
    adapter: StateAdapter
    conditions: tuple[Condition, ...]
    trigger_types: tuple[type, ...]
    pipeline_count: int
    decision_features: dict
    exclusions: tuple[dict, ...]
    write_set: tuple[str, ...]
    period_s: float | None = None


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
    writes = set()
    if config.selection.orders:
        writes.add("orders[buffered]")
    if config.selection.pick_lists:
        writes.add("pick_lists[buffered]")
    writes.update(
        f"tours[{status.value}]" for status in config.selection.tours
        if accepts_for(status, config.scope.freeze_from,
                       config.scope.frozen_accepts) != Accepts.NONE
    )
    return writes


def _derive_problem_class(config: DecisionConfig, raw) -> str:
    """Infer the CoSy family from selected work and the declared output stage."""
    if config.solver is None:
        return "release_only"
    components = tuple(path.rsplit(".", 1)[-1]
                       for path in raw.solver.repo.components)
    output = raw.solver.get("output")
    if output not in {"batching", "routing", "sequencing", "scheduling", "positioning"}:
        raise ValueError("solver.output must be batching, routing, sequencing, scheduling, or positioning")
    if output == "batching":
        return "OSBP" if "OrderSplitter" in components else "OBP"
    if output == "routing":
        if any(status.value == "started" for status in config.selection.tours):
            return "ATIP"
        return "OBRP" if config.selection.orders else "ORP"
    if output == "sequencing":
        return "OBRSP"
    if output in {"scheduling", "positioning"}:
        if config.selection.pick_lists:
            return "RORSP" if any(
                accepts_for(status, config.scope.freeze_from,
                            config.scope.frozen_accepts) == Accepts.REBUILD
                for status in config.selection.tours
            ) else "ORSP"
        return "OBRSP"
    raise ValueError(f"Unsupported decision output {output}")


def resolve(cfg: DictConfig, data_card) -> dict[str, DecisionSetup]:
    """Validate and build every named decision exactly once."""
    if "decisions" not in cfg:
        raise ValueError("Configuration has no decisions group")
    definitions = DecisionDefinitions.model_validate({
        "on_violation": cfg.get("on_violation", "reject"),
        "decisions": OmegaConf.to_container(cfg.decisions, resolve=True),
    })
    setups: dict[str, DecisionSetup] = {}
    trigger_owner: dict[type, str] = {}
    for name, raw in cfg.decisions.items():
        model = definitions.decisions[name]
        problem_class = _derive_problem_class(model, raw)
        for condition in model.conditions:
            target = condition.get("_target_", "")
            if target.rsplit(".", 1)[-1] in {"NbrOrdersCondition", "NbrBatchesCondition"}:
                if condition.get("threshold", 1) > 1:
                    raise ValueError(
                        f"{name}: {target.rsplit('.', 1)[-1]} with threshold > 1 "
                        "is a waiting rule; configure decision.waiting instead"
                    )
        solver = (instantiate(
            raw.solver, problem_class=problem_class,
            instances_dir=Path(cfg.instances_base),
            cache_dir=Path(cfg.cache_base) / cfg.data_card.name,
            output_dir=cfg.experiment.output_dir,
            instance_name=cfg.experiment.instance_name,
            verbose=True, luigi_cfg=cfg.luigi,
        ) if model.solver is not None else None)
        decision_card = deepcopy(data_card)
        decision_card.problem_class = problem_class
        decision_card.decision_features = {
            "type": "decision",
            "features": {
                "incumbent": any(
                    accepts_for(status, model.scope.freeze_from,
                                model.scope.frozen_accepts) == Accepts.REBUILD
                    for status in model.selection.tours
                ),
                "membership_ops": ["append"] if model.selection.orders else [],
                "routing_origin": ("live" if any(status.value == "started"
                                                  for status in model.selection.tours)
                                   else "depot"),
            },
        }
        mapper = DomainAlgorithmMapper(TAXONOMY)
        configured_components = ({path.rsplit(".", 1)[-1]
                                  for path in raw.solver.repo.components}
                                 if solver is not None else set())
        exclusions = tuple(
            {"algorithm": card.algo_name, "reasons": reasons}
            for card in (solver.algorithm_cards if solver is not None else ())
            if card.implementation.get("component_name", card.algo_name)
            in configured_components
            if (reasons := mapper.rejection_reasons(card, decision_card))
        )
        if solver is not None and problem_class == "OBRSP":
            policy_names = {
                "no_waiting": "NoWaiting",
                "henn": "HennWaiting",
                "analytic_stochastic": "AnalyticStochasticWaiting",
            }
            if model.waiting.policy == "order_count":
                labels = ("One", "Two", "Three", "Four")
                if model.waiting.k > len(labels):
                    raise ValueError("Order-count waiting supports k=1 through 4")
                policy_name = f"WaitFor{labels[model.waiting.k - 1]}"
            else:
                policy_name = policy_names[model.waiting.policy]
            card = next((card for card in solver.algorithm_cards
                         if card.algo_name == policy_name), None)
            if card is None:
                raise ValueError(f"No algorithm card for waiting policy {policy_name}")
            reasons = [reason for reason in mapper.rejection_reasons(card, decision_card)
                       if not reason.startswith("problem type ")]
            if reasons:
                raise ValueError(f"Waiting policy {policy_name} is incompatible: "
                                 + "; ".join(reasons))
        if solver is not None:
            solver.build_pipelines(decision_card)
        trigger_types = tuple(_event_type(event) for event in model.triggers
                              if isinstance(event, str))
        periods = [event.every_s for event in model.triggers
                   if isinstance(event, PeriodicTrigger)]
        if len(periods) > 1:
            raise ValueError(f"{name}: only one every_s trigger is allowed")
        for event_type in trigger_types:
            if event_type in trigger_owner:
                raise ValueError(
                    f"{event_type.__name__} triggers both {trigger_owner[event_type]} and {name}"
                )
            trigger_owner[event_type] = name
        setups[name] = DecisionSetup(
            name=name, problem_class=problem_class,
            config=model, solver=solver,
            commitment=model.commitment,
            waiting=model.waiting,
            adapter=StateAdapter(selection=model.selection, scope=model.scope),
            conditions=tuple(instantiate(condition) for condition in raw.get("conditions", ())),
            trigger_types=trigger_types,
            pipeline_count=len(solver.pipelines) if solver is not None else 0,
            decision_features=decision_card.decision_features["features"],
            exclusions=exclusions,
            write_set=tuple(sorted(_write_set(model))),
            period_s=periods[0] if periods else None,
        )
    for left_index, (left_name, left) in enumerate(setups.items()):
        for right_name, right in list(setups.items())[left_index + 1:]:
            shared = _write_set(left.config) & _write_set(right.config)
            if shared and left.config.precedence == right.config.precedence:
                raise ValueError(
                    f"{left_name} and {right_name} both write {sorted(shared)}; "
                    "give them different precedence values"
                )
    return setups
