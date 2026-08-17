"""Tests for the formal CASIM decision-configuration model.

Each test targets a formal invariant (V1-V8) rather than an implementation
branch.  The tests verify that the configuration compiler rejects
semantically invalid configurations and accepts valid ones, and that the
compiled runtime is explainable from the semantic owners.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ware_ops_algos.domain_models import load_and_flatten_data_card

from casim.decision_card import (
    CONDITION_SPECS,
    DECISION_CARDS,
    DecisionCard,
    DecisionCompatibilityMapper,
    PROJECTION_SPECS,
)
from casim.events.operational_events import (
    InterventionRequest,
    PickerIdle,
    PlanningRun,
    WMSRun,
)
from casim.pipelines.solution_ranker import SolutionRanker
from casim.pipelines.taxonomy import solution_kind
from casim.setup import (
    SUPPORTED_DECISIONS,
    _compile_bindings,
    compile_engine,
)


ROOT = Path(__file__).parents[1]
SCENARIO = ROOT / "scenarios" / "scenario_intervention_stress"


# ───────────────────────── helpers ─────────────────────────


def _problem_cfg(**kwargs):
    return OmegaConf.create(kwargs)


def _mapper():
    return DecisionCompatibilityMapper()


def _validate(problem_class, replanning, problem_cfg, trigger_classes=None):
    return _mapper().validate(
        problem_class, replanning, problem_cfg,
        trigger_classes=trigger_classes or [],
    )


def _minimal_cfg(problems, tmp_path):
    return OmegaConf.create({
        "experiment": {
            "output_dir": str(tmp_path),
            "working_dir": str(tmp_path),
            "instance_name": "test",
        },
        "instances_base": str(tmp_path),
        "cache_base": str(tmp_path / "cache"),
        "data_card": {"name": "test"},
        "luigi": None,
        "engines": {"problems": problems},
    })


def _intervention_cfg(overrides, tmp_path):
    config_dir = str((SCENARIO / "config").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="intervention_stress_config", overrides=overrides)
    OmegaConf.update(cfg, "project_root", str(ROOT), merge=False)
    OmegaConf.update(cfg, "instances_base", str(ROOT / "scenarios"), merge=False)
    OmegaConf.update(cfg, "cache_base", str(tmp_path / "cache"), merge=False)
    OmegaConf.update(cfg, "experiment.output_dir", str(tmp_path), merge=False)
    OmegaConf.update(cfg, "experiment.working_dir", str(tmp_path / "work"), merge=False)
    return cfg


# ───────────────────────── V1: registered binding ─────────────────────


def test_every_registered_binding_compiles():
    for binding in sorted(SUPPORTED_DECISIONS):
        card = DECISION_CARDS[binding]
        assert card.exposure_dict(due_horizon_s=None, limit=None, congestion_penalty=0.0)


@pytest.mark.parametrize(
    "problem_class, replanning",
    [
        ("OBP", "unstarted"), ("OBP", "active"),
        ("ORSP", "active"), ("OBRSP", "unstarted"),
        ("OBRSP", "active"), ("OBRP", "unstarted"),
        ("ORP", "none"), ("ORP", "unstarted"),
        ("UNKNOWN", "none"),
    ],
)
def test_unregistered_binding_rejected(problem_class, replanning):
    with pytest.raises(ValueError, match="Unsupported decision binding"):
        _validate(problem_class, replanning, _problem_cfg())


# ───────────────────────── V2: closed syntax ──────────────────────────


def test_unknown_decision_field_rejected():
    cfg = _problem_cfg(requires={"orders": 1}, bogus_field=True)
    with pytest.raises(ValueError, match="unknown field"):
        _validate("OBP", "none", cfg)


def test_unknown_requires_key_rejected():
    cfg = _problem_cfg(requires={"orders": 1, "tours": 5})
    with pytest.raises(ValueError, match="unknown requires key 'tours'"):
        _validate("OBP", "none", cfg)


def test_unknown_commit_field_rejected():
    cfg = _problem_cfg(commit={"n_jobs": 1, "bogus": 5})
    with pytest.raises(ValueError, match="unknown field"):
        _validate("ORSP", "none", cfg)


def test_unknown_cosy_solver_field_rejected():
    cfg = _problem_cfg(solver={"type": "cosy", "objective": "distance", "bogus": 5})
    with pytest.raises(ValueError, match="unknown field"):
        _validate("OBRSP", "none", cfg)


# ───────────────────────── V3: parameter validation ───────────────────


def test_relevant_parameter_accepted():
    _validate("ORSP", "none", _problem_cfg(due_horizon_s=3600, limit=50))
    _validate("ORP", "active", _problem_cfg(congestion_penalty=1.5),
              trigger_classes=[InterventionRequest])
    _validate("OBP", "none", _problem_cfg(limit=50))


def test_irrelevant_parameter_rejected():
    with pytest.raises(ValueError, match="does not consume parameter 'due_horizon_s'"):
        _validate("OBP", "none", _problem_cfg(due_horizon_s=3600))
    with pytest.raises(ValueError, match="does not consume parameter 'congestion_penalty'"):
        _validate("ORSP", "none", _problem_cfg(congestion_penalty=1000.0))
    with pytest.raises(ValueError, match="does not consume parameter 'limit'"):
        _validate("ORP", "active", _problem_cfg(limit=50),
                  trigger_classes=[InterventionRequest])


def test_invalid_parameter_domain_rejected():
    with pytest.raises(ValueError, match="invalid value for 'limit'"):
        _validate("OBP", "none", _problem_cfg(limit=-5))
    with pytest.raises(ValueError, match="invalid value for 'limit'"):
        _validate("OBP", "none", _problem_cfg(limit=0))
    with pytest.raises(ValueError, match="invalid value for 'due_horizon_s'"):
        _validate("ORSP", "none", _problem_cfg(due_horizon_s=-1))
    with pytest.raises(ValueError, match="invalid value for 'congestion_penalty'"):
        _validate("ORP", "active", _problem_cfg(congestion_penalty=-1),
                  trigger_classes=[InterventionRequest])


def test_card_exposure_dict_is_deterministic():
    for binding in sorted(SUPPORTED_DECISIONS):
        for due, limit, cong in [(None, None, 0.0), (3600, 50, 1.5)]:
            first = DECISION_CARDS[binding].exposure_dict(
                due_horizon_s=due, limit=limit, congestion_penalty=cong)
            second = DECISION_CARDS[binding].exposure_dict(
                due_horizon_s=due, limit=limit, congestion_penalty=cong)
            assert first == second, binding


# ───────────────────────── V4: condition validation ──────────────────


def test_condition_on_unavailable_fact_rejected():
    with pytest.raises(ValueError, match="cannot require 'orders'"):
        _validate("ORSP", "none", _problem_cfg(requires={"orders": 1}))
    with pytest.raises(ValueError, match="cannot require 'batches'"):
        _validate("OBP", "none", _problem_cfg(requires={"batches": 1}))


def test_impossible_cardinality_guard_rejected():
    cfg = _problem_cfg(requires={"pickers": 2})
    with pytest.raises(ValueError, match="impossible"):
        _validate("ORP", "active", cfg, trigger_classes=[InterventionRequest])


def test_redundant_but_true_guard_accepted():
    cfg = _problem_cfg(requires={"pickers": 1})
    _validate("ORP", "active", cfg, trigger_classes=[InterventionRequest])


def test_dock_capacity_uses_leq_operator():
    spec = CONDITION_SPECS["dock_capacity"]
    assert spec.operator == "<="
    assert spec.render("dock_capacity", 5) == "dock_capacity <= 5"


def test_not_on_break_render():
    spec = CONDITION_SPECS["not_on_break"]
    assert spec.operator == "false"
    assert spec.render("not_on_break", True) == "not_on_break"


def test_not_on_break_accepted_for_active_and_ordinary():
    _validate("ORP", "active",
              _problem_cfg(requires={"not_on_break": True}),
              trigger_classes=[InterventionRequest])
    _validate("OBP", "none", _problem_cfg(requires={"not_on_break": True}))


def test_universal_conditions_accepted_for_every_binding():
    for binding in sorted(SUPPORTED_DECISIONS):
        card = DECISION_CARDS[binding]
        triggers = [InterventionRequest] if card.required_trigger_capabilities() else []
        _validate(*binding, _problem_cfg(requires={"not_on_break": True, "dock_capacity": 5}),
                  trigger_classes=triggers)


def test_condition_specs_are_the_five_runtime_conditions():
    assert set(CONDITION_SPECS) == {
        "orders", "batches", "pickers", "not_on_break", "dock_capacity"
    }


# ───────────────────────── V5: trigger validation ─────────────────────


def test_active_with_compatible_trigger_accepted():
    _validate("ORP", "active", _problem_cfg(congestion_penalty=1.0),
              trigger_classes=[InterventionRequest])
    _validate("OBRP", "active", _problem_cfg(congestion_penalty=1.0),
              trigger_classes=[InterventionRequest])


def test_active_with_incompatible_trigger_rejected():
    with pytest.raises(ValueError, match="requires"):
        _validate("ORP", "active", _problem_cfg(),
                  trigger_classes=[PlanningRun])
    with pytest.raises(ValueError, match="requires"):
        _validate("ORP", "active", _problem_cfg(),
                  trigger_classes=[PickerIdle])


def test_active_with_no_trigger_rejected():
    with pytest.raises(ValueError, match="no trigger is configured"):
        _validate("ORP", "active", _problem_cfg())


def test_non_active_accepts_any_trigger():
    _validate("OBP", "none", _problem_cfg(), trigger_classes=[WMSRun])
    _validate("OBRSP", "none", _problem_cfg(), trigger_classes=[PickerIdle])


def test_intervention_request_has_active_tour_capabilities():
    assert InterventionRequest.decision_capabilities == frozenset({
        "picker_id", "tour_id", "route_version", "resumes_execution"
    })


# ───────────────────────── V6: solver / ranking ──────────────────────


def test_scheduling_objectives_accepted():
    for obj in ("distance", "makespan", "tardiness"):
        _validate("ORSP", "none", _problem_cfg(solver={"type": "cosy", "objective": obj}))


def test_unsupported_objective_rejected():
    with pytest.raises(ValueError, match="does not support ranking objective"):
        _validate("OBRSP", "none", _problem_cfg(solver={"type": "cosy", "objective": "makespn"}))


def test_objective_on_unranked_batching_rejected():
    with pytest.raises(ValueError, match="not ranked by objective"):
        _validate("OBP", "none", _problem_cfg(solver={"type": "cosy", "objective": "tardiness"}))


def test_unranked_without_objective_accepted():
    _validate("OBP", "none", _problem_cfg(solver={"type": "cosy"}))


def test_routing_only_distance_accepted():
    _validate("OBRP", "none", _problem_cfg(solver={"type": "cosy", "objective": "distance"}))
    _validate("ORP", "active", _problem_cfg(congestion_penalty=0.0,
              solver={"type": "cosy", "objective": "distance"}),
              trigger_classes=[InterventionRequest])


def test_routing_non_distance_rejected():
    with pytest.raises(ValueError, match="does not support ranking objective"):
        _validate("OBRP", "none", _problem_cfg(solver={"type": "cosy", "objective": "makespan"}))


def test_supported_objectives_match_ranker():
    assert SolutionRanker.supported_objectives("OBP") == frozenset()
    assert SolutionRanker.supported_objectives("ORP") == frozenset({"distance"})
    assert SolutionRanker.supported_objectives("ORSP") == frozenset({
        "distance", "makespan", "tardiness"
    })


def test_direct_solver_skips_objective_validation():
    _validate("OBRSP", "none", _problem_cfg(solver={"type": "direct", "_target_": "x.S"}))


# ───────────────────────── V7: commitment ────────────────────────────


def test_scheduling_commit_accepted():
    _validate("ORSP", "none", _problem_cfg(commit={"n_jobs": 1}))
    _validate("OBRSP", "none", _problem_cfg(commit={"n_jobs": 1}))


def test_commit_on_non_scheduling_rejected():
    with pytest.raises(ValueError, match="cannot use a SchedulingCommitmentPolicy"):
        _validate("OBP", "none", _problem_cfg(commit={"n_jobs": 1}))
    with pytest.raises(ValueError, match="cannot use a SchedulingCommitmentPolicy"):
        _validate("OBRP", "none", _problem_cfg(commit={"n_jobs": 1}))
    with pytest.raises(ValueError, match="cannot use a SchedulingCommitmentPolicy"):
        _validate("ORP", "active", _problem_cfg(congestion_penalty=0.0, commit={"n_jobs": 1}),
                  trigger_classes=[InterventionRequest])


def test_solution_kind_from_taxonomy():
    assert solution_kind("OBP") == "BatchingSolution"
    assert solution_kind("ORP") == "CombinedRoutingSolution"
    assert solution_kind("ORSP") == "SchedulingSolution"


# ───────────────────────── V8: global uniqueness ─────────────────────


def test_duplicate_binding_rejected(tmp_path):
    cfg = _minimal_cfg([
        {"problem_class": "OBRSP", "replanning": "none", "triggers": ["OrderArrival"],
         "requires": {"orders": 1, "pickers": 1},
         "solver": {"type": "cosy", "objective": "makespan"}},
        {"problem_class": "OBRSP", "replanning": "none", "triggers": ["PickerIdle"],
         "requires": {"orders": 1, "pickers": 1},
         "solver": {"type": "cosy", "objective": "makespan"}},
    ], tmp_path)
    with pytest.raises(ValueError, match="Duplicate decision binding"):
        _compile_bindings(cfg)


def test_duplicate_trigger_rejected(tmp_path):
    cfg = _minimal_cfg([
        {"problem_class": "OBRSP", "replanning": "none",
         "triggers": ["OrderArrival", "PickerIdle"],
         "requires": {"orders": 1, "pickers": 1},
         "solver": {"type": "cosy", "objective": "makespan"}},
        {"problem_class": "OBP", "replanning": "none",
         "triggers": ["OrderArrival"],
         "requires": {"orders": 1},
         "solver": {"type": "cosy"}},
    ], tmp_path)
    with pytest.raises(ValueError, match="already bound"):
        _compile_bindings(cfg)


# ───────────────────────── no applicable pipeline ────────────────────


def test_no_applicable_pipeline_rejected(tmp_path):
    cfg = _intervention_cfg(["engines=insertion", "intervention_repo=fill_nn"], tmp_path)
    OmegaConf.update(cfg, "intervention_repo.components", [
        "casim.pipelines.problem_based_template.InstanceLoader",
        "casim.pipelines.problem_based_template.OrdersProvider",
        "casim.pipelines.subproblems.item_assignment.GreedyIA",
        "casim.pipelines.subproblems.batching.FiFo",
        "casim.pipelines.subproblems.picker_routing.SShape",
        "casim.pipelines.problem_based_template.ResultAggregationRouting",
    ], merge=False)
    card = load_and_flatten_data_card(OmegaConf.to_container(cfg.data_card, resolve=True))
    with pytest.raises(ValueError, match="form a pipeline"):
        compile_engine(cfg, card)


# ───────────────────── inspectability + explainability ───────────────


def test_compile_engine_populates_applicability(tmp_path):
    cfg = _intervention_cfg(["engines=routing", "intervention_repo=routing_tsp"], tmp_path)
    card = load_and_flatten_data_card(OmegaConf.to_container(cfg.data_card, resolve=True))
    compiled = compile_engine(cfg, card)
    obrsp = compiled.decisions[("OBRSP", "none")]
    assert obrsp.applicability is not None
    assert obrsp.applicability.retained_components
    assert obrsp.applicability.pipelines


def test_explain_derives_from_owners():
    from casim.decision_card import CompiledCondition
    from casim.decision_engine import SchedulingCommitmentPolicy
    from casim.setup import CompiledDecision, CompiledEngine
    from casim.simulation_engine.state_adapter import StateAdapter

    adapter = StateAdapter(
        problem_class="ORSP", replanning="unstarted",
        batches={"source": "buffered_and_replannable",
                 "due_horizon_s": 21600, "limit": 96},
        resources={"source": "available"},
    )
    conditions = (
        CompiledCondition("batches", CONDITION_SPECS["batches"], 1),
        CompiledCondition("pickers", CONDITION_SPECS["pickers"], 1),
        CompiledCondition("not_on_break", CONDITION_SPECS["not_on_break"], True),
    )
    decision = CompiledDecision(
        binding=("ORSP", "unstarted"),
        adapter=adapter,
        trigger_classes=(PlanningRun,),
        conditions=conditions,
        solver=None,
        policy=SchedulingCommitmentPolicy(
            max_jobs_per_picker=1, planning_horizon_s=14400
        ),
    )
    compiled = CompiledEngine(
        decisions={("ORSP", "unstarted"): decision},
        triggers_map={PlanningRun: ("ORSP", "unstarted")},
        data_card=type("D", (), {"problem_class": "ORSP"})(),
    )
    report = compiled.explain(("ORSP", "unstarted"))
    assert "ORSP / unstarted" in report
    assert "batches: buffered + replannable" in report
    assert "due_horizon_s = 21600" in report
    assert "limit = 96" in report
    assert "resources: available" in report
    assert "PlanningRun" in report
    assert "batches >= 1" in report
    assert "pickers >= 1" in report
    assert "not_on_break" in report
    assert "routing" in report
    assert "scheduling" in report
    assert "SchedulingSolution" in report
    assert "replannable_unstarted_work" in report
    assert "resource_ready_times" in report
    assert "max_jobs_per_picker = 1" in report
    assert "planning_horizon_s = 14400" in report
    assert "direct solver" in report


def test_explain_dock_capacity_uses_leq():
    from casim.decision_card import CompiledCondition
    from casim.setup import CompiledDecision, CompiledEngine
    from casim.simulation_engine.state_adapter import StateAdapter

    adapter = StateAdapter(problem_class="OBRSP", replanning="none",
                           orders={"source": "buffered"},
                           resources={"source": "dispatchable", "scope": "trigger_if_present"})
    conditions = (
        CompiledCondition("orders", CONDITION_SPECS["orders"], 1),
        CompiledCondition("pickers", CONDITION_SPECS["pickers"], 1),
        CompiledCondition("dock_capacity", CONDITION_SPECS["dock_capacity"], 5),
    )
    decision = CompiledDecision(
        binding=("OBRSP", "none"), adapter=adapter,
        trigger_classes=(), conditions=conditions,
        solver=None, policy=None,
    )
    compiled = CompiledEngine(
        decisions={("OBRSP", "none"): decision},
        triggers_map={}, data_card=None,
    )
    report = compiled.explain(("OBRSP", "none"))
    assert "dock_capacity <= 5" in report
    assert "dock_capacity >=" not in report
    assert "commitment" in report
    assert "  none" in report


# ───────────────────── projection spec ownership ────────────────────


def test_projection_specs_own_parameter_consumption():
    assert PROJECTION_SPECS[("orders", "buffered")].parameters == frozenset({"limit"})
    assert PROJECTION_SPECS[("batches", "buffered")].parameters == frozenset({"limit", "due_horizon_s"})
    assert PROJECTION_SPECS[("active_tour", "residual")].parameters == frozenset({"congestion_penalty"})
    assert PROJECTION_SPECS[("resources", "all")].parameters == frozenset()


def test_card_parameters_aggregate_from_projections():
    assert DECISION_CARDS[("OBP", "none")].parameters() == frozenset({"limit"})
    assert DECISION_CARDS[("ORSP", "none")].parameters() == frozenset({"limit", "due_horizon_s"})
    assert DECISION_CARDS[("ORP", "active")].parameters() == frozenset({"congestion_penalty"})
    assert DECISION_CARDS[("OBRP", "active")].parameters() == frozenset({"limit", "congestion_penalty"})


def test_card_facts_include_base_facts():
    for binding in sorted(SUPPORTED_DECISIONS):
        facts = DECISION_CARDS[binding].facts()
        assert "dynamic.is_break" in facts
        assert "dynamic.n_staged_pallets" in facts


def test_active_card_has_fixed_cardinalities():
    card = DECISION_CARDS[("ORP", "active")]
    assert card.fixed_cardinalities() == {"batches.count": 1, "resources.count": 1}


# ───────────────────── cleanup: frozen + strict validators ────────────


def test_compiled_decision_is_frozen():
    from dataclasses import FrozenInstanceError

    from casim.setup import CompiledDecision

    decision = CompiledDecision(
        binding=("OBP", "none"),
        adapter=None,
        trigger_classes=(),
        conditions=(),
        solver=None,
        policy=None,
    )
    with pytest.raises(FrozenInstanceError):
        decision.binding = ("ORSP", "none")


def test_compiled_engine_is_frozen():
    from dataclasses import FrozenInstanceError

    from casim.setup import CompiledEngine

    compiled = CompiledEngine(decisions={}, triggers_map={}, data_card=None)
    with pytest.raises(FrozenInstanceError):
        compiled.decisions = {}


def test_commit_rejects_float_for_integer_fields():
    with pytest.raises(ValueError, match="invalid value for 'n_jobs'"):
        _validate("ORSP", "none", _problem_cfg(commit={"n_jobs": 1.5}))
    with pytest.raises(ValueError, match="invalid value for 'max_jobs_per_picker'"):
        _validate("ORSP", "none", _problem_cfg(commit={"max_jobs_per_picker": 2.0}))


def test_commit_rejects_bool_for_integer_fields():
    with pytest.raises(ValueError, match="invalid value for 'n_jobs'"):
        _validate("ORSP", "none", _problem_cfg(commit={"n_jobs": True}))
    with pytest.raises(ValueError, match="invalid value for 'max_jobs_per_picker'"):
        _validate("ORSP", "none", _problem_cfg(commit={"max_jobs_per_picker": False}))


def test_commit_accepts_float_for_planning_horizon_s():
    _validate("ORSP", "none", _problem_cfg(commit={"planning_horizon_s": 3600.5}))


def test_commit_rejects_bool_for_planning_horizon_s():
    with pytest.raises(ValueError, match="invalid value for 'planning_horizon_s'"):
        _validate("ORSP", "none", _problem_cfg(commit={"planning_horizon_s": True}))


# ───────────────────── cleanup: explain wording ─────────────────────


def test_explain_policy_none_renders_none_not_commit_all():
    from casim.decision_card import CompiledCondition
    from casim.setup import CompiledDecision, CompiledEngine
    from casim.simulation_engine.state_adapter import StateAdapter

    adapter = StateAdapter(
        problem_class="OBP", replanning="none",
        orders={"source": "buffered"}, resources={"source": "all"},
    )
    decision = CompiledDecision(
        binding=("OBP", "none"), adapter=adapter,
        trigger_classes=(), conditions=(),
        solver=None, policy=None,
    )
    compiled = CompiledEngine(
        decisions={("OBP", "none"): decision},
        triggers_map={}, data_card=None,
    )
    report = compiled.explain(("OBP", "none"))
    assert "commitment" in report
    assert "  none" in report
    assert "commit all (no scheduling policy)" not in report


def test_explain_active_tour_resources_not_all():
    from casim.decision_card import CompiledCondition
    from casim.setup import CompiledDecision, CompiledEngine
    from casim.simulation_engine.state_adapter import StateAdapter

    adapter = StateAdapter(
        problem_class="ORP", replanning="active",
        active_tour={"source": "residual", "congestion_penalty": 0.0},
    )
    decision = CompiledDecision(
        binding=("ORP", "active"), adapter=adapter,
        trigger_classes=(InterventionRequest,), conditions=(),
        solver=None, policy=None,
    )
    compiled = CompiledEngine(
        decisions={("ORP", "active"): decision},
        triggers_map={InterventionRequest: ("ORP", "active")},
        data_card=None,
    )
    report = compiled.explain(("ORP", "active"))
    assert "active_tour: residual" in report
    assert "resources: all" not in report
    assert "active tour; single picker" in report


# ───────────────────── parameter domain validators ────────────────────


def test_limit_accepts_positive_int():
    _validate("OBP", "none", _problem_cfg(limit=1))
    _validate("OBP", "none", _problem_cfg(limit=50))


def test_limit_accepts_none():
    _validate("OBP", "none", _problem_cfg())


def test_limit_rejects_zero():
    with pytest.raises(ValueError, match="invalid value for 'limit'"):
        _validate("OBP", "none", _problem_cfg(limit=0))


def test_limit_rejects_negative():
    with pytest.raises(ValueError, match="invalid value for 'limit'"):
        _validate("OBP", "none", _problem_cfg(limit=-5))


def test_limit_rejects_float():
    with pytest.raises(ValueError, match="invalid value for 'limit'"):
        _validate("OBP", "none", _problem_cfg(limit=1.5))


def test_limit_rejects_bool():
    with pytest.raises(ValueError, match="invalid value for 'limit'"):
        _validate("OBP", "none", _problem_cfg(limit=True))
    with pytest.raises(ValueError, match="invalid value for 'limit'"):
        _validate("OBP", "none", _problem_cfg(limit=False))


def test_due_horizon_s_accepts_nonneg_int():
    _validate("ORSP", "none", _problem_cfg(due_horizon_s=0))
    _validate("ORSP", "none", _problem_cfg(due_horizon_s=3600))


def test_due_horizon_s_accepts_nonneg_float():
    _validate("ORSP", "none", _problem_cfg(due_horizon_s=3600.5))


def test_due_horizon_s_accepts_none():
    _validate("ORSP", "none", _problem_cfg())


def test_due_horizon_s_rejects_negative():
    with pytest.raises(ValueError, match="invalid value for 'due_horizon_s'"):
        _validate("ORSP", "none", _problem_cfg(due_horizon_s=-1))
    with pytest.raises(ValueError, match="invalid value for 'due_horizon_s'"):
        _validate("ORSP", "none", _problem_cfg(due_horizon_s=-0.5))


def test_due_horizon_s_rejects_bool():
    with pytest.raises(ValueError, match="invalid value for 'due_horizon_s'"):
        _validate("ORSP", "none", _problem_cfg(due_horizon_s=True))
    with pytest.raises(ValueError, match="invalid value for 'due_horizon_s'"):
        _validate("ORSP", "none", _problem_cfg(due_horizon_s=False))


def test_congestion_penalty_accepts_nonneg_int():
    _validate(
        "ORP", "active",
        _problem_cfg(congestion_penalty=0),
        trigger_classes=[InterventionRequest],
    )


def test_congestion_penalty_accepts_nonneg_float():
    _validate(
        "ORP", "active",
        _problem_cfg(congestion_penalty=1.5),
        trigger_classes=[InterventionRequest],
    )
    _validate(
        "ORP", "active",
        _problem_cfg(congestion_penalty=0.0),
        trigger_classes=[InterventionRequest],
    )


def test_congestion_penalty_rejects_negative():
    with pytest.raises(ValueError, match="invalid value for 'congestion_penalty'"):
        _validate(
            "ORP", "active",
            _problem_cfg(congestion_penalty=-1),
            trigger_classes=[InterventionRequest],
        )
    with pytest.raises(ValueError, match="invalid value for 'congestion_penalty'"):
        _validate(
            "ORP", "active",
            _problem_cfg(congestion_penalty=-0.1),
            trigger_classes=[InterventionRequest],
        )


def test_congestion_penalty_rejects_bool():
    with pytest.raises(ValueError, match="invalid value for 'congestion_penalty'"):
        _validate(
            "ORP", "active",
            _problem_cfg(congestion_penalty=True),
            trigger_classes=[InterventionRequest],
        )
    with pytest.raises(ValueError, match="invalid value for 'congestion_penalty'"):
        _validate(
            "ORP", "active",
            _problem_cfg(congestion_penalty=False),
            trigger_classes=[InterventionRequest],
        )


# ───────────────────── condition domain validators ────────────────────


def test_condition_rejects_float():
    with pytest.raises(ValueError, match="invalid value for 'orders'"):
        _validate("OBP", "none", _problem_cfg(requires={"orders": 1.5}))
    with pytest.raises(ValueError, match="invalid value for 'pickers'"):
        _validate("OBP", "none", _problem_cfg(requires={"pickers": 2.0}))


def test_condition_rejects_bool():
    with pytest.raises(ValueError, match="invalid value for 'orders'"):
        _validate("OBP", "none", _problem_cfg(requires={"orders": True}))
    with pytest.raises(ValueError, match="invalid value for 'pickers'"):
        _validate("OBP", "none", _problem_cfg(requires={"pickers": False}))


def test_condition_rejects_none():
    with pytest.raises(ValueError, match="invalid value for 'orders'"):
        _validate("OBP", "none", _problem_cfg(requires={"orders": None}))


def test_condition_rejects_zero():
    with pytest.raises(ValueError, match="invalid value for 'orders'"):
        _validate("OBP", "none", _problem_cfg(requires={"orders": 0}))


def test_condition_rejects_negative():
    with pytest.raises(ValueError, match="invalid value for 'orders'"):
        _validate("OBP", "none", _problem_cfg(requires={"orders": -1}))


def test_dock_capacity_rejects_float():
    with pytest.raises(ValueError, match="invalid value for 'dock_capacity'"):
        _validate(
            "OBRSP", "none",
            _problem_cfg(requires={"orders": 1, "pickers": 1, "dock_capacity": 5.5}),
        )
