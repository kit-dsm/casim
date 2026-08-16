"""Tests for the decision-configuration space derived from (problem_class, replanning)."""

import pytest
from omegaconf import OmegaConf

from casim.setup import (
    SUPPORTED_DECISIONS,
    _derive_exposure,
    _build_adapter,
)


@pytest.mark.parametrize(
    "problem_class, replanning",
    sorted(SUPPORTED_DECISIONS),
)
def test_derive_exposure_returns_nonempty_dict_for_every_supported_binding(
    problem_class, replanning
):
    exposure = _derive_exposure(
        problem_class,
        replanning,
        due_horizon_s=None,
        limit=None,
        congestion_penalty=0.0,
    )
    assert isinstance(exposure, dict)
    assert len(exposure) > 0


@pytest.mark.parametrize(
    "problem_class, replanning, expected_source_keys",
    [
        ("OBP", "none", {"orders", "resources"}),
        ("ORSP", "none", {"batches", "resources"}),
        ("ORSP", "unstarted", {"batches", "resources"}),
        ("OBRSP", "none", {"orders", "resources"}),
        ("OBRP", "none", {"orders", "resources"}),
        ("ORP", "active", {"active_tour"}),
        ("OBRP", "active", {"active_tour", "orders"}),
    ],
)
def test_derive_exposure_produces_correct_exposure_keys(
    problem_class, replanning, expected_source_keys
):
    exposure = _derive_exposure(
        problem_class,
        replanning,
        due_horizon_s=None,
        limit=None,
        congestion_penalty=0.0,
    )
    assert set(exposure) == expected_source_keys


@pytest.mark.parametrize(
    "problem_class, replanning, key, source",
    [
        ("OBP", "none", "orders", "buffered"),
        ("OBP", "none", "resources", "all"),
        ("ORSP", "none", "batches", "buffered"),
        ("ORSP", "none", "resources", "nonactive"),
        ("ORSP", "unstarted", "batches", "buffered_and_replannable"),
        ("ORSP", "unstarted", "resources", "available"),
        ("OBRSP", "none", "orders", "buffered"),
        ("OBRSP", "none", "resources", "dispatchable"),
        ("OBRP", "none", "orders", "buffered"),
        ("OBRP", "none", "resources", "dispatchable"),
        ("ORP", "active", "active_tour", "residual"),
        ("OBRP", "active", "active_tour", "residual"),
        ("OBRP", "active", "orders", "buffered"),
    ],
)
def test_derive_exposure_produces_correct_sources(
    problem_class, replanning, key, source
):
    exposure = _derive_exposure(
        problem_class,
        replanning,
        due_horizon_s=None,
        limit=None,
        congestion_penalty=0.0,
    )
    assert exposure[key]["source"] == source


@pytest.mark.parametrize(
    "problem_class, replanning",
    [
        ("OBP", "unstarted"),
        ("OBP", "active"),
        ("ORSP", "active"),
        ("OBRSP", "unstarted"),
        ("OBRSP", "active"),
        ("OBRP", "unstarted"),
        ("ORP", "none"),
        ("ORP", "unstarted"),
        ("UNKNOWN", "none"),
    ],
)
def test_derive_exposure_raises_for_unsupported_bindings(problem_class, replanning):
    with pytest.raises(ValueError, match="Unsupported decision binding"):
        _derive_exposure(
            problem_class,
            replanning,
            due_horizon_s=None,
            limit=None,
            congestion_penalty=0.0,
        )


@pytest.mark.parametrize(
    "problem_class, replanning",
    sorted(SUPPORTED_DECISIONS),
)
def test_build_adapter_sets_problem_class_and_replanning(problem_class, replanning):
    cfg = OmegaConf.create(
        {
            "problem_class": problem_class,
            "replanning": replanning,
        }
    )
    pc, rep, adapter = _build_adapter(cfg)
    assert pc == problem_class
    assert rep == replanning
    assert adapter.problem_class == problem_class
    assert adapter.replanning == replanning


def test_build_adapter_defaults_replanning_to_none():
    cfg = OmegaConf.create({"problem_class": "OBP"})
    pc, rep, adapter = _build_adapter(cfg)
    assert rep == "none"
    assert adapter.replanning == "none"


@pytest.mark.parametrize("replanning", ["planned", "all", "", "NONE"])
def test_build_adapter_rejects_invalid_replanning(replanning):
    cfg = OmegaConf.create(
        {"problem_class": "OBP", "replanning": replanning}
    )
    with pytest.raises(ValueError, match="replanning must be one of"):
        _build_adapter(cfg)


def test_supported_decisions_is_exactly_the_documented_set():
    assert SUPPORTED_DECISIONS == {
        ("OBP", "none"),
        ("ORSP", "none"),
        ("ORSP", "unstarted"),
        ("OBRSP", "none"),
        ("OBRP", "none"),
        ("ORP", "active"),
        ("OBRP", "active"),
    }


def test_research_params_threaded_into_exposure():
    exposure = _derive_exposure(
        "ORSP",
        "none",
        due_horizon_s=3600,
        limit=50,
        congestion_penalty=0.0,
    )
    assert exposure["batches"]["due_horizon_s"] == 3600
    assert exposure["batches"]["limit"] == 50


def test_congestion_penalty_threaded_into_active_tour():
    exposure = _derive_exposure(
        "ORP",
        "active",
        due_horizon_s=None,
        limit=None,
        congestion_penalty=1.5,
    )
    assert exposure["active_tour"]["congestion_penalty"] == 1.5
