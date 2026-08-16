from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ware_ops_algos.algorithms.algorithm_cards import (
    load_packaged_algo_cards,
)
from ware_ops_algos.domain_algo_mapper.domain_algo_mapper import (
    DomainAlgorithmMapper,
)

from casim.pipelines.taxonomy import TAXONOMY
from casim.simulation_engine.state_adapter import StateAdapter
from casim.setup import build_runtime
from ware_ops_algos.domain_models import load_and_flatten_data_card


ROOT = Path(__file__).parents[1]
SCENARIO = ROOT / "scenarios" / "scenario_intervention_stress"


def _card(features):
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str((SCENARIO / "config").resolve()),
    ):
        cfg = compose(config_name="intervention_stress_config")
    card = load_and_flatten_data_card(
        OmegaConf.to_container(cfg.data_card, resolve=True)
    )
    card = deepcopy(card)
    card.problem_class = "OBRP"
    card.warehouse_info["features"].update(
        {feature: True for feature in features}
    )
    return card


_ORDER_WINDOW_FEATURES = StateAdapter(
    problem_class="OBRSP",
    replanning="none",
    orders={"source": "buffered"},
    resources={"source": "dispatchable", "scope": "trigger_if_present"},
).projected_features()

_ACTIVE_TOUR_FEATURES = StateAdapter(
    problem_class="OBRP",
    replanning="active",
    active_tour={"source": "residual"},
    orders={"source": "buffered"},
).projected_features()


def test_adapter_context_selects_residual_algorithms_only_when_needed():
    cards = load_packaged_algo_cards()
    mapper = DomainAlgorithmMapper(TAXONOMY)

    ordinary = {
        card.algo_name
        for card in mapper.filter(
            cards,
            _card(_ORDER_WINDOW_FEATURES),
        )
    }
    active = {
        card.algo_name
        for card in mapper.filter(
            cards,
            _card(_ACTIVE_TOUR_FEATURES),
        )
    }

    assert "FiFo" in ordinary
    assert "SShape" in ordinary
    assert "FiFo" not in active
    assert "SShape" not in active
    assert {"ExactSolving", "NearestNeighbourhood", "ResidualFiFo"} <= active


def test_residual_fifo_static_cart_requirements_are_checked_by_mapper():
    card = _card(_ACTIVE_TOUR_FEATURES)
    card.resources["features"]["box_can_mix_orders"] = True

    applicable = {
        algorithm.algo_name
        for algorithm in DomainAlgorithmMapper(TAXONOMY).filter(
            load_packaged_algo_cards(),
            card,
        )
    }

    assert "ResidualFiFo" not in applicable


def test_incompatible_intervention_repository_fails_during_setup(tmp_path):
    config_dir = str((SCENARIO / "config").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="intervention_stress_config",
            overrides=["engines=insertion", "intervention_repo=fill_nn"],
        )
    OmegaConf.update(cfg, "project_root", str(ROOT), merge=False)
    OmegaConf.update(cfg, "instances_base", str(ROOT / "scenarios"), merge=False)
    OmegaConf.update(cfg, "cache_base", str(tmp_path / "cache"), merge=False)
    OmegaConf.update(cfg, "experiment.output_dir", str(tmp_path), merge=False)
    OmegaConf.update(
        cfg,
        "experiment.working_dir",
        str(tmp_path / "work"),
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "intervention_repo.components",
        [
            "casim.pipelines.problem_based_template.InstanceLoader",
            "casim.pipelines.problem_based_template.OrdersProvider",
            "casim.pipelines.subproblems.item_assingment.GreedyIA",
            "casim.pipelines.subproblems.batching.FiFo",
            "casim.pipelines.subproblems.picker_routing.SShape",
            "casim.pipelines.problem_based_template.ResultAggregationRouting",
        ],
        merge=False,
    )
    card = load_and_flatten_data_card(
        OmegaConf.to_container(cfg.data_card, resolve=True)
    )

    with pytest.raises(ValueError, match="form a pipeline"):
        build_runtime(cfg, card)
