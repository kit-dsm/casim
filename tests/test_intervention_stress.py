from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scenarios.scenario_reopt.loader import ReoptDataLoader
from scenarios.scenario_intervention_stress.experiment_intervention_stress import (
    run,
)


ROOT = Path(__file__).parents[1]
SCENARIO = ROOT / "scenarios" / "scenario_intervention_stress"


def test_stress_fixture_is_multi_picker_and_conventional():
    domain = ReoptDataLoader(SCENARIO).load(
        SCENARIO / "data" / "intervention_stress.json"
    )
    assert len(domain.orders.orders) == 64
    assert len(domain.resources.resources) == 8
    assert domain.layout.graph_data.n_aisles == 6
    assert domain.layout.graph_data.n_blocks == 2
    assert all(
        resource.pick_cart.n_boxes == 3
        for resource in domain.resources.resources
    )
    assert len(
        {
            id(resource.pick_cart)
            for resource in domain.resources.resources
        }
    ) == 8


def test_stress_policies_use_expected_intervention_pipeline():
    config_dir = str((SCENARIO / "config").resolve())
    expected = {
        "reopt": ("reopt", None, None),
        "routing_tsp": ("routing", "ORP", "PickListProvider"),
        "fill_tsp": ("insertion", "OBRP", "ResidualFiFo"),
        "fill_nn": ("insertion", "OBRP", "NearestNeighbourhood"),
    }
    with initialize_config_dir(
        version_base="1.3",
        config_dir=config_dir,
    ):
        for policy, (engine, problem, component) in expected.items():
            cfg = compose(
                config_name="intervention_stress_config",
                overrides=[
                    f"engines={engine}",
                    f"intervention_repo={policy}",
                ],
            )
            assert cfg.intervention_repo.name == policy
            problem_classes = [
                entry.problem_class for entry in cfg.engines.problems
            ]
            assert "OBRSP" in problem_classes
            if problem is None:
                assert problem_classes == ["OBRSP"]
                continue
            assert problem in problem_classes
            entry = next(
                entry
                for entry in cfg.engines.problems
                if entry.problem_class == problem
            )
            components = entry.solver.repo.components
            assert any(component in value for value in components)


@pytest.mark.parametrize("threshold", [2, 64, 65])
def test_stress_drains_all_orders_for_low_and_high_thresholds(
    tmp_path,
    threshold,
):
    config_dir = str((SCENARIO / "config").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="intervention_stress_config",
            overrides=["engines=insertion", "intervention_repo=fill_nn"],
        )
    run_dir = tmp_path / str(threshold)
    OmegaConf.update(cfg, "project_root", str(ROOT), merge=False)
    OmegaConf.update(
        cfg,
        "instances_base",
        str(ROOT / "scenarios"),
        merge=False,
    )
    OmegaConf.update(cfg, "cache_base", str(run_dir / "cache"), merge=False)
    OmegaConf.update(
        cfg,
        "experiment.output_dir",
        str(run_dir),
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "experiment.working_dir",
        str(run_dir / "work"),
        merge=False,
    )
    OmegaConf.update(cfg, "viz.record", False, merge=False)
    OmegaConf.update(
        cfg,
        "engines.problems.0.requires.orders",
        threshold,
        merge=False,
    )

    result = run(cfg)

    assert result["status"] == "success"
    assert result["completion_reason"] == "drained"
    assert result["completed_orders"] == 64
    assert not any(result["unfinished_work"].values())
    if threshold >= 64:
        assert result["interventions"] == 0
        assert result["configuration_warnings"]
