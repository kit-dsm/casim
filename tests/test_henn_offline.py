from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ware_ops_algos.algorithms import CombinedRoutingSolution

from scenarios.experiment_commons import (
    load_and_flatten_data_card,
    setup_decision_engine,
    setup_scenario,
)
from scenarios.scenario_henn.scenario_specific_hooks import build_sim_hooks
from casim.events.operational_events import OrderArrival, FlushRemainingOrders


SCENARIO_ROOT = (
    Path(__file__).parents[1] / "scenarios" / "scenario_henn"
).resolve()


def _compose(*overrides):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(SCENARIO_ROOT / "config"),
    ):
        return compose(
            config_name="henn_config",
            overrides=list(overrides),
        )


@pytest.mark.parametrize("batching", ["fcfs", "cw_like", "ls"])
@pytest.mark.parametrize("selection", ["first", "short", "long", "sav"])
def test_hydra_composes_the_twelve_study_variants(batching, selection):
    cfg = _compose(f"batching={batching}", f"selection={selection}")
    assert cfg.batching.name == batching
    assert cfg.selection.name == selection
    assert cfg.data_card.problem_class == "OBRP"
    assert cfg.cosy_repo.components[-1].endswith(
        "ResultAggregationRouting"
    )


@pytest.mark.parametrize("batching", ["fcfs", "cw_like", "ls"])
def test_existing_setup_discovers_one_pipeline_per_batching_variant(
    batching,
    tmp_path,
):
    cfg = _compose(f"batching={batching}")
    project_root = Path(__file__).parents[1].resolve()
    OmegaConf.update(cfg, "project_root", str(project_root), merge=False)
    OmegaConf.update(
        cfg,
        "instances_base",
        str(project_root / "scenarios"),
        merge=False,
    )
    OmegaConf.update(cfg, "cache_base", str(tmp_path / "cache"), merge=False)
    OmegaConf.update(
        cfg,
        "experiment.output_dir",
        str(tmp_path),
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "experiment.instance_name",
        f"smoke-{batching}",
        merge=False,
    )
    OmegaConf.update(cfg, "luigi.runtime", 1, merge=False)
    data_card = load_and_flatten_data_card(cfg.data_card)
    simulation = setup_scenario(cfg)
    decision_engine = setup_decision_engine(cfg, data_card)

    simulation.reset(hooks=build_sim_hooks(cfg))

    assert len(decision_engine.solver_map["OBRP"].pipelines) == 1
    assert sum(
        isinstance(event, OrderArrival)
        for event in simulation.events
    ) == 40
    assert sum(
        isinstance(event, FlushRemainingOrders)
        for event in simulation.events
    ) == 1
    done, snapshot = simulation.run()
    assert not done
    result = decision_engine.solver_map["OBRP"].solve(snapshot, action=None)
    assert result is not None
    solution, _, _ = result
    assert isinstance(solution, CombinedRoutingSolution)
    assert len(solution.routes) == 1
