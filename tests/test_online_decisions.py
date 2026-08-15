from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from casim.events.operational_events import OrderArrival
from casim.setup import build_simulation
from scenarios.scenario_henn.algorithm import HennWakeUp
from scenarios.scenario_henn.scenario_specific_hooks import build_sim_hooks


SCENARIO_ROOT = (
    Path(__file__).parents[1] / "scenarios" / "scenario_henn"
).resolve()


def _config():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(SCENARIO_ROOT / "config"),
    ):
        return compose(config_name="henn_config")


def test_reset_hooks_recreate_the_same_order_stream_without_duplicates(
    tmp_path,
):
    cfg = _config()
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
    simulation = build_simulation(cfg)

    simulation.reset(hooks=build_sim_hooks(cfg))
    first = [
        (event.order_id, event.time)
        for event in simulation.events
        if isinstance(event, OrderArrival)
    ]

    simulation.reset(hooks=build_sim_hooks(cfg))
    second = [
        (event.order_id, event.time)
        for event in simulation.events
        if isinstance(event, OrderArrival)
    ]

    assert len(first) == len(second) == 40
    assert sorted(first) == sorted(second)
    assert HennWakeUp in simulation.triggers_map
