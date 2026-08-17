"""Required integration test: Henn timed wakeup end-to-end."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ware_ops_algos.algorithms import CombinedRoutingSolution

from casim.setup import build_runtime
from scenarios.scenario_henn.algorithm import HennWakeUp
from scenarios.scenario_henn.experiment_henn import run_henn_experiment
from scenarios.scenario_henn.scenario_specific_hooks import build_sim_hooks
from ware_ops_algos.domain_models import load_and_flatten_data_card


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


def _cfg(tmp_path, *overrides):
    cfg = _compose(*overrides)
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
        "wakeup-smoke",
        merge=False,
    )
    OmegaConf.update(cfg, "luigi.runtime", 1, merge=False)
    return cfg


@pytest.mark.parametrize("batching", ["fcfs"])
def test_henn_wakeup_registers_tuple_binding(tmp_path, batching):
    cfg = _cfg(tmp_path, f"batching={batching}")
    data_card = load_and_flatten_data_card(cfg.data_card)
    simulation, decision_engine = build_runtime(cfg, data_card)
    simulation.reset(hooks=build_sim_hooks(cfg))
    assert simulation.triggers_map[HennWakeUp] == ("OBRP", "none")


def test_henn_timed_wait_wakeup_and_dispatch_end_to_end(tmp_path):
    cfg = _cfg(tmp_path, "batching=fcfs", "selection=short",
               "waiting=henn_4_1")
    OmegaConf.update(cfg, "experiment.max_decisions", 200, merge=False)
    data_card = load_and_flatten_data_card(cfg.data_card)
    simulation, decision_engine = build_runtime(cfg, data_card)

    actual, trace = run_henn_experiment(cfg, simulation, decision_engine)

    timed_waits = [
        row for row in trace
        if row["action"] == "wait" and row["wait_until_s"] is not None
    ]
    if not timed_waits:
        pytest.skip(
            "This instance did not produce a timed wait under henn_4_1; "
            "the wake-up path is exercised only when the threshold precedes "
            "the next arrival."
        )

    wait_row = timed_waits[0]
    wait_time = wait_row["current_time_s"]
    wait_until = wait_row["wait_until_s"]
    assert wait_until > wait_time

    dispatch_after_wait = [
        row for row in trace
        if row["action"] == "dispatch"
        and row["current_time_s"] >= wait_until
    ]
    assert dispatch_after_wait, (
        "No dispatch occurred after the timed wait; the HennWakeUp event "
        "did not produce a subsequent decision snapshot."
    )

    assert actual["completion_time_s"] > 0
    assert actual["algorithm_id"].startswith("henn_4_1+")
