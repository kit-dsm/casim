"""Luigi tasks built for separate solvers keep their own pipeline paths."""

from pathlib import Path

from hydra import compose, initialize_config_dir

from casim.pipelines.problem_based_template import InstanceLoader
from scenarios.experiment_commons import (
    load_and_flatten_data_card, setup_experiment,
)
from scenarios.scenario_stochastic_waiting.scenario_specific_hooks import (
    add_orders_hook, picker_arrival_hook,
)


CONFIG_DIR = (Path(__file__).resolve().parents[1] / "scenarios"
              / "scenario_stochastic_waiting" / "config")


def _instance_loaders(task):
    if isinstance(task, InstanceLoader):
        return (task,)
    return tuple(loader for dependency in task.requires().values()
                 for loader in _instance_loaders(dependency))


def test_two_solvers_keep_distinct_cache_paths(tmp_path):
    """A second CoSy solver must read its own dynamic domain cache."""
    runs = []
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        for name in ("first", "second"):
            directory = tmp_path / name
            cfg = compose(config_name="stochastic_waiting_config", overrides=[
                f"instances_base={directory.as_posix()}",
                f"cache_base={directory.as_posix()}",
                f"experiment.output_dir={directory.as_posix()}",
            ])
            card = load_and_flatten_data_card(cfg.data_card)
            simulation, decision_engine = setup_experiment(cfg, card)
            solver = decision_engine.solver_map["OBRSPW"]
            simulation.reset(hooks=[add_orders_hook, picker_arrival_hook])
            done, snapshot = simulation.run()
            assert not done
            runs.append((solver, snapshot))

    assert runs[0][0].cache_path != runs[1][0].cache_path
    for solver, snapshot in runs:
        loaders = tuple(loader for pipeline in solver.pipelines
                        for loader in _instance_loaders(pipeline))
        assert loaders
        assert all(loader.pipeline_params.domain_path == str(solver.cache_path)
                   for loader in loaders)
        solution, _, _, _ = solver.solve(snapshot)
        assert solution is not None
