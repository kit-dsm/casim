"""Luigi tasks built for separate solvers keep their own pipeline paths."""

from casim.pipelines.problem_based_template import InstanceLoader
from casim.run import prepare
from tests.scenario_configs import analytic_config


def _instance_loaders(task):
    if isinstance(task, InstanceLoader):
        return (task,)
    return tuple(loader for dependency in task.requires().values()
                 for loader in _instance_loaders(dependency))


def test_two_solvers_keep_distinct_cache_paths(tmp_path):
    """A second CoSy solver must read its own dynamic domain cache."""
    runs = []
    for name in ("first", "second"):
        prepared = prepare(analytic_config(tmp_path / name))
        simulation, decision_engine = prepared.sim, prepared.engine
        solver = decision_engine.solver_map["OBRSP"]
        prepared.reset()
        while True:
            done, snapshot = simulation.run()
            assert not done
            if snapshot.dynamic_warehouse_info.context.solve:
                break
            simulation.step(decision_engine.on_trigger(snapshot), snapshot)
        runs.append((solver, snapshot))

    assert runs[0][0].cache_path != runs[1][0].cache_path
    for solver, snapshot in runs:
        loaders = tuple(loader for pipeline in solver.pipelines
                        for loader in _instance_loaders(pipeline))
        assert loaders
        assert all(loader.pipeline_params.domain_path == str(solver.cache_path)
                   for loader in loaders)
        candidates = solver.solve(snapshot)
        assert candidates
        assert all(candidate is not None for candidate in candidates.values())
