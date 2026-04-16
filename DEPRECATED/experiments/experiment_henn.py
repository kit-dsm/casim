import time
from pathlib import Path

import hydra
from omegaconf import DictConfig
from ware_ops_algos.domain_models import datacard_from_instance
from ware_ops_pipes.utils.experiment_utils import RankingEvaluatorDistance

from experiments.experiment_commons import build_state_transformers, make_execution, build_data_loader, \
    build_trigger_map, build_req_policy, CoSyRunner, OnlineRunner
from ware_ops_sim.sim import SimWarehouseDomain, WarehouseSimulation
from ware_ops_sim.sim.decision_engine.sim_control import DecisionEngine
from ware_ops_sim.sim.events.events import PickerArrival


@hydra.main(config_path="config", config_name="henn_online_config")
def main(cfg: DictConfig):
    project_root = Path(cfg.project_root)
    instances_dir = Path(cfg.instances_base) / cfg.data_card.name
    cache_dir = Path(cfg.cache_base) / cfg.data_card.name
    cache_path = Path(cfg.cache_base) / "dynamic_info.pkl"

    cache_dir.mkdir(parents=True, exist_ok=True)

    orders_path = instances_dir / "orders.csv"
    layout_path = instances_dir / "layout.csv"


    state_transformers = build_state_transformers(cfg)
    runner = OnlineRunner(instance_set_name=cfg.data_card.name,
                          instances_dir=instances_dir,
                          cache_dir=cache_dir,
                          project_root=project_root,
                          instance_name=cfg.experiment.instance_name,
                          verbose=True)
    execution = make_execution(cfg)

    loader_kwargs = {k: instances_dir / v
                     for k, v in cfg.data_card.source.items()
                     if k.endswith("path")}

    loader = build_data_loader(cfg)

    domain = loader.load(**loader_kwargs)
    dc = datacard_from_instance(domain, "initial_dc")

    for key, value in execution.items():
        dc.problem_class = key
        if isinstance(execution[key], CoSyRunner):
            execution[key].build_pipelines(dc)

    for key, value in execution.items():
        dc.problem_class = key
        if (isinstance(execution[key], OnlineRunner) or
                isinstance(execution[key], CoSyRunner)):
            execution[key].build_pipelines(dc)
    # execution["OBSRP"].build_pipelines(dc)
    # runner.build_pipelines(data_card=dc)
    ranker = RankingEvaluatorDistance(output_dir="", instance_name="")
    trigger_map = build_trigger_map(cfg)
    req_policy = build_req_policy(cfg)
    learnable_problems = cfg.decision_engine.learnable_problems
    sim_control = DecisionEngine(execution=runner,
                                 selector=ranker,
                                 requirements_policies=req_policy,
                                 triggers=trigger_map,
                                 learnable_problems=learnable_problems,
                                 execution_map=execution
                                 )

    sim = WarehouseSimulation(
        state_transformers=state_transformers,
        control=sim_control,
        data_loader=loader,
        domain_cache_path=str(cache_path),
        loader_kwargs=loader_kwargs
    )

    def picker_arrival_hook(sim: WarehouseSimulation,
                            domain: SimWarehouseDomain):
        print("n_pickers", domain.resources.resources)
        min_order_date = 99999999999999
        for o in domain.orders.orders:
            if o.order_date < min_order_date:
                min_order_date = o.order_date
        for resource in domain.resources.resources:
            sim.add_event(PickerArrival(time=min_order_date,
                                        picker_id=resource.id))

    sim.reset(hooks=[picker_arrival_hook])
    start = time.time()
    sim.run()
    end = time.time()
    makespan = sim.state.tracker.final_makespan / 1000
    # window = cfg.simulation.conditions.OBRP.order_window

    print(f"[RESULT] | makespan={makespan:.2f} | runtime={end - start:.2f}s")

    return makespan

if __name__ == "__main__":
    main()
