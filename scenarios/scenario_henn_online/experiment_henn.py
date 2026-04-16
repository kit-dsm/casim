import cProfile
import logging
import pstats
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig
from ware_ops_algos.domain_models import datacard_from_instance

from casim.decision_engine.decision_engine import DecisionEngine
from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.operational_events import PickerArrival
from casim.pipelines.objective_evaluator import ObjectiveEvaluator
from casim.pipelines.pipeline_runner import build_pipeline_runner, build_trigger_map
from casim.simulation_engine import SimulationEngine
from casim.state.conditions import build_condition_policies
from casim.state.state_snapshot import build_state_snapshots

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

@hydra.main(config_path="config", config_name="henn_online_config")
def main(cfg: DictConfig):

    # Configuration
    instances_dir = Path(cfg.instances_base) / cfg.data_card.name
    cache_dir = Path(cfg.cache_base) / cfg.data_card.name
    cache_path = Path(cfg.cache_base) / "dynamic_info.pkl"
    cache_dir.mkdir(parents=True, exist_ok=True)


    state_transformers = build_state_snapshots(cfg)

    pipeline_runner = build_pipeline_runner(cfg)

    loader_kwargs = {k: instances_dir / v
                     for k, v in cfg.data_card.source.items()
                     if k.endswith("path")}

    loader = build_data_loader(cfg)

    domain = loader.load(**loader_kwargs)
    dc = datacard_from_instance(domain, "initial_dc")

    for key, value in pipeline_runner.items():
        dc.problem_class = key
        pipeline_runner[key].build_pipelines(dc)

    trigger_map = build_trigger_map(cfg)
    req_policy = build_condition_policies(cfg)

    evaluators_map = {"OBRSP": ObjectiveEvaluator(objective="makespan")}

    sim_control = DecisionEngine(conditions_map=req_policy,
                                 trigger_map=trigger_map,
                                 execution_map=execution,
                                 evaluators_map=evaluators_map
                                 )

    sim = SimulationEngine(
        state_transformers=state_transformers,
        control=sim_control,
        data_loader=loader,
        domain_cache_path=str(cache_path),
        loader_kwargs=loader_kwargs
    )

    def picker_arrival_hook(sim: SimulationEngine,
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
