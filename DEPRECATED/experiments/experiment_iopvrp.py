import csv
import pickle
import time
from abc import ABC

from pathlib import Path
from typing import Tuple

import hydra
import luigi
from hydra.utils import instantiate
from omegaconf import DictConfig

from ware_ops_algos.algorithms import PlanningState, CombinedRoutingSolution, AssignmentSolution, RoundRobinAssigner, \
    PickList, AlgorithmSolution, Algorithm, I, O, GreedyItemAssignment
from ware_ops_algos.domain_models import BaseWarehouseDomain, datacard_from_instance
from ware_ops_algos.domain_models.taxonomy import SUBPROBLEMS
from ware_ops_pipes.pipelines import set_pipeline_params

from experiments.experiment_commons import build_state_transformers, build_trigger_map, build_req_policy, \
    dump_pipelines_csv, OnlineRunner, build_data_loader, make_execution, CoSyRunner
from ware_ops_sim.data_loaders import IWSPELoader
from ware_ops_sim.sim import WarehouseSimulation, SimWarehouseDomain
from ware_ops_sim.sim.conditions import ShiftStartNumbOrdersCondition, NumberOrdersCondition
from ware_ops_sim.sim.decision_engine.sim_control import DecisionEngine
from ware_ops_sim.sim.events.events import OrderArrival, RoutingDone, OrderPriorityChange, PickerArrival, \
    PickerTourQuery, \
    TourEnd, BreakStart, ShiftStart
from ware_ops_sim.sim.state.state_transformer import OnlineStateSnapshot
from ware_ops_algos.algorithms.algorithm_filter import AlgorithmFilter
from ware_ops_pipes.utils.experiment_utils import PipelineRunner, RankingEvaluatorDistance
from ware_ops_algos.utils.io_helpers import load_pickle, dump_pickle, dump_json


@hydra.main(config_path="config", config_name="iopvrp_config")
def main(cfg: DictConfig):

    # Configuration
    project_root = Path(cfg.project_root)
    instances_dir = Path(cfg.instances_base) / cfg.data_cards.name
    cache_dir = Path(cfg.cache_base) / cfg.data_cards.name
    cache_path = Path(cfg.cache_base) / "dynamic_info.pkl"

    cache_dir.mkdir(parents=True, exist_ok=True)

    orders_path = instances_dir / "orders.csv"
    layout_path = instances_dir / "layout.csv"

    state_transformers = build_state_transformers(cfg)

    ranker = RankingEvaluatorDistance(output_dir="", instance_name="")
    execution = make_execution(cfg)
    runner = OnlineRunner(instance_set_name=cfg.data_cards.name,
                          instances_dir=instances_dir,
                          cache_dir=cache_dir,
                          project_root=project_root,
                          instance_name=cfg.experiment.instance_name,
                          verbose=True)

    loader_kwargs = {k: instances_dir / v
                     for k, v in cfg.data_cards.source.items()
                     if k.endswith("path")}

    loader = build_data_loader(cfg)

    domain = loader.load(**loader_kwargs)
    dc = datacard_from_instance(domain, "initial_dc")
    # dc.problem_class = "OBP"
    # execution["OBP"].build_pipelines(dc)
    # # dc.problem_class = "ORP"
    for key, value in execution.items():
        dc.problem_class = key
        if (isinstance(execution[key], OnlineRunner) or
                isinstance(execution[key], CoSyRunner)):
            execution[key].build_pipelines(dc)
    # execution["OBSRP"].build_pipelines(dc)
    # runner.build_pipelines(data_card=dc)

    trigger_map = build_trigger_map(cfg)
    req_policy = build_req_policy(cfg)
    learnable_problems = cfg.simulation.learnable_problems

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
        order_list_path=orders_path,
        order_line_path=layout_path,
        loader_kwargs=loader_kwargs
    )

    def pre_shift_start_hook(sim: WarehouseSimulation,
                             domain: SimWarehouseDomain):
        sim.add_event(ShiftStart(time=0))

    def picker_arrival_hook(sim: WarehouseSimulation,
                            domain: SimWarehouseDomain):
        print("n_pickers", domain.resources.resources)
        for resource in domain.resources.resources:
            sim.add_event(PickerArrival(time=1,
                                        picker_id=resource.id))

    sim.reset(hooks=[pre_shift_start_hook, picker_arrival_hook])
    start = time.time()
    sim.run()
    end = time.time()
    print("runtime", end-start)


    # import cProfile
    # import pstats
    #
    # with cProfile.Profile() as pr:
    #     sim.reset(hooks=[picker_arrival_hook, pre_shift_start_hook])
    #     sim.run()
    # stats = pstats.Stats(pr)
    # stats.sort_stats('cumulative')
    # stats.print_stats(20)




    # dump_pipelines_csv("./used_pipelines.csv", runner.used_pipelines)
    # dump_json("./used_pipelines.json", runner.used_pipelines)


if __name__ == "__main__":
    main()
