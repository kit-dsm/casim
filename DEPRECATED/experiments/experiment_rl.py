import csv
import pickle
from abc import ABC

from pathlib import Path
from typing import Tuple

import hydra
import luigi
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from ware_ops_algos.algorithms import PlanningState, CombinedRoutingSolution, AssignmentSolution, RoundRobinAssigner, \
    PickList, AlgorithmSolution, Algorithm, I, O
from ware_ops_algos.domain_models import BaseWarehouseDomain, datacard_from_instance
from ware_ops_algos.domain_models.taxonomy import SUBPROBLEMS
from ware_ops_pipes.pipelines import set_pipeline_params

from experiments.experiment_commons import build_state_transformers, build_trigger_map, build_req_policy, OnlineRunner, \
    build_data_loader, make_execution, CoSyRunner
from ware_ops_sim.data_loaders import IWSPELoader
from ware_ops_sim.sim import WarehouseSimulation, SimWarehouseDomain
from ware_ops_sim.sim.conditions import ShiftStartNumbOrdersCondition, NumberOrdersCondition
from ware_ops_sim.sim.decision_engine.sim_control import DecisionEngine
from ware_ops_sim.sim.events.events import OrderArrival, RoutingDone, OrderPriorityChange, PickerArrival, \
    PickerTourQuery, \
    TourEnd, BreakStart, OrderSelectionDone, ShiftStart
from ware_ops_sim.sim.state.state_transformer import OnlineStateSnapshot
from ware_ops_algos.algorithms.algorithm_filter import AlgorithmFilter
from ware_ops_pipes.utils.experiment_utils import PipelineRunner, RankingEvaluatorDistance
from ware_ops_algos.utils.io_helpers import load_pickle, dump_pickle, dump_json

from ware_ops_sim.rl import WarehouseEnv


from sb3_contrib import MaskablePPO

def mask_fn(env: WarehouseEnv):
    return env.get_action_mask()

@hydra.main(config_path="config", config_name="rl_config")
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

    runner = OnlineRunner(instance_set_name=cfg.data_cards.name,
                          instances_dir=instances_dir,
                          cache_dir=cache_dir,
                          project_root=project_root,
                          instance_name=cfg.experiment.instance_name,
                          verbose=False)

    trigger_map = build_trigger_map(cfg)
    req_policy = build_req_policy(cfg)
    learnable_problems = cfg.simulation.learnable_problems
    execution = make_execution(cfg)

    sim_control = DecisionEngine(execution=runner,
                                 selector=ranker,
                                 requirements_policies=req_policy,
                                 triggers=trigger_map,
                                 learnable_problems=learnable_problems,
                                 execution_map=execution
                                 )


    def make_sim():
        loader_kwargs = {k: instances_dir / v
                         for k, v in cfg.data_cards.source.items()
                         if k.endswith("path")}
        loader = build_data_loader(cfg)
        domain = loader.load(**loader_kwargs)
        dc = datacard_from_instance(domain, "initial_dc")
        for key, value in execution.items():
            dc.problem_class = key
            if (isinstance(execution[key], OnlineRunner) or
                    isinstance(execution[key], CoSyRunner)):
                execution[key].build_pipelines(dc)
        sim = WarehouseSimulation(
            state_transformers=state_transformers,
            control=sim_control,
            data_loader=loader,
            domain_cache_path=str(cache_path),
            order_list_path=orders_path,
            order_line_path=layout_path,
            loader_kwargs=loader_kwargs
        )
        return sim

    def picker_arrival_hook(sim: WarehouseSimulation,
                            domain: SimWarehouseDomain):
        for resource in domain.resources.resources:
            sim.add_event(PickerArrival(time=0,
                                        picker_id=resource.id))

    def pre_shift_start_hook(sim: WarehouseSimulation,
                             domain: SimWarehouseDomain):
        sim.add_event(ShiftStart(time=0))

    run = wandb.init(
        # name=cfg.experiment.id,
        entity="j4b",
        project="ware_ops",
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,
        save_code=False,
        settings=wandb.Settings(console="off")
    )

    wandb_callback = WandbCallback(log="all")

    env = WarehouseEnv(
        sim_factory=make_sim,
        reset_hooks=[picker_arrival_hook, pre_shift_start_hook]
    )


    env = ActionMasker(env, action_mask_fn=mask_fn)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        tensorboard_log="./direct_actions_tensorboard/")

    model.learn(
        total_timesteps=363*3000,
        tb_log_name="./ppo_direct",
        callback=[wandb_callback],
        log_interval=1)

##########################################################################
#########################################################################
    # import cProfile
    # import pstats
    #
    # with cProfile.Profile() as pr:
    #     model.learn(
    #         total_timesteps=363,
    #         tb_log_name="./ppo_direct",
    #         callback=[wandb_callback],
    #         log_interval=1)
    # stats = pstats.Stats(pr)
    # stats.sort_stats('cumulative')
    # stats.print_stats(20)
    #
    # model.learn(
    #     total_timesteps=363 * 10,
    #     tb_log_name="./ppo_direct",
    #     callback=[wandb_callback],
    #     log_interval=1
    # )

    # Example with explicit algorithm from outside of sim

    # from ware_ops_algos.algorithms import GreedyOrderSelection
    #
    # selector = GreedyOrderSelection()
    #
    # obs, info = env.reset()
    # print(obs, info)
    # done = False
    # while not done:
    #     action = selector.solve(info["dynamic_info"].orders.orders)
    #     # TODO This should be instantiated and retrieved based on config for each problem
    #     obs, reward, done, _, info = env.step(action)

    # Run model inference
    # obs, _ = env.reset()
    # done = False
    # while not done:
    #     # action_masks = get_action_masks(env)
    #     action, _states = model.predict(obs, action_masks=mask_fn)
    #     obs, reward, done, truncated, info = env.step(action)

    # Run decision from outside

    # while not done:
    #     action = selector.solve(info["dynamic_info"].orders.orders)
    #     print("Action", action)
    #     # solution = action_to_solution(action, ctx)
    #     solution = action
    #     events = env.sim._order_selection_to_events(solution)
    #     for e in events:
    #         env.sim.add_event(e)
    #     problem = info["dynamic_info"].problem_class
    #     state_transformer = env.sim.state_transformers[problem]
    #     state_transformer.on_success(env.sim.state, solution)
    #     done, ctx = env.sim.run()

    # while not done:
    #     action = decision_maker.solve()
    #     env.step(action)

if __name__ == "__main__":
    main()
