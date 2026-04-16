from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from experiments.callbacks import CustomWandbCallback
from experiments.experiment_commons import build_state_transformers, build_trigger_map, build_req_policy, OnlineRunner, \
    build_data_loader
from ware_ops_sim.data_loaders import IWSPELoader
from ware_ops_sim.sim import WarehouseSimulation, SimWarehouseDomain
from ware_ops_sim.sim.conditions import ShiftStartNumbOrdersCondition, NumberOrdersCondition
from ware_ops_sim.sim.decision_engine.sim_control import DecisionEngine
from ware_ops_sim.sim.events.events import OrderArrival, RoutingDone, OrderPriorityChange, PickerArrival, \
    PickerTourQuery, \
    TourEnd, BreakStart, OrderSelectionDone
from ware_ops_sim.sim.state.state_transformer import OnlineStateSnapshot
from ware_ops_pipes.utils.experiment_utils import RankingEvaluatorDistance

from ware_ops_algos.domain_models import datacard_from_instance

from ware_ops_sim.rl import WarehouseEnvIndirect

from stable_baselines3 import DQN, PPO


@hydra.main(config_path="config", config_name="rl_indirect_config")
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

    loader_kwargs = {k: instances_dir / v
                     for k, v in cfg.data_cards.source.items()
                     if k.endswith("path")}

    loader = build_data_loader(cfg)
    domain = loader.load(**loader_kwargs)
    dc = datacard_from_instance(domain, "initial_iwspe")
    runner.build_pipelines(data_card=dc)

    trigger_map = build_trigger_map(cfg)
    req_policy = build_req_policy(cfg)
    learnable_problems = cfg.simulation.learnable_problems

    sim_control = DecisionEngine(execution=runner,
                                 selector=ranker,
                                 requirements_policies=req_policy,
                                 triggers=trigger_map,
                                 learnable_problems=learnable_problems
                                 )

    def make_sim():
        loader_kwargs = {k: instances_dir / v
                         for k, v in cfg.data_cards.source.items()
                         if k.endswith("path")}
        loader = build_data_loader(cfg)
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

    run = wandb.init(
        # name=cfg.experiment.id,
        entity="j4b",
        project="ware_ops",
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,
        save_code=False
    )

    env = WarehouseEnvIndirect(
        sim_factory=make_sim,
        reset_hooks=[picker_arrival_hook]
    )

    env = Monitor(env)
    # SubprocVecEnv
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # env = ActionMasker(env, action_mask_fn=mask_fn)
    # model = MaskablePPO(MaskableActorCriticPolicy,
    #                     env,
    #                     verbose=1,
    #                     tensorboard_log="./indirect_actions_tensorboard/",
    #                     n_steps=20)

    # model = DQN(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=1e-4,
    #     buffer_size=100_000,
    #     learning_starts=363,
    #     batch_size=64,
    #     train_freq=4,
    #     gradient_steps=4,
    #     target_update_interval=1000,
    #     exploration_fraction=0.15,
    #     exploration_final_eps=0.05,
    #     gamma=0.99,
    #     verbose=1,
    #     tensorboard_log="./indirect_actions_tensorboard/",
    # )

    model = PPO(
        "MlpPolicy",
        env,
        # n_steps=20,
        # n_epochs=4,
        # learning_rate=1e-4,
        verbose=1,
        tensorboard_log="./indirect_actions_tensorboard/",
    )
    wandb_callback = WandbCallback()
    # wandb_callback = CustomWandbCallback(
    #     # model_save_path=f"models/{run.id}",
    #     verbose=2,
    #     # log="all",  # Log all variables
    #     log_interval=1,
    # )

    model.learn(
        total_timesteps=363*30,
        tb_log_name="./dqn_indirect",
        callback=[wandb_callback],
        log_interval=1
        )


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
