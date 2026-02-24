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
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from ware_ops_algos.algorithms import PlanningState, CombinedRoutingSolution, AssignmentSolution, RoundRobinAssigner, \
    PickList, AlgorithmSolution, Algorithm, I, O
from ware_ops_algos.domain_models import BaseWarehouseDomain
from ware_ops_algos.domain_models.taxonomy import SUBPROBLEMS
from ware_ops_pipes.pipelines import set_pipeline_params

from experiments.experiment_commons import build_state_transformers, build_trigger_map, build_req_policy, OnlineRunner
from ware_ops_sim.data_loaders import IWSPELoader
from ware_ops_sim.sim import WarehouseSimulation, SimWarehouseDomain
from ware_ops_sim.sim.conditions import ShiftStartNumbOrdersCondition, NumberOrdersCondition
from ware_ops_sim.sim.decision_engine.sim_control import DecisionEngine
from ware_ops_sim.sim.events.events import OrderArrival, RoutingDone, OrderPriorityChange, PickerArrival, \
    PickerTourQuery, \
    TourEnd, BreakStart, OrderSelectionDone
from ware_ops_sim.sim.state.state_transformer import OnlineStateTransformer
from ware_ops_algos.algorithms.algorithm_filter import AlgorithmFilter
from ware_ops_pipes.utils.experiment_utils import PipelineRunner, RankingEvaluatorDistance
from ware_ops_algos.utils.io_helpers import load_pickle, dump_pickle, dump_json

from ware_ops_sim.rl import WarehouseEnv


from sb3_contrib import MaskablePPO

# class OnlineRunner(PipelineRunner):
#     """Runner for IOPVRP instances (paired files)"""
#     def __init__(self, instance_set_name: str, instances_dir: Path, cache_dir: Path,
#                  project_root: Path, instance_name: str, **loader_kwargs):
#         runner_kwargs = {k: v for k, v in loader_kwargs.items()
#                          if k in ['max_pipelines', 'verbose', 'cleanup']}
#
#         super().__init__(instance_set_name, instances_dir, cache_dir, project_root, **runner_kwargs)
#         self.instance_name = instance_name
#         self.run_id = -1
#         self.used_pipelines = {}
#         self.implementation_module = {
#             "GreedyIA": "ware_ops_pipes.pipelines.components.item_assignment.greedy_item_assignment",
#             # "NNIA": "ware_ops_pipes.pipelines.components.item_assignment.nn_item_assignment",
#             # "SinglePosIA": "ware_ops_pipes.pipelines.components.item_assignment.single_pos_item_assignment",
#             # "MinMinIA": "ware_ops_pipes.pipelines.components.item_assignment.min_min_item_assignment",
#             # "MinMaxIA": "ware_ops_pipes.pipelines.components.item_assignment.min_max_item_assignment",
#             "DummyOS": "ware_ops_pipes.pipelines.components.order_selection.dummy_order_selection",
#             # "MinMaxArticlesOS": "ware_ops_pipes.pipelines.components.order_selection.min_max_articles_os",
#             # "MinMaxAislesOS": "ware_ops_pipes.pipelines.components.order_selection.min_max_aisles_os",
#             "GreedyOS": "ware_ops_pipes.pipelines.components.order_selection.greedy_order_selection",
#             # "MinAisleConflictsOS": "ware_ops_pipes.pipelines.components.order_selection.min_aisle_conflicts_os",
#             # "MinDistOS": "ware_ops_pipes.pipelines.components.order_selection.min_dist_os",
#             # "MinSharedAislesOS": "ware_ops_pipes.pipelines.components.order_selection.min_shared_aisles_os",
#             # "SShape": "ware_ops_pipes.pipelines.components.routing.s_shape",
#             # "NearestNeighbourhood": "ware_ops_pipes.pipelines.components.routing.nn",
#             "PLRouting": "ware_ops_pipes.pipelines.components.routing.pl",
#             # "LargestGap": "ware_ops_pipes.pipelines.components.routing.largest_gap",
#             # "Midpoint": "ware_ops_pipes.pipelines.components.routing.midpoint",
#             # "Return": "ware_ops_pipes.pipelines.components.routing.return_algo",
#             # "ExactSolving": "ware_ops_pipes.pipelines.components.routing.exact_algo",
#             # "RatliffRosenthal": "ware_ops_pipes.pipelines.components.routing.sprp",
#             # "FiFo": "ware_ops_pipes.pipelines.components.batching.fifo",
#             # "OrderNrFiFo": "ware_ops_pipes.pipelines.components.batching.order_nr_fifo",
#             # "DueDate": "ware_ops_pipes.pipelines.components.batching.due_date",
#             # "Random": "ware_ops_pipes.pipelines.components.batching.random",
#             # "CombinedBatchingRoutingAssigning": "ware_ops_pipes.pipelines.components.routing.joint_batching_routing_assigning",
#             # "ClosestDepotMinDistanceSeedBatching": "ware_ops_pipes.pipelines.components.batching.seed",
#             # "ClosestDepotMaxSharedArticlesSeedBatching": "ware_ops_pipes.pipelines.components.batching.seed_shared_articles",
#             # "ClarkAndWrightSShape": "ware_ops_pipes.pipelines.components.batching.clark_and_wright_sshape",
#             # "ClarkAndWrightNN": "ware_ops_pipes.pipelines.components.batching.clark_and_wright_nn",
#             # "ClarkAndWrightRR": "ware_ops_pipes.pipelines.components.batching.clark_and_wright_rr",
#             # "LSBatchingRR": "ware_ops_pipes.pipelines.components.batching.ls_rr",
#             # "LSBatchingNNRand": "ware_ops_pipes.pipelines.components.batching.ls_nn_rand",
#             # "LSBatchingNNDueDate": "ware_ops_pipes.pipelines.components.batching.ls_nn_due",
#             # "LSBatchingNNFiFo": "ware_ops_pipes.pipelines.components.batching.ls_nn_fifo",
#             # "SPTScheduling": "ware_ops_pipes.pipelines.components.sequencing.spt_scheduling",
#             # "LPTScheduling": "ware_ops_pipes.pipelines.components.sequencing.lpt_scheduling",
#             "EDDScheduling": "ware_ops_pipes.pipelines.components.sequencing.edd_scheduling",
#             # "EDDSequencing": "ware_ops_pipes.pipelines.components.sequencing.edd_sequencing",
#             # "RRAssigner": "ware_ops_pipes.pipelines.components.picker_assignment.round_robin_assignment"
#         }
#         self.pipelines = None
#
#     def dump_domain(self, dynamic_domain: BaseWarehouseDomain):
#         dump_pickle(str(self.cache_path), dynamic_domain)
#
#     def load_domain(self, instance_name: str, file_paths: list[Path]) -> BaseWarehouseDomain:
#         return load_pickle(str(self.cache_path))
#
#     def discover_instances(self) -> list[Tuple[str, list[Path]]]:
#         pass
#
#
#     def solve(self, dynamic_domain: BaseWarehouseDomain) -> AlgorithmSolution:
#         self.dump_domain(dynamic_domain)
#
#         # Filter applicable algorithms
#         algo_filter = AlgorithmFilter(SUBPROBLEMS)
#         models_applicable = algo_filter.filter(
#             algorithms=self.models,
#             instance=dynamic_domain,
#             verbose=self.verbose
#         )
#         self.run_id += 1
#         self._import_models(models_applicable)
#         dynamic_instance_name = self.instance_name + "_" + str(self.run_id)
#         output_folder = (
#                 self.project_root / "experiments" / "online" / "output"
#                 / dynamic_instance_name
#         )
#         output_folder.mkdir(parents=True, exist_ok=True)
#
#         set_pipeline_params(
#             output_folder=str(output_folder),
#             instance_set_name=self.instance_set_name,
#             instance_name=dynamic_instance_name,
#             instance_path="",
#             domain_path=str(self.cache_path)
#         )
#
#         # Build and run pipelines
#         if not self.pipelines:
#             print("Building pipelines")
#             self.pipelines = self._build_pipelines()
#             print(self.pipelines)
#         else:
#             print("Using cached pipelines")
#
#         if self.pipelines:
#             print(f"\n✓ Running {len(self.pipelines)} pipelines...\n")
#             luigi.interface.InterfaceLogging.setup(type('opts',
#                                                         (),
#                                                         {'background': None,
#                                                          'logdir': None,
#                                                          'logging_conf_file': None,
#                                                          'log_level': 'CRITICAL'  # <<<<<<<<<<
#                                                          }))
#             luigi.build([self.pipelines[0]], local_scheduler=True)
#
#             if self.cleanup:
#                 self._cleanup(output_folder)
#
#             sequencing_plans = self.load_sequencing_solutions(output_folder)
#             routing_plans = self.load_routing_solutions(output_folder)
#
#             # based on the resulting plans retrieve the best solution via simple ranking best -> worst
#             if dynamic_domain.problem_class in ["OBSRP", "OSRP"]:
#                 best_key, best_dist = None, float("inf")
#                 for k, plan in sequencing_plans.items():
#                     plan: PlanningState
#                     dist = sum(a.distance for a in plan.sequencing_solutions.jobs)
#                     if dist < best_dist:
#                         best_key, best_dist = k, dist
#                 print(sequencing_plans)
#                 print("best key", best_key)
#                 # print(sequencing_plans[best_key].provenance["instance_solving"]["algo"])
#                 solution_object = sequencing_plans[best_key].sequencing_solutions
#                 for o in dynamic_domain.orders.orders:
#                     self.used_pipelines[o.order_id] = sequencing_plans[best_key].provenance
#
#             elif dynamic_domain.problem_class in ["OBRP", "SPRP", "ORP"]:
#                 best_key, best_dist = None, float("inf")
#                 for k, plan in routing_plans.items():
#                     plan: PlanningState
#                     dist = sum(r.route.distance for r in plan.routing_solutions)
#                     if dist < best_dist:
#                         best_key, best_dist = k, dist
#                 solution = routing_plans[best_key].routing_solutions
#                 routes = []
#                 for r in solution:
#                     routes.append(r.route)
#                 solution_object = CombinedRoutingSolution(routes=routes)
#                 for o in dynamic_domain.orders.orders:
#                     self.used_pipelines[o.order_id] = routing_plans[best_key].provenance
#             else:
#                 raise ValueError
#             return solution_object
#         else:
#             print("⚠ No valid pipelines found!")
#
#     @staticmethod
#     def load_routing_solutions(base_dir: str):
#         sol_files = Path(base_dir).glob("**/*routing_plan.pkl")
#         solutions = {}
#         for f in sol_files:
#             with open(f, "rb") as fh:
#                 try:
#                     solutions[f.name] = pickle.load(fh)
#                 except Exception as e:
#                     print(f"❌ Failed to load {f}: {e}")
#         return solutions
#
#     @staticmethod
#     def load_sequencing_solutions(base_dir: str):
#         sol_files = Path(base_dir).glob("**/*scheduling_plan.pkl")
#         solutions = {}
#         for f in sol_files:
#             with open(f, "rb") as fh:
#                 try:
#                     solutions[f.name] = pickle.load(fh)
#                 except Exception as e:
#                     print(f"❌ Failed to load {f}: {e}")
#         return solutions

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

    sim_control = DecisionEngine(execution=runner,
                                 selector=ranker,
                                 requirements_policies=req_policy,
                                 triggers=trigger_map,
                                 learnable_problems=learnable_problems
                                 )

    def make_sim():
        sim = WarehouseSimulation(
            state_transformers=state_transformers,
            control=sim_control,
            data_loader=IWSPELoader(
                instances_dir=instances_dir,
                cfg=cfg),
            domain_cache_path=str(cache_path),
            order_list_path=orders_path,
            order_line_path=layout_path,
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

    # env = WarehouseEnv(
    #     sim_factory=make_sim,
    #     reset_hooks=[picker_arrival_hook]
    # )
    #
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
    wandb_callback = WandbCallback(log="all")




    # env = ActionMasker(env, action_mask_fn=mask_fn)
    # env = Monitor(env)
    # env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)

    def make_env(mask_fn):
        def _init():
            env = WarehouseEnv(
                sim_factory=make_sim,
                reset_hooks=[picker_arrival_hook]
            )
            env = ActionMasker(env, action_mask_fn=mask_fn)
            env = Monitor(env)
            return env

        return _init

    n_envs = 8  # start conservative, you have 20 logical cores
    env = SubprocVecEnv([make_env(mask_fn) for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        n_epochs=1,
        n_steps=363,
        # n_epochs=4,
        # learning_rate=1e-4,
        tensorboard_log="./direct_actions_tensorboard/")

    model.learn(
        total_timesteps=363,
        tb_log_name="./ppo_direct",
        callback=[wandb_callback],
        log_interval=1)

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
