import copy
import csv
import pickle
import time

from pathlib import Path

import hydra
from omegaconf import DictConfig

from ware_ops_algos.domain_models import datacard_from_instance

from experiments.experiment_commons import build_state_transformers, build_trigger_map, build_req_policy, \
    dump_pipelines_csv, OnlineRunner, build_data_loader, make_execution, CoSyRunner

from ware_ops_sim.sim import WarehouseSimulation, SimWarehouseDomain
from ware_ops_sim.sim.decision_engine.sim_control import DecisionEngine
from ware_ops_sim.sim.events.events import OrderArrival, RoutingDone, OrderPriorityChange, PickerArrival, \
    PickerTourQuery, \
    TourEnd, BreakStart, ShiftStart
from ware_ops_sim.sim.state.state_transformer import OnlineStateSnapshot
from ware_ops_algos.algorithms.algorithm_filter import AlgorithmFilter
from ware_ops_pipes.utils.experiment_utils import PipelineRunner, RankingEvaluatorDistance
from ware_ops_algos.utils.io_helpers import load_pickle, dump_pickle, dump_json


def profile_deepcopy_components(sim):
    """Call this after sim.reset() + first sim.run() returns False."""
    state = sim.state

    components = {
        "layout_manager": state.layout_manager,
        "resource_manager": state.resource_manager,
        "order_manager": state.order_manager,
        "tour_manager": state.tour_manager,
        "storage_manager": state.storage_manager,
        "tracker": state.tracker,
        "_layout": state._layout,
        "_articles": state._articles,
        "_storage": state._storage,
        "_resources": state._resources,
        "events_heap": sim.events,
        "full_sim": sim,
    }

    results = {}
    for name, obj in components.items():
        t = time.perf_counter()
        for _ in range(5):
            _ = copy.deepcopy(obj)
        elapsed = (time.perf_counter() - t) / 5
        results[name] = elapsed
        print(f"{name:30s}: {elapsed * 1000:.1f} ms")

    return results

def profile_tour_manager_fields(sim):
    import copy, time, sys
    tm = sim.state.tour_manager
    for name, val in vars(tm).items():
        t = time.perf_counter()
        for _ in range(3):
            _ = copy.deepcopy(val)
        elapsed = (time.perf_counter() - t) / 3
        n = len(val) if hasattr(val, '__len__') else '?'
        print(f"  {name:35s}: {elapsed*1000:.1f} ms | len: {n}")


def profile_tour_internals(sim):
    import copy, time

    tm = sim.state.tour_manager

    # Sample a few tours
    sample_tours = list(tm.all_tours.values())[:5]

    print(f"Total tours: {len(tm.all_tours)}")
    print(f"Profiling fields of first tour:\n")

    for name, val in vars(sample_tours[0]).items():
        t = time.perf_counter()
        for _ in range(10):
            _ = copy.deepcopy(val)
        elapsed = (time.perf_counter() - t) / 10
        n = len(val) if hasattr(val, '__len__') else '?'
        print(f"  {name:35s}: {elapsed * 1000:.2f} ms | len: {n} | type: {type(val).__name__}")

    # Also check cost scales linearly with n_tours
    print("\nCost vs n_tours:")
    all_ids = list(tm.all_tours.keys())
    for n in [1, 10, 50, 100, 363]:
        subset = {k: tm.all_tours[k] for k in all_ids[:n]}
        t = time.perf_counter()
        for _ in range(3):
            _ = copy.deepcopy(subset)
        elapsed = (time.perf_counter() - t) / 3
        print(f"  {n:5d} tours: {elapsed * 1000:.1f} ms")


def profile_picklist(sim):
    import copy, time

    tm = sim.state.tour_manager
    # Get a tour with a non-None picklist
    sample_tour = list(tm.all_tours.values())[0]
    pl = sample_tour.pick_list

    print(f"PickList type: {type(pl)}")
    print(f"PickList fields:")
    for name, val in vars(pl).items():
        t = time.perf_counter()
        for _ in range(20):
            _ = copy.deepcopy(val)
        elapsed = (time.perf_counter() - t) / 20
        n = len(val) if hasattr(val, '__len__') else '?'
        print(f"  {name:35s}: {elapsed * 1000:.3f} ms | len: {n} | type: {type(val).__name__}")


def profile_remaining(sim):
    import copy, time

    for manager_name in ['order_manager', 'storage_manager', 'tracker']:
        obj = getattr(sim.state, manager_name)
        print(f"\n{manager_name}:")
        for name, val in vars(obj).items():
            t = time.perf_counter()
            for _ in range(5):
                _ = copy.deepcopy(val)
            elapsed = (time.perf_counter() - t) / 5
            n = len(val) if hasattr(val, '__len__') else '?'
            print(f"  {name:35s}: {elapsed * 1000:.2f} ms | len: {n} | type: {type(val).__name__}")


def verify_deepcopy_overrides(sim):
    import copy

    # Verify StorageManager override is called
    sm = sim.state.storage_manager
    sm_copy = copy.deepcopy(sm)
    print("StorageManager._storage same object:", sm._storage is sm_copy._storage)
    print("StorageManager._articles same object:", sm._articles is sm_copy._articles)

    # Verify SimulationState override is called
    state = sim.state
    state_copy = copy.deepcopy(state)
    print("SimulationState._storage same object:", state._storage is state_copy._storage)
    print("SimulationState.storage_manager._storage same object:",
          state.storage_manager._storage is state_copy.storage_manager._storage)

    # Verify tracker override
    print("tracker.processed_orders same object:",
          state.tracker.processed_orders is state_copy.tracker.processed_orders)
    print("tracker.logs is empty:", state_copy.tracker.logs == [])


def profile_remaining_cost(sim):
    import copy, time

    state = sim.state

    # Profile each manager that doesn't have a __deepcopy__ override
    print("Remaining mutable state cost:")

    components = {
        'layout_manager': state.layout_manager,
        'resource_manager': state.resource_manager,
        'tour_manager.active_only': {
            tid: t for tid, t in state.tour_manager.all_tours.items()
            if t.status.name != 'DONE'
        },
        'full_state': state,
        'full_sim': sim,
    }

    for name, obj in components.items():
        t = time.perf_counter()
        for _ in range(10):
            _ = copy.deepcopy(obj, {})
        elapsed = (time.perf_counter() - t) / 10
        print(f"  {name:35s}: {elapsed * 1000:.1f} ms")


def profile_sim_fields(sim):
    import copy, time

    print("WarehouseSimulation fields:")
    for name, val in sim.__dict__.items():
        if name == 'state':
            continue
        t = time.perf_counter()
        for _ in range(5):
            _ = copy.deepcopy(val)
        elapsed = (time.perf_counter() - t) / 5
        print(f"  {name:35s}: {elapsed * 1000:.1f} ms")

def estimate_mcts_budget(sim):
    import copy, time

    # Measure one full rollout from current point
    sim_copy = copy.deepcopy(sim)
    t = time.perf_counter()
    sim_copy.run()
    rollout_ms = (time.perf_counter() - t) * 1000

    copy_ms = 114
    iteration_ms = copy_ms + rollout_ms
    iterations_in_budget = int(180_000 / iteration_ms)

    print(f"Copy: {copy_ms}ms")
    print(f"Rollout: {rollout_ms:.0f}ms")
    print(f"Iteration total: {iteration_ms:.0f}ms")
    print(f"Iterations in 180s budget: {iterations_in_budget}")

@hydra.main(config_path="config", config_name="config")
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

    for key, value in execution.items():
        dc.problem_class = key
        if (isinstance(execution[key], OnlineRunner) or
                isinstance(execution[key], CoSyRunner)):
            execution[key].build_pipelines(dc)

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
            sim.add_event(PickerArrival(time=0,
                                        picker_id=resource.id))


    sim.reset(hooks=[pre_shift_start_hook, picker_arrival_hook])

    sim_copy = copy.deepcopy(sim)
    t = time.perf_counter()
    sim_copy.run()  # full rollout
    full_rollout_ms = (time.perf_counter() - t) * 1000
    print(f"Full rollout: {full_rollout_ms:.0f}ms")

    # Estimate shallow rollout (3 tours)
    per_tour_ms = full_rollout_ms / sim_copy.state.tracker.finished_tours
    print(f"Per tour: {per_tour_ms:.1f}ms")
    print(f"3-tour rollout estimate: {per_tour_ms * 3:.0f}ms")

    # MCTS budget
    for n_tours in [1, 3, 5, 10]:
        iteration_ms = 8.5 + per_tour_ms * n_tours
        iterations = int(180_000 / iteration_ms)
        print(f"n={n_tours} tours: {iteration_ms:.0f}ms/iter → {iterations} iterations")

    # profile_sim_fields(sim)
    # print(hasattr(WarehouseSimulation, '__deepcopy__'))
    #
    # # Also verify it's being called
    # original = WarehouseSimulation.__deepcopy__
    #
    # def patched(self, memo):
    #     print("WarehouseSimulation.__deepcopy__ called")
    #     return original(self, memo)
    #
    # WarehouseSimulation.__deepcopy__ = patched
    #
    # _ = copy.deepcopy(sim)
    # estimate_mcts_budget(sim)
    #
    #
    # sim.run()  # get to first decision point
    #
    # t = time.time()
    # profile_deepcopy_components(sim)
    # # for _ in range(10):
    # #     _ = copy.deepcopy(sim)
    # print(f"deepcopy: {(time.time() - t) / 10:.3f}s per copy")
    #
    # profile_tour_manager_fields(sim)
    #
    # profile_tour_internals(sim)
    #
    # profile_picklist(sim)
    #
    # profile_remaining_cost(sim)


    # start = time.time()
    # sim.run()
    # end = time.time()
    # print("runtime", end-start)


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
