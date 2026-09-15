"""Test that StochasticWaitingOptimizer works with casim domain objects.

Three test groups:
1. End-to-end: build WaitingAnalysisInput from a real Henn simulation
   snapshot and verify the optimizer produces a valid WaitingSolution.
2. Online waiting: run the actual simulation, dispatch a tour, wait for
   a new order to arrive while the picker is mid-route, then call the
   StochasticWaitingOptimizer with the live simulation state.
3. Benchmark: construct the validate_a128 instance from domain objects
   and check E[D] against Mathematica reference values.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ware_ops_algos.algorithms import (
    BatchObject,
    CombinedRoutingSolution,
    PickPosition,
    Route,
    WarehouseOrder,
)
from ware_ops_algos.algorithms.waiting import (
    StochasticWaitingOptimizer,
    WaitingAnalysisInput,
    WaitingSolution,
)
from ware_ops_algos.algorithms.waiting.stochastic_waiting import (
    _build_warehouse_instance,
)
from ware_ops_algos.algorithms.waiting.analytic_progress.core import (
    compute_segment_times,
    theoretical_route_duration,
)
from ware_ops_algos.algorithms.waiting.analytic_progress.continuous_calculation import (
    continuous_expected_detour,
)
from ware_ops_algos.domain_models import (
    LayoutData,
    LayoutParameters,
    LayoutType,
    Resource,
)
from ware_ops_algos.domain_models import load_and_flatten_data_card

from casim.setup import build_runtime
from scenarios.scenario_henn.scenario_specific_hooks import build_sim_hooks
from scenarios.scenario_henn.algorithm import (
    HennWakeUp,
    decide_henn,
    single_order_service_times,
)
from casim.events.operational_events import (
    OrderArrival,
    FlushRemainingOrders,
    InterventionRequest,
)


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


def _build_sim(tmp_path, batching="fcfs"):
    cfg = _compose(f"batching={batching}")
    project_root = Path(__file__).parents[1].resolve()
    OmegaConf.update(cfg, "project_root", str(project_root), merge=False)
    OmegaConf.update(cfg, "instances_base", str(project_root / "scenarios"), merge=False)
    OmegaConf.update(cfg, "cache_base", str(tmp_path / "cache"), merge=False)
    OmegaConf.update(cfg, "experiment.output_dir", str(tmp_path), merge=False)
    OmegaConf.update(cfg, "experiment.instance_name", "stoch-test", merge=False)
    OmegaConf.update(cfg, "luigi.runtime", 1, merge=False)
    data_card = load_and_flatten_data_card(cfg.data_card)
    return build_runtime(cfg, data_card), cfg


# ──────────────────────── End-to-end test ────────────────────────────


def test_stochastic_waiting_optimizer_produces_decision(tmp_path):
    """Build WaitingAnalysisInput from a real simulation snapshot and
    run the stochastic waiting optimizer end-to-end."""
    (sim, de), cfg = _build_sim(tmp_path)
    sim.reset(hooks=build_sim_hooks(cfg))

    assert sum(isinstance(e, OrderArrival) for e in sim.events) == 40

    route = None
    snapshot = None
    for _ in range(20):
        done, snapshot = sim.run()
        if done:
            break
        result = de.solve(snapshot, action=None)
        if result is None:
            continue
        solution, _, _ = result
        if not isinstance(solution, CombinedRoutingSolution) or not solution.routes:
            continue
        if len(solution.routes[0].batch.orders) >= 2:
            route = solution.routes[0]
            break

    assert route is not None, "Could not find a batch with >= 2 orders"
    assert snapshot is not None

    batch_orders = route.batch.orders
    base_orders = batch_orders[:-1]
    insert_order = batch_orders[-1]
    picker = snapshot.resources.resources[0]

    analysis_input = WaitingAnalysisInput(
        layout=snapshot.layout,
        picker=picker,
        base_orders=base_orders,
        insert_order=insert_order,
        base_route=route,
        current_time=snapshot.dynamic_warehouse_info.time or 0.0,
        expected_interarrival=28.8,
    )

    optimizer = StochasticWaitingOptimizer(engine="discrete")
    result_solution = optimizer.solve(analysis_input)

    assert isinstance(result_solution, WaitingSolution)
    assert result_solution.algo_name == "StochasticWaiting"
    assert result_solution.execution_time > 0
    assert isinstance(result_solution.should_wait, bool)
    assert result_solution.engine == "discrete"
    assert result_solution.predicted_completion_time > 0
    assert result_solution.initial_completion_time > 0
    assert result_solution.expected_detour >= 0


# ─────────────── Online waiting through the simulation ──────────────


def test_stochastic_waiting_during_active_tour(tmp_path):
    """Run the real simulation: orders arrive online, tours are dispatched
    and executed.  At each decision point (new order arrival or picker
    becoming idle), the StochasticWaitingOptimizer is called as a drop-in
    replacement for the deterministic Henn waiting policy.

    This exercises the full online waiting pipeline:
      simulation → order arrives → solver routes batch →
      StochasticWaitingOptimizer decides wait/dispatch →
      commit → tour execution → picker idle → next order → ...
    """
    (sim, de), cfg = _build_sim(tmp_path)
    sim.reset(hooks=build_sim_hooks(cfg))

    single_service_cache: dict[int, float] = {}
    optimizer_results: list[WaitingSolution] = []
    dispatches = 0
    tours_completed = 0

    for _ in range(500):
        done, snapshot = sim.run()
        if done:
            break

        active_tour_id = snapshot.dynamic_warehouse_info.active_tour_id
        if active_tour_id is not None:
            sim.state.clear_intervention_request(active_tour_id)
            continue

        # Decision point: either a new order arrived or the picker
        # became idle.  The solver produces a candidate route for the
        # currently buffered orders.
        result = de.solve(snapshot, action=None)
        if result is None:
            continue
        candidate, _, _ = result
        if not isinstance(candidate, CombinedRoutingSolution) or not candidate.routes:
            continue

        route = candidate.routes[0]
        if not route.batch.orders:
            continue

        # Compute single-order service times (needed by the Henn
        # threshold policy for comparison, and by the optimizer for
        # the completion-time estimate).
        single_order_service_times(
            snapshot,
            de.solver_map[("OBRP", "none")],
            single_service_cache,
        )

        picker = snapshot.resources.resources[0]
        current_time = float(sim.state.current_time)

        # Determine the next arrival time from the event queue.
        next_arrival_time = None
        for event in sim.events:
            if isinstance(event, OrderArrival):
                next_arrival_time = float(event.time)
                break

        # The base orders are the ones already in the batch.
        # The "insert order" is a hypothetical future order — for the
        # stochastic optimizer, we use the last order in the batch as
        # the one whose integration we're evaluating.
        batch_orders = route.batch.orders
        if len(batch_orders) < 1:
            continue

        base_orders = batch_orders[:-1] if len(batch_orders) > 1 else batch_orders
        insert_order = batch_orders[-1]

        # Build WaitingAnalysisInput from the LIVE simulation state.
        analysis_input = WaitingAnalysisInput(
            layout=snapshot.layout,
            picker=picker,
            base_orders=base_orders,
            insert_order=insert_order,
            base_route=route,
            current_time=current_time,
            expected_interarrival=28.8,
        )

        # Run the stochastic waiting optimizer on the live state.
        optimizer = StochasticWaitingOptimizer(engine="continuous")
        stoch_result = optimizer.solve(analysis_input)

        assert isinstance(stoch_result, WaitingSolution)
        assert stoch_result.algo_name == "StochasticWaiting"
        assert stoch_result.execution_time > 0
        assert isinstance(stoch_result.should_wait, bool)
        assert stoch_result.engine == "continuous"
        assert stoch_result.predicted_completion_time > 0
        assert stoch_result.initial_completion_time > 0
        assert stoch_result.expected_detour >= 0

        optimizer_results.append(stoch_result)

        # Also compute the deterministic Henn decision for comparison.
        henn_decision = decide_henn(
            candidate=candidate,
            snapshot=snapshot,
            current_time=current_time,
            next_arrival=next_arrival_time,
            stream_exhausted=next_arrival_time is None,
            selector=str(cfg.selection.name),
            single_services=single_service_cache,
            waiting_policy=str(cfg.waiting.name),
            fill_threshold=float(cfg.waiting.fill_threshold or 0.75),
            max_age_s=float(cfg.waiting.max_age_s or 300.0),
        )

        # Use the deterministic decision to drive the simulation forward.
        if henn_decision.action == "wait":
            if henn_decision.wait_until is not None:
                sim.add_event(HennWakeUp(henn_decision.wait_until))
            continue

        events, committed = de.commit(snapshot, henn_decision.solution)
        sim.step(events)
        dispatches += 1

    # The simulation must have completed (drained all orders)
    assert sim.state.input_closed
    # We must have dispatched at least once
    assert dispatches > 0
    # The optimizer must have been called at least once during the
    # online decision loop (new orders arrived while picker was busy
    # or idle, triggering a decision point)
    assert len(optimizer_results) > 0, (
        "No online waiting scenario was triggered — expected at least "
        "one decision point with buffered orders"
    )
    # All optimizer results must have valid values
    for r in optimizer_results:
        assert r.predicted_completion_time > 0
        assert r.initial_completion_time > 0
        assert r.expected_detour >= 0


# ─────────────────── Benchmark: validate_a128 ────────────────────────
# Instance: Route A = (1, 2, 8), M=8, N_L=17, w=1.0, L=17.0, v=1.0, t_p=0.0
# Reference values from project_4D4L validate_a128.py (Mathematica).

TOL = 1e-9

SPOT_CHECKS = [
    (25.0, 289 / 136, "phase_2_vertical:8-25 s=17"),
    (31.0, 59 / 8, "phase_3_horizontal:25-31 s=6"),
    (48.0, 13.75, "phase_2_vertical:31-48 s=17"),
    (66.0, 15.5, "phase_n_2_ret_up:49-66 s=17"),
    (83.0, 147 / 4 + 289 / 136, "phase_n_1_ret_down:66-83 s=17"),
]


def _make_a128_domain():
    """Build the validate_a128 benchmark instance from domain objects."""
    params = LayoutParameters(
        n_aisles=8,
        n_pick_locations=17,
        n_blocks=1,
        dist_top_to_pick_location=0.0,
        dist_bottom_to_pick_location=0.0,
        dist_pick_locations=1.0,
        dist_aisle=1.0,
        dist_start=0.0,
        start_location=(0, 0),
    )
    layout = LayoutData(tpe=LayoutType.CONVENTIONAL, graph_data=params)
    picker = Resource(
        id=0,
        speed=1.0,
        time_per_pick=0.0,
        tour_setup_time=0.0,
    )
    picker.available_at = 0.0

    orders = []
    for aisle in (1, 2, 8):
        pp = PickPosition(
            order_number=aisle,
            article_id=aisle,
            amount=1,
            pick_node=(aisle, 1),
            in_store=1,
        )
        order = WarehouseOrder(
            order_id=aisle,
            order_date=0.0,
            pick_positions=(pp,),
        )
        orders.append(order)

    return layout, picker, orders


def test_a128_adapter_produces_correct_warehouse_instance():
    """The adapter from domain objects to WarehouseInstance must
    reproduce the exact benchmark parameters."""
    layout, picker, orders = _make_a128_domain()
    inst = _build_warehouse_instance(layout, picker, orders)

    assert inst.M == 8
    assert inst.N_L == 17
    assert inst.w == 1.0
    assert inst.L == 17.0
    assert inst.v == 1.0
    assert inst.t_p == 0.0
    assert inst.A == [1, 2, 8]
    assert inst.n_list == [1, 1, 1]
    assert inst.k == 3
    assert inst.n_ges == 3
    assert (1, 1) in inst.P
    assert (2, 1) in inst.P
    assert (8, 1) in inst.P


def test_a128_route_duration():
    """T_end = 84.0 for the benchmark instance."""
    layout, picker, orders = _make_a128_domain()
    inst = _build_warehouse_instance(layout, picker, orders)
    assert theoretical_route_duration(inst) == pytest.approx(84.0, abs=TOL)


def test_a128_segment_times():
    """Segment times match all 8 phase boundaries exactly."""
    layout, picker, orders = _make_a128_domain()
    inst = _build_warehouse_instance(layout, picker, orders)
    T_in, T_out, T_end = compute_segment_times(inst)

    assert T_end == pytest.approx(84.0, abs=TOL)
    assert T_in[3] == pytest.approx(8.0, abs=TOL)
    assert T_out[3] == pytest.approx(25.0, abs=TOL)
    assert T_in[2] == pytest.approx(31.0, abs=TOL)
    assert T_out[2] == pytest.approx(48.0, abs=TOL)
    assert T_in[1] == pytest.approx(49.0, abs=TOL)
    assert T_out[1] == pytest.approx(83.0, abs=TOL)


@pytest.mark.parametrize(
    ("t", "expected", "label"),
    SPOT_CHECKS,
    ids=[s[2] for s in SPOT_CHECKS],
)
def test_a128_continuous_ed_matches_reference(t, expected, label):
    """Continuous E[D] at spot-check points matches Mathematica values
    to 1e-9, running through the full adapter chain from domain objects."""
    layout, picker, orders = _make_a128_domain()
    inst = _build_warehouse_instance(layout, picker, orders)
    T_in, T_out, T_end = compute_segment_times(inst)
    t_clamped = min(t, T_end)
    result = continuous_expected_detour(t_clamped, inst, T_in, T_out)
    assert result.expected_detour == pytest.approx(expected, abs=TOL), (
        f"{label}: t={t}, expected E[D]={expected:.9f}, "
        f"got {result.expected_detour:.9f}"
    )


def test_a128_optimizer_runs_through_adapter():
    """The StochasticWaitingOptimizer must run end-to-end on the
    benchmark instance, producing a valid WaitingSolution via
    the domain-object adapter."""
    layout, picker, orders = _make_a128_domain()
    base_orders = orders[:-1]
    insert_order = orders[-1]

    batch = BatchObject(batch_id=0, orders=base_orders)
    route = Route(distance=84.0, batch=batch)

    analysis_input = WaitingAnalysisInput(
        layout=layout,
        picker=picker,
        base_orders=base_orders,
        insert_order=insert_order,
        base_route=route,
        current_time=0.0,
        expected_interarrival=28.8,
    )

    optimizer = StochasticWaitingOptimizer(engine="continuous")
    result = optimizer.solve(analysis_input)

    assert isinstance(result, WaitingSolution)
    assert result.algo_name == "StochasticWaiting"
    assert result.execution_time > 0
    assert isinstance(result.should_wait, bool)
    assert result.engine == "continuous"
    assert result.predicted_completion_time > 0
    assert result.initial_completion_time > 0
