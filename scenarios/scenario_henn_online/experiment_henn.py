import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.operational_events import (
    FlushRemainingOrders,
    NodeArrival,
    PickerArrival,
)
from casim.io_helpers import dump_json, dump_pickle
from casim.simulation_engine.simulation_engine import SimulationEngine
from casim.viz.app import launch
from scenarios.experiment_commons import (
    load_and_flatten_data_card,
    setup_decision_engine,
    setup_scenario,
)
from scenarios.scenario_henn_online.algorithm import (
    HennWakeUp,
    decide_henn,
    insert_into_active_tour,
    single_order_service_times,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def picker_arrival_hook(sim: SimulationEngine, domain: SimWarehouseDomain):
    first_order_time = min(order.order_date for order in domain.orders.orders)
    for resource in domain.resources.resources:
        sim.add_event(PickerArrival(first_order_time, resource.id))


def add_orders_hook(sim: SimulationEngine, domain: SimWarehouseDomain):
    for order in domain.orders.orders:
        sim.add_order(order)
    last_order_time = max(order.order_date for order in domain.orders.orders)
    sim.add_event(FlushRemainingOrders(last_order_time))


def _add_wakeup_once(sim: SimulationEngine, wait_until: float) -> None:
    already_pending = any(
        isinstance(event, HennWakeUp)
        and abs(event.time - wait_until) < 1e-9
        for event in sim.events
    )
    if not already_pending:
        sim.add_event(HennWakeUp(wait_until))


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="henn_online_config",
)
def main(cfg: DictConfig):
    """Run the Henn study as an explicit observe/decide/commit loop."""
    datacard = load_and_flatten_data_card(cfg.data_card)
    sim = setup_scenario(cfg)
    decision_engine = setup_decision_engine(cfg, datacard)

    # A wake-up is scenario policy, so it is registered here rather than in
    # CASIM's generic event registry.
    sim.triggers_map[HennWakeUp] = "OBRP"
    sim.reset(hooks=[add_orders_hook, picker_arrival_hook])

    service_time_cache: dict[int, float] = {}
    waiting_decisions: list[dict[str, object]] = []

    while True:
        done, snapshot = sim.run()
        if done:
            break

        active_tour_id = snapshot.dynamic_warehouse_info.active_tour_id
        if active_tour_id is not None:
            result = decision_engine.solve(snapshot)
            if result is None:
                version = sim.state.resume_active_tour(active_tour_id)
                sim.add_event(NodeArrival(
                    sim.state.current_time, active_tour_id, version
                ))
                continue
            candidate, _, _ = result
            tour = sim.state.tour_manager.get_tour(active_tour_id)
            picker = sim.state.resource_manager.get_resource(
                tour.assigned_resource
            )
            replacement = insert_into_active_tour(
                candidate,
                tour,
                picker.current_location,
                snapshot.layout,
            )
            if replacement is None:
                version = sim.state.resume_active_tour(active_tour_id)
                action = "resume"
            else:
                version = sim.state.commit_active_plan(
                    active_tour_id,
                    replacement,
                    picker_id=tour.assigned_resource,
                    expected_version=snapshot.dynamic_warehouse_info.route_version,
                )
                action = "insert"
            waiting_decisions.append({
                "time": sim.state.current_time,
                "action": action,
                "tour_id": active_tour_id,
                "route_version": version,
            })
            sim.add_event(NodeArrival(
                sim.state.current_time, active_tour_id, version
            ))
            continue

        result = decision_engine.solve(snapshot)
        if result is None:
            continue
        candidate, _, _ = result
        single_services = single_order_service_times(
            snapshot,
            decision_engine.get_solver(snapshot.problem_class),
            service_time_cache,
        )
        decision = decide_henn(
            candidate=candidate,
            snapshot=snapshot,
            current_time=sim.state.current_time,
            input_closed=sim.state.input_closed,
            selector=cfg.henn.selector,
            single_services=single_services,
            waiting_policy=cfg.henn.waiting_policy,
            fill_threshold=cfg.henn.fill_threshold,
            max_age_s=cfg.henn.max_age_s,
        )
        waiting_decisions.append({
            "time": sim.state.current_time,
            "action": decision.action,
            "reason": decision.reason,
            "wait_until": decision.wait_until,
            **decision.details,
        })

        if decision.action == "wait":
            _add_wakeup_once(sim, decision.wait_until)
            continue

        events, committed = decision_engine.commit(snapshot, decision.solution)
        sim.step(events, snapshot.problem_class, committed)

    output_dir = Path(cfg.experiment.output_dir)
    tracker = decision_engine.decision_tracker
    dump_pickle(str(output_dir / "decisions.pkl"), {
        "decisions": tracker.decisions,
        "pipeline_counts": dict(tracker.pipeline_counts),
    })
    dump_json(str(output_dir / "waiting_decisions.json"), waiting_decisions)

    if cfg.viz.launch:
        from casim.viz.gantt_chart import gantt_chart

        viz_dir = output_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        gantt_chart(sim.state.tracker).write_html(str(viz_dir / "gantt.html"))
        launch(viz_dir, port=cfg.viz.port, debug=False)


if __name__ == "__main__":
    main()
