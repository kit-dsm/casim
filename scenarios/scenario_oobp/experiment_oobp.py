import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.operational_events import OrderStreamClosed, PickerArrival
from casim.io_helpers import dump_pickle
from casim.runner import apply_decision
from casim.simulation_engine.simulation_engine import SimulationEngine
from scenarios.experiment_commons import load_and_flatten_data_card, setup_experiment

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def picker_arrival_hook(sim: SimulationEngine,
                        domain: SimWarehouseDomain):
    min_order_date = min(order.order_date for order in domain.orders.orders)
    for resource in domain.resources.resources:
        sim.add_event(PickerArrival(time=min_order_date,
                                    picker_id=resource.id,
                                    picker_available=True))


def add_orders_hook(sim: SimulationEngine,
                    domain: SimWarehouseDomain):
    for order in domain.orders.orders:
        sim.add_order(order)
    sim.add_event(OrderStreamClosed(max(order.order_date for order in domain.orders.orders)))


@hydra.main(config_path="config", config_name="oobp_config")
def main(cfg: DictConfig):
    datacard = load_and_flatten_data_card(cfg.data_card)

    sim, decision_engine = setup_experiment(cfg, datacard)

    sim.reset(hooks=[add_orders_hook,
                     picker_arrival_hook,
                     ])

    done = False
    while not done:
        done, state_snapshot = sim.run()
        if done:
            dt = decision_engine.decision_tracker
            dump_pickle(str(Path(cfg.experiment.output_dir) / "decisions.pkl"), {
                "decisions": dt.decisions,
                "pipeline_counts": dict(dt.pipeline_counts),
            })
            print(f"'Total decisions':{dt.num_decisions}")
            break
        result = decision_engine.on_trigger(state_snapshot)
        apply_decision(sim, decision_engine, result, state_snapshot)
    if cfg.viz.launch:
        from casim.viz.gantt_chart import gantt_chart
        fig = gantt_chart(sim.state.tracker)
        fig.write_html(str(Path(cfg.experiment.output_dir) / "viz" / "gantt.html"))


if __name__ == "__main__":
    main()
