import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from scenarios.experiment_commons import load_and_flatten_data_card, setup_scenario, setup_decision_engine
from casim.io_helpers import dump_pickle
from casim.viz.gantt_chart import gantt_chart
from scenarios.scenario_ijpe.scenario_specific_hooks import build_sim_hooks
from scenarios.scenario_ijpe.scripts.make_summary_scenarios_multirun import export_multiday_metrics

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


@hydra.main(config_path="config", config_name="grocery_retailer_config", version_base="1.3")
def main(cfg: DictConfig):
    datacard = load_and_flatten_data_card(cfg.data_card)

    out_dir = Path("./data")

    sim = setup_scenario(cfg)
    decision_engine = setup_decision_engine(cfg, datacard)

    event_map = {cls_name: instantiate(cls) for cls_name, cls in cfg.engines.decision_engine.event_map.items()}
    decision_engine.event_map = event_map

    # picks = pd.read_csv("data/picks_anonym.csv")
    # master = pd.read_csv("data/master_anonym.csv")
    #
    # build_order_stream(
    #     picks, master,
    #     out_csv=out_dir / "order_stream.csv",
    #     maps_path=out_dir / "id_maps.json",
    #     gen_cfg=GenerationConfig(n_days=n_days, seed=42),
    #     # scenario=SCENARIOS["monday_driver_outage"],
    # )

    sim.reset(hooks=build_sim_hooks(cfg))

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
            for pipeline, count in sorted(dt.pipeline_counts.items(),
                                          key=lambda x: -x[1]):
                pct = 100 * count / dt.num_decisions
                print(f"{pipeline} {count:} {pct}%")
            break
        events_to_add, solution = decision_engine.on_trigger(state_snapshot)
        sim.step(events_to_add, state_snapshot.problem_class, solution)

    # print(sim.state.tracker.delayed_expected_finish)
    for shifted_order in sim.state.tracker.shifted_orders:
        o = shifted_order[0]
        from_time = shifted_order[1]
        to_time = shifted_order[2]
        print(o.order_id, o.due_date, from_time, to_time)

    if cfg.viz.launch:
        fig = gantt_chart(sim.state.tracker)
        # ensure the viz output directory exists before writing the file
        viz_dir = Path(cfg.experiment.output_dir) / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(viz_dir / "gantt.html"))
        # viz_dir = Path(cfg.experiment.output_dir) / "viz"
        # launch(viz_dir, port=cfg.viz.port, debug=False)

    due_time_by_order = {
        order.order_id: order.due_date
        for order in sim._initial_domain.orders.orders
    }
    
    scenario_name = HydraConfig.get().runtime.choices["simulation"]
    summary = export_multiday_metrics(
        tracker=sim.state.tracker,
        out_dir=Path(cfg.experiment.output_dir) / "kpis",
        scenario_name=scenario_name,
        n_days=6,
        day_sec=86400,
        pickers_per_day=[25, 17, 19, 20, 18, 7],
        shift_start_hour=5.5,
        shift_end_hour=14.5,
        breaks=[
            {"start_hour": 8.0, "duration_minutes": 30},
            {"start_hour": 11.0, "duration_minutes": 30},
        ],
        due_time_by_order=due_time_by_order,
        dock_capacity=98,
        picker_ids=sorted(r.id for r in sim._initial_domain.resources.resources),
    )

    print(summary)
if __name__ == "__main__":
    main()
