"""The reference layout and analytical waiting policy use the same geometry."""

from pathlib import Path
from copy import deepcopy

import pytest
from hydra import compose, initialize_config_dir

from ware_ops_algos.algorithms import (
    BatchObject, Job, PickPosition, Route, ScheduledJob, WaitingInput,
    WarehouseOrder,
)
from ware_ops_algos.algorithms.waiting.analytic_progress import geometry as g
from ware_ops_algos.algorithms.waiting.analytic_progress.continuous_calculation import continuous_expected_detour
from ware_ops_algos.algorithms.waiting.analytic_progress.core import compute_segment_times
from ware_ops_algos.algorithms.waiting.analytic_progress.optimal_waiting import solve_optimal_wait
from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards
from ware_ops_algos.domain_algo_mapper.domain_algo_mapper import DomainAlgorithmMapper

from casim.pipelines.taxonomy import TAXONOMY
from scenarios.experiment_commons import load_and_flatten_data_card
from scenarios.scenario_stochastic_waiting.loader import StochasticWaitingDataLoader


CONFIG_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "scenario_stochastic_waiting" / "config"


@pytest.mark.parametrize(("section", "feature", "value"), [
    ("orders", "order_lines_per_order", 2),
    ("orders", "amount", 2),
    ("information", "incoming_orders.type", "different_order_process"),
    ("information", "incoming_orders.mean_interarrival_time_s", None),
    ("information", "incoming_orders.mean_interarrival_time_s", -1.0),
    ("layout", "n_blocks", 2),
])
def test_analytical_waiting_card_filters_declared_assumptions(section, feature, value):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="stochastic_waiting_config")
    card = load_and_flatten_data_card(cfg.data_card)
    algorithm = next(model for model in load_packaged_algo_cards()
                     if model.algo_name == "AnalyticStochasticWaiting")
    mapper = DomainAlgorithmMapper(TAXONOMY)

    assert mapper.filter([algorithm], card) == [algorithm]
    incompatible = deepcopy(card)
    if value is None:
        getattr(incompatible, section)["features"].pop(feature)
    else:
        getattr(incompatible, section)["features"][feature] = value
    assert mapper.filter([algorithm], incompatible) == []


def test_reference_data_matches_its_card_and_uses_paper_router():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="stochastic_waiting_config")
    card = load_and_flatten_data_card(cfg.data_card)
    domain = StochasticWaitingDataLoader(CONFIG_DIR, cfg).load()

    assert card.orders["features"]["order_lines_per_order"] == 1
    assert card.orders["features"]["amount"] == 1
    assert all(len(order.order_positions) == 1 and
               order.order_positions[0].amount == 1
               for order in domain.orders.orders)
    assert "casim.pipelines.subproblems.picker_routing.WalkOrWaitSShape" in cfg.cosy_repo.components


def test_reference_route_and_wait_match_the_paper_example():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="stochastic_waiting_config")
    domain = StochasticWaitingDataLoader(CONFIG_DIR, cfg).load()
    orders = tuple(
        WarehouseOrder(
            order_id=index,
            order_date=float(index),
            pick_positions=(PickPosition(
                order_number=index, article_id=index, amount=1,
                pick_node=(2, 1), in_store=1,
            ),),
        )
        for index in range(3)
    )
    batch = BatchObject(batch_id=0, orders=list(orders))
    route = Route(distance=0.0, batch=batch)
    job = Job(
        job_id=0, processing_time=0.0, release_time=2.0,
        due_date=float("inf"), n_picks=3, route=route, batch=batch,
    )
    data = WaitingInput(
        candidates=(ScheduledJob(job, 0, 2.0, 2.0),),
        current_time=2.0,
        input_closed=False,
        picker=domain.resources.resources[0],
        layout=domain.layout,
        information=domain.information,
    )
    times_in, times_out, route_duration = compute_segment_times(data)
    depot_detour = continuous_expected_detour(0.0, data, times_in, times_out)
    later_detour = continuous_expected_detour(20.0, data, times_in, times_out)
    optimum = solve_optimal_wait(data, mean_interarrival_time=28.8)

    assert (g.M(data), g.N_L(data), g.L(data)) == (8, 16, 17.0)
    assert g.arrival_times(data) == [0.0, 1.0, 2.0]
    assert route_duration == 38.0
    assert depot_detour.expected_detour == 5.25
    assert later_detour.expected_detour == pytest.approx(35.00735294117647)
    assert optimum.wait_duration == pytest.approx(9.085034848335033)
    assert domain.objective == "mean_order_completion_time"
