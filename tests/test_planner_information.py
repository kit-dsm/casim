"""Planner information crosses the decision boundary without future events."""

from dataclasses import dataclass
from copy import copy
from pathlib import Path
from typing import ClassVar

import pytest
from hydra import compose, initialize_config_dir

from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards
from ware_ops_algos.domain_algo_mapper.domain_algo_mapper import DomainAlgorithmMapper
from ware_ops_algos.domain_models import (
    ExponentialSingleLineUniformLocationOrderStream, PlannerInformation,
)
from ware_ops_algos.domain_models.datacards import load_and_flatten_data_card as load_card_path

from casim.simulation_engine.state_adapter import HennWaitingAdapter
from scenarios.experiment_commons import (
    load_and_flatten_data_card, setup_decision_engine, setup_scenario,
)
from scenarios.scenario_stochastic_waiting.scenario_specific_hooks import (
    add_orders_hook, picker_arrival_hook,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "scenario_stochastic_waiting" / "config"


def configured_study(tmp_path):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="stochastic_waiting_config", overrides=[
            f"instances_base={tmp_path.as_posix()}",
            f"cache_base={(tmp_path / 'cache').as_posix()}",
            f"experiment.output_dir={tmp_path.as_posix()}",
        ])


def test_waiting_information_crosses_decision_boundary_without_future_orders(tmp_path):
    cfg = configured_study(tmp_path)
    card = load_and_flatten_data_card(cfg.data_card)
    assert card.information["features"] == {
        "incoming_orders": ExponentialSingleLineUniformLocationOrderStream.representation,
    }
    assert load_card_path(CONFIG_DIR / "data_card" / "stochastic_waiting.yaml").information == card.information
    sim = setup_scenario(cfg)
    sim.reset(hooks=[add_orders_hook, picker_arrival_hook])
    done, snapshot = sim.run()
    assert not done
    assert snapshot.information is not sim.state.information
    assert snapshot.information.require(
        "incoming_orders", ExponentialSingleLineUniformLocationOrderStream
    ).mean_interarrival_time_s == 28.8
    assert all(order.order_date <= snapshot.dynamic_warehouse_info.time
               for order in snapshot.orders.orders)
    assert not hasattr(snapshot.information, "arrival_times_s")

    card = next(c for c in load_packaged_algo_cards()
                if c.algo_name == "AnalyticStochasticWaiting")
    mapper = DomainAlgorithmMapper({"OBRSPW": {"variables": ["waiting"]}})
    assert mapper.filter([card], snapshot) == [card]
    snapshot.information = None
    assert mapper.filter([card], snapshot) == []
    invalid_card = copy(card)
    invalid_card.requirements = {"misspelled_information": {"type": ["process_information"]}}
    assert mapper.filter([invalid_card], snapshot) == []


@pytest.mark.parametrize("processes, message", [
    ([{"id": "incoming_orders", "type": "unknown", "mean_interarrival_time_s": 28.8}], "Unsupported"),
    ([{"id": "incoming_orders", "type": "exponential_single_line_uniform_location_order_stream",
       "mean_interarrival_time_s": 0}], "finite positive"),
])
def test_scenario_loader_rejects_a_forecast_it_cannot_construct(tmp_path, processes, message):
    cfg = configured_study(tmp_path)
    cfg.data_card.information.processes = processes
    load_and_flatten_data_card(cfg.data_card)
    sim = setup_scenario(cfg)
    with pytest.raises(ValueError, match=message):
        sim.reset(hooks=[add_orders_hook, picker_arrival_hook])


def test_duplicate_process_ids_fail_when_card_loads(tmp_path):
    cfg = configured_study(tmp_path)
    cfg.data_card.information.processes = list(cfg.data_card.information.processes) * 2
    with pytest.raises(ValueError, match="Duplicate"):
        load_and_flatten_data_card(cfg.data_card)


def test_mapper_excludes_missing_or_incompatible_information_before_simulation(tmp_path):
    cfg = configured_study(tmp_path)
    card = load_and_flatten_data_card(cfg.data_card)
    algorithm = next(c for c in load_packaged_algo_cards()
                     if c.algo_name == "AnalyticStochasticWaiting")
    mapper = DomainAlgorithmMapper({"OBRSPW": {"variables": ["waiting"]}})
    card.information = {"type": None, "features": {}}
    assert mapper.filter([algorithm], card) == []
    with pytest.raises(ValueError, match="No CoSy pipeline"):
        setup_decision_engine(cfg, card)

    card.information = {"type": "process_information", "features": {
        "incoming_orders": "different_order_process",
    }}
    assert mapper.filter([algorithm], card) == []
    with pytest.raises(ValueError, match="No CoSy pipeline"):
        setup_decision_engine(cfg, card)


@dataclass(frozen=True)
class ExampleAttendanceInformation:
    """Contract check only; no attendance policy or realization is introduced."""

    representation: ClassVar[str] = "example_attendance_probability"
    probability: float


def test_second_process_fits_same_snapshot_boundary(tmp_path):
    cfg = configured_study(tmp_path)
    sim = setup_scenario(cfg)
    sim.reset(hooks=[add_orders_hook, picker_arrival_hook])
    original = sim.state.information.require("incoming_orders", ExponentialSingleLineUniformLocationOrderStream)
    sim.state.information = PlannerInformation({
        "incoming_orders": original,
        "picker_attendance": ExampleAttendanceInformation(probability=0.9),
    })
    projection = HennWaitingAdapter().transform_state(sim.state, "OBRSPW")
    assert projection.information.require("picker_attendance", ExampleAttendanceInformation).probability == 0.9
    assert projection.information.get_features() == {
        "incoming_orders": ExponentialSingleLineUniformLocationOrderStream.representation,
        "picker_attendance": ExampleAttendanceInformation.representation,
    }
