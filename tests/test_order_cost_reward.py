import numpy as np
import pytest

from scenarios.scenario_henn_rl.rewards import OrderCostReward


@pytest.mark.parametrize(
    ("power", "threshold", "expected"),
    [
        (1.0, 0.0, 21.0),
        (2.0, 0.0, 165.0),
        (1.0, 5.0, 7.0),
        (2.0, 5.0, 29.0),
    ],
)
def test_completed_objective_matches_closed_form(power, threshold, expected):
    reward = OrderCostReward(
        {0: 0.0, 1: 2.0, 2: 5.0},
        power=power,
        thresholds_s=threshold,
    )
    completions = {0: 4.0, 1: 9.0, 2: 15.0}
    assert reward.objective(completions) == pytest.approx(expected)
    assert reward.accrued_cost(15.0, completions) == pytest.approx(expected)


def test_incremental_return_is_independent_of_step_granularity():
    reward = OrderCostReward(
        {0: 0.0, 1: 2.0, 2: 5.0},
        power=2.0,
        weights={0: 1.0, 1: 2.0, 2: 0.5},
        thresholds_s={0: 0.0, 1: 1.0, 2: 3.0},
        normalizer=10.0,
    )
    completions = {0: 4.0, 1: 9.0, 2: 15.0}
    previous = reward.accrued_cost(0.0, completions)
    total_return = 0.0
    for current_time in (1.0, 2.0, 4.0, 5.0, 9.0, 12.0, 15.0):
        current = reward.accrued_cost(current_time, completions)
        total_return += reward.incremental_reward(previous, current)
        previous = current
    assert total_return == pytest.approx(
        -reward.objective(completions) / reward.normalizer
    )


def test_convex_power_prefers_balanced_flow_with_equal_mean():
    arrivals = {0: 0.0, 1: 0.0}
    balanced = {0: 5.0, 1: 5.0}
    tail_heavy = {0: 1.0, 1: 9.0}
    linear = OrderCostReward(arrivals, power=1.0)
    squared = OrderCostReward(arrivals, power=2.0)
    assert linear.objective(balanced) == linear.objective(tail_heavy)
    assert squared.objective(balanced) < squared.objective(tail_heavy)


def test_sla_and_priority_parameters_have_explicit_effects():
    arrivals = {0: 0.0, 1: 0.0}
    completions = {0: 4.0, 1: 9.0}
    tardiness = OrderCostReward(arrivals, thresholds_s=5.0)
    weighted = OrderCostReward(arrivals, weights={0: 1.0, 1: 3.0})
    assert tardiness.objective(completions) == pytest.approx(4.0)
    assert weighted.objective(completions) == pytest.approx(31.0)


def test_absolute_due_dates_include_overdue_cost_at_arrival():
    reward = OrderCostReward(
        {0: 5.0, 1: 5.0, 2: 5.0},
        due_times={0: 10.0, 1: 5.0, 2: 2.0},
    )
    completions = {0: 14.0, 1: 9.0, 2: 9.0}
    assert reward.accrued_cost(4.0, completions) == 0.0
    assert reward.accrued_cost(5.0, completions) == pytest.approx(3.0)
    assert reward.objective(completions) == pytest.approx(15.0)

    previous = reward.accrued_cost(4.0, completions)
    total_return = 0.0
    for current_time in (5.0, 7.0, 9.0, 14.0):
        current = reward.accrued_cost(current_time, completions)
        total_return += reward.incremental_reward(previous, current)
        previous = current
    assert total_return == pytest.approx(-reward.objective(completions))


def test_random_event_partitions_preserve_objective_identity():
    rng = np.random.default_rng(42)
    for power in (1.0, 1.5, 2.0, 3.0):
        arrivals = {index: float(value) for index, value in enumerate(
            np.sort(rng.uniform(0.0, 20.0, size=20))
        )}
        completions = {
            order_id: arrival + float(rng.uniform(1.0, 30.0))
            for order_id, arrival in arrivals.items()
        }
        reward = OrderCostReward(
            arrivals,
            power=power,
            weights={order_id: float(rng.uniform(0.2, 3.0)) for order_id in arrivals},
            thresholds_s={
                order_id: float(rng.uniform(0.0, 10.0)) for order_id in arrivals
            },
            normalizer=17.0,
        )
        times = sorted(
            {0.0, *arrivals.values(), *completions.values(), max(completions.values())}
        )
        previous = reward.accrued_cost(times[0], completions)
        total_return = 0.0
        for current_time in times[1:]:
            current = reward.accrued_cost(current_time, completions)
            total_return += reward.incremental_reward(previous, current)
            previous = current
        assert total_return == pytest.approx(
            -reward.objective(completions) / reward.normalizer,
            rel=1e-12,
        )


def test_invalid_reward_parameters_and_trajectories_fail_loudly():
    with pytest.raises(ValueError, match="power"):
        OrderCostReward({0: 0.0}, power=0.5)
    with pytest.raises(ValueError, match="normalizer"):
        OrderCostReward({0: 0.0}, normalizer=0.0)
    with pytest.raises(ValueError, match="Missing weights"):
        OrderCostReward({0: 0.0, 1: 0.0}, weights={0: 1.0})
    with pytest.raises(ValueError, match="not both"):
        OrderCostReward(
            {0: 0.0}, thresholds_s=1.0, due_times={0: 2.0}
        )
    reward = OrderCostReward({0: 2.0})
    with pytest.raises(ValueError, match="Missing completions"):
        reward.objective({})
    with pytest.raises(ValueError, match="before arrival"):
        reward.accrued_cost(3.0, {0: 1.0})
    with pytest.raises(ValueError, match="cannot decrease"):
        reward.incremental_reward(2.0, 1.0)
