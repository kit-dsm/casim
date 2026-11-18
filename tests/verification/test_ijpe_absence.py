"""The fitted absence model of the IJPE scenario (casim/sources/absence.py)."""

import numpy as np

from casim.sources.absence import AbsenceModel


def test_rates_are_absences_per_present_picker_hour():
    # Day 1: picker 0 present 1:00-2:00, picker 1 1:30-3:00; absences at 1:05
    # and 1:40, one at 2:10. Day 2: picker 0 present 1:00-2:00, no absence.
    # Hour 1: 1 + 0.5 + 1 = 2.5 present hours, 2 absences; hour 2: 1 hour, 1.
    day1 = ({0: (3600, 7200), 1: (5400, 10800)}, [(3900, 60), (6000, 120), (7800, 30)])
    day2 = ({0: (3600, 7200)}, [])
    model = AbsenceModel.fit([day1, day2])
    assert model.rates == {1: 2 / 2.5, 2: 1.0}
    assert model.samples == ((3900, 60), (6000, 120), (7800, 30))


def test_draws_repeat_with_the_seed_and_stay_in_the_roster():
    samples = tuple((3600 + 60 * i, 100 + i) for i in range(40))
    model = AbsenceModel(rates={1: 3.0, 2: 3.0}, samples=samples)
    roster = {0: (3600, 9000), 1: (4000, 7000)}
    first = model.draw(roster, np.random.default_rng(7))
    assert first == model.draw(roster, np.random.default_rng(7))
    assert first != model.draw(roster, np.random.default_rng(8))
    assert all(roster[p][0] <= t <= roster[p][1] for p, t, _ in first)
    assert {d for _, _, d in first} <= {d for _, d in samples}


def test_a_duration_comes_from_absences_near_the_same_clock_time():
    # 20 absences at 1:00 last 10 s, 20 at 5:00 last 500 s: a request at 1:10
    # draws only from the first group.
    samples = tuple([(3600, 10)] * 20 + [(18000, 500)] * 20)
    model = AbsenceModel(rates={}, samples=samples)
    rng = np.random.default_rng(0)
    assert {model._duration(4200, rng) for _ in range(50)} == {10}
