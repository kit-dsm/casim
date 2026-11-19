"""The fitted absence model: scheduled breaks and allowance (casim/sources/absence.py)."""

import numpy as np
import pytest

from casim.sources.absence import PARTS, AbsenceModel, Allowance, Break

H = 3600
WINDOWS = ((7 * H, 9.5 * H), (10.5 * H, 13.5 * H))


def test_breaks_and_allowance_are_fitted_from_the_recorded_absences():
    # Pickers 0 and 2 work 6:00-14:00, picker 1 10:00-14:00 (not present in
    # the morning window). Picker 0: 30 min at 8:30 is its morning break; 5 min
    # at 9:06 is allowance; 40 min at 12:00 its midday break. Picker 1: 15 min
    # at 11:00 its midday break; 2 min at 12:00 allowance. Picker 2: 10 min at
    # 6:30, in its first hour (start-up allowance). Morning: 1 break of 2
    # present; midday: 2 of 3. Allowance per working hour (present time not
    # absent): start-up 1 in 1 + 1 + 5/6 h; regular 2 in (6 - 1/2 - 1/12 - 2/3)
    # + (2 - 1/4 - 1/30) + 6 h; wind-down 0 in 3 h.
    roster = {0: (6 * H, 14 * H), 1: (10 * H, 14 * H), 2: (6 * H, 14 * H)}
    absences = [(0, 8.5 * H, 1800), (0, 9.1 * H, 300), (0, 12 * H, 2400),
                (1, 11 * H, 900), (1, 12 * H, 120), (2, 6.5 * H, 600)]
    model = AbsenceModel.fit([(roster, absences)], WINDOWS, break_min_s=600)
    morning, midday = model.breaks
    assert (morning.share, morning.samples) == (0.5, ((8.5 * H, 1800),))
    assert (midday.share, midday.samples) == (2 / 3, ((11 * H, 900), (12 * H, 2400)))
    start_up, regular, wind_down = (model.allowance[name] for name in PARTS)
    assert start_up.durations == (600,) and start_up.rate == pytest.approx(1 / (2 + 5 / 6))
    regular_hours = (6 - 1 / 2 - 1 / 12 - 2 / 3) + (2 - 1 / 4 - 1 / 30) + 6
    assert regular.durations == (120, 300) and regular.rate == pytest.approx(2 / regular_hours)
    assert wind_down == Allowance(0.0, ())


def test_draws_repeat_with_the_seed_and_stay_in_the_roster():
    model = AbsenceModel(
        breaks=(Break(WINDOWS[0], 1.0, ((8.5 * H, 1800), (9 * H, 1500))),
                Break(WINDOWS[1], 1.0, ((12 * H, 2400),))),
        allowance={name: Allowance(1.5, (120, 300, 600)) for name in PARTS})
    roster = {0: (6 * H, 14 * H), 1: (6.5 * H, 13 * H)}
    first = model.draw(roster, np.random.default_rng(7))
    assert first == model.draw(roster, np.random.default_rng(7))
    assert first != model.draw(roster, np.random.default_rng(8))
    assert all(roster[p][0] <= t <= roster[p][1] for p, t, _ in first)
    for picker in roster:
        durations = {d for p, _, d in first if p == picker}
        assert 2400 in durations and durations & {1800, 1500}
        assert durations <= {120, 300, 600, 1500, 1800, 2400}


def test_a_picker_not_present_in_a_break_window_takes_no_break_there():
    model = AbsenceModel(breaks=(Break(WINDOWS[0], 1.0, ((8.5 * H, 1800),)),),
                         allowance={name: Allowance(0.0, (60,)) for name in PARTS})
    assert model.draw({0: (10 * H, 14 * H)}, np.random.default_rng(0)) == []


def test_no_absence_starts_while_the_picker_is_away():
    # A high allowance rate and long breaks: a break that falls due during an
    # allowance starts when it ends, and no allowance starts during a break.
    model = AbsenceModel(
        breaks=(Break(WINDOWS[0], 1.0, ((8.5 * H, 1800),)), Break(WINDOWS[1], 1.0, ((12 * H, 2400),))),
        allowance={name: Allowance(3.0, (300, 900)) for name in PARTS})
    roster = {picker: (6 * H, 14 * H) for picker in range(20)}
    drawn = model.draw(roster, np.random.default_rng(1))
    for picker in roster:
        spells = sorted((t, t + d) for p, t, d in drawn if p == picker)
        assert all(later[0] >= earlier[1] for earlier, later in zip(spells, spells[1:]))
        assert {round(b - a) for a, b in spells} >= {1800, 2400}
