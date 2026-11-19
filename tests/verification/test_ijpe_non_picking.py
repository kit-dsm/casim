"""The fitted non-picking time of the IJPE day.

Observation and fit (scripts/ijpe/fit_non_picking.py) on synthetic days, and
the realization of breaks, line and setup times in the source
(scenarios/ijpe/source.py) with common random numbers.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scenarios.ijpe.source import ScenarioSource, draw_breaks, process_truth
from scripts.ijpe.fit_non_picking import Observations, fit, observe, order_effect, quantity_class
from casim.sources.recorded import RecordedRow

H = 3600
WINDOWS = ((7 * H, 9.5 * H), (10.5 * H, 13.5 * H))
START, END, A, B = (0, -1), (-1, -1), (1, 0), (2, 0)


def _replay(rows, speed=1.0):
    """A replay-like object: one tour per order, depot to A 10, A to B 5, B to depot 20."""
    nodes = [START, END, A, B]
    d = {(START, A): 10, (START, B): 15, (A, B): 5, (A, END): 12, (B, END): 20}
    matrix = pd.DataFrame([[0.0 if x == y else d.get((x, y), d.get((y, x), 0.0))
                            for y in nodes] for x in nodes], index=nodes, columns=nodes)
    tours = {}
    for row in sorted(rows, key=lambda r: r.start_s):
        tours.setdefault((row.picker, row.order), []).append(row)
    tours = sorted(((p, ls) for (p, _), ls in tours.items()), key=lambda t: t[1][0].start_s)
    roster = {p: (min(r.start_s for r in rows if r.picker == p),
                  max(r.end_s for r in rows if r.picker == p)) for p in {r.picker for r in rows}}
    network = SimpleNamespace(distance_matrix=matrix, start_node=START, end_node=END)
    return SimpleNamespace(tours=tours, roster=lambda: roster,
                           time_model=SimpleNamespace(speed=speed),
                           layout=SimpleNamespace(layout_network=network))


def row(order, node, start, end, quantity=6, picker=0, article=1):
    return RecordedRow(picker, order, article, quantity, node, float(start), float(end))


def test_parts_are_measured_on_time_without_records():
    t = 6 * H
    rows = [
        row(1, A, t, t + 30), row(1, B, t + 40, t + 60, quantity=20),   # gap 10, walk 5
        row(2, A, 27000, 27020),                                       # 07:30
        # 40 min after tour 2 (07:30) and a 15 min gap inside tour 3, both
        # starting in the morning window: the longer one is the break.
        row(3, A, 29420, 29440), row(3, B, 30340, 30360),
        # Tour 4 holds tour 5's line inside its only line: that line is
        # bookkeeping, the line inside it is kept.
        row(4, A, 30400, 30700), row(5, B, 30500, 30520),
    ]
    foreign = {0: [(t + 200, t + 300)]}            # a record outside the lines
    obs = observe(_replay(rows), foreign, WINDOWS, break_min_s=600)
    assert obs.breaks == [[(27020.0, 2400.0)], []]
    assert obs.covered_s == 100
    assert obs.break_spells == [(0, 27020.0, 2400.0)]
    # Setup: time without a record minus the walk B-depot (20) and depot-A (10);
    # tour 3 has only the break (0 - 30, clipped); tour 4: 40 - 30; tour 5
    # starts inside tour 4: 0.
    assert obs.setups == [27000 - (t + 60) - 100 - 30, 0.0, 10.0, 0.0]
    assert obs.clipped_setups == 2
    assert sorted(obs.lines) == sorted([(6, 30.0), (20, 20 + 10 - 5), (6, 20.0),
                                        (6, 20.0), (6, 20 + 900 - 5), (6, 20.0)])
    assert obs.lines_bookkeeping == 1 and obs.lines_as_breaks == 0
    assert obs.non_picking_s()["breaks"] == 2400


def test_a_long_line_in_a_window_can_be_the_break():
    rows = [row(1, A, 8 * H, 8 * H + 1500), row(1, B, 8 * H + 1500, 8 * H + 1520)]
    obs = observe(_replay(rows), {}, WINDOWS, break_min_s=600)
    assert obs.breaks[0] == [(8 * H, 1500.0)]
    assert obs.lines_as_breaks == 1
    assert obs.lines == [(6, 20 - 5)]


def test_fit_pools_the_held_out_days():
    day = Observations(windows=WINDOWS, breaks=[[(8 * H, 1800)], []], present=[2, 2],
                       setups=[0.0, 100.0, 500.0], lines=[(6, 10.0), (6, 20.0), (30, 60.0)])
    other = Observations(windows=WINDOWS, breaks=[[(8.5 * H, 1200)], [(12 * H, 2400)]],
                         present=[2, 1], setups=[300.0], lines=[(12, 30.0)])
    result = fit([day, other])
    assert result["breaks"][0] == {"window": list(WINDOWS[0]), "share": 0.5,
                                   "samples": [[8 * H, 1800], [8.5 * H, 1200]]}
    assert result["breaks"][1]["share"] == pytest.approx(1 / 3)
    assert result["setup_s"] == [0.0, 100.0, 300.0, 500.0]
    # Class 13-24 has no line: it takes all lines.
    assert result["line_s"] == [[10.0, 20.0], [30.0], [10.0, 20.0, 30.0, 60.0], [60.0]]
    assert [quantity_class(q) for q in (1, 6, 7, 12, 13, 24, 25)] == [0, 0, 1, 1, 2, 2, 3]


FIT = {"breaks": [{"window": list(WINDOWS[0]), "share": 1.0, "samples": [[8 * H, 1800]]}],
       "setup_s": [float(s) for s in range(0, 1000, 7)], "quantity_bounds": [6, 12, 24],
       "line_s": [[float(s) for s in range(1, 400)] for _ in range(4)]}


def _order(order_id, lines):
    return SimpleNamespace(order_id=order_id, order_positions=[
        SimpleNamespace(article_id=a, amount=q) for a, q in lines])


def _pick(order, article, amount):
    return SimpleNamespace(order_number=order, article_id=article, amount=amount)


def test_line_and_setup_times_are_common_random_numbers():
    orders = [_order(10, [(1, 6), (2, 6), (1, 6)]), _order(11, [(1, 30)])]
    truth = process_truth(FIT, seed=3, day=20260413, orders=orders, pickers=[0, 1],
                          first_setup_s=215)
    # The same seed gives the same times whatever the order of the orders.
    again = process_truth(FIT, seed=3, day=20260413, orders=orders[::-1], pickers=[1, 0],
                          first_setup_s=215)
    assert truth.line_s == again.line_s and truth.setup_s == again.setup_s
    assert len(truth.line_s[(10, 1, 6)]) == 2 and truth.line_s[(10, 1, 6)][0] != truth.line_s[(10, 1, 6)][1]
    # Each picker's first setup is the fixed one, then drawn per picker.
    assert truth.setup_s[0][0] == truth.setup_s[1][0] == 215
    assert truth.setup_s[0][1:] != truth.setup_s[1][1:]
    assert truth.pick(_pick(11, 1, 30)) == truth.line_s[(11, 1, 30)][0]
    reseeded = process_truth(FIT, seed=4, day=20260413, orders=orders, pickers=[0, 1],
                             first_setup_s=215)
    assert reseeded.line_s != truth.line_s


def _day_with_order_effect(rng, sd, orders=400, lines=30):
    """Line samples of 20 s times an order factor (mean 1, ``sd``) times noise."""
    day = Observations(windows=WINDOWS, breaks=[[], []], present=[0, 0])
    for order in range(orders):
        factor = rng.lognormal(-np.log1p(sd ** 2) / 2, np.sqrt(np.log1p(sd ** 2)))
        for _ in range(lines):
            day.lines.append((6, 20.0 * factor * rng.gamma(4.0, 0.25)))
            day.line_orders.append(order)
    return day


def test_fit_recovers_the_spread_of_an_order_factor():
    rng = np.random.default_rng(0)
    days = [_day_with_order_effect(rng, 0.3) for _ in range(3)]
    sd, factors = order_effect(days)
    assert sd == pytest.approx(0.3, abs=0.03)
    # Deflated pools keep the within-order spread only (gamma(4) has cv 0.5).
    result = fit(days)
    pool = np.asarray(result["line_s"][0])
    assert result["order_sd"] == sd
    assert pool.std() / pool.mean() == pytest.approx(0.5, abs=0.04)
    # ... and the mean line time of the raw samples.
    raw = np.mean([s for day in days for _, s in day.lines])
    assert pool.mean() == pytest.approx(raw, rel=1e-9)
    # Without order structure the factor is about zero.
    flat = [_day_with_order_effect(rng, 0.0) for _ in range(3)]
    assert order_effect(flat)[0] < 0.03


def test_lines_of_an_order_share_its_factor():
    orders = [_order(o, [(a, 6) for a in range(1, 41)]) for o in range(300)]
    plain = process_truth(FIT, seed=1, day=20260413, orders=orders, pickers=[0], first_setup_s=215)
    shared = process_truth({**FIT, "order_sd": 0.4}, seed=1, day=20260413, orders=orders,
                           pickers=[0], first_setup_s=215)
    ratios = []
    for o in range(300):
        keys = [(o, a, 6) for a in range(1, 41)]
        r = {shared.line_s[k][0] / plain.line_s[k][0] for k in keys}
        assert max(r) - min(r) < 1e-9          # one factor per order, same line draws
        ratios.append(r.pop())
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.06)
    assert np.std(ratios) == pytest.approx(0.4, abs=0.06)


def test_breaks_are_drawn_for_present_pickers_only():
    drawn = draw_breaks(FIT, {0: (6 * H, 14 * H), 1: (10 * H, 14 * H)}, seed=0, day=20260413)
    assert drawn == [(0, 8 * H, 1800.0)]


def test_orders_that_overlap_another_order_of_the_login_give_no_sample():
    # Picker 0 scans orders 1 and 2 alternately (two people, or wrong records):
    # neither gives a line sample, the gap before order 3 no setup sample, and
    # the picker-day no break presence. Picker 1 works one order at a time.
    rows = [row(1, A, 8 * H, 8 * H + 20), row(2, B, 8 * H + 30, 8 * H + 50),
            row(1, B, 8 * H + 60, 8 * H + 80), row(2, A, 8 * H + 90, 8 * H + 110),
            row(3, A, 8 * H + 200, 8 * H + 220),
            row(4, A, 8 * H, 8 * H + 20, picker=1), row(4, B, 8 * H + 40, 8 * H + 60, picker=1),
            row(5, A, 8 * H + 400, 8 * H + 420, picker=1)]
    obs = observe(_replay(rows), {}, WINDOWS, break_min_s=600)
    assert (obs.orders, obs.orders_overlapping) == (5, 2)
    assert (obs.picker_days, obs.picker_days_left_out) == (2, 1)
    assert obs.present == [1, 0]
    # Breaks are still found on the left-out picker-day (a WMS order holding
    # one is stretched by it), but give no break sample.
    assert obs.breaks == [[], []]
    # Lines of orders 3, 4 and 5 only; setups 1 -> 2 and 2 -> 3 left out, 4 -> 5 kept.
    assert sorted(obs.lines) == [(6, 20.0), (6, 20.0), (6, 20.0), (6, 20 + 20 - 5)]
    assert obs.setups_left_out == 2 and obs.setups == [400 - 60 - 20 - 10]
    # The check measures every line on its own order's scans.
    mine = [s for _, s, overlapping in obs.own_lines if overlapping]
    assert mine == [20.0, 20 + 40 - 5, 20.0, 20 + 40 - 5]


DATA_DIR = Path(__file__).resolve().parents[2] / "scenarios" / "ijpe" / "data"


@pytest.mark.skipif(not (DATA_DIR / "layout.csv").exists(), reason="confidential IJPE data")
def test_extra_pickers_join_the_logins_and_never_leave():
    settings = dict(day="2026-04-17", shift_start_s=19800.0, pause_threshold_s=300.0,
                    speed=2306.0, time_per_pick=16.26, tour_setup_time=215.0,
                    non_picking="fixed", breaks=[[28800.0, 30600.0]])
    plain = ScenarioSource(DATA_DIR, presence="arrival", **settings)
    source = ScenarioSource(DATA_DIR, presence="arrival", extra_pickers=2, **settings)
    n = len(plain.replay.pickers)
    domain = source.initial_state()
    assert [p.id for p in domain.resources.resources] == list(range(n + 2))
    events = list(source.events())
    arrivals = {e.picker_id: e.time for e in events if type(e).__name__ == "PickerArrival"}
    departures = [e for e in events if type(e).__name__ == "PickerDeparture"]
    roster = plain.replay.roster()
    assert arrivals[n] == arrivals[n + 1] == np.median([a for a, _ in roster.values()])
    assert all(d.duration is not None for d in departures)          # breaks only
    assert {d.picker_id for d in departures} == set(range(n + 2))   # extras take the break too
    with pytest.raises(ValueError, match="presence"):
        ScenarioSource(DATA_DIR, presence="recorded", extra_pickers=1, **settings)
