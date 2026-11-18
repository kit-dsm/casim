"""The recorded replay against the hand derivation of scenarios/recorded_replay/README.md."""

from pathlib import Path

import pandas as pd
import pytest

from casim.run import compose_scenario, run

SCENARIO = Path(__file__).resolve().parents[2] / "scenarios" / "recorded_replay"


def _run(tmp_path_factory, *overrides):
    output_dir = tmp_path_factory.mktemp("recorded_replay")
    run(compose_scenario(SCENARIO, list(overrides), output_dir=output_dir))
    return (pd.read_csv(output_dir / "facts" / "picks.csv"),
            pd.read_csv(output_dir / "facts" / "tours.csv", dtype={"order_ids": str}),
            pd.read_csv(output_dir / "facts" / "orders.csv"))


@pytest.fixture(scope="module")
def default(tmp_path_factory):
    return _run(tmp_path_factory)


def test_picks_follow_the_recorded_sequence_across_interleaved_orders(default):
    picks, _, _ = default
    assert picks[["picker_id", "order_id", "node_x", "node_y", "walk_distance",
                  "pick_start", "pick_end"]].values.tolist() == [
        [0, 1, 1, 2, 3, 18, 20], [0, 2, 1, 3, 1, 21, 23], [0, 1, 1, 4, 1, 24, 26],
        [1, 4, 2, 2, 4, 49, 51], [0, 3, 2, 1, 3, 109, 111]]


def test_pauses_hold_each_tour_until_its_recorded_start(default):
    _, tours, _ = default
    assert tours[["picker_id", "start", "setup_end", "end", "distance", "order_ids"]
                 ].values.tolist() == [
        [0, 10, 15, 31, 10, "1 2"], [1, 40, 45, 55, 8, "4"], [0, 101, 106, 114, 6, "3"]]


def test_orders_complete_with_their_tours(default):
    _, _, orders = default
    assert orders[["order_id", "lines", "tour_start", "first_pick_start", "last_pick_end",
                   "completion"]].values.tolist() == [
        [1, 2, 10, 18, 26, 31], [2, 1, 10, 21, 23, 31], [3, 1, 101, 109, 111, 114],
        [4, 1, 40, 49, 51, 55]]


def test_chained_replay_runs_tours_back_to_back_without_the_pause(tmp_path_factory):
    picks, tours, _ = _run(tmp_path_factory, "variants=chained")
    assert tours.loc[tours.order_ids == "3", ["start", "setup_end", "end"]
                     ].values.tolist() == [[31, 36, 44]]
    assert picks.loc[picks.order_id == 3, ["pick_start", "pick_end"]].values.tolist() == [[39, 41]]


def test_anchored_replay_does_not_extend_a_busy_picker_break(tmp_path_factory, default):
    _, tours, _ = _run(tmp_path_factory, "variants=anchored")
    assert tours.loc[tours.order_ids == "3", "start"].tolist() == [100]
    assert default[1].loc[default[1].order_ids == "3", "start"].tolist() == [101]


def test_a_model_faster_than_the_record_stays_anchored_but_not_chained(tmp_path_factory):
    # Pick 0.5 s: picker 0's first tour sets up 10-15, picks 18-18.5, 19.5-20, 21-21.5 and
    # ends at 26.5, before its recorded end 30. Anchored, order 3's tour still starts at its
    # recorded start 100; chained (threshold 50), the break only begins at 30, so the picker
    # starts order 3's tour at 26.5, as a picker ahead of a clock-time break would.
    _, anchored, _ = _run(tmp_path_factory, "variants=anchored", "source.time_per_pick=0.5")
    _, chained, _ = _run(tmp_path_factory, "source.time_per_pick=0.5")
    assert anchored.loc[anchored.order_ids == "1 2", "end"].tolist() == [26.5]
    assert anchored.loc[anchored.order_ids == "3", "start"].tolist() == [100]
    assert chained.loc[chained.order_ids == "3", "start"].tolist() == [26.5]


def test_a_line_pick_time_model_sets_each_pick_in_execution(tmp_path_factory):
    # Pick 1 s + 3 s per package, one package per line: 4 s per pick. Picker 0's first tour:
    # setup 10-15, walk 3, pick 18-22, walk 1, pick 23-27, walk 1, pick 28-32, back 5, end 37.
    picks, tours, _ = _run(tmp_path_factory, "source.time_per_pick=1.0", "source.per_unit_s=3.0")
    first = picks[picks.tour_id == picks.tour_id.iloc[0]]
    assert first[["pick_start", "pick_end"]].values.tolist() == [[18, 22], [23, 27], [28, 32]]
    assert tours.loc[tours.order_ids == "1 2", "end"].tolist() == [37]


def test_interleaved_orders_at_one_node_keep_the_recorded_pick_order(tmp_path):
    cfg = compose_scenario(SCENARIO, output_dir=tmp_path)
    cfg.source.lines = [
        [0, 1, 1, 1, [1, 2], 10, 20],
        [0, 2, 2, 1, [1, 2], 20, 22],
        [0, 1, 3, 1, [1, 2], 22, 24],
    ]
    run(cfg)
    picks = pd.read_csv(tmp_path / "facts" / "picks.csv").sort_values("pick_start")
    assert picks["order_id"].tolist() == [1, 2, 1]
    assert picks["article_id"].tolist() == [1, 2, 3]


def test_anchored_setup_can_be_placed_before_the_first_wms_line(tmp_path_factory):
    # First recorded line begins at 10, setup lasts 5, so the first tour's
    # setup begins at 5 and its first modeled walk begins at 10.
    picks, tours, _ = _run(tmp_path_factory, "variants=anchored",
                          "source.setup_in_gap=true")
    assert tours.loc[tours.order_ids == "1 2", ["start", "setup_end"]].values.tolist() == [[5, 10]]
    assert picks.iloc[0][["walk_start", "pick_start", "pick_end"]].tolist() == [10, 13, 15]


def _kpis(tmp_path_factory, *overrides):
    import json
    output_dir = tmp_path_factory.mktemp("recorded_replay")
    run(compose_scenario(SCENARIO, list(overrides), output_dir=output_dir))
    return (json.loads((output_dir / "kpis.json").read_text(encoding="utf-8")),
            pd.read_csv(output_dir / "facts" / "tours.csv", dtype={"order_ids": str}),
            pd.read_csv(output_dir / "facts" / "orders.csv"))


def test_recorded_presence_keeps_the_timing_and_counts_absence(tmp_path_factory, default):
    # Picker 0 arrives at 10, pauses 31-101 (70 s, from its recorded gap), and its
    # last line ends at 110, inside its last tour (101-114), so it leaves at 114,
    # when the run ends: absent 10 + 70. Picker 1 arrives at 40, its last line
    # ends at 50, inside its tour (40-55), so it leaves at 55: absent 40 + 59.
    kpis, tours, orders = _kpis(tmp_path_factory, "variants=recorded_presence")
    assert tours.values.tolist() == default[1].values.tolist()
    assert kpis["absent_time_by_picker"] == {"0": 80.0, "1": 99.0}
    assert kpis["busy_time_by_picker"] == {"0": 34.0, "1": 15.0}
    assert orders.completion.notna().all()


def test_recorded_presence_leaves_work_open_after_the_last_recorded_line(tmp_path_factory):
    # Pick 40 s: picker 0's first tour runs 10-145 (setup 5, walks 3, 1, 1 and 5,
    # three picks of 40). Its pause (requested at 30) and its departure after
    # the last recorded line (110) both wait for that tour and take effect in
    # order: the pause 145-215, then the departure, without return, so order 3's
    # tour never starts. The run ends at 215: picker 0 is absent 10 + 70, picker
    # 1 before 40 and after its tour, 93-215. With the shift-long presence the
    # picker returns from its pause at 215 and picks order 3.
    kpis, tours, orders = _kpis(tmp_path_factory, "variants=recorded_presence",
                                "source.time_per_pick=40.0")
    assert tours[["picker_id", "start", "end", "order_ids"]].values.tolist() == [
        [0, 10, 145, "1 2"], [1, 40, 93, "4"]]
    assert orders.loc[orders.completion.isna(), "order_id"].tolist() == [3]
    assert kpis["absent_time_by_picker"] == {"0": 80.0, "1": 162.0}
    _, shift, _ = _kpis(tmp_path_factory, "source.time_per_pick=40.0")
    assert shift.loc[shift.order_ids == "3", ["start", "end"]].values.tolist() == [[215, 266]]


def test_arrival_presence_never_forces_a_departure(tmp_path_factory):
    # Pick 40 s as above, with presence "arrival": picker 0 arrives at 10, its
    # pause (70 s) runs after its first tour, 145-215, and it picks order 3 at
    # 215-266, as with the shift-long presence. Absent: picker 0 before 10 and
    # during the pause; picker 1 only before 40.
    kpis, tours, orders = _kpis(tmp_path_factory, "variants=recorded_presence",
                                "source.presence=arrival", "source.time_per_pick=40.0")
    assert tours[["picker_id", "start", "end", "order_ids"]].values.tolist() == [
        [0, 10, 145, "1 2"], [1, 40, 93, "4"], [0, 215, 266, "3"]]
    assert orders.completion.notna().all()
    assert kpis["absent_time_by_picker"] == {"0": 80.0, "1": 40.0}


def test_stops_inside_a_tour_become_pauses_of_their_recorded_length():
    # One tour of one order, threshold 300 s, 19 s per line. Lines at 10-30,
    # 130-150 (gap 100: no stop), 520-540 (gap 370: a stop from 150 of 370 s)
    # and 540-940 (400 s, 381 s over the pick time: a stop from 559 of 381 s).
    from casim.sources.recorded import RecordedReplay, RecordedRow, TimeModel

    rows = [RecordedRow(picker=0, order=1, article=a, quantity=1, node=(1, a),
                        start_s=s, end_s=e)
            for a, (s, e) in enumerate([(10, 30), (130, 150), (520, 540), (540, 940)])]
    model = TimeModel(speed=1.0, time_per_pick=19.0, tour_setup_time=5.0)

    def stops(enabled):
        replay = RecordedReplay(rows, None, model, shift_start_s=0.0, pause_threshold_s=300.0,
                                stops_in_tours=enabled)
        return [(d.time, d.duration) for d in replay._pauses()], replay.summary

    assert stops(False)[0] == []
    found, summary = stops(True)
    assert sorted(found) == [(150, 370), (559, 381)]
    assert (summary.stops_in_tours, summary.stop_s_in_tours) == (2, 751)
