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


def test_each_recorded_order_is_a_tour_of_its_own(default):
    # Order 2 lies inside order 1's recorded interval; the replay does not merge
    # them into a batch: order 1's tour, then order 2's.
    picks, _, _ = default
    assert picks[["picker_id", "order_id", "node_x", "node_y", "walk_distance",
                  "pick_start", "pick_end"]].values.tolist() == [
        [0, 1, 1, 2, 3, 18, 20], [0, 1, 1, 4, 2, 22, 24], [0, 2, 1, 3, 4, 38, 40],
        [1, 4, 2, 2, 4, 49, 51], [0, 3, 2, 1, 3, 122, 124]]


def test_pauses_hold_each_tour_until_its_recorded_start(default):
    _, tours, _ = default
    assert tours[["picker_id", "start", "setup_end", "end", "distance", "order_ids"]
                 ].values.tolist() == [
        [0, 10, 15, 29, 10, "1"], [0, 29, 34, 44, 8, "2"], [1, 40, 45, 55, 8, "4"],
        [0, 114, 119, 127, 6, "3"]]


def test_orders_complete_with_their_tours(default):
    _, _, orders = default
    assert orders[["order_id", "lines", "tour_start", "first_pick_start", "last_pick_end",
                   "completion"]].values.tolist() == [
        [1, 2, 10, 18, 24, 29], [2, 1, 29, 38, 40, 44], [3, 1, 114, 122, 124, 127],
        [4, 1, 40, 49, 51, 55]]


def test_chained_replay_runs_tours_back_to_back_without_the_pause(tmp_path_factory):
    picks, tours, _ = _run(tmp_path_factory, "variants=chained")
    assert tours.loc[tours.order_ids == "3", ["start", "setup_end", "end"]
                     ].values.tolist() == [[44, 49, 57]]
    assert picks.loc[picks.order_id == 3, ["pick_start", "pick_end"]].values.tolist() == [[52, 54]]


def test_anchored_replay_does_not_extend_a_busy_picker_break(tmp_path_factory, default):
    _, tours, _ = _run(tmp_path_factory, "variants=anchored")
    assert tours.loc[tours.order_ids == "3", "start"].tolist() == [100]
    assert default[1].loc[default[1].order_ids == "3", "start"].tolist() == [114]


def test_a_model_faster_than_the_record_stays_anchored_but_not_chained(tmp_path_factory):
    # Pick 0.5 s: picker 0's tour of order 1 runs 10-26 and its tour of order 2
    # 26-39.5, past the recorded end 30. Anchored, order 3's tour still starts at
    # its recorded start 100; chained (threshold 50), the 70 s break requested at
    # 30 begins when order 2's tour ends, 39.5-109.5, and order 3 follows.
    _, anchored, _ = _run(tmp_path_factory, "variants=anchored", "source.time_per_pick=0.5")
    _, chained, _ = _run(tmp_path_factory, "source.time_per_pick=0.5")
    assert anchored.loc[anchored.order_ids == "2", "end"].tolist() == [39.5]
    assert anchored.loc[anchored.order_ids == "3", "start"].tolist() == [100]
    assert chained.loc[chained.order_ids == "3", "start"].tolist() == [109.5]


def test_a_line_pick_time_model_sets_each_pick_in_execution(tmp_path_factory):
    # Pick 1 s + 3 s per package, one package per line: 4 s per pick. Order 1's tour:
    # setup 10-15, walk 3, pick 18-22, walk 2, pick 24-28, back 5, end 33.
    picks, tours, _ = _run(tmp_path_factory, "source.time_per_pick=1.0", "source.per_unit_s=3.0")
    first = picks[picks.tour_id == picks.tour_id.iloc[0]]
    assert first[["pick_start", "pick_end"]].values.tolist() == [[18, 22], [24, 28]]
    assert tours.loc[tours.order_ids == "1", "end"].tolist() == [33]


def test_time_on_another_order_is_not_a_gap():
    # Order 1: lines 0-10 and 400-410; order 2 in between, 10-400. Order 1's
    # tour has a 390 s gap between its lines, but the picker had order 2's line
    # open throughout, so there is no stop.
    from casim.sources.recorded import RecordedReplay, RecordedRow, TimeModel

    rows = [RecordedRow(picker=0, order=o, article=a, quantity=1, node=(1, 1),
                        start_s=s, end_s=e)
            for a, (o, s, e) in enumerate([(1, 0, 10), (2, 10, 400), (1, 400, 410)])]
    replay = RecordedReplay(rows, None, TimeModel(speed=1.0, time_per_pick=1000.0,
                                                  tour_setup_time=5.0),
                            shift_start_s=0.0, pause_threshold_s=300.0, stops_in_tours=True)
    assert [[r.order for r in lines] for _, lines in replay.tours] == [[1, 1], [2]]
    assert replay._pauses() == []
    assert replay.summary.gaps_inside_tours == 0


def test_anchored_setup_can_be_placed_before_the_first_wms_line(tmp_path_factory):
    # First recorded line begins at 10, setup lasts 5, so the first tour's
    # setup begins at 5 and its first modeled walk begins at 10.
    picks, tours, _ = _run(tmp_path_factory, "variants=anchored",
                          "source.setup_in_gap=true")
    assert tours.loc[tours.order_ids == "1", ["start", "setup_end"]].values.tolist() == [[5, 10]]
    assert picks.iloc[0][["walk_start", "pick_start", "pick_end"]].tolist() == [10, 13, 15]


def _kpis(tmp_path_factory, *overrides):
    import json
    output_dir = tmp_path_factory.mktemp("recorded_replay")
    run(compose_scenario(SCENARIO, list(overrides), output_dir=output_dir))
    return (json.loads((output_dir / "kpis.json").read_text(encoding="utf-8")),
            pd.read_csv(output_dir / "facts" / "tours.csv", dtype={"order_ids": str}),
            pd.read_csv(output_dir / "facts" / "orders.csv"))


def test_recorded_presence_strands_work_a_late_picker_cannot_reach(tmp_path_factory):
    # Picker 0 arrives at 10 and does order 1 (10-29) and order 2 (29-44) one
    # after the other, 14 s past its recorded end 30. Its 70 s pause, requested
    # at 30, runs 44-114; its departure after the last recorded line (110) then
    # takes effect, so order 3's tour never starts. Absent: picker 0 before 10
    # and during the pause; picker 1 before 40 and after its tour (55-114).
    kpis, tours, orders = _kpis(tmp_path_factory, "variants=recorded_presence")
    assert tours.loc[tours.start.isna(), "order_ids"].tolist() == ["3"]
    assert kpis["absent_time_by_picker"] == {"0": 80.0, "1": 99.0}
    assert kpis["busy_time_by_picker"] == {"0": 34.0, "1": 15.0}
    assert orders.loc[orders.completion.isna(), "order_id"].tolist() == [3]


def test_recorded_presence_leaves_work_open_after_the_last_recorded_line(tmp_path_factory):
    # Pick 40 s: order 1's tour runs 10-105 (setup 5, walks 3, 2 and 5, two
    # picks of 40). Its pause (requested at 30) and its departure after the last
    # recorded line (110) take effect in order: the pause 105-175, then the
    # departure, without return, so the tours of orders 2 and 3 never start. The
    # run ends at 175: picker 0 is absent 10 + 70, picker 1 before 40 and after
    # its tour, 93-175. With the shift-long presence the picker returns at 175
    # and picks orders 2 and 3.
    kpis, tours, orders = _kpis(tmp_path_factory, "variants=recorded_presence",
                                "source.time_per_pick=40.0")
    started = tours[tours.start.notna()]
    assert started[["picker_id", "start", "end", "order_ids"]].values.tolist() == [
        [0, 10, 105, "1"], [1, 40, 93, "4"]]
    # The tours committed to picker 0, who left, stay in the table, scheduled
    # and never started.
    stranded = tours[tours.start.isna()]
    assert stranded[["picker_id", "status", "order_ids"]].values.tolist() == [
        [0, "scheduled", "2"], [0, "scheduled", "3"]]
    assert orders.loc[orders.completion.isna(), "order_id"].tolist() == [2, 3]
    assert kpis["absent_time_by_picker"] == {"0": 80.0, "1": 122.0}
    _, shift, _ = _kpis(tmp_path_factory, "source.time_per_pick=40.0")
    assert shift.loc[shift.order_ids == "3", ["start", "end"]].values.tolist() == [[228, 279]]


def test_arrival_presence_never_forces_a_departure(tmp_path_factory):
    # Pick 40 s as above, with presence "arrival": picker 0 arrives at 10, its
    # pause (70 s) runs after order 1's tour, 105-175, and it picks order 2 at
    # 175-228 and order 3 at 228-279, as with the shift-long presence. Absent: picker 0 before 10 and
    # during the pause; picker 1 only before 40.
    kpis, tours, orders = _kpis(tmp_path_factory, "variants=recorded_presence",
                                "source.presence=arrival", "source.time_per_pick=40.0")
    assert tours[["picker_id", "start", "end", "order_ids"]].values.tolist() == [
        [0, 10, 105, "1"], [0, 175, 228, "2"], [1, 40, 93, "4"], [0, 228, 279, "3"]]
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
