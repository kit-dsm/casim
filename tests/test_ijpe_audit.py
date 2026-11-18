"""The derivations of the IJPE data audit on a hand-built line table (no WMS data needed)."""

import pandas as pd
import pytest

from casim.sources import wms_export as export
from scripts.ijpe import audit


def _lines(rows):
    frame = pd.DataFrame(rows, columns=["order", "date", "start_clock", "end_clock", "picker",
                                        "area", "customer", "position"])
    frame.insert(0, "row", range(len(frame)))
    frame["date"] = pd.to_datetime(frame["date"])
    frame["start"] = frame["date"] + pd.to_timedelta(frame["start_clock"])
    frame["end"] = frame["date"] + pd.to_timedelta(frame["end_clock"])
    return frame


LINES = _lines([
    # order A on day 1, one split line booked a day later inside A's interval
    ("A", "2026-04-13", "06:00:00", "06:01:00", "p1", 1, "c1", 1),
    ("A", "2026-04-13", "06:01:00", "06:02:00", "p1", 1, "c1", 2),
    ("A", "2026-04-14", "06:01:30", "06:01:40", "p1", 1, "c1", 2),
    ("A", "2026-04-13", "06:05:00", "06:06:00", "p1", 1, "c1", 3),
    # order B of the same picker opens while A is open (06:04-06:08)
    ("B", "2026-04-13", "06:04:00", "06:04:30", "p1", 7, "c1", 1),
    ("B", "2026-04-13", "06:07:00", "06:08:00", "p1", 7, "c1", 2),
])


def test_minority_booking_date_moves_to_the_order_pick_date():
    frame = audit.with_pick_date(LINES)
    assert frame["rebooked"].tolist() == [False, False, True, False, False, False]
    moved = frame.loc[frame["rebooked"]].iloc[0]
    assert moved["pick_date"] == pd.Timestamp("2026-04-13")
    assert moved["start"] == pd.Timestamp("2026-04-13 06:01:30")


def test_union_counts_overlapping_time_once():
    intervals = audit.with_pick_date(LINES)[["start", "end"]]
    # 06:00-06:02, 06:04-06:04:30, 06:05-06:06, 06:07-06:08
    assert audit._union(intervals) == pd.Timedelta(minutes=4, seconds=30)


def test_concurrency_splits_open_time_by_number_of_open_orders():
    table = audit.concurrency(LINES)
    # A open 06:00-06:06, B open 06:04-06:08: one order 6 min, two orders 2 min
    assert table.loc[1, "hours"] * 60 == pytest.approx(6)
    assert table.loc[2, "hours"] * 60 == pytest.approx(2)


def test_excluded_article_is_matched_by_hash_only():
    articles = pd.Series(["123", "456"])
    assert not export.is_excluded_article(articles).any()
