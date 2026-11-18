"""Service measures calculated from realised orders and picker state."""

from __future__ import annotations


def _orders(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["completion"] is not None]
    timed = [row for row in completed if row["due_date"] is not None]
    tardiness = [max(0.0, row["completion"] - row["due_date"])
                 for row in timed]
    return {
        "orders": len(rows),
        "completed_orders": len(completed),
        "open_orders": len(rows) - len(completed),
        "orders_with_due_date": sum(row["due_date"] is not None for row in rows),
        "on_time_rate": (sum(delay == 0 for delay in tardiness) / len(timed)
                         if timed else None),
        "total_tardiness_s": sum(tardiness),
        "maximum_tardiness_s": max(tardiness, default=0.0),
        "makespan_s": max((row["completion"] for row in completed), default=None),
    }


def service_measures(rows: list[dict], tracker) -> dict:
    """Keep affected and unaffected denominators explicit.

    On-time rate uses completed orders with a due date. Open orders are
    reported separately; a run that ends with open work cannot silently call
    those orders on time. Picker presence is busy plus idle time. Absence is
    reported outside that denominator.
    """
    groups = {
        "all": _orders(rows),
        "affected": _orders([row for row in rows if row["affected"]]),
        "unaffected": _orders([row for row in rows if not row["affected"]]),
    }
    busy = tracker.busy_time_by_picker
    idle = tracker.idle_time_by_picker
    absent = tracker.absent_time_by_picker
    picker_ids = set(busy) | set(idle) | set(absent)
    picker_rows = []
    for picker in sorted(picker_ids):
        busy_s, idle_s, absent_s = (busy.get(picker, 0.0), idle.get(picker, 0.0),
                                   absent.get(picker, 0.0))
        presence_s = busy_s + idle_s
        picker_rows.append({"picker_id": picker, "presence_s": presence_s,
                            "busy_s": busy_s, "idle_s": idle_s,
                            "absent_s": absent_s,
                            "busy_share_of_presence": (busy_s / presence_s
                                                       if presence_s else None)})
    return {"orders": groups, "pickers": picker_rows}
