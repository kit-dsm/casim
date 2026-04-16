"""Snapshot-derived KPIs. Pure functions of a single snapshot + static data."""
from ware_ops_algos.algorithms import TourStates


def outstanding_count(snap) -> int:
    """Orders in the order buffer, not yet batched."""
    return len(snap["buffered_order_ids"])


def batched_count(snap) -> int:
    """Orders in pick lists, not yet assigned to a tour."""
    return sum(len(pl) for pl in snap["pick_list_buffer"])


def tours_by_status(snap) -> dict:
    """Count of tours grouped by status."""
    counts = {}
    for t in snap["tours"].values():
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    return counts


def in_progress_order_count(snap) -> int:
    return sum(
        len(t["order_ids"]) for t in snap["tours"].values()
        if t["status"] == TourStates.STARTED
    )


def completed_order_count(snap) -> int:
    return sum(
        len(t["order_ids"]) for t in snap["tours"].values()
        if t["status"] == TourStates.DONE
    )


def active_tour_for_picker(snap, picker_id: int):
    """Returns the active tour dict for a picker, or None."""
    tour_id = snap["active_picker_tour"].get(picker_id)
    if tour_id is None:
        return None
    return snap["tours"].get(tour_id)


def active_picklist_order_ids(snap, picker_id: int) -> list[int]:
    """Order ids currently being picked by this picker."""
    tour = active_tour_for_picker(snap, picker_id)
    return list(tour["order_ids"]) if tour else []


def order_status(snap, order_id: int) -> str:
    """Derived status label for one order at this snapshot."""
    if order_id in snap["buffered_order_ids"]:
        return "outstanding"
    for pl in snap["pick_list_buffer"]:
        if order_id in pl:
            return "batched"
    for tour in snap["tours"].values():
        if order_id in tour["order_ids"]:
            s = tour["status"]
            if s == TourStates.DONE:
                return "completed"
            if s == TourStates.STARTED:
                return "in_progress"
            return "assigned"
    return "unknown"


def order_status_map(snap) -> dict[int, str]:
    """All order ids → status, computed in one pass."""
    result = {}
    for oid in snap["buffered_order_ids"]:
        result[oid] = "outstanding"
    for pl in snap["pick_list_buffer"]:
        for oid in pl:
            result[oid] = "batched"
    for tour in snap["tours"].values():
        s = tour["status"]
        if s == TourStates.DONE:
            label = "completed"
        elif s == TourStates.STARTED:
            label = "in_progress"
        else:
            label = "assigned"
        for oid in tour["order_ids"]:
            result[oid] = label
    return result


def summary(snap) -> dict:
    """One-shot KPI bundle for the header/panel."""
    return {
        "outstanding": outstanding_count(snap),
        "batched": batched_count(snap),
        "in_progress": in_progress_order_count(snap),
        "completed": completed_order_count(snap),
        "tours_by_status": tours_by_status(snap),
    }