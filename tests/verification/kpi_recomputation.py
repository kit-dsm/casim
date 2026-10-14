"""Recompute tour KPIs independently of ExperimentTracker.

The tracker records ``process_times[i] = end - start`` where ``start`` is the
TourStart event time (after setup, per ``tour_manager.start_tour`` line 105).
So ``total_tour_times`` and ``avg_tour_makespan`` exclude setup time.  This
module mirrors that definition and also reports ``occupation_including_setup``
separately, so the setup gap is visible without being flagged as an
inconsistency.

Distance is recomputed from the independently recorded route legs.  When a
leg's distance is ``None`` (marked not recomputable by the capture), the
distance KPI is reported as ``"not_recomputable"`` with the reason, and no
distance comparison is made.

Time model (from the design document, Section 7.3)::

    end = start + s + d / v + n * t_pick

where *s* is ``tour_setup_time``, *d* the route distance, *v* the picker speed
and *n* the number of picks.  The tracker's ``end - start`` excludes *s*
because ``start`` is the TourStart time (after setup), so the recomputation
also excludes it.  ``occupation_including_setup`` adds it back.
"""

from __future__ import annotations

from collections import defaultdict
from math import isclose


def recompute_kpis(capture: dict, tracker: dict) -> dict[str, dict[str, object]]:
    """Return every tracker KPI that disagrees with captured tour facts."""
    tours = capture["completed_tours"]
    legs = capture["route_legs"]
    order_arrivals = capture["order_arrivals"]

    by_tour = defaultdict(float)
    by_picker = defaultdict(float)
    not_recomputable = []
    for leg in legs:
        if leg["distance"] is None:
            not_recomputable.append(leg.get("recompute_reason", "unknown"))
            continue
        distance = leg["distance"] * leg["completed_fraction"]
        by_tour[leg["tour_id"]] += distance
        by_picker[str(leg["picker_id"])] += distance

    completion = []
    for tour in tours:
        for order_id in tour["order_ids"]:
            arrival = order_arrivals.get(str(order_id))
            if arrival is not None:
                completion.append(tour["end"] - arrival)
    durations = [tour["end"] - tour["start"] for tour in tours]
    occupation = [
        tour["tour_setup_time"] + by_tour[tour["tour_id"]] / tour["speed"]
        + tour["picks"] * tour["time_per_pick"]
        for tour in tours
    ]

    distance_recomputable = not not_recomputable
    computed = {
        "num_tours": len(tours),
        "num_orders_completed": len(completion),
        "makespan": max((tour["end"] for tour in tours), default=0.0),
        "mean_order_completion_time": (sum(completion) / len(completion)
                                       if completion else 0.0),
        "total_tour_times": sum(durations),
        "avg_tour_makespan": sum(durations) / len(durations) if durations else 0.0,
    }
    if distance_recomputable:
        computed["total_distance"] = sum(by_tour.values())
        computed["distance_by_picker"] = dict(by_picker)
    else:
        computed["total_distance"] = "not_recomputable"
        computed["distance_by_picker"] = "not_recomputable"

    differences = {}
    for name, independent in computed.items():
        reported = tracker.get(name)
        if isinstance(independent, str):
            differences[name] = {"independent": independent, "tracker": reported}
            continue
        if isinstance(independent, dict):
            keys = set(independent) | set(reported or {})
            equal = all(isclose(independent.get(key, 0.0),
                                (reported or {}).get(key, 0.0),
                                abs_tol=1e-6, rel_tol=1e-9) for key in keys)
        else:
            equal = isclose(independent, reported, abs_tol=1e-6, rel_tol=1e-9)
        if not equal:
            differences[name] = {"independent": independent, "tracker": reported}

    if distance_recomputable and isclose(sum(occupation), computed["total_tour_times"],
                                          abs_tol=1e-6, rel_tol=1e-9):
        pass
    else:
        differences["occupation_including_setup"] = {
            "independent": sum(occupation),
            "recorded_tour_times": computed["total_tour_times"],
            "setup_gap": sum(tour["tour_setup_time"] for tour in tours),
            "note": ("Tracker total_tour_times excludes setup (start is "
                     "TourStart time, after setup). occupation_including_setup "
                     "adds tour_setup_time per tour. Difference is a definition "
                     "mismatch, not an inconsistency. Documented for Task 4.14."),
        }

    if not_recomputable:
        differences["distance_not_recomputable"] = {
            "reasons": list(set(not_recomputable)),
            "count": len(not_recomputable),
        }

    return differences
